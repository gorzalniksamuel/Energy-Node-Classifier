import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import os
import timm


# Build selected classification model (swin needs different image format - 1024 the others use 1280x1280)
def build_classification_model(model_key: str, num_classes: int):
    timm_names = {
        "effnet": "tf_efficientnetv2_m",
        "resnet": "resnet50",
        "swin": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        "convnext_large": "convnext_large"
    }

    model_name = timm_names.get(model_key, "convnext_large")
    print(f"Creating TIMM model: {model_name} with {num_classes} classes")

    kwargs = {}

    # Swin was trained at 1024x1024
    if "swin" in model_key:
        kwargs["img_size"] = 1024

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        **kwargs
    )
    return model

def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensors to NCHW for Grad-CAM math.
    Some timm models/hooks return activations in NHWC.
    """
    if x is None:
        return x

    if x.dim() == 4:
        b, d1, d2, d3 = x.shape
        # Heuristic: NHWC looks like (B, H, W, C) where C is relatively large
        # and H/W are spatial-ish.
        if d3 > 8 and d1 <= 1024 and d2 <= 1024:
            return x.permute(0, 3, 1, 2).contiguous()

    return x

def resolve_effnetv2_target_layer(model: nn.Module) -> nn.Module | None:
    """
    EffNetV2 CAM layer:
    Prefer a higher-resolution feature stage than the last one.
    Empirically, blocks[-2] often localizes better than blocks[-1] for small facilities.
    """
    if hasattr(model, "blocks"):
        try:
            blocks = model.blocks
            if len(blocks) >= 2:
                return blocks[-2]  # higher spatial resolution than last stage
            return blocks[-1]
        except Exception:
            pass

    if hasattr(model, "conv_head"):
        return model.conv_head

    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d):
            return m
    return None

def _normalize_cam_percentile(cam: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """
    Robust CAM normalization:
    - subtract min
    - divide by high percentile (not max) to prevent single-pixel domination
    - clip to [0,1]
    """
    cam = cam.astype(np.float32)
    cam -= float(cam.min())
    denom = float(np.percentile(cam, pct))
    if denom <= 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    cam = np.clip(cam / denom, 0.0, 1.0)
    return cam


def _smooth_cam(cam: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Mild spatial smoothing for prettier/less noisy CAMs.
    """
    if sigma <= 0:
        return cam.astype(np.float32)
    return cv2.GaussianBlur(cam.astype(np.float32), (0, 0), sigmaX=float(sigma))

# GradCAM for classifier explanations
class GradCAM:
    """
    Robust Grad-CAM:
    - captures forward activations
    - captures gradients w.r.t. those activations using tensor.register_hook
      (more reliable than module backward hooks on timm blocks)
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._printed = False
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        print("[DBG] hook fired, output requires_grad:", getattr(output, "requires_grad", None))
        # output should be a Tensor (ideally B,C,H,W)
        self.activations = output
        self.gradients = None

        # Register gradient hook directly on the activation tensor (key fix)
        if isinstance(output, torch.Tensor) and output.requires_grad:
            def _grad_hook(grad):
                self.gradients = grad
                if not self._printed:
                    try:
                        print(f"[GradCAM] act={tuple(self.activations.shape)} grad={tuple(self.gradients.shape)}")
                    except Exception:
                        pass
                    self._printed = True
            output.register_hook(_grad_hook)

        if not self._printed:
            try:
                shp = getattr(output, "shape", None)
                print(f"[GradCAM] activation shape: {tuple(shp) if shp is not None else shp}")
            except Exception:
                pass

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        # hard-guard: GradCAM needs gradients even in an inference service
        with torch.enable_grad():
            if not input_tensor.requires_grad:
                input_tensor = input_tensor.requires_grad_(True)

            self.model.zero_grad(set_to_none=True)

            logits = self.model(input_tensor)

            if class_idx is None:
                class_idx = int(torch.argmax(logits, dim=1).item())

            score = logits[0, class_idx]
            score.backward(retain_graph=True)

            acts = self.activations
            grads = self.gradients

        if not isinstance(acts, torch.Tensor) or not isinstance(grads, torch.Tensor):
            # fallback blank map
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Ensure NCHW
        if acts.dim() == 4 and acts.shape[-1] > 8 and acts.shape[1] <= 8:
            acts = acts.permute(0, 3, 1, 2).contiguous()
        if grads.dim() == 4 and grads.shape[-1] > 8 and grads.shape[1] <= 8:
            grads = grads.permute(0, 3, 1, 2).contiguous()

        if acts.dim() != 4 or grads.dim() != 4:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # ---- LayerCAM (usually tighter than classic Grad-CAM on EfficientNet) ----
        # pixel-wise channel weighting: A * ReLU(dY/dA)
        cam = (acts * torch.relu(grads)).sum(dim=1)  # (B,H,W)
        cam = torch.relu(cam)[0]

        cam = cam.detach().cpu().numpy().astype(np.float32)

        # robust normalization (percentile beats max for spiky cams)
        cam -= float(cam.min())
        denom = float(np.percentile(cam, 99.0))
        if denom > 1e-8:
            cam = np.clip(cam / denom, 0.0, 1.0)
        else:
            cam[:] = 0.0

        # mild smoothing for visual stability
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=2.0)

        return cam

class SwinGradCAM:
    """
    Swin-only Grad-CAM.
    Keeps existing GradCAM untouched for CNN-based models.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, reshape_transform):
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform

        self.activations = None
        self.gradients = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        self.gradients = None

        if isinstance(output, torch.Tensor) and output.requires_grad:
            def _grad_hook(grad):
                self.gradients = grad
            output.register_hook(_grad_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        with torch.enable_grad():
            if not input_tensor.requires_grad:
                input_tensor = input_tensor.requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            logits = self.model(input_tensor)

            if class_idx is None:
                class_idx = int(torch.argmax(logits, dim=1).item())

            score = logits[0, class_idx]
            score.backward(retain_graph=True)

            acts = self.activations
            grads = self.gradients

        if acts is None or grads is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Swin token output -> spatial map
        acts = self.reshape_transform(acts)
        grads = self.reshape_transform(grads)

        if acts.dim() != 4 or grads.dim() != 4:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        cam = (acts * torch.relu(grads)).sum(dim=1)
        cam = torch.relu(cam)[0]

        cam = cam.detach().cpu().numpy().astype(np.float32)

        cam -= float(cam.min())
        denom = float(np.percentile(cam, 99.0))
        if denom > 1e-8:
            cam = np.clip(cam / denom, 0.0, 1.0)
        else:
            cam[:] = 0.0

        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=2.0)
        return cam


# Classification wrapper
class EnergyClassifier:
    def __init__(self, model_dir: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.models = {}

        # Standard (resnet, effnet, ...) transform (1280x1208px)
        self.transform_standard = A.Compose([
            A.Resize(height=1280, width=1280, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # Swin transform (1024x1024px)
        self.transform_swin = A.Compose([
            A.Resize(height=1024, width=1024),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    # Helper function for Swin GradCAM reshaping
    def swin_reshape_transform(self, tensor):
        # Handle (B, L, C)
        if tensor.dim() == 3:
            B, L, C = tensor.shape
            s = int(L ** 0.5)
            if s * s != L:
                raise ValueError(f"Cannot reshape Swin tokens: sequence length {L} is not a square")
            return tensor.transpose(1, 2).reshape(B, C, s, s)

        # Handle (B, H, W, C)
        if tensor.dim() == 4 and tensor.shape[-1] > 8:
            return tensor.permute(0, 3, 1, 2).contiguous()

        # Already (B, C, H, W)
        return tensor


    def load_model(self, model_key: str):
        if model_key in self.models:
            return self.models[model_key]

        filename_map = {
            "effnet": "best_effnet.pt",
            "resnet": "best_resnet.pt",
            "swin": "best_swin.pt",
            "convnext_large": "best_convnext.pt"
        }

        filename = filename_map.get(model_key, "best_model.pt")
        path = os.path.join(self.model_dir, filename)

        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return None

        try:
            print(f"Loading Model: {filename}...")
            checkpoint = torch.load(path, map_location=self.device)

            state_dict = None
            class_names = []

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint: state_dict = checkpoint["state_dict"]
                elif "model_state" in checkpoint: state_dict = checkpoint["model_state"]
                else: state_dict = checkpoint
                class_names = checkpoint.get("class_names", [])
            else:
                state_dict = checkpoint

            if state_dict is None:
                raise ValueError("Could not find state_dict")

            # Dynamic class mapping
            num_classes = 10
            for key in ["classifier.bias", "fc.bias", "head.bias", "head.fc.bias"]:
                if key in state_dict:
                    num_classes = state_dict[key].shape[0]
                    break

            # Fallback class names if not in checkpoint
            if not class_names or class_names[0].startswith("Class_"):
                standard_10 = [
                    "Biomass", "Coal", "Compressor Metering Station", "Gas Plant", "Hydro",
                    "Industry", "Nuclear", "Solar", "Substation", "Wind"
                ]
                if num_classes == 10:
                    class_names = standard_10
                elif num_classes == 11:
                    class_names = sorted(standard_10 + ["NEG"])
                else:
                    class_names = [f"Class_{i}" for i in range(num_classes)]

            print(f"Model {model_key} loaded with {num_classes} classes: {class_names}")

            # Build Model Architecture
            model = build_classification_model(model_key, num_classes)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing: print(f"Warning: Missing keys: {missing[:3]}...")

            model.to(self.device)
            model.eval()

            grad_cam = None
            target_layer = None

            try:
                if "effnet" in model_key:
                    target_layer = resolve_effnetv2_target_layer(model)
                elif "resnet" in model_key:
                    target_layer = model.layer4[-1]
                elif "convnext" in model_key:
                    if hasattr(model, "stages"):
                        target_layer = model.stages[-1].blocks[-1]
                    else:
                        target_layer = list(model.modules())[-2]
                elif "swin" in model_key:
                    if hasattr(model, "layers"):
                        target_layer = model.layers[-1].blocks[-1]
            except Exception as e:
                print(f"Could not automatically resolve GradCAM layer: {e}")

            if target_layer is not None:
                if "swin" in model_key:
                    grad_cam = SwinGradCAM(
                        model,
                        target_layer,
                        reshape_transform=self.swin_reshape_transform
                    )
                else:
                    grad_cam = GradCAM(model, target_layer)


            entry = {"model": model, "grad_cam": grad_cam, "class_names": class_names}
            self.models[model_key] = entry
            return entry
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def predict(self, image: Image.Image, model_key="convnext", gradcam_all: bool = False):
        entry = self.load_model(model_key)
        if not entry:
            return [], None, None, None  # (results, overlay_pil, heatmap_top_pil, heatmaps_by_label)

        model = entry["model"]
        grad_cam = entry["grad_cam"]
        class_names = entry["class_names"]

        img_np = np.array(image.convert("RGB"))

        # transforms
        if "swin" in model_key:
            augmented = self.transform_swin(image=img_np)
        else:
            augmented = self.transform_standard(image=img_np)

        img_tensor = augmented["image"].unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True

        # forward
        logits = model(img_tensor)

        # multi-label => sigmoid
        probs = torch.sigmoid(logits)[0]  # shape: (num_classes,)

        # results list
        results = []
        for i, score in enumerate(probs):
            if i < len(class_names):
                results.append({"label": class_names[i], "score": float(score.detach().cpu())})
        results.sort(key=lambda x: x["score"], reverse=True)

        overlay_pil = None
        heatmap_top_pil = None
        heatmaps_by_label = None  # dict[str, PIL.Image]

        if grad_cam:
            try:
                H, W = img_np.shape[0], img_np.shape[1]

                # Without negative class => CAM on low confidence images -> noise
                top_score = float(probs.max().detach().cpu())
                MIN_CAM_SCORE = 0.75  # tune 0.60–0.85

                # If user asked for gradcam_all, don't gate (they explicitly want all maps)
                if (not gradcam_all) and (top_score < MIN_CAM_SCORE):
                    # no heatmap generated
                    return results, overlay_pil, None, None

                # Aggregate Grad-CAM over top-k classes
                top_k = 1
                top_vals, top_idxs = torch.topk(probs, k=min(top_k, probs.numel()))

                # Sharpen weights so top-1 dominates if you ever raise top_k
                w = top_vals.detach().cpu().numpy().astype(np.float32)
                gamma = 4.0
                w = np.power(w, gamma)
                wsum = float(w.sum())
                if wsum <= 1e-8:
                    w = np.ones_like(w, dtype=np.float32)
                    wsum = float(w.sum())
                w /= wsum

                agg = np.zeros((H, W), dtype=np.float32)

                for weight, idx in zip(w, top_idxs.detach().cpu().numpy().astype(int).tolist()):
                    cam = grad_cam(img_tensor, int(idx))  # raw cam (Hc,Wc)

                    if cam.shape != (H, W):
                        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

                    # robust normalize + smooth
                    cam = _normalize_cam_percentile(cam, pct=99.0)
                    cam = _smooth_cam(cam, sigma=2.0)

                    agg += float(weight) * cam

                # normalize final aggregate robustly
                agg = _normalize_cam_percentile(agg, pct=99.0)
                agg = _smooth_cam(agg, sigma=1.5)

                # render overlay
                heat_uint8 = np.uint8(255 * agg)
                heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
                heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

                alpha = 0.45
                overlay = (
                        img_np.astype(np.float32) * (1 - alpha) +
                        heat_rgb.astype(np.float32) * alpha
                ).astype(np.uint8)

                heatmap_top_pil = Image.fromarray(overlay)

                # OPTIONAL debug: per-class maps (keeps your existing functionality)
                if gradcam_all:
                    heatmaps_by_label = {}
                    for idx, name in enumerate(class_names):
                        hh = grad_cam(img_tensor, int(idx))
                        if hh.shape != (H, W):
                            hh = cv2.resize(hh, (W, H), interpolation=cv2.INTER_LINEAR)

                        hh = _normalize_cam_percentile(hh, pct=99.0)
                        hh = _smooth_cam(hh, sigma=2.0)

                        hh_uint8 = np.uint8(255 * hh)
                        hh_bgr = cv2.applyColorMap(hh_uint8, cv2.COLORMAP_JET)
                        hh_rgb = cv2.cvtColor(hh_bgr, cv2.COLOR_BGR2RGB)

                        overlay_i = (
                                img_np.astype(np.float32) * (1 - alpha) +
                                hh_rgb.astype(np.float32) * alpha
                        ).astype(np.uint8)

                        heatmaps_by_label[name] = Image.fromarray(overlay_i)

            except Exception as e:
                print(f"GradCAM generation failed: {e}")

        return results, overlay_pil, heatmap_top_pil, heatmaps_by_label

# Object detection wrapper
class ObjectDetector:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = {}
        # We map the long class names to shorter ones such that they dont block too much content in the image
        self.short_names = {
            "Tank or Silo for Fuel or By-Products": "Tank/Silo",
            "Biogas Dome": "Biogas",
            "Air Coolers": "Cooler",
            "Nuclear Containment Building": "Nuclear CB",
            "Pylon": "Pylon",
            "Inlet Scrubber or Absorber or Pig Receivers or Metering": "Gas Piping",
            "Water Tank or Pond": "Water Tank",
            "Chimney Side": "Chimney", # Both chimney classes will be mapped to the same output label
            "Chimney Top": "Chimney",
            "Gas Compressor": "Compressor",
            "Electrical Substation": "Substation",
            "Solar Array": "Solar",
            "Wind Turbine": "Wind",
            "Coal Heap": "Coal",
            "Conveyor": "Conveyor",
            "Cooling Tower": "Cooling Tower"
        }

    def load_model(self, model_key: str):
        if model_key in self.models:
            return self.models[model_key]

        filename_map = {
            "yolo11": "best_yolo11.pt",
            "yolo26": "best_yolo26.pt"
        }
        filename = filename_map.get(model_key, "best_yolo11.pt")
        path = os.path.join(self.model_dir, filename)

        if not os.path.exists(path):
            print(f"YOLO Model file not found: {path}")
            return None

        try:
            print(f"Loading YOLO Model: {filename}...")
            model = YOLO(path)
            self.models[model_key] = model
            return model
        except Exception as e:
            print(f"Error loading YOLO {filename}: {e}")
            return None

    def predict(self, image: Image.Image, model_key="yolo11"):
        model = self.load_model(model_key)
        if not model:
            return [], None

        try:
            # Run inference
            results = model.predict(image, conf=0.25, verbose=False)
            if not results:
                return [], image

            result = results[0]

            # Apply abbreviations / short names dict to result.names
            if isinstance(getattr(result, "names", None), dict) and result.names:
                result.names = {
                    cls_id: self.short_names.get(name, name)
                    for cls_id, name in result.names.items()
                }

            detections: list[dict] = []

            # Helper: map class id -> label
            def _label_for(cls_id: int, fallback: str = "") -> str:
                names = getattr(result, "names", None)
                if isinstance(names, dict):
                    return names.get(cls_id, fallback or str(cls_id))
                return fallback or str(cls_id)

            # 1) PRIMARY extraction: result.boxes (most stable)
            try:
                boxes = getattr(result, "boxes", None)
                if boxes is not None:
                    # len(boxes) works in ultralytics Boxes
                    if len(boxes) > 0:
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        cls = boxes.cls.detach().cpu().numpy()
                        conf = boxes.conf.detach().cpu().numpy()

                        for i in range(len(xyxy)):
                            cls_id = int(cls[i])
                            detections.append({
                                "label": _label_for(cls_id),
                                "confidence": float(conf[i]),
                                "bbox": [
                                    float(xyxy[i][0]),
                                    float(xyxy[i][1]),
                                    float(xyxy[i][2]),
                                    float(xyxy[i][3]),
                                ],
                            })
            except Exception as e:
                print(f"[YOLO] boxes extraction failed: {e}")
                detections = []

            # 2) SECONDARY extraction: result.summary()
            if len(detections) == 0:
                try:
                    summ = result.summary()
                    if isinstance(summ, list):
                        for it in summ:
                            cls_id = int(it.get("class", -1))
                            confv = float(it.get("confidence", 0.0))
                            box = it.get("box", None)

                            label = _label_for(cls_id, it.get("name", str(cls_id)))

                            # Normalize bbox into [x1,y1,x2,y2]
                            if isinstance(box, dict):
                                bbox = [
                                    float(box.get("x1", 0.0)),
                                    float(box.get("y1", 0.0)),
                                    float(box.get("x2", 0.0)),
                                    float(box.get("y2", 0.0)),
                                ]
                            elif isinstance(box, (list, tuple)) and len(box) >= 4:
                                bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                            else:
                                bbox = [0.0, 0.0, 0.0, 0.0]

                            detections.append({
                                "label": label,
                                "confidence": confv,
                                "bbox": bbox,
                            })
                except Exception as e:
                    print(f"[YOLO] summary() extraction failed: {e}")
                    detections = []

            # 3) TERTIARY extraction: result.to_json() (new API)
            if len(detections) == 0:
                try:
                    import json as _json
                    if hasattr(result, "to_json"):
                        raw = result.to_json()
                        items = _json.loads(raw) if raw else []
                        for it in items:
                            cls_id = int(it.get("class", -1))
                            confv = float(it.get("confidence", 0.0))
                            box = it.get("box", {}) or {}
                            label = _label_for(cls_id, it.get("name", str(cls_id)))

                            detections.append({
                                "label": label,
                                "confidence": confv,
                                "bbox": [
                                    float(box.get("x1", 0.0)),
                                    float(box.get("y1", 0.0)),
                                    float(box.get("x2", 0.0)),
                                    float(box.get("y2", 0.0)),
                                ],
                            })

                        if len(detections) > 0:
                            print(f"[YOLO] to_json() fallback used: {len(detections)} dets")
                    else:
                        print("[YOLO] to_json() not available on this Ultralytics Results version")
                except Exception as e:
                    print(f"[YOLO] to_json() fallback failed: {e}")
                    detections = []

            # Log (make this consistent with actual boxes)
            try:
                boxes = getattr(result, "boxes", None)
                boxes_len = len(boxes) if boxes is not None else 0
            except Exception:
                boxes_len = -1
            print(f"[YOLO] boxes_len={boxes_len} | detections_json={len(detections)}")

            # Annotated image
            try:
                annotated_bgr = result.plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                annotated_pil = Image.fromarray(annotated_rgb)
            except Exception as e:
                print(f"[YOLO] plot() failed: {e}")
                annotated_pil = image

            return detections, annotated_pil

        except Exception as e:
            print(f"Det Inference Error: {e}")
            return [], image