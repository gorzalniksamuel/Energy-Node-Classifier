"""
- timm backbones (EffNet / ConvNeXt / SwinV2, etc.)
- Loss options: BCE / Focal / ASL
- Optional class-balanced sampling
- Early stopping on val macro-F1 @ 0.5
- Per-class threshold tuning on val, then evaluate test with tuned thresholds

SwinV2 regularization:
- Model init: drop_rate/drop_path_rate/attn_drop_rate
- Layer-wise LR decay
- Warmup and Cosine LR
- EMA of weights (very helpful for ViT/Swin)
- MixUp for multi-label (soft labels supported)

- Negative column -> 10 classes + implicit negatives (--drop_negative arg)
"""

import os
import json
import time
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

try:
    import timm
    HAVE_TIMM = True
except Exception:
    timm = None
    HAVE_TIMM = False


# repro
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# label csv read
def read_multilabel_csv(csv_path: str, drop_negative: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError(f"'filename' column missing in: {csv_path}")

    ignore = {"filename", "Unlabeled", "q", "susbt"}
    class_names = [c for c in df.columns if c not in ignore]

    if drop_negative and "Negative" in class_names:
        class_names = [c for c in class_names if c != "Negative"]

    if not class_names:
        raise ValueError(f"No class columns found after filtering. Columns were: {df.columns.tolist()}")

    return df, class_names


# lat/lon as group keys to avoid leakage
def extract_location_key(filename: str) -> str:
    parts = filename.split("_")
    lat = None
    lon = None
    for p in parts:
        if p.startswith("lat"):
            lat = p[3:]
        elif p.startswith("lon"):
            lon = p[3:]
    if lat is None or lon is None:
        return filename
    return f"{lat}_{lon}"


def split_modeling_test_grouped(df_all: pd.DataFrame, test_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_all.copy()
    df["group"] = df["filename"].apply(extract_location_key)
    groups = df["group"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    modeling_idx, test_idx = next(gss.split(df, groups=groups))
    df_modeling = df.iloc[modeling_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    return df_modeling, df_test


def assert_no_group_overlap(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str):
    ga = set(a["group"].astype(str).tolist())
    gb = set(b["group"].astype(str).tolist())
    inter = ga.intersection(gb)
    if inter:
        ex = next(iter(inter))
        raise RuntimeError(f"GROUP LEAKAGE between {name_a} and {name_b}: {len(inter)} overlapping groups. Example: {ex}")



# Dataset
class MultiLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_names: List[str], img_dir: str, transform: A.Compose):
        self.df = df.reset_index(drop=True)
        self.class_names = class_names
        self.img_dir = img_dir
        self.transform = transform

        self.paths = [os.path.join(img_dir, fn) for fn in self.df["filename"].tolist()]
        self.labels = self.df[class_names].values.astype(np.float32)

        missing = [p for p in self.paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing. Example: {missing[0]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        y = self.labels[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        aug = self.transform(image=img_np)
        x = aug["image"]
        return x, torch.from_numpy(y), os.path.basename(path)


# Albumentations transforms
def _random_resized_crop(img_size: int):
    try:
        return A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.70, 1.0),
            ratio=(0.90, 1.10),
            p=1.0,
        )
    except TypeError:
        return A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.70, 1.0),
            ratio=(0.90, 1.10),
            p=1.0,
        )


def _coarse_dropout(img_size: int):
    try:
        return A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(int(img_size * 0.03), int(img_size * 0.10)),
            hole_width_range=(int(img_size * 0.03), int(img_size * 0.10)),
            fill=0,
            p=0.35,
        )
    except TypeError:
        return A.CoarseDropout(
            max_holes=8,
            max_height=int(img_size * 0.10),
            max_width=int(img_size * 0.10),
            min_holes=1,
            fill_value=0,
            p=0.35,
        )


def build_transforms(img_size: int, strong: bool = False):
    train_list = [
        _random_resized_crop(img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.35),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(0.0, 0.03),
            rotate=(-10, 10),
            shear=(-8, 8),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.35,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.25),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=5, p=0.15),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.10),
    ]

    if strong:
        train_list += [
            _coarse_dropout(img_size),
            A.GaussianBlur(blur_limit=(3, 5), p=0.10),
        ]

    train_list += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    train_tfms = A.Compose(train_list)

    eval_tfms = A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tfms, eval_tfms


# SwinV2 regularization
def build_model(
        model_name: str,
        num_classes: int,
        img_size: int,
        drop_rate: float,
        drop_path_rate: float,
        attn_drop_rate: float,
) -> nn.Module:
    if not HAVE_TIMM:
        raise RuntimeError("timm not installed. pip install timm")

    # timm will ignore unknown args for many models; Swin/Vit use these
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        img_size=img_size,                 # important for some Swin/Vit variants
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        attn_drop_rate=attn_drop_rate,
    )
    return model


# Losses
def compute_pos_weight(train_labels: np.ndarray) -> torch.Tensor:
    eps = 1e-6
    P = train_labels.sum(axis=0)
    N = train_labels.shape[0] - P
    return torch.tensor((N + eps) / (P + eps), dtype=torch.float32)


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor([]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=(self.pos_weight if self.pos_weight.numel() else None),
            reduction="none",
        )
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        mod = (1 - pt).pow(self.gamma)
        return (mod * bce).mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gn = float(gamma_neg)
        self.gp = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()

        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)

        loss_pos = targets * torch.log(torch.clamp(xs_pos, min=self.eps))
        loss_neg = (1 - targets) * torch.log(torch.clamp(xs_neg, min=self.eps))
        loss = loss_pos + loss_neg

        pt = xs_pos * targets + xs_neg * (1 - targets)
        gamma = self.gp * targets + self.gn * (1 - targets)
        focal = (1 - pt).pow(gamma)

        return -(focal * loss).mean()



# MixUp (multi-label)
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y2 = y[idx]
    x_mix = lam * x + (1 - lam) * x2
    y_mix = lam * y + (1 - lam) * y2
    return x_mix, y_mix

# EMA
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = {}
        self._init(model)

    def _init(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    def apply_to(self, model: nn.Module):
        # returns backup to restore later
        backup = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            backup[n] = p.detach().clone()
            p.data.copy_(self.shadow[n].data)
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n].data)

# LLRD
def build_param_groups_llrd(model: nn.Module, base_lr: float, weight_decay: float, layer_decay: float):
    try:
        from timm.optim.optim_factory import param_groups_layer_decay
        return param_groups_layer_decay(model, weight_decay=weight_decay, layer_decay=layer_decay, no_weight_decay_list=[])
    except Exception:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": weight_decay}]

# Metrics
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "P": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "R": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def tune_thresholds_per_class(y_true: np.ndarray, y_prob: np.ndarray, step: float = 0.01) -> np.ndarray:
    C = y_true.shape[1]
    thresholds = np.zeros(C, dtype=np.float32)
    grid = np.arange(0.05, 0.95 + 1e-9, step)
    for c in range(C):
        best_f1, best_t = -1.0, 0.5
        yt = y_true[:, c].astype(int)
        yp = y_prob[:, c]
        for t in grid:
            yhat = (yp >= t).astype(int)
            f1 = f1_score(yt, yhat, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[c] = best_t
    return thresholds


def compute_metrics_with_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= thresholds[None, :]).astype(int)
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "P": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "R": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }


@torch.no_grad()
def eval_epoch(model, loader, device, use_amp: bool, loss_fn: Optional[nn.Module] = None):
    model.eval()
    loss_meter = 0.0
    n = 0
    probs_all, targets_all = [], []

    amp_enabled = use_amp and (device.type == "cuda")

    for x, y, _ in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(x)
            loss = (loss_fn(logits, y) if loss_fn is not None
                    else F.binary_cross_entropy_with_logits(logits, y))

        bs = x.size(0)
        loss_meter += float(loss.item()) * bs
        n += bs

        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

    return (loss_meter / max(1, n)), np.concatenate(probs_all, 0), np.concatenate(targets_all, 0)


def plot_curve(xs, ys, title, xlabel, ylabel, out_path):
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_balanced_sampler(labels: np.ndarray, pos_weight: np.ndarray) -> WeightedRandomSampler:
    y = labels.astype(np.float32)
    pw = pos_weight.astype(np.float32)
    w = np.ones((y.shape[0],), dtype=np.float32)
    for i in range(y.shape[0]):
        pos = np.where(y[i] > 0.5)[0]
        w[i] = float(np.max(pw[pos])) if len(pos) > 0 else 1.0
    w = w / (w.mean() + 1e-8)
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)


def train_fold(
        fold_dir: str,
        model_name: str,
        img_size: int,
        num_classes: int,
        class_names: List[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device,
        lr: float,
        weight_decay: float,
        epochs: int,
        patience: int,
        accumulation_steps: int,
        use_amp: bool,
        loss_name: str,
        pos_weight: torch.Tensor,
        focal_gamma: float,
        asl_gn: float,
        asl_gp: float,
        asl_clip: float,
        grad_clip: float,
        # Swin/transformer regs
        drop_rate: float,
        drop_path_rate: float,
        attn_drop_rate: float,
        layer_decay: float,
        warmup_epochs: int,
        ema_decay: float,
        mixup_alpha: float,
) -> Dict[str, float]:
    ensure_dir(fold_dir)

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        img_size=img_size,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        attn_drop_rate=attn_drop_rate,
    ).to(device)

    # loss
    if loss_name == "bce":
        loss_fn: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    elif loss_name == "focal":
        loss_fn = FocalLossWithLogits(gamma=focal_gamma, pos_weight=pos_weight.to(device))
    elif loss_name == "asl":
        loss_fn = AsymmetricLoss(gamma_neg=asl_gn, gamma_pos=asl_gp, clip=asl_clip)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    # optimizer + LLRD
    param_groups = build_param_groups_llrd(model, base_lr=lr, weight_decay=weight_decay, layer_decay=layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    # warmup + cosine
    def lr_lambda(ep: int):
        if warmup_epochs <= 0:
            return 1.0
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        # cosine for remaining
        t = (ep - warmup_epochs) / max(1, (epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    amp_enabled = use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    best_path = os.path.join(fold_dir, "best.pt")
    history_path = os.path.join(fold_dir, "history.jsonl")

    best_macro = -1.0
    bad = 0
    val_losses: List[float] = []
    val_macro: List[float] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss_meter = 0.0
        n = 0

        for step, (x, y, _) in enumerate(tqdm(train_loader, desc=f"Train e{epoch:02d}", leave=False), start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # MixUp (multi-label, soft labels)
            x, y = mixup_batch(x, y, mixup_alpha)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)
                loss = loss_fn(logits, y) / max(1, accumulation_steps)

            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            bs = x.size(0)
            loss_meter += float(loss.item()) * bs * max(1, accumulation_steps)
            n += bs

            if step % max(1, accumulation_steps) == 0:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(model)

        # flush partial accumulation
        if step % max(1, accumulation_steps) != 0:
            if amp_enabled:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        scheduler.step()

        train_loss = loss_meter / max(1, n)

        # Evaluate using EMA weights if enabled for val stability
        if ema is not None:
            backup = ema.apply_to(model)
            val_loss, val_prob, val_true = eval_epoch(model, val_loader, device, use_amp=use_amp, loss_fn=loss_fn)
            ema.restore(model, backup)
        else:
            val_loss, val_prob, val_true = eval_epoch(model, val_loader, device, use_amp=use_amp, loss_fn=loss_fn)

        m = compute_metrics(val_true, val_prob, threshold=0.5)

        val_losses.append(float(val_loss))
        val_macro.append(float(m["macro_f1"]))

        secs = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"micro_f1={m['micro_f1']:.4f} macro_f1={m['macro_f1']:.4f} | "
            f"P={m['P']:.4f} R={m['R']:.4f} | {secs:.1f}s"
        )

        rec = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            **m,
            "seconds": float(secs),
            "lr": float(optimizer.param_groups[0].get("lr", lr)),
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if m["macro_f1"] > best_macro:
            best_macro = float(m["macro_f1"])
            bad = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "img_size": img_size,
                    "best_val_macro_f1": best_macro,
                    "ema_decay": ema_decay,
                },
                best_path,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping: no macro_f1 improvement for {patience} epochs.")
                break

    xs = list(range(1, len(val_losses) + 1))
    plot_curve(xs, val_losses, "Val Loss", "Epoch", "Loss", os.path.join(fold_dir, "curve_val_loss.png"))
    plot_curve(xs, val_macro, "Val macro-F1 (thr=0.5)", "Epoch", "F1", os.path.join(fold_dir, "curve_val_macro_f1.png"))

    # load best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    best_model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        img_size=img_size,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        attn_drop_rate=attn_drop_rate,
    ).to(device)
    best_model.load_state_dict(ckpt["model_state"])
    best_model.eval()

    # tune thresholds on val set
    _, val_prob2, val_true2 = eval_epoch(best_model, val_loader, device, use_amp=use_amp, loss_fn=loss_fn)
    thresholds = tune_thresholds_per_class(val_true2, val_prob2, step=0.01)

    with open(os.path.join(fold_dir, "thresholds.json"), "w") as f:
        json.dump({c: float(t) for c, t in zip(class_names, thresholds.tolist())}, f, indent=2)

    # test with val thresholds on test set
    test_loss, test_prob, test_true = eval_epoch(best_model, test_loader, device, use_amp=use_amp, loss_fn=loss_fn)
    tm = compute_metrics_with_thresholds(test_true, test_prob, thresholds)

    out = {
        "loss": float(test_loss),
        "micro_f1_tuned": float(tm["micro_f1"]),
        "macro_f1_tuned": float(tm["macro_f1"]),
        "P_tuned": float(tm["P"]),
        "R_tuned": float(tm["R"]),
        "best_val_macro_f1_thr05": float(ckpt.get("best_val_macro_f1", float("nan"))),
    }
    with open(os.path.join(fold_dir, "test_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"[TEST tuned] loss={out['loss']:.4f} micro_f1={out['micro_f1_tuned']:.4f} "
        f"macro_f1={out['macro_f1_tuned']:.4f} P={out['P_tuned']:.4f} R={out['R_tuned']:.4f}"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run_name", required=True)

    # model
    ap.add_argument("--model", default="swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
                    help="Any timm model name. SwinV2 recommended.")
    ap.add_argument("--img_size", type=int, default=1024)

    # dataloader
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accumulation_steps", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)

    # training
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)  # transformers like higher wd
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # data flags
    ap.add_argument("--drop_negative", action="store_true")
    ap.add_argument("--strong_aug", action="store_true")
    ap.add_argument("--balanced_sampler", action="store_true")

    # imbalance handling
    ap.add_argument("--loss", choices=["bce", "focal", "asl"], default="asl")
    ap.add_argument("--pos_weight_cap", type=float, default=20.0)
    ap.add_argument("--pos_weight_sqrt", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--asl_gn", type=float, default=4.0)
    ap.add_argument("--asl_gp", type=float, default=1.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)

    # Swin/transformer regularization knobs
    ap.add_argument("--drop_rate", type=float, default=0.10)
    ap.add_argument("--drop_path_rate", type=float, default=0.20)     # stochastic depth
    ap.add_argument("--attn_drop_rate", type=float, default=0.0)
    ap.add_argument("--layer_decay", type=float, default=0.75)        # LLRD
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)

    args = ap.parse_args()

    if not HAVE_TIMM:
        raise SystemExit("timm is required. pip install timm")

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_dir = os.path.join(args.data_root, "train")
    csv_path = os.path.join(train_dir, "_classes.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}")

    df_all, class_names = read_multilabel_csv(csv_path, drop_negative=args.drop_negative)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    labels_all = df_all[class_names].values.astype(np.float32)
    neg_only = float((labels_all.sum(axis=1) == 0).mean())
    avg_labels = float(labels_all.sum(axis=1).mean())
    print(f"Dataset: n={len(df_all)} | all-zero={neg_only:.3f} | avg_labels/img={avg_labels:.2f}")

    df_all = df_all.copy()
    df_all["group"] = df_all["filename"].apply(extract_location_key)

    df_modeling, df_test = split_modeling_test_grouped(df_all, test_frac=args.test_frac, seed=args.seed)
    assert_no_group_overlap(df_modeling, df_test, "modeling", "test")

    print(f"Holdout test: n={len(df_test)} ({args.test_frac*100:.1f}%)")
    print(f"Modeling: n={len(df_modeling)} | n_folds={args.n_folds}")

    train_tfms, eval_tfms = build_transforms(args.img_size, strong=args.strong_aug)

    # test loader
    df_test_nog = df_test.drop(columns=["group"], errors="ignore")
    test_ds = MultiLabelDataset(df_test_nog, class_names, train_dir, eval_tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    run_dir = os.path.join(args.data_root, "runs", args.run_name)
    ensure_dir(run_dir)

    fold_results = []
    gkf = GroupKFold(n_splits=args.n_folds)
    groups = df_modeling["group"].values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df_modeling, groups=groups), start=1):
        print(f"\n========== Fold {fold}/{args.n_folds} ==========")

        # leakage guard
        tr_groups = set(df_modeling.iloc[tr_idx]["group"].astype(str).tolist())
        va_groups = set(df_modeling.iloc[va_idx]["group"].astype(str).tolist())
        if tr_groups.intersection(va_groups):
            raise RuntimeError(f"GROUP LEAKAGE in fold {fold}")

        df_tr = df_modeling.iloc[tr_idx].drop(columns=["group"], errors="ignore").reset_index(drop=True)
        df_va = df_modeling.iloc[va_idx].drop(columns=["group"], errors="ignore").reset_index(drop=True)

        tr_labels = df_tr[class_names].values.astype(np.float32)
        va_labels = df_va[class_names].values.astype(np.float32)

        pos_weight = compute_pos_weight(tr_labels)
        if args.pos_weight_sqrt:
            pos_weight = torch.sqrt(pos_weight)
        pos_weight = torch.clamp(pos_weight, max=args.pos_weight_cap)

        print(
            f"Train: n={len(df_tr)} | all-zero={float((tr_labels.sum(axis=1)==0).mean()):.3f} | "
            f"avg_labels={float(tr_labels.sum(axis=1).mean()):.2f}"
        )
        print(
            f"Val:   n={len(df_va)} | all-zero={float((va_labels.sum(axis=1)==0).mean()):.3f} | "
            f"avg_labels={float(va_labels.sum(axis=1).mean()):.2f}"
        )
        print(f"Test:  n={len(test_ds)} (fixed holdout)")

        train_ds = MultiLabelDataset(df_tr, class_names, train_dir, train_tfms)
        val_ds = MultiLabelDataset(df_va, class_names, train_dir, eval_tfms)

        sampler = None
        shuffle = True
        if args.balanced_sampler:
            sampler = build_balanced_sampler(tr_labels, pos_weight.numpy())
            shuffle = False

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        fold_dir = os.path.join(run_dir, f"fold_{fold}")
        metrics = train_fold(
            fold_dir=fold_dir,
            model_name=args.model,
            img_size=args.img_size,
            num_classes=num_classes,
            class_names=class_names,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            accumulation_steps=args.accumulation_steps,
            use_amp=args.amp,
            loss_name=args.loss,
            pos_weight=pos_weight,
            focal_gamma=args.focal_gamma,
            asl_gn=args.asl_gn,
            asl_gp=args.asl_gp,
            asl_clip=args.asl_clip,
            grad_clip=args.grad_clip,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            attn_drop_rate=args.attn_drop_rate,
            layer_decay=args.layer_decay,
            warmup_epochs=args.warmup_epochs,
            ema_decay=args.ema_decay,
            mixup_alpha=args.mixup_alpha,
        )

        fold_results.append({"fold": fold, **metrics})

    # summary
    micro = [r["micro_f1_tuned"] for r in fold_results]
    macro = [r["macro_f1_tuned"] for r in fold_results]
    summary = {
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "n_folds": args.n_folds,
        "test_frac": args.test_frac,
        "drop_negative": bool(args.drop_negative),
        "loss": args.loss,
        "strong_aug": bool(args.strong_aug),
        "balanced_sampler": bool(args.balanced_sampler),
        "swin_regs": {
            "drop_rate": args.drop_rate,
            "drop_path_rate": args.drop_path_rate,
            "attn_drop_rate": args.attn_drop_rate,
            "layer_decay": args.layer_decay,
            "warmup_epochs": args.warmup_epochs,
            "ema_decay": args.ema_decay,
            "mixup_alpha": args.mixup_alpha,
        },
        "fold_results": fold_results,
        "cv_summary": {
            "micro_f1_tuned_mean": float(np.mean(micro)),
            "micro_f1_tuned_std": float(np.std(micro, ddof=1)) if len(micro) > 1 else 0.0,
            "macro_f1_tuned_mean": float(np.mean(macro)),
            "macro_f1_tuned_std": float(np.std(macro, ddof=1)) if len(macro) > 1 else 0.0,
        },
    }
    with open(os.path.join(run_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCV Summary (tuned thresholds):")
    print(f"micro_f1: {summary['cv_summary']['micro_f1_tuned_mean']:.4f} ± {summary['cv_summary']['micro_f1_tuned_std']:.4f}")
    print(f"macro_f1: {summary['cv_summary']['macro_f1_tuned_mean']:.4f} ± {summary['cv_summary']['macro_f1_tuned_std']:.4f}")


if __name__ == "__main__":
    main()