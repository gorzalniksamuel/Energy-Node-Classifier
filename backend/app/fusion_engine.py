import math
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple

CLASSIFIER_THRESHOLDS = {
    "Biomass": 0.3,
    "Industry": 0.60,
    "Nuclear": 0.80,
    "Hydro": 0.20,
    "Compressor Metering Station": 0.50,
}

DEFAULT_CLASSIFIER_THRESHOLD = 0.40

REGENERATIVE_CLASSES = {"Solar", "Wind", "Hydro", "Biomass"}
NON_REGENERATIVE_CLASSES = {
    "Coal",
    "Gas Plant",
    "Nuclear",
    "Oil",
    "Compressor Metering Station",
}

# Helpers
def get_classifier_threshold(label: str) -> float:
    return CLASSIFIER_THRESHOLDS.get(label, DEFAULT_CLASSIFIER_THRESHOLD)


def build_detection_summary(detections: List[Dict]) -> Dict[str, Any]:
    counts = defaultdict(int)
    conf_sum = defaultdict(float)

    for d in detections or []:
        lbl = d.get("label")
        conf = float(d.get("confidence", 0.0))
        if not lbl:
            continue
        counts[lbl] += 1
        conf_sum[lbl] += conf
    avg_conf = {
        k: (conf_sum[k] / counts[k]) if counts[k] > 0 else 0.0
        for k in counts
    }

    return {
        "counts": dict(counts),
        "avg_conf": avg_conf,
    }

def select_classifier_candidates(
        classification: List[Dict],
        detections: List[Dict],
) -> List[Tuple[str, float]]:
    """
    Rules:
    - up to 2 non-regenerative classes
    - up to 2 regenerative classes
    - Substation may be added additionally
    - no further non-regenerative classes
    - Nuclear requires detector support
    - Coal preferred over Nuclear if coal-specific detector evidence exists
    """
    det = build_detection_summary(detections)

    ranked = []
    for c in classification or []:
        label = c.get("label")
        score = float(c.get("score", 0.0))
        if not label:
            continue

        thr = get_classifier_threshold(label)

        if label == "Nuclear":
            nuclear_cb_ok = (
                    detection_present(det, "Nuclear CB") or
                    detection_present(det, "Cooling Tower")
            )
            if score >= thr and nuclear_cb_ok:
                ranked.append((label, score))
            continue

        if score >= thr:
            ranked.append((label, score))

    ranked.sort(key=lambda x: x[1], reverse=True)

    coal_specific = (
            detection_present(det, "Conveyor") or
            detection_present(det, "Coal Heap") or
            det["counts"].get("Tank/Silo", 0) >= 15
    )

    # coal beats nuclear when both are classifier candidates and coal-specific evidence exists
    labels_in_ranked = {lbl for lbl, _ in ranked}
    if coal_specific and "Coal" in labels_in_ranked and "Nuclear" in labels_in_ranked:
        reranked = []
        coal_score = None
        nuclear_score = None
        for lbl, sc in ranked:
            if lbl == "Coal":
                coal_score = sc
            elif lbl == "Nuclear":
                nuclear_score = sc

        for lbl, sc in ranked:
            if lbl == "Coal":
                reranked.append((lbl, max(sc, (nuclear_score or 0.0) + 0.001)))
            elif lbl == "Nuclear":
                reranked.append((lbl, min(sc, (coal_score or sc) - 0.001)))
            else:
                reranked.append((lbl, sc))
        ranked = sorted(reranked, key=lambda x: x[1], reverse=True)

    selected = []
    selected_labels = set()
    nonreg_count = 0
    regen_count = 0
    substation_added = False

    for label, score in ranked:
        if label in selected_labels:
            continue

        if label == "Substation":
            if not substation_added:
                selected.append((label, score))
                selected_labels.add(label)
                substation_added = True
            continue

        if label in REGENERATIVE_CLASSES:
            if regen_count < 2:
                selected.append((label, score))
                selected_labels.add(label)
                regen_count += 1
            continue

        if label in NON_REGENERATIVE_CLASSES:
            if nonreg_count < 2:
                selected.append((label, score))
                selected_labels.add(label)
                nonreg_count += 1
            continue

    return selected


def detection_present(det_summary: Dict[str, Any], label: str, min_count: int = 1, min_avg_conf: float = 0.0) -> bool:
    count = det_summary["counts"].get(label, 0)
    avg_conf = det_summary["avg_conf"].get(label, 0.0)
    return count >= min_count and avg_conf >= min_avg_conf

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def map_database_type_to_classes(raw_type: str) -> List[str]:
    t = normalize_text(raw_type)

    t = t.replace("under construction", " ")
    t = re.sub(r"\s+", " ", t).strip()

    classes = set()

    if any(x in t for x in [
        "substation",
        "substations power plants",
        "transformer",
        "converter station",
        "converter station back to back"
    ]):
        classes.add("Substation")

    if any(x in t for x in [
        "compressor",
        "metering station",
        "border point"
    ]):
        classes.add("Compressor Metering Station")

    if any(x in t for x in [
        "coal",
        "hard coal",
        "brown coal",
        "lignite",
        "petcoke",
        "coal derived gas"
    ]):
        classes.add("Coal")

    if any(x in t for x in [
        "biomass",
        "biogas",
        "waste",
        "non renewable waste"
    ]):
        classes.add("Biomass")

    if any(x in t for x in [
        "hydro",
        "hydro pumped storage",
        "hydro mixed pump storage",
        "hydro pure storage",
        "hydro pure pump storage",
        "hydro water reservoir",
        "run of river",
        "wave",
        "tidal",
        "marine"
    ]):
        classes.add("Hydro")

    if "nuclear" in t:
        classes.add("Nuclear")

    if any(x in t for x in [
        "solar",
        "photovoltaic"
    ]):
        classes.add("Solar")

    if any(x in t for x in [
        "wind",
        "wind farm",
        "wind offshore",
        "wind onshore",
        "wind park"
    ]):
        classes.add("Wind")

    if any(x in t for x in [
        "gas",
        "natural gas",
        "fossil gas",
        "lng",
        "gathering and processing",
        "lng facilities",
        "natural gas compressor stations",
        "natural gas flaring detections",
        "injection and disposal"
    ]):
        classes.add("Gas Plant")

    if any(x in t for x in [
        "hydrogen",
        "h2"
    ]):
        classes.add("Hydrogen")

    if any(x in t for x in [
        "oil",
        "crude oil",
        "petroleum",
        "petroleum terminals",
        "crude oil refineries",
        "oil refinery",
        "offshore platforms",
        "tank batteries",
        "stations other",
        "diesel",
        "fuel oil",
        "kerosene",
        "gasoline",
        "jet fuel",
        "asphalt",
        "bitumen",
        "refined petroleum",
        "refined products",
        "oil shale",
        "condensate"
    ]):
        classes.add("Oil")

    if any(x in t for x in [
        "industrial",
        "industry",
        "equipment and components",
        "fossil fuel",
        "mixed fossil fuels",
        "other fossil fuels",
        "other fuels",
        "other or unspecified energy sources"
    ]):
        classes.add("Industry")

    return sorted(classes)

class MultiModalFusionEngine:
    def run(
            self,
            lat: float,
            lon: float,
            classifier_classes: List[str],
            classification: List[Dict],
            detections: List[Dict],
            osm_features: List[Dict],
            db_records: List[Dict],
            agent_prediction: str,
            agent_confidence: float,
            weights: Dict[str, float],
            active_modalities: Dict[str, bool],
            use_distance_weighting: bool = True,
            lambda_m: float = 1500,
    ) -> Dict[str, Any]:

        modal_scores = {
            "image": defaultdict(float),
            "osm": defaultdict(float),
            "database": defaultdict(float),
            "agent": defaultdict(float)
        }

        filtered_weights = {
            k: v for k, v in weights.items()
            if active_modalities.get(k, False)
        }

        if not filtered_weights:
            return {
                "classes": [],
                "scores": {},
                "explanation": ["No active modalities selected."]
            }

        weights = normalize_weights(filtered_weights)

        scores = defaultdict(float)
        explanation = []

        det = build_detection_summary(detections or [])
        detection_counts = defaultdict(int, det["counts"])
        detection_avg = det["avg_conf"]

        def avg(label):
            return float(detection_avg.get(label, 0.0))

        # Classifier
        selected_classifier = select_classifier_candidates(classification or [], detections or [])

        for label, prob in selected_classifier:
            if label in classifier_classes:
                contrib = weights["image"] * prob
                scores[label] += contrib
                modal_scores["image"][label] += contrib
                explanation.append(f"Classifier selected: {label} ({prob:.2f})")

        # YOLO detector
        for obj in ["Compressor", "Gas Piping"]:
            if detection_counts[obj] > 0:
                boost = max(avg(obj), 0.2)
                contrib = weights["image"] * boost
                scores["Compressor Metering Station"] += contrib
                modal_scores["image"]["Compressor Metering Station"] += contrib
                explanation.append(f"{obj} detected")

        if detection_counts["Chimney"] > 0:
            boost = max(avg("Chimney"), 0.3)
            for cls, factor in [
                ("Coal", 0.6),
                ("Gas Plant", 0.4),
                ("Biomass", 0.3),
                ("Industry", 0.5),
                ("Nuclear", 0.2),
            ]:
                contrib = weights["image"] * factor * boost
                scores[cls] += contrib
                modal_scores["image"][cls] += contrib
            explanation.append("Chimney detected")

        if detection_counts["Solar"] > 0:
            contrib = weights["image"] * avg("Solar")
            scores["Solar"] += contrib
            modal_scores["image"]["Solar"] += contrib

        if detection_counts["Wind"] > 0:
            contrib = weights["image"] * avg("Wind")
            scores["Wind"] += contrib
            modal_scores["image"]["Wind"] += contrib

        if detection_counts["Biogas"] > 0:
            contrib = weights["image"] * max(avg("Biogas"), 0.35)
            scores["Biomass"] += contrib
            modal_scores["image"]["Biomass"] += contrib
            explanation.append("Biogas detected -> Biomass support")

        if detection_counts["Nuclear CB"] > 0:
            contrib = weights["image"] * max(avg("Nuclear CB"), 0.3)
            scores["Nuclear"] += contrib
            modal_scores["image"]["Nuclear"] += contrib
            explanation.append("Nuclear CB detected")

        if detection_counts["Conveyor"] > 0:
            contrib = weights["image"] * max(avg("Conveyor"), 0.3)
            scores["Coal"] += contrib
            modal_scores["image"]["Coal"] += contrib
            explanation.append("Conveyor detected")

        if detection_counts["Coal Heap"] > 0:
            contrib = weights["image"] * max(avg("Coal Heap"), 0.3)
            scores["Coal"] += contrib
            modal_scores["image"]["Coal"] += contrib
            explanation.append("Coal Heap detected")

        if detection_counts["Tank/Silo"] >= 15:
            contrib = weights["image"] * 0.35
            scores["Coal"] += contrib
            modal_scores["image"]["Coal"] += contrib
            explanation.append("15+ Tank/Silo detected")

        # OSM
        hydrogen_flag = False
        osm_hydrogen_flag = False

        for f in osm_features or []:
            cat = f.get("category")
            f_lat = f.get("lat")
            f_lon = f.get("lon")

            if f_lat is None or f_lon is None:
                continue

            dist = haversine_m(lat, lon, f_lat, f_lon)
            w_dist = math.exp(-dist / lambda_m) if use_distance_weighting else 1.0

            if cat in ["biomass_power_plant", "biogas_power_plant"]:
                contrib = weights["osm"] * w_dist
                scores["Biomass"] += contrib
                modal_scores["osm"]["Biomass"] += contrib

            elif cat == "coal_power_plant":
                contrib = weights["osm"] * w_dist
                scores["Coal"] += contrib
                modal_scores["osm"]["Coal"] += contrib

            elif cat == "gas_power_plant":
                contrib = weights["osm"] * w_dist
                scores["Gas Plant"] += contrib
                modal_scores["osm"]["Gas Plant"] += contrib

            elif cat == "oil_power_plant":
                contrib = weights["osm"] * w_dist
                scores["Oil"] += contrib
                modal_scores["osm"]["Oil"] += contrib

            elif cat == "nuclear_power_plant":
                contrib = weights["osm"] * w_dist
                scores["Nuclear"] += contrib
                modal_scores["osm"]["Nuclear"] += contrib

            elif cat == "hydro":
                contrib = weights["osm"] * w_dist
                scores["Hydro"] += contrib
                modal_scores["osm"]["Hydro"] += contrib

            elif cat == "solar_farm":
                contrib = weights["osm"] * w_dist
                scores["Solar"] += contrib
                modal_scores["osm"]["Solar"] += contrib

            elif cat == "wind_farm":
                contrib = weights["osm"] * w_dist
                scores["Wind"] += contrib
                modal_scores["osm"]["Wind"] += contrib

            elif cat in ["gas_compressor_station", "gas_substation_prs"]:
                contrib = weights["osm"] * w_dist
                scores["Compressor Metering Station"] += contrib
                modal_scores["osm"]["Compressor Metering Station"] += contrib

            elif cat in ["gas_storage", "gas_pipeline", "gas_pipeline_marker"]:
                contrib = weights["osm"] * 0.35 * w_dist
                scores["Gas Plant"] += contrib
                modal_scores["osm"]["Gas Plant"] += contrib

            elif cat == "electrical_substation":
                contrib = weights["osm"] * w_dist
                scores["Substation"] += contrib
                modal_scores["osm"]["Substation"] += contrib

            elif cat == "industrial_area":
                contrib = weights["osm"] * w_dist
                scores["Industry"] += contrib
                modal_scores["osm"]["Industry"] += contrib

            elif cat == "hydrogen":
                hydrogen_flag = True
                osm_hydrogen_flag = True
                contrib = weights["osm"] * w_dist
                scores["Hydrogen"] += contrib
                modal_scores["osm"]["Hydrogen"] += contrib

        # AWS db
        for r in db_records or []:
            mapped = map_database_type_to_classes(r.get("type"))
            if not mapped:
                continue

            if "Hydrogen" in mapped:
                hydrogen_flag = True
            non_hydrogen = [m for m in mapped if m != "Hydrogen"]
            if not non_hydrogen:
                continue

            share = (weights["database"] * 1) / len(non_hydrogen)
            for m in non_hydrogen:
                scores[m] += share
                modal_scores["database"][m] += share

        # agent
        if agent_prediction:
            if agent_prediction == "Hydrogen":
                hydrogen_flag = True

            if agent_prediction in (
                    set(classifier_classes)
                    | {"Hydrogen", "Biomass", "Coal", "Compressor Metering Station", "Gas Plant",
                       "Hydro", "Industry", "Nuclear", "Oil", "Solar", "Substation", "Wind"}
            ):
                contrib = weights["agent"] * float(agent_confidence or 0)
                scores[agent_prediction] += contrib
                modal_scores["agent"][agent_prediction] += contrib

        # overrides (coal indicators)
        coal_specific = (
                detection_counts["Conveyor"] > 0 or
                detection_counts["Coal Heap"] > 0 or
                detection_counts["Tank/Silo"] >= 15
        )

        if scores["Coal"] > 0 and scores["Nuclear"] > 0 and coal_specific:
            scores["Coal"] = max(scores["Coal"], scores["Nuclear"] + 0.05)
            scores["Nuclear"] *= 0.75
            explanation.append("Override: Coal preferred over Nuclear due to coal-specific detections")

        if scores["Compressor Metering Station"] > 0.50 and scores["Gas Plant"] > 0.50:
            scores["Compressor Metering Station"] *= 0.70
            explanation.append("Override: Gas Plant preferred over Compressor Metering Station")

        # Final selection
        THRESHOLD = 0.4

        # Weak Biomass + Biogas detection -> promote Biomass
        if detection_counts["Biogas"] > 0 and scores["Biomass"] >= 0.20 and scores["Biomass"] < THRESHOLD:
            scores["Biomass"] = THRESHOLD
            explanation.append("Promotion: weak Biomass upgraded due to Biogas detection")

        prelim = [c for c, s in scores.items() if s >= THRESHOLD]

        if not prelim and detection_counts["Biogas"] > 0:
            prelim = ["Biomass"]
            explanation.append("Fallback: no class passed threshold, Biogas detected -> Biomass")

        regen = []
        nonreg = []
        substation = []

        for c in sorted(prelim, key=lambda x: scores[x], reverse=True):
            if c == "Substation":
                substation.append(c)
            elif c in REGENERATIVE_CLASSES:
                regen.append(c)
            elif c in NON_REGENERATIVE_CLASSES:
                nonreg.append(c)

        final_classes = nonreg[:2] + regen[:2]

        if substation:
            final_classes.append("Substation")

        # OSM-derived h2 => always retain regardless of score
        if osm_hydrogen_flag and "Hydrogen" not in final_classes:
            final_classes.append("Hydrogen")
            explanation.append("OSM Hydrogen detected -> Hydrogen retained irrespective of score")

        # Non-OSM Hydrogen remains special-case append logic
        elif hydrogen_flag and "Hydrogen" not in final_classes:
            if len([c for c in final_classes if c not in REGENERATIVE_CLASSES and c != "Substation"]) < 2:
                final_classes.append("Hydrogen")
            else:
                current_nonreg = [c for c in final_classes if c not in REGENERATIVE_CLASSES and c != "Substation"]
                if current_nonreg:
                    weakest = min(current_nonreg, key=lambda x: scores.get(x, 0.0))
                    if scores.get("Hydrogen", 0.0) > scores.get(weakest, 0.0):
                        final_classes.remove(weakest)
                        final_classes.append("Hydrogen")

        return {
            "classes": list(dict.fromkeys(final_classes)),
            "scores": dict(scores),
            "modal_scores": {
                k: dict(v) for k, v in modal_scores.items()
            },
            "explanation": explanation
        }
