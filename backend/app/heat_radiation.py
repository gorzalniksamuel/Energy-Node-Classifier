import base64
from io import BytesIO
from typing import Any, Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import planetary_computer as pc
import rasterio
import geopandas as gpd

from pystac_client import Client
from shapely.geometry import Point, mapping
from pyproj import CRS
from rasterio.mask import mask



HEAT_CLASSES = [
    "large_persistent_heat_source",
    "moderate_or_intermittent_heat",
    "no_detectable_heat",
]

# Geometry defaults
DEFAULT_AOI_M = 1000
DEFAULT_RING_INNER_M = 2000
DEFAULT_RING_OUTER_M = 4500

# Hotspot if AOI>(bg_p95+offset)
DEFAULT_HOTSPOT_OFFSET_C = 1.0

# Scene thresholds
HOT_FRAC_SCENE_THRESH = 0.01    # ->some hotspot exists
HOT_FRAC_PERSIST_THRESH = 0.03  # -> meaningful hotspot area

MIN_VALID_AOI_PX = 25
MIN_VALID_BG_PX = 50

# Scoring normalization
TAIL_OFFSET_C = 0.3
TAIL_SCALE_C = 3.5
AREA_NORM = 0.08

LARGE_INTENSITY_CUTOFF = 0.55
LARGE_PERSIST_CUTOFF = 0.20

# Peak triggers
LARGE_PEAK_D99_C = 2.2
LARGE_PEAK_HOTFRAC = 0.04


# geometry helpers
def utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def buffer_poly_wgs84(lat: float, lon: float, buffer_m: float):
    wgs84 = CRS.from_epsg(4326)
    utm = utm_crs_from_lonlat(lon, lat)
    pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs=wgs84)
    poly_utm = pt.to_crs(utm).buffer(buffer_m).iloc[0]
    poly_wgs = gpd.GeoSeries([poly_utm], crs=utm).to_crs(wgs84).iloc[0]
    return poly_wgs


def donut_poly_wgs84(lat: float, lon: float, inner_m: float, outer_m: float):
    wgs84 = CRS.from_epsg(4326)
    utm = utm_crs_from_lonlat(lon, lat)
    pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs=wgs84)
    outer = pt.to_crs(utm).buffer(outer_m).iloc[0]
    inner = pt.to_crs(utm).buffer(inner_m).iloc[0]
    donut_utm = outer.difference(inner)
    donut_wgs = gpd.GeoSeries([donut_utm], crs=utm).to_crs(wgs84).iloc[0]
    return donut_wgs


# raster helpers
def read_clip_as_array(href: str, geom_wgs84, geom_crs=CRS.from_epsg(4326)) -> np.ndarray:
    with rasterio.open(href) as ds:
        geom_proj = gpd.GeoSeries([geom_wgs84], crs=geom_crs).to_crs(ds.crs).iloc[0]
        data, _ = mask(ds, [mapping(geom_proj)], crop=True, filled=True)
        arr = data[0].astype("float32")
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan
        return arr


def get_scale_offset(asset, href: str):
    rb = asset.extra_fields.get("raster:bands")
    if isinstance(rb, list) and rb and isinstance(rb[0], dict):
        scale = rb[0].get("scale")
        offset = rb[0].get("offset")
        try:
            if scale is not None and offset is not None:
                return float(scale), float(offset)
        except Exception:
            pass

    try:
        with rasterio.open(href) as ds:
            t = ds.tags(1)
        scale = t.get("scale_factor") or t.get("SCALE_FACTOR") or t.get("ScaleFactor")
        offset = t.get("add_offset") or t.get("ADD_OFFSET") or t.get("AddOffset")
        if scale is not None and offset is not None:
            return float(scale), float(offset)
    except Exception:
        pass

    return None, None


def mask_bad_pixels(temp_arr: np.ndarray, qa_arr: np.ndarray) -> np.ndarray:
    qa = qa_arr.astype(np.uint16)

    def bit_is_set(x, bit):
        return (x & (1 << bit)) != 0

    bad = (
            bit_is_set(qa, 0) |  # fill
            bit_is_set(qa, 1) |  # dilated cloud
            bit_is_set(qa, 2) |  # cirrus
            bit_is_set(qa, 3) |  # cloud
            bit_is_set(qa, 4) |  # cloud shadow
            bit_is_set(qa, 5)    # snow
    )
    out = temp_arr.astype("float32")
    out[bad] = np.nan
    return out


# stac helpers
def stac_search_landsat_c2_l2(intersects_geojson, start_date: str, end_date: str, max_cloud: float):
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        intersects=intersects_geojson,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


def pick_assets(item):
    thermal_candidates = ["lwir11", "ST_B10", "ST_B6", "surface_temperature"]
    qa_candidates = ["qa_pixel", "qa"]
    thermal_key = next((k for k in thermal_candidates if k in item.assets), None)
    qa_key = next((k for k in qa_candidates if k in item.assets), None)
    return thermal_key, qa_key


# heatmap (png base64)
def fetch_landsat_heatmap_png_b64(
        lat: float,
        lon: float,
        buffer_m: float,
        start_date: str,
        end_date: str,
        max_cloud: float,
        prefer: str = "least_cloudy",
) -> Dict[str, Any]:
    aoi = buffer_poly_wgs84(lat, lon, buffer_m)
    items = stac_search_landsat_c2_l2(aoi.__geo_interface__, start_date, end_date, max_cloud)
    if not items:
        raise RuntimeError("No scenes found for heatmap. Relax date range/max_cloud.")

    if prefer == "least_cloudy":
        items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 9999))
        item = items[0]
    elif prefer == "most_recent":
        items.sort(key=lambda it: it.properties.get("datetime", ""))
        item = items[-1]
    else:
        item = items[0]

    item = pc.sign(item)
    thermal_key, qa_key = pick_assets(item)
    if thermal_key is None or qa_key is None:
        raise RuntimeError(f"Missing thermal/QA asset. Available: {list(item.assets.keys())}")

    thermal_asset = item.assets[thermal_key]
    qa_asset = item.assets[qa_key]

    raw_t = read_clip_as_array(thermal_asset.href, aoi)
    raw_q = read_clip_as_array(qa_asset.href, aoi)

    scale, offset = get_scale_offset(thermal_asset, thermal_asset.href)
    if scale is None:
        scale = 0.00341802
    if offset is None:
        offset = 149.0

    temp_c = (raw_t * scale + offset) - 273.15
    temp_c = mask_bad_pixels(temp_c, raw_q)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(temp_c, interpolation="nearest")
    ax.set_title(
        f"Landsat thermal (°C)\n"
        f"({lat:.5f}, {lon:.5f}) buffer={buffer_m}m | "
        f"cloud={item.properties.get('eo:cloud_cover','NA')}% | "
        f"date={str(item.properties.get('datetime','NA'))[:10]} | asset={thermal_key}"
    )
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Thermal temperature proxy (°C)")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "item_id": item.id,
        "datetime": item.properties.get("datetime"),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "thermal_asset_used": thermal_key,
        "qa_asset_used": qa_key,
        "heatmap_png_b64": png_b64,
    }


# multi-scene thermal features
def compute_landsat_thermal_features(
        lat: float,
        lon: float,
        aoi_m: float = DEFAULT_AOI_M,
        ring_inner_m: float = DEFAULT_RING_INNER_M,
        ring_outer_m: float = DEFAULT_RING_OUTER_M,
        start_date: str = "2024-06-01",
        end_date: str = "2024-09-30",
        max_cloud: float = 70.0,
        max_scenes: int = 20,
        hotspot_offset_c: float = DEFAULT_HOTSPOT_OFFSET_C,
        min_valid_aoi_px: int = MIN_VALID_AOI_PX,
        min_valid_bg_px: int = MIN_VALID_BG_PX,
        min_scenes_required: int = 4,
        debug_print: bool = False,
) -> Dict[str, Any]:
    aoi = buffer_poly_wgs84(lat, lon, aoi_m)
    ring = donut_poly_wgs84(lat, lon, ring_inner_m, ring_outer_m)

    items = stac_search_landsat_c2_l2(aoi.__geo_interface__, start_date, end_date, max_cloud)
    if not items:
        raise RuntimeError("no scenes found.")

    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 9999))
    items = items[:max_scenes]

    d95_list, d99_list, hotf_list = [], [], []
    used, skipped = 0, 0

    for it in items:
        it = pc.sign(it)
        thermal_key, qa_key = pick_assets(it)
        if thermal_key is None or qa_key is None:
            skipped += 1
            continue

        thermal_asset = it.assets[thermal_key]
        qa_asset = it.assets[qa_key]

        raw_t_aoi = read_clip_as_array(thermal_asset.href, aoi)
        raw_q_aoi = read_clip_as_array(qa_asset.href, aoi)
        if np.all(np.isnan(raw_t_aoi)):
            skipped += 1
            continue

        scale, offset = get_scale_offset(thermal_asset, thermal_asset.href)
        if scale is None:
            scale = 0.00341802
        if offset is None:
            offset = 149.0

        temp_aoi = (raw_t_aoi * scale + offset) - 273.15
        temp_aoi = mask_bad_pixels(temp_aoi, raw_q_aoi)

        raw_t_bg = read_clip_as_array(thermal_asset.href, ring)
        raw_q_bg = read_clip_as_array(qa_asset.href, ring)
        temp_bg = (raw_t_bg * scale + offset) - 273.15
        temp_bg = mask_bad_pixels(temp_bg, raw_q_bg)

        aoi_valid = temp_aoi[np.isfinite(temp_aoi)]
        bg_valid = temp_bg[np.isfinite(temp_bg)]
        if aoi_valid.size < min_valid_aoi_px or bg_valid.size < min_valid_bg_px:
            skipped += 1
            continue

        aoi_p95 = float(np.nanpercentile(aoi_valid, 95))
        aoi_p99 = float(np.nanpercentile(aoi_valid, 99))
        bg_p95 = float(np.nanpercentile(bg_valid, 95))
        bg_p99 = float(np.nanpercentile(bg_valid, 99))

        d95 = aoi_p95 - bg_p95
        d99 = aoi_p99 - bg_p99

        thr = bg_p95 + hotspot_offset_c
        hotf = float(np.mean(aoi_valid > thr))

        d95_list.append(d95)
        d99_list.append(d99)
        hotf_list.append(hotf)
        used += 1

        if debug_print:
            dt = str(it.properties.get("datetime", ""))[:10]
            cc = it.properties.get("eo:cloud_cover", "NA")
            print(dt, "cloud=", cc, "d95=", round(d95, 2), "d99=", round(d99, 2),
                  "hotf=", round(hotf, 3), "bg_p95=", round(bg_p95, 2), "bg_p99=", round(bg_p99, 2))

    if used < min_scenes_required:
        raise RuntimeError(
            f"Insufficient usable scenes (used={used}, skipped={skipped}). "
        )

    d95 = np.array(d95_list, dtype="float32")
    d99 = np.array(d99_list, dtype="float32")
    hotf = np.array(hotf_list, dtype="float32")

    return {
        "n_scenes_used": int(used),

        "deltaT95_median": float(np.nanmedian(d95)),
        "deltaT95_p90": float(np.nanpercentile(d95, 90)),
        "deltaT99_median": float(np.nanmedian(d99)),
        "deltaT99_p90": float(np.nanpercentile(d99, 90)),

        # peak
        "deltaT99_max": float(np.nanmax(d99)),
        "hot_frac_max": float(np.nanmax(hotf)),

        # hotspot area
        "hot_frac_median": float(np.nanmedian(hotf)),
        "hot_frac_p90": float(np.nanpercentile(hotf, 90)),

        # frequency
        "hot_frac_any": float(np.mean(hotf > HOT_FRAC_SCENE_THRESH)),
        "hot_frac_persist": float(np.mean(hotf > HOT_FRAC_PERSIST_THRESH)),

        # variability
        "d95_std": float(np.nanstd(d95)),
        "d99_std": float(np.nanstd(d99)),
        "hotf_std": float(np.nanstd(hotf)),

        "d_95": d95,
        "d_99": d99,
        "hot_frac": hotf,
    }


# scoring
def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _softmax(vals):
    x = np.array(vals, dtype="float32")
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def heat_heuristic_score(features: Dict[str, Any]):
    d95_med = float(features.get("deltaT95_median", 0.0))
    d99_med = float(features.get("deltaT99_median", 0.0))
    d99_max = float(features.get("deltaT99_max", 0.0))

    hotf_med = float(features.get("hot_frac_median", 0.0))
    hotf_max = float(features.get("hot_frac_max", 0.0))
    hot_any = float(features.get("hot_frac_any", 0.0))
    hot_pers = float(features.get("hot_frac_persist", 0.0))

    n = int(features.get("n_scenes_used", 0))
    d95_std = float(features.get("d95_std", 0.0))

    # base intensity
    tail_intensity = _clamp01((0.6 * d95_med + 0.4 * d99_med - TAIL_OFFSET_C) / TAIL_SCALE_C)
    area_intensity = _clamp01(hotf_med / AREA_NORM)
    base_intensity = _clamp01(0.65 * tail_intensity + 0.35 * area_intensity)

    # peak intensity
    peak_tail = _clamp01((d99_max - 1.2) / 3.2)
    peak_area = _clamp01(hotf_max / 0.10)
    peak_intensity = _clamp01(0.6 * peak_tail + 0.4 * peak_area)

    persistence = _clamp01(hot_pers)
    intermittency = _clamp01(hot_any)

    n_conf = _clamp01((n - 2) / 10.0)
    var_penalty = _clamp01(d95_std / 5.0)
    reliability = _clamp01(0.85 * n_conf + 0.15 * (1.0 - var_penalty))
    reliability = max(0.35, reliability)

    large_regime = _clamp01((base_intensity - LARGE_INTENSITY_CUTOFF) / 0.30) * _clamp01((persistence - LARGE_PERSIST_CUTOFF) / 0.45)

    peak_large = _clamp01((d99_max - LARGE_PEAK_D99_C) / 2.5) + _clamp01((hotf_max - LARGE_PEAK_HOTFRAC) / 0.08)
    peak_large = _clamp01(peak_large / 2.0)

    some_heat = _clamp01((base_intensity - 0.18) / 0.40) + 0.7 * intermittency + 0.5 * peak_intensity
    some_heat = _clamp01(some_heat / 2.0)

    s_large = (2.2 * large_regime + 1.5 * peak_large + 0.7 * base_intensity + 0.6 * persistence) * reliability
    s_mod   = (1.4 * some_heat + 0.8 * peak_intensity + 0.4 * intermittency + 0.2 * (1.0 - large_regime)) * reliability
    s_none  = (1.6 * (1.0 - some_heat) + 0.9 * (1.0 - base_intensity) + 0.6 * (1.0 - peak_intensity)) * reliability

    EPS = 1e-6
    if hotf_max > EPS:
        s_none = -1e9

    probs = _softmax([s_large, s_mod, s_none])

    out = {
        "large_persistent_heat_source": float(probs[0]),
        "moderate_or_intermittent_heat": float(probs[1]),
        "no_detectable_heat": float(probs[2]),
    }
    ranked = sorted(out.items(), key=lambda kv: kv[1], reverse=True)

    composites = {
        "base_intensity": float(base_intensity),
        "peak_intensity": float(peak_intensity),
        "persistence": float(persistence),
        "intermittency": float(intermittency),
        "large_regime": float(large_regime),
        "peak_large": float(peak_large),
        "reliability": float(reliability),
    }
    raw = {
        "large_persistent_heat_source": float(s_large),
        "moderate_or_intermittent_heat": float(s_mod),
        "no_detectable_heat": float(s_none),
    }
    return out, ranked, raw, composites

# Pipeline wrappaer
def run_heat_radiation(
        lat: float,
        lon: float,
        start_date: str = "2024-06-01",
        end_date: str = "2024-09-30",
        max_cloud: float = 75.0,
        max_scenes: int = 20,
        aoi_m: float = DEFAULT_AOI_M,
        ring_inner_m: float = DEFAULT_RING_INNER_M,
        ring_outer_m: float = DEFAULT_RING_OUTER_M,
        hotspot_offset_c: float = DEFAULT_HOTSPOT_OFFSET_C,
) -> Tuple[Dict[str, Any], str]:
    heatmap = fetch_landsat_heatmap_png_b64(
        lat=lat,
        lon=lon,
        buffer_m=aoi_m,
        start_date=start_date,
        end_date=end_date,
        max_cloud=max_cloud,
        prefer="least_cloudy",
    )

    feats = compute_landsat_thermal_features(
        lat=lat,
        lon=lon,
        aoi_m=aoi_m,
        ring_inner_m=ring_inner_m,
        ring_outer_m=ring_outer_m,
        start_date=start_date,
        end_date=end_date,
        max_cloud=max_cloud,
        max_scenes=max_scenes,
        hotspot_offset_c=hotspot_offset_c,
        min_scenes_required=2,
        debug_print=False,
    )

    probs, ranked, raw, comps = heat_heuristic_score(feats)

    feats_api = dict(feats)
    feats_api.pop("d_95", None)
    feats_api.pop("d_99", None)
    feats_api.pop("hot_frac", None)


    api_result = {
        "lat": lat,
        "lon": lon,
        "window": (start_date, end_date),
        "thermal_features": feats_api,
        "heat_probs": probs,
        "heat_ranked": ranked,
        "heat_raw_scores": raw,
        "heat_composites": comps,
        "heatmap_meta": {
            "item_id": heatmap.get("item_id"),
            "datetime": heatmap.get("datetime"),
            "cloud_cover": heatmap.get("cloud_cover"),
            "thermal_asset_used": heatmap.get("thermal_asset_used"),
            "qa_asset_used": heatmap.get("qa_asset_used"),
        },
    }

    return api_result, heatmap["heatmap_png_b64"]

