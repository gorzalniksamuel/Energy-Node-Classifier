import re
import json
import math
import time
import random
import base64
import threading
import requests
import pandas as pd
import time, traceback

from pydantic import BaseModel, Field
from typing import Optional, Any, List, Dict

from re import search
from requests.exceptions import Timeout, ConnectionError, HTTPError

import folium
from folium.plugins import MarkerCluster

import html as _html
from collections import Counter


from fastapi import Depends
from fastapi import BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

import concurrent.futures
from contextlib import asynccontextmanager

from PIL import Image, ImageDraw
from batch_manager import BatchManager

from google import genai
from google.genai import types
from io import BytesIO

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import matplotlib
matplotlib.use("Agg")

from heat_radiation import run_heat_radiation
from ml_engine import EnergyClassifier, ObjectDetector
from fusion_engine import MultiModalFusionEngine
from pipeline_wrapper import run_single_pipeline
from models import KeyValidationRequest, OsmFeature, PredictionRequest, PredictionResponse, BatchRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load default models on startup
    print("Pre-loading models...")
    classifier.load_model("convnext_large")
    detector.load_model("yolo26")
    yield
    # Clean up if needed
    print("Shutting down")



batch_manager = BatchManager()


# Fast API init
app = FastAPI(title="Thesis Backend API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Helpers
def google_places_search(lat: float, lon: float, radius_m: int, api_key: str, keywords: List[str]) -> List[dict]:
    """
    Queries Google Places Nearby Search for multiple keywords and merges results.
    """
    if not api_key:
        return []

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    results = []
    seen_place_ids = set()

    for kw in keywords:
        params = {
            "location": f"{lat},{lon}",
            "radius": radius_m,
            "keyword": kw,
            "key": api_key
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            for item in data.get("results", []):
                pid = item.get("place_id")
                if pid and pid not in seen_place_ids:
                    seen_place_ids.add(pid)
                    results.append(item)

        except Exception as e:
            print(f"[Places] query failed ({kw}): {e}")

    return results

def build_places_memo(places: List[dict], origin_lat: float, origin_lon: float) -> str:
    """
    Converts Places results into a readable memo for Gemini.
    """
    lines = []

    for p in places[:10]:
        name = p.get("name", "Unnamed")
        loc = p.get("geometry", {}).get("location", {})
        lat = loc.get("lat")
        lon = loc.get("lng")

        if lat is None or lon is None:
            continue

        maps_url = f"https://maps.google.com/?q={lat},{lon}"

        lines.append(
            f"- {name} ({lat:.5f},{lon:.5f}) {maps_url}"
        )

    if not lines:
        return "No relevant infrastructure found via Google Places."

    return "\n".join(lines)


def pil_to_b64_png(img: Image.Image) -> Optional[str]:
    if img is None:
        return None
    try:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return None

def extract_json(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        lines = t.splitlines()
        if lines and lines[0].strip().lower() == "json":
            t = "\n".join(lines[1:]).strip()
    return t

def extract_first_json_object(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    if s.startswith("```"):
        s = s.strip("`").strip()
        lines = s.splitlines()
        if lines and lines[0].strip().lower() in ("json", "javascript"):
            s = "\n".join(lines[1:]).strip()

    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return ""

    stack = []
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            opener = stack.pop()
            if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                continue
            if not stack:
                return s[start:i+1]

    return ""


def gemini_grounded_search(gemini_api_key: str, query: str, model: str = "gemini-3.0-flash-preview") -> dict:
    if not gemini_api_key:
        return {"memo": "", "urls": [], "error": "Missing Gemini API key"}

    try:
        client = genai.Client(api_key=gemini_api_key)
        tool = types.Tool(google_search=types.GoogleSearch())

        prompt = f"""
                Use Google Search to find publicly available, high-level information relevant to this location.
                
                Query:
                {query}
                
                Return a short memo (5-10 bullets) summarizing what you found.
                Each bullet must include a URL in parentheses.
                Do not include operational/security details.
                """

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[tool],
                temperature=0.2,
            ),
        )

        text = (resp.text or "").strip()
        urls = re.findall(r"https?://[^\s)\]]+", text)
        return {"memo": text, "urls": urls[:15], "error": None}

    except Exception as e:
        return {"memo": "", "urls": [], "error": str(e)}

ALLOWED_AGENT_CLASSES = [
    "Biomass",
    "Coal",
    "Compressor Metering Station",
    "Gas Plant",
    "Hydro",
    "Hydrogen",
    "Industry",
    "Nuclear",
    "Oil",
    "Solar",
    "Substation",
    "Wind",
    "Negative"
]

def build_agent_prompt_image_web_only(lat, lon, buffer_km, web_memo: str, web_error: str = "") -> str:
    web_status = "available" if web_memo.strip() else "empty"
    if web_error:
        web_status = f"unavailable ({web_error})"

    classes_str = ", ".join([f'"{c}"' for c in ALLOWED_AGENT_CLASSES])

    return f"""
    You are an Energy Infrastructure Analyst.
    
    INPUT DATA:
    1. Satellite Image (attached)
    2. Coordinates: {lat}, {lon} (Buffer: {buffer_km}km)
    3. Web Research Memo: 
    "{web_memo or 'No relevant web results found.'}"

    TASK:
    Classify the facility shown in the image/coordinates into EXACTLY one of these categories:
    [{classes_str}]

    INSTRUCTIONS:
    - PRIORITIZE the Web Research Memo if it identifies a specific facility (e.g. "Heilbronn Coal Plant").
    - If the Web Memo is empty or vague, analyze the Satellite Image visually.
    - If the Image shows wind turbines, solar panels (not rooftop panels but those which belong to solar parks), or cooling towers, use that visual evidence.
    - If no energy infrastructure is visible and no web results exist, choose "Negative".

    OUTPUT REQUIREMENT:
    Return pure JSON with this structure:
    {{
      "final_prediction": "Class Name",
      "confidence": 0.95,
      "review_summary": "1-2 sentences summarizing the Web Memo findings.",
      "rationale": "Explain WHY you chose the class. If web failed, describe the visual features (e.g., 'Visible white 3-blade turbines').",
      "key_evidence": [
        "Web: [Name of facility found or 'None']",
        "Visual: [Specific visual feature 1]",
        "Visual: [Specific visual feature 2]"
      ]
    }}
    """.strip()


def call_gemini_text(gemini_api_key: str, prompt: str) -> str:
    if not gemini_api_key:
        raise ValueError("Missing Gemini API key")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 800,
            "responseMimeType": "application/json"
        }
    }

    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(data)[:2000]


def repair_json_with_gemini(gemini_api_key: str, broken_text: str) -> str:
    prompt = (
        "You will be given a response that MUST be valid JSON but may be malformed.\n"
        "Return ONLY a corrected JSON object. No markdown. No commentary.\n\n"
        f"INPUT:\n{broken_text}"
    )
    return call_gemini_text(gemini_api_key, prompt)


def validate_gemini_key(gemini_api_key: str) -> (bool, str):
    if not gemini_api_key or len(gemini_api_key.strip()) == 0:
        return False, "Missing Gemini API key"

    # Minimal generateContent request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": "ping"}]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 5
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=10)

        # Common auth failures are 401/403
        if r.status_code in (401, 403):
            return False, "Invalid Gemini API key"

        r.raise_for_status()

        data = r.json()
        # If we got candidates back, it's valid enough
        if data.get("candidates"):
            return True, "Gemini key verified"

        # Some failures return 200 with error ish shapes
        return False, "Gemini validation returned no candidates"

    except Exception as e:
        return False, f"Gemini validation failed: {str(e)}"

def call_gemini_with_image(gemini_api_key: str, prompt: str, pil_image: Image.Image) -> str:
    if not gemini_api_key:
        raise ValueError("Missing Gemini API key")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"

    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json"
        }
    }

    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    try:
        cand = (data.get("candidates") or [])[0] or {}
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        texts = []
        for p in parts:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                texts.append(p["text"])
        return "\n".join(texts).strip()
    except Exception:
        return json.dumps(data)


# Initialize Engines
MODEL_DIR = "/app/ml_models"
classifier = EnergyClassifier(MODEL_DIR)
detector = ObjectDetector(MODEL_DIR)

CLASSIFIER_LOCK = threading.Lock()
DETECTOR_LOCK = threading.Lock()

fusion_engine = MultiModalFusionEngine()

AWS_API_URL = "https://4i015zjr5d.execute-api.eu-north-1.amazonaws.com/prod/nearby"

def clean_category(cat_data):
    if isinstance(cat_data, list):
        return ", ".join(str(x) for x in cat_data)
    if isinstance(cat_data, str):
        return cat_data.replace('{', '').replace('}', '').replace('"', '')
    return str(cat_data)

def fetch_aws_task(lat, lon, radius, api_key, limit=25):
    if not api_key or len(api_key.strip()) == 0:
        return [], "Missing Key"

    params = {"lat": lat, "lon": lon, "radius_km": radius, "limit": limit}
    headers = {"X-API-Key": api_key}

    try:
        r = requests.get(AWS_API_URL, params=params, headers=headers, timeout=30)

        # Explicit auth handling
        if r.status_code in (401, 403):
            return [], "Invalid Key"

        r.raise_for_status()
        data = r.json()

        items = []
        if isinstance(data, dict):
            items = data.get("items", []) or data.get("results", []) or []
        elif isinstance(data, list):
            items = data
        else:
            items = []

        parsed_records = []
        for idx, item in enumerate(items):
            raw_cat = item.get("category", item.get("type", "Infrastructure"))

            # distance_km is the real field in your payload
            dist_val = item.get("distance_km", item.get("distance", item.get("dist", 0)))

            record = {
                "id": idx + 1,

                # Frontend expects: type, name, dist, source, ts, remarks
                "type": clean_category(raw_cat),

                # There is no "name" in payload => use id/source/category as a readable label
                "name": str(item.get("id", "Unnamed")),

                "dist": f"{float(dist_val):.2f} km" if dist_val is not None else "-",
                "source": item.get("source", "-"),

                # No timestamp in payload but we keep the field with placehoder
                "ts": "-",

                "remarks": str(item.get("remarks", "") or "-"),

                "lat": item.get("lat"),
                "lon": item.get("lon"),
            }
            parsed_records.append(record)

        return parsed_records, None

    except Exception as e:
        return [], str(e)

# OSM
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# for url rortation as overpass is unstable often times for large queries
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
]
TRANSIENT_HTTP_CODES = {429, 502, 503, 504}

# Filter for following TAGS
GAS_INFRA_TAGS = [
    '["pipeline"]', '["pipeline"="marker"]', '["pipeline"="valve"]',
    '["man_made"="pipeline_valve"]', '["pipeline"="substation"]',
    '["facility"="pipeline_compressor"]', '["man_made"="compressor"]',
    '["man_made"="gasometer"]', '["man_made"="storage_tank"]["content"="gas"]',
    '["industrial"="gas"]'
]

ENERGY_TAGS = [
    # generic power infra
    '["power"="plant"]',
    '["power"="generator"]',
    '["power"="substation"]',
    '["power"="station"]',
    '["power"="solar_farm"]',
    '["power"="wind_farm"]',
    '["power"="transformer"]',
    '["power"~"pole|tower"]',

    # source-based
    '["plant:source"]',
    '["generator:source"]',

    # biogas explicit
    '["plant:source"="biogas"]',
    '["generator:source"="biogas"]',
    '["generator:method"="anaerobic_digestion"]',
    '["man_made"="storage_tank"]["substance"="biogas"]',
    '["man_made"="storage_tank"]["content"="biogas"]',
    '["substance"="biogas"]',
    '["building"="digester"]',
    '["man_made"="biogas_dome"]',
    '["industrial"="biogas"]',
]

# Hydrogen infrastructure
ENERGY_TAGS += [
    '["plant:source"="hydrogen"]',
    '["generator:source"="hydrogen"]',
    '["substance"="hydrogen"]',
    '["content"="hydrogen"]',
    '["industrial"="hydrogen"]',
    '["product"="hydrogen"]',
    '["storage"="hydrogen"]',
]

INDUSTRIAL_TAGS = [
    '["landuse"="industrial"]', '["building"="industrial"]', '["man_made"="works"]'
]

ALL_TAG_GROUPS = GAS_INFRA_TAGS + ENERGY_TAGS + INDUSTRIAL_TAGS


def build_query_for_tags(bbox, tag_list, overpass_timeout_s=60, out_stmt="out tags center qt;"):
    """
    Build a query for a given list of tag filters.
    Keeps semantics: same tags, same bbox.
    """
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"
    query_blocks = "\n".join(f"nwr{tag}({bbox_str});" for tag in tag_list)
    return f"[out:json][timeout:{int(overpass_timeout_s)}];\n(\n{query_blocks}\n);\n{out_stmt}"

def _chunk_list(xs, chunk_size):
    if chunk_size <= 0:
        return [xs]
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]

def _is_transient_error(exc: Exception) -> bool:
    # Network-level timeouts / connection problems are transient
    if isinstance(exc, (Timeout, ConnectionError)):
        return True
    if isinstance(exc, HTTPError):
        resp = getattr(exc, "response", None)
        if resp is not None and resp.status_code in TRANSIENT_HTTP_CODES:
            return True
    return False

def _post_overpass(url: str, query: str, headers: dict, request_timeout_s: int):
    r = requests.post(url, data={"data": query}, headers=headers, timeout=request_timeout_s)
    # Explicitly raise for status to enter HTTPError branch
    r.raise_for_status()
    return r.json()

def km_to_bbox(lat, lon, buffer_km):
    km_per_deg_lat = 111.32
    lat_delta = buffer_km / km_per_deg_lat
    km_per_deg_lon = km_per_deg_lat * math.cos(math.radians(lat))
    if km_per_deg_lon == 0: km_per_deg_lon = 0.001
    lon_delta = buffer_km / km_per_deg_lon
    return (lat - lat_delta, lon - lon_delta, lat + lat_delta, lon + lon_delta)

def build_query(bbox):
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"
    query_blocks = "\n".join(f"nwr{tag}({bbox_str});" for tag in ALL_TAG_GROUPS)
    return f"[out:json][timeout:90];\n(\n{query_blocks}\n);\nout body center;"

def fetch_osm_task_old(lat, lon, buffer):
    try:
        bbox = km_to_bbox(lat, lon, buffer)
        query = build_query(bbox)
        headers = {"User-Agent": "EnergyNodeClassifier/1.0", "Accept": "*/*"}
        r = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=180)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return {"elements": []}, str(e)


def fetch_osm_task(
        lat,
        lon,
        buffer,
        *,
        # Robustness knobs
        max_endpoints_to_try: int = 3,      # how many different servers to try
        retries_per_endpoint: int = 2,      # how many retries per server
        base_backoff_s: float = 0.6,        # jittered exponential backoff base
        max_backoff_s: float = 4.0,
        request_timeout_s: int = 45,        # http timeout
        overpass_timeout_s: int = 35,       # timeout inside overpass query
        split_into_batches: bool = True,
        batch_size: int = 5,               # number of nwr[...] blocks per query
):
    print(f"[OSM] START lat={lat} lon={lon} buffer={buffer}")
    try:
        bbox = km_to_bbox(lat, lon, buffer)

        if split_into_batches:
            tag_batches = _chunk_list(ALL_TAG_GROUPS, batch_size)
        else:
            tag_batches = [ALL_TAG_GROUPS]

        urls = OVERPASS_URLS[:]
        random.shuffle(urls)

        headers = {"User-Agent": "EnergyNodeClassifier/1.0", "Accept": "*/*"}

        merged_elements = []
        batch_errors = []

        for batch_idx, tag_list in enumerate(tag_batches, start=1):
            query = build_query_for_tags(
                bbox=bbox,
                tag_list=tag_list,
                overpass_timeout_s=overpass_timeout_s,
                out_stmt="out tags center qt;",
            )

            batch_success = False
            last_err = None

            for u_i, url in enumerate(urls[:max_endpoints_to_try], start=1):
                for attempt in range(retries_per_endpoint + 1):
                    try:
                        data = _post_overpass(url, query, headers, request_timeout_s)
                        merged_elements.extend(data.get("elements", []) or [])
                        batch_success = True
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e

                        if not _is_transient_error(e):
                            break

                        exp = min(max_backoff_s, base_backoff_s * (2 ** attempt))
                        sleep_s = random.uniform(exp * 0.5, exp * 1.5)
                        time.sleep(sleep_s)

                if batch_success:
                    break
                # if not successful ==> rotate to next endpoint

            if not batch_success:
                # Keep going: partial results are better than total failure
                msg = str(last_err)
                batch_errors.append(f"batch {batch_idx}/{len(tag_batches)} failed: {msg}")

        # Deduplicate by type_id if present
        seen = set()
        deduped = []
        for el in merged_elements:
            key = (el.get("type"), el.get("id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(el)

        result = {"elements": deduped}
        print(f"[OSM] END success elements={len(deduped)}")

        if batch_errors and not deduped:
            return {"elements": []}, " | ".join(batch_errors)
        if batch_errors:
            return result, "PARTIAL: " + " | ".join(batch_errors)

        return result, None

    except Exception as e:
        return {"elements": []}, str(e)


def categorize(tags):
    if not tags:
        return "other"

    def norm(s: str) -> str:
        return (s or "").strip().lower()

    # Common fields
    power = norm(tags.get("power", ""))
    man_made = norm(tags.get("man_made", ""))
    pipeline = norm(tags.get("pipeline", ""))
    name = norm(tags.get("name", ""))
    plant_source = norm(tags.get("plant:source", ""))
    gen_source = norm(tags.get("generator:source", ""))
    gen_method = norm(tags.get("generator:method", ""))
    gen_type = norm(tags.get("generator:type", ""))
    substance = norm(tags.get("substance", "")) or norm(tags.get("content", ""))

    fuel_blob = " ".join([
        plant_source, gen_source, gen_method, gen_type, substance,
        " ".join([str(v).lower() for v in tags.values() if isinstance(v, (str, int, float))])
    ])

    if power in ("pole", "tower") or man_made in ("utility_pole",):
        return "electricity_pole"

    if "geothermal" in fuel_blob:
        return "geothermal_power_plant"

    if power == "substation":
        return "electrical_substation"

    # h2
    hydrogen_terms = [
        "hydrogen",
        "h2",
        "electrolyser",
        "electrolyzer",
        "electrolysis",
        "power to gas",
        "power-to-gas",
        "power to x",
        "power-to-x",
    ]

    if any(term in fuel_blob for term in hydrogen_terms):
        return "hydrogen"

    # solar/wind
    if power == "solar_farm" or "photovoltaic" in fuel_blob or "solar" in fuel_blob:
        return "solar_farm"
    if power == "wind_farm" or "wind_turbine" in fuel_blob or "wind" in fuel_blob:
        return "wind_farm"

    # gas infra
    if "compressor" in fuel_blob:
        return "gas_compressor_station"
    if man_made == "gasometer" or ("storage_tank" in man_made and "gas" in fuel_blob):
        return "gas_storage"
    if pipeline == "substation" or ("pressure" in name and "regulation" in name):
        return "gas_substation_prs"
    if pipeline in ["valve", "marker"] or man_made == "pipeline_valve":
        return "gas_pipeline_marker"
    if "pipeline" in tags or "gas" in fuel_blob:
        if "water" not in fuel_blob and "heat" not in fuel_blob:
            return "gas_pipeline"

    # Power plant / generator classification
    if power in ["plant", "generator", "station"]:
        if "nuclear" in fuel_blob:
            return "nuclear_power_plant"
        if "coal" in fuel_blob or "lignite" in fuel_blob or "hardcoal" in fuel_blob or "hard_coal" in fuel_blob:
            return "coal_power_plant"
        if "gas" in fuel_blob or "natural_gas" in fuel_blob or "lng" in fuel_blob:
            return "gas_power_plant"
        if "oil" in fuel_blob or "diesel" in fuel_blob or "fuel_oil" in fuel_blob:
            return "oil_power_plant"
        if "biomass" in fuel_blob or "waste" in fuel_blob or "refuse" in fuel_blob:
            return "biomass_power_plant"
        if "biogas" in fuel_blob or "anaerobic" in fuel_blob:
            return "biogas_power_plant"
        if "hydro" in fuel_blob or "water" in fuel_blob:
            return "hydro"
        return "power_plant_other"

    if tags.get("landuse") == "industrial" or tags.get("building") == "industrial" or man_made == "works":
        return "industrial_area"

    return "other"



def _build_popup_html(cat: str, name: str, lat: float, lon: float, tags: dict) -> str:
    safe_cat = _html.escape((cat or "other").replace("_", " ").upper())
    safe_name = _html.escape(name or "Unnamed")

    # tags can have non-primitive values; stringify safely
    tag_rows = []
    if isinstance(tags, dict) and tags:
        for k in sorted(tags.keys()):
            v = tags.get(k)
            try:
                vs = _html.escape(str(v))
            except Exception:
                vs = _html.escape(repr(v))
            ks = _html.escape(str(k))
            tag_rows.append(
                f"<tr>"
                f"<td style='padding:2px 6px; font-family:monospace; vertical-align:top; white-space:nowrap;'>{ks}</td>"
                f"<td style='padding:2px 6px; vertical-align:top; word-break:break-word;'>{vs}</td>"
                f"</tr>"
            )
    else:
        tag_rows.append("<tr><td style='padding:6px; opacity:.75;'>No tags available</td></tr>")

    rows_html = "".join(tag_rows)

    return f"""
    <div style="max-width:380px;">
      <div style="font-weight:700; margin-bottom:4px;">{safe_cat}</div>
      <div style="margin-bottom:6px;">{safe_name}</div>
      <div style="font-size:11px; opacity:.75; margin-bottom:8px;">
        {lat:.5f}, {lon:.5f}
      </div>
      <div style="max-height:240px; overflow:auto; border:1px solid #ddd; border-radius:6px;">
        <table style="width:100%; border-collapse:collapse; font-size:12px;">
          {rows_html}
        </table>
      </div>
    </div>
    """

def generate_map_html(lat, lon, buffer_km, data):
    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        el_lat = el.get("lat") or el.get("center", {}).get("lat")
        el_lon = el.get("lon") or el.get("center", {}).get("lon")
        if el_lat is not None and el_lon is not None:
            rows.append({
                "lat": float(el_lat),
                "lon": float(el_lon),
                "category": categorize(tags),
                "name": tags.get("name", "Unnamed"),
                "tags": tags
            })

    class_counts = dict(Counter(r["category"] for r in rows))
    features = rows

    # Base map
    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.map.CustomPane("font_awesome_fix").add_to(m)
    m.get_root().header.add_child(
        folium.Element(
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">'
        )
    )

    folium.Circle([lat, lon], radius=buffer_km * 1000, color="blue", fill=True, opacity=0.1).add_to(m)
    folium.Marker(
        [lat, lon],
        popup=f"Center<br>{len(rows)} items found",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

    style_config = {
        "electrical_substation": {"color": "purple", "icon": "bolt"},
        "geothermal_power_plant": {"color": "darkgreen", "icon": "thermometer-half", "prefix": "fa"},
        "coal_power_plant": {"color": "black", "icon": "industry", "prefix": "fa"},
        "nuclear_power_plant": {"color": "darkpurple", "icon": "radiation", "prefix": "fa"},
        "gas_power_plant": {"color": "darkred", "icon": "fire"},
        "oil_power_plant": {"color": "darkred", "icon": "tint", "prefix": "fa"},
        "biomass_power_plant": {"color": "green", "icon": "leaf", "prefix": "fa"},
        "biogas_power_plant": {"color": "green", "icon": "leaf", "prefix": "fa"},
        "hydro": {"color": "blue", "icon": "tint", "prefix": "fa"},
        "gas_compressor_station": {"color": "red", "icon": "arrows-alt", "prefix": "fa"},
        "gas_storage": {"color": "red", "icon": "database", "prefix": "fa"},
        "gas_substation_prs": {"color": "orange", "icon": "random", "prefix": "fa"},
        "gas_pipeline": {"color": "orange", "icon": "minus", "prefix": "fa"},
        "gas_pipeline_marker": {"color": "orange", "icon": "map-marker", "prefix": "fa"},
        "solar_farm": {"color": "orange", "icon": "sun-o", "prefix": "fa"},
        "wind_farm": {"color": "lightblue", "icon": "leaf", "prefix": "fa"},
        "industrial_area": {"color": "gray", "icon": "building", "prefix": "fa"},
        "power_plant_other": {"color": "darkred", "icon": "industry", "prefix": "fa"},
        "electricity_pole": {"color": "cadetblue", "icon": "flash", "prefix": "glyphicon"},
        "other": {"color": "lightgray", "icon": "info-sign"},
    }

    # new FeatureGroup per category -> toggle
    category_groups = {}  # category -> folium.FeatureGroup
    for cat in class_counts.keys():
        category_groups[cat] = folium.FeatureGroup(name=cat, show=True)
        category_groups[cat].add_to(m)

    # Markers into the category feature groups (cluster per category group)
    clusters = {}
    for cat, fg in category_groups.items():
        clusters[cat] = MarkerCluster(name=f"{cat}_cluster").add_to(fg)

    for r in rows:
        cat = r["category"]
        style = style_config.get(cat, style_config["other"])
        icon_args = {"color": style["color"], "icon": style["icon"], "prefix": style.get("prefix", "glyphicon")}

        popup_content = _build_popup_html(cat, r["name"], r["lat"], r["lon"], r.get("tags", {}) or {})

        folium.Marker(
            [r["lat"], r["lon"]],
            popup=folium.Popup(popup_content, max_width=420),
            tooltip=f"{cat}: {r['name']}",
            icon=folium.Icon(**icon_args),
        ).add_to(clusters.get(cat, clusters[list(clusters.keys())[0]]))

    cat_js = {cat: fg.get_name() for cat, fg in category_groups.items()}
    cat_js_json = json.dumps(cat_js)

    bridge_js = f"""
    <script>
      (function() {{
        const CAT_LAYERS = {cat_js_json}; // category -> leaflet layer var name (string)
        function getMapInstance() {{
          for (const k in window) {{
            try {{
              const v = window[k];
              if (v && v instanceof L.Map) return v;
            }} catch(e) {{}}
          }}
          return null;
        }}

        function setVisibleCategories(categories) {{
          const map = getMapInstance();
          if (!map) return;

          const wanted = new Set((categories || []).map(String));
          const showAll = wanted.size === 0;

          Object.keys(CAT_LAYERS).forEach(cat => {{
            const varName = CAT_LAYERS[cat];
            const layer = window[varName];
            if (!layer) return;

            const shouldShow = showAll || wanted.has(cat);

            const onMap = map.hasLayer(layer);
            if (shouldShow && !onMap) layer.addTo(map);
            if (!shouldShow && onMap) map.removeLayer(layer);
          }});
        }}

        window.addEventListener("message", function(ev) {{
          const msg = ev && ev.data;
          if (!msg || typeof msg !== "object") return;
          if (msg.type !== "OSM_FILTER") return;
          setVisibleCategories(msg.categories || []);
        }});
      }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(bridge_js))

    html_out = m.get_root().render()
    return html_out, class_counts, features

# Mapbox Satellite Image API call
def fetch_mapbox_image(lat, lon, buffer_km, api_key):
    if not api_key:
        raise ValueError("Mapbox API Key is missing")

    REQ_WIDTH = 640
    REQ_HEIGHT = 640
    buffer_meters = buffer_km * 1000
    span_meters = buffer_meters * 2
    required_m_per_px = span_meters / REQ_WIDTH
    lat_rad = math.radians(lat)
    scale_factor = math.cos(lat_rad)
    initial_resolution = 156543.03392 * scale_factor
    zoom = math.log2(initial_resolution / required_m_per_px)
    zoom_level = round(zoom - 0.1, 2)
    zoom_level = max(0, min(zoom_level, 20))

    style_id = "mapbox/satellite-v9"
    url = f"https://api.mapbox.com/styles/v1/{style_id}/static/{lon},{lat},{zoom_level}/{REQ_WIDTH}x{REQ_HEIGHT}@2x"

    response = requests.get(url, params={"access_token": api_key}, timeout=30)

    if response.status_code == 200:
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        raise Exception(f"Mapbox Error {response.status_code}: {response.text[:200]}")


def run_classification_task_from_image(raw_image: Image.Image, model_key: str):
    t0 = time.time()
    try:
        if raw_image is None:
            return [], None, None, None, "Missing satellite image"

        print(f"[CLS] start model={model_key}")
        t1 = time.time()
        with CLASSIFIER_LOCK:
            results, _, heatmap_top, heatmaps_by_label = classifier.predict(raw_image, model_key, gradcam_all=True)
        print(f"[CLS] predict done in {time.time()-t1:.2f}s total={time.time()-t0:.2f}s")
        return results, raw_image, heatmap_top, heatmaps_by_label, None
    except Exception as e:
        print(f"[CLS] failed after {time.time()-t0:.2f}s: {e}")
        traceback.print_exc()
        return [], None, None, None, str(e)


def run_detection_task_from_image(image: Image.Image, model_key: str):
    try:
        if image is None:
            return [], None, "Missing satellite image"

        # detector.predict -> detections+ annotated_image
        with DETECTOR_LOCK:
            results, annotated_image = detector.predict(image, model_key)
        return results, annotated_image, None
    except Exception as e:
        return [], None, str(e)









#####################
### API ENDPOINTS ###
#####################
@app.post("/validate-key")
def validate_key(req: KeyValidationRequest):
    is_valid = False
    message = "Unknown Service"

    if req.service == 'aws':
        # cheap dummy request
        records, error = fetch_aws_task(49.4438, 8.5020, 0.2, req.key, limit=1)

        if error == "Invalid Key":
            return {"valid": False, "message": "Invalid AWS API key"}
        if error:
            return {"valid": False, "message": f"Connection Failed: {error}"}
        return {"valid": True, "message": "AWS Key Verified"}

    elif req.service == 'gemini':
        ok, msg = validate_gemini_key(req.key)
        return {"valid": ok, "message": msg}

    elif req.service == 'sentinel':
        if req.key and len(req.key.strip()) > 10:
            return {"valid": True, "message": "Sentinel key accepted (not used by Landsat heat pipeline)."}
        return {"valid": False, "message": "Sentinel key too short"}

    elif req.service == 'mapbox':
        test_url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/0,0,0/1x1"
        try:
            r = requests.get(test_url, params={"access_token": req.key}, timeout=5)
            if r.status_code == 200:
                is_valid = True
                message = "Mapbox Key Verified"
            elif r.status_code == 401 or r.status_code == 403:
                is_valid = False
                message = "Invalid Mapbox Token"
            else:
                is_valid = False
                message = f"Mapbox Error {r.status_code}"
        except Exception as e:
            is_valid = False
            message = f"Connection Failed: {str(e)}"


    return {"valid": is_valid, "message": message}



@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    summary_parts = [f"Analysis for ({request.latitude}, {request.longitude})"]

    osm_result = None
    osm_class_counts = None
    osm_features = None

    db_records = []

    agent_report = None
    agent_confidence = None
    agent_review_summary = None
    agent_rationale = None
    agent_key_evidence = None
    agent_reviewed_inputs = ["image", "web"]
    final_prediction = None

    classification_data = []
    detection_data = []

    satellite_image_base64 = None
    heatmap_images_base64 = None
    heatmap_image_base64 = None
    detection_image_base64 = None

    heat_radiation_image_b64 = None
    heat_radiation_result = None

    # Shared satellite image for all image-based stages (classification / detection / agent)
    shared_satellite_img = None

    # Local helper to avoid duplicate encoding code
    def _pil_to_b64_png(img: Optional[Image.Image]) -> Optional[str]:
        if img is None:
            return None
        try:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"PNG encode error: {e}")
            return None

    # Fetch satellite image ONCE if any image-dependent stage is enabled
    need_sat_image = request.run_classification or request.run_object_detection or request.run_agent
    if need_sat_image:
        try:
            shared_satellite_img = fetch_mapbox_image(
                request.latitude,
                request.longitude,
                request.buffer,
                request.mapbox_api_key
            )
            satellite_image_base64 = _pil_to_b64_png(shared_satellite_img)
            summary_parts.append("Satellite: OK")
        except Exception as e:
            summary_parts.append(f"Satellite: Failed ({e})")
            print(f"Satellite fetch error: {e}")

    # Run network / external tasks concurrently.
    # ML image inference is intentionally run SEQUENTIALLY later for stability.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_osm = None
        future_aws = None
        future_heat = None

        if request.run_osm:
            # Cap OSM buffer for stability on dense areas / large queries
            osm_buffer = min(request.buffer, 5.0)
            if osm_buffer < request.buffer:
                summary_parts.append(f"OSM buffer capped to {osm_buffer}km")
            future_osm = executor.submit(fetch_osm_task, request.latitude, request.longitude, osm_buffer)

        if request.run_database:
            future_aws = executor.submit(
                fetch_aws_task,
                request.latitude,
                request.longitude,
                request.buffer,
                request.aws_api_key
            )

        if request.run_heat_radiation:
            future_heat = executor.submit(
                run_heat_radiation,
                request.latitude,
                request.longitude,
                request.heat_start_date,
                request.heat_end_date,
                request.heat_max_cloud,
                request.heat_max_scenes,
                request.heat_aoi_m,
                request.heat_ring_inner_m,
                request.heat_ring_outer_m,
                request.heat_hotspot_offset_c,
            )

        # OSM
        if future_osm:
            try:
                data, error = future_osm.result()
            except Exception as e:
                data, error = {"elements": []}, str(e)

            elements = (data or {}).get("elements", []) if isinstance(data, dict) else []
            if error and not elements:
                summary_parts.append("OSM: Failed")
                print(f"OSM Error: {error}")
            else:
                if error:
                    summary_parts.append(f"OSM: Partial ({error})")
                    print(f"OSM Partial Warning: {error}")

                count = len(elements)
                summary_parts.append(f"OSM: {count} elements")

                try:
                    html, class_counts, features = generate_map_html(
                        request.latitude,
                        request.longitude,
                        min(request.buffer, 5.0) if request.run_osm else request.buffer,
                        data
                    )
                    osm_result = html
                    osm_class_counts = class_counts
                    osm_features = features
                except Exception as e:
                    summary_parts.append("OSM Map: Gen Error")
                    print(f"Map Gen Error: {e}")

        # AWS DB
        if future_aws:
            try:
                records, error = future_aws.result()
            except Exception as e:
                records, error = [], str(e)

            if error:
                summary_parts.append("DB: Failed")
                print(f"DB Error: {error}")
            else:
                db_records = records
                summary_parts.append(f"DB: {len(db_records)} records")

        # Heat Radiation
        if future_heat:
            try:
                heat_radiation_result, heat_radiation_image_b64 = future_heat.result()
                if isinstance(heat_radiation_result, dict) and heat_radiation_result.get("heat_ranked"):
                    top_cls, top_p = heat_radiation_result["heat_ranked"][0]
                    summary_parts.append(f"Heat: {top_cls} ({top_p:.2f})")
                else:
                    summary_parts.append("Heat: Completed")
            except Exception as e:
                summary_parts.append("Heat: Failed")
                heat_radiation_result = {"error": str(e)}
                heat_radiation_image_b64 = None
                print(f"Heat Radiation Error: {e}")

    # ---- Sequential ML stages (more stable than parallel on shared ML/GPU backends) ----

    # Classification
    if request.run_classification:
        if shared_satellite_img is None:
            summary_parts.append("Classify: Skipped (no satellite image)")
        else:
            try:
                # Reuse existing helper; fetches image internally in old version, so do direct model call here
                t0 = time.time()
                with CLASSIFIER_LOCK:
                    # classifier.predict returns (results, raw_img, heatmap_obj) in your wrapper usage
                    # In your current code you ignore the 2nd return and keep the original image.
                    results, _, heatmap_obj, heatmaps_by_label = classifier.predict(
                        shared_satellite_img.copy(),
                        request.classification_model,
                        gradcam_all=False,
                    )

                classification_data = results or []

                # Encode ALL heatmaps
                if heatmaps_by_label:
                    heatmap_images_base64 = {
                        label: _pil_to_b64_png(pil_img)
                        for label, pil_img in heatmaps_by_label.items()
                        if pil_img is not None
                    }

                # Keep legacy single heatmap too (top class)
                if heatmap_obj:
                    heatmap_image_base64 = _pil_to_b64_png(heatmap_obj)

                classification_data = results or []
                top = classification_data[0]["label"] if classification_data else "Unknown"
                summary_parts.append(f"Classify: {top}")
                print(f"[CLS] done in {time.time()-t0:.2f}s")

            except Exception as e:
                summary_parts.append(f"Classify: Failed ({e})")
                print(f"[CLS] Error: {e}")

    # Object Detection
    if request.run_object_detection:
        if shared_satellite_img is None:
            summary_parts.append("Detect: Skipped (no satellite image)")
        else:
            try:
                t0 = time.time()
                with DETECTOR_LOCK:
                    # detector.predict returns (detections, annotated_image)
                    res, annotated_img_obj = detector.predict(shared_satellite_img.copy(), request.detection_model)

                detection_data = res or []
                summary_parts.append(f"Detect: {len(detection_data)} objs")
                print(f"[DET] done in {time.time()-t0:.2f}s")

                if annotated_img_obj:
                    detection_image_base64 = _pil_to_b64_png(annotated_img_obj)

            except Exception as e:
                summary_parts.append(f"Detect: Failed ({e})")
                print(f"[DET] Error: {e}")

    # ---- Websearch Agent (reuses same satellite image) ----
    if request.run_agent:
        places_keywords = [
            "power plant",
            "wind farm",
            "solar farm",
            "electrical substation",
            "gas compressor station",
            "biogas plant",
            "industrial plant",
        ]

        places = google_places_search(
            request.latitude,
            request.longitude,
            int(request.buffer * 1000),
            request.google_maps_api_key,
            places_keywords
        )

        places_memo = build_places_memo(
            places,
            request.latitude,
            request.longitude
        )

        try:
            if shared_satellite_img is None:
                raise ValueError("No satellite image available for agent")

            # get address:
            geolocator = Nominatim(user_agent="Energy Node Classifier")

            # enforces min delay between requests to avoid rate limiting errors
            reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

            def reverse_geocode(lat, lon):
                location = reverse((lat, lon), exactly_one=True)
                return location.address if location else None

            try:
                address = reverse_geocode(request.latitude, request.longitude)
            except Exception as e:
                address = None

            # search based on coordinates
            search_query = (
                f"wind park, solar park, power station, biogas facility, industry, "
                f"electrical substation or gas infrastructure near {address}" 
                f" with exact coordinates ({request.latitude}, {request.longitude})"
            )
            print(f"Search query: {search_query}")

            # grounded google search
            web = gemini_grounded_search(request.gemini_api_key, search_query)

            web_memo = web.get("memo", "")
            web_error = web.get("error", "")

            places_memo = build_places_memo(
                google_places_search(
                    request.latitude,
                    request.longitude,
                    int(request.buffer * 1000),
                    request.google_maps_api_key,
                    places_keywords
                ),
                request.latitude,
                request.longitude
            )

            combined_memo = f"""
            Google Places Results:
            {places_memo}
            
            Web Search Results:
            {web_memo}
            """
            web_error = web.get("error", "")

            # prompt
            prompt = build_agent_prompt_image_web_only(
                request.latitude,
                request.longitude,
                request.buffer,
                combined_memo,
                web_error
            )

            # call model with json (use shared image)
            agent_text = call_gemini_with_image(
                request.gemini_api_key,
                prompt,
                shared_satellite_img.copy()
            )

            # safe json parsing
            raw_json = extract_first_json_object(agent_text)
            if not raw_json:
                fixed = repair_json_with_gemini(request.gemini_api_key, agent_text)
                raw_json = extract_first_json_object(fixed)

            if not raw_json:
                raise ValueError("Agent response did not contain parseable JSON")

            agent_json = json.loads(raw_json)

            # normalize to fixed classes
            pred_raw = str(agent_json.get("final_prediction", "Negative"))
            found_class = "Negative"
            for c in ALLOWED_AGENT_CLASSES:
                if pred_raw.lower() == c.lower():
                    found_class = c
                    break
            if "turbine" in pred_raw.lower():
                found_class = "Wind"
            if "panel" in pred_raw.lower() or "photovoltaic" in pred_raw.lower():
                found_class = "Solar"

            final_prediction = found_class
            conf = float(agent_json.get("confidence", 0.0))

            review_summary = agent_json.get("review_summary") or "No web summary provided."
            rationale = agent_json.get("rationale") or "Classified based on available satellite imagery features."
            evidence = agent_json.get("key_evidence") or ["Visual: Features consistent with prediction"]

            if not isinstance(evidence, list):
                evidence = [str(evidence)]

            agent_report = (
                    f"Final: {final_prediction} (confidence={conf:.2f})\n\n"
                    f"Review summary: {review_summary}\n\n"
                    f"Rationale: {rationale}\n\n"
                    f"Evidence:\n- " + "\n- ".join([str(x) for x in evidence])
            )

            summary_parts.append(f"Agent: {final_prediction}")

            agent_confidence = conf
            agent_review_summary = review_summary
            agent_rationale = rationale
            agent_key_evidence = [str(x) for x in evidence]

        except Exception as e:
            print(f"Agent Error: {e}")
            summary_parts.append("Agent: Failed")
            agent_report = f"Agent failed: {str(e)}"
            final_prediction = "Negative"
            agent_confidence = 0.0

    fusion_result = None

    try:
        cls_entry = classifier.models.get(request.classification_model)
        if cls_entry:
            classifier_classes = cls_entry["class_names"]
        else:
            classifier_classes = [r["label"] for r in classification_data]

        fusion_result = fusion_engine.run(
            lat=request.latitude,
            lon=request.longitude,
            classifier_classes=classifier_classes,
            classification=classification_data if request.run_classification else [],
            detections=detection_data if request.run_object_detection else [],
            osm_features=osm_features if request.run_osm else [],
            db_records=db_records if request.run_database else [],
            agent_prediction=final_prediction if request.run_agent else None,
            agent_confidence=agent_confidence if request.run_agent else 0.0,
            weights=request.fusion_weights or {
                "image": 0.4,
                "osm": 0.25,
                "database": 0.25,
                "agent": 0.10
            },
            active_modalities={
                "image": request.run_classification or request.run_object_detection,
                "osm": request.run_osm,
                "database": request.run_database,
                "agent": request.run_agent
            },
            lambda_m=request.buffer * 1000.0
        )

        summary_parts.append(f"Fusion: {fusion_result['classes']}")

    except Exception as e:
        print("Fusion failed:", e)

    return PredictionResponse(
        summary=" | ".join(summary_parts),

        osm_html=osm_result,
        osm_class_counts=osm_class_counts,
        osm_features=osm_features,

        database_records=db_records,
        classification_results=classification_data,
        heatmap_images=heatmap_images_base64,
        detection_results=detection_data,

        satellite_image=satellite_image_base64,
        heatmap_image=heatmap_image_base64,
        detection_image=detection_image_base64,

        agent_report=agent_report,
        final_prediction=final_prediction,
        agent_confidence=agent_confidence,
        agent_review_summary=agent_review_summary,
        agent_rationale=agent_rationale,
        agent_key_evidence=agent_key_evidence,
        agent_reviewed_inputs=agent_reviewed_inputs,

        heat_radiation_image=heat_radiation_image_b64,
        heat_radiation_result=heat_radiation_result,

        prediction="Processed",

        fusion_result=fusion_result,
    )




@app.post("/batch")
async def run_batch(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),

        # --- API Keys ---
        aws_api_key: str = Form(""),
        mapbox_api_key: str = Form(""),
        gemini_api_key: str = Form(""),
        google_maps_api_key: str = Form(""),

        # --- Feature Flags ---
        run_osm: bool = Form(False),
        run_database: bool = Form(False),
        run_classification: bool = Form(False),
        run_object_detection: bool = Form(False),
        run_agent: bool = Form(False),
        run_heat_radiation: bool = Form(False),

        classification_model: str = Form("convnext_large"),
        detection_model: str = Form("yolo26"),

        # fusion weights (json string)
        fusion_weights: str = Form("{}"),
):
    """
    Batch processing endpoint.
    Accepts:
        - CSV / XLSX file
        - All sidebar configuration via FormData
    """

    # parse uploaded files
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

    # Validate required columns
    required_cols = {"latitude", "longitude", "buffer"}
    if not required_cols.issubset(df.columns):
        return {
            "error": "Input file must contain columns: latitude, longitude, buffer"
        }

    rows = []
    for _, r in df.iterrows():
        try:
            rows.append({
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "buffer": float(r["buffer"]),
            })
        except Exception:
            continue

    if len(rows) == 0:
        return {"error": "No valid rows found in file"}

    # parse fusion weights
    try:
        fusion_weights_dict = json.loads(fusion_weights)
        if not isinstance(fusion_weights_dict, dict):
            fusion_weights_dict = {}
    except Exception:
        fusion_weights_dict = {}

    # building param dict
    parameters = {
        "aws_api_key": aws_api_key,
        "mapbox_api_key": mapbox_api_key,
        "gemini_api_key": gemini_api_key,
        "google_maps_api_key": google_maps_api_key,

        "run_osm": run_osm,
        "run_database": run_database,
        "run_classification": run_classification,
        "run_object_detection": run_object_detection,
        "run_agent": run_agent,
        "run_heat_radiation": run_heat_radiation,

        "classification_model": classification_model,
        "detection_model": detection_model,

        "fusion_weights": fusion_weights_dict,
    }


    job_id = batch_manager.create_job()

    # start background task
    background_tasks.add_task(
        batch_manager.run_batch,
        job_id=job_id,
        rows=rows,
        pipeline_callable=lambda lat, lon, buffer, parameters, fusion_engine:
        run_single_pipeline(
            lat=lat,
            lon=lon,
            buffer=buffer,
            parameters=parameters,
            fusion_engine=fusion_engine,
            fetch_osm_task=fetch_osm_task,
            generate_map_html=generate_map_html,
            fetch_aws_task=fetch_aws_task,
            fetch_mapbox_image=fetch_mapbox_image,
            classifier=classifier,
            detector=detector,
            classifier_lock=CLASSIFIER_LOCK,
            detector_lock=DETECTOR_LOCK,
            reverse_geocode_fn=lambda la, lo: Nominatim(
                user_agent="EnergyNodeClassifier"
            ).reverse((la, lo), exactly_one=True).address
        ),
        parameters=parameters,
        fusion_engine=fusion_engine
    )

    return {"job_id": job_id}

@app.get("/batch/{job_id}")
def get_batch_status(job_id: str):
    job = batch_manager.get_job(job_id)
    if not job:
        return {"error": "not found"}
    return job


@app.get("/batch/{job_id}/download")
def download_batch(job_id: str):
    job = batch_manager.get_job(job_id)
    if not job or job["status"] != "finished":
        return {"error": "not ready"}

    return FileResponse(job["file"], filename=f"{job_id}.csv")

@app.get("/")
def read_root():
    return {"message": "Backend is running"}
