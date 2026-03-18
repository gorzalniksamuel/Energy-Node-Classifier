from pydantic import BaseModel, Field
from typing import Optional, Any, List, Dict


class KeyValidationRequest(BaseModel):
    service: str
    key: str

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    buffer: float

    aws_api_key: Optional[str] = ""
    mapbox_api_key: Optional[str] = ""
    gemini_api_key: Optional[str] = ""
    google_maps_api_key: Optional[str] = ""
    sentinel_api_key: Optional[str] = ""

    # Feature Flags
    run_osm: bool = False
    run_database: bool = False
    run_classification: bool = False
    run_object_detection: bool = False
    run_agent: bool = False

    run_heat_radiation: bool = False

    # Model Selection
    classification_model: str = "convnext_large"
    detection_model: str = "yolo11"

    # Heat emissions parameters
    heat_start_date: str = "2024-06-01"
    heat_end_date: str = "2024-09-30"
    heat_max_cloud: float = 75.0
    heat_max_scenes: int = 20
    heat_aoi_m: float = 1000
    heat_ring_inner_m: float = 2000
    heat_ring_outer_m: float = 4500
    heat_hotspot_offset_c: float = 1.0

    fusion_weights: Optional[Dict[str, float]] = None

class OsmFeature(BaseModel):
    lat: float
    lon: float
    category: str
    name: str
    tags: Dict[str, Any]

class PredictionResponse(BaseModel):
    summary: str
    osm_html: Optional[str] = None
    osm_class_counts: Optional[Dict[str, int]] = None

    osm_features: Optional[List[OsmFeature]] = None

    database_records: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    classification_results: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    detection_results: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    satellite_image: Optional[str] = None
    heatmap_image: Optional[str] = None
    heatmap_images: Optional[Dict[str, str]] = None
    detection_image: Optional[str] = None

    agent_report: Optional[str] = None
    final_prediction: Optional[str] = None

    agent_confidence: Optional[float] = None
    agent_review_summary: Optional[str] = None
    agent_rationale: Optional[str] = None
    agent_key_evidence: Optional[List[str]] = None
    agent_reviewed_inputs: Optional[List[str]] = None

    heat_radiation_image: Optional[str] = None
    heat_radiation_result: Optional[Dict[str, Any]] = None

    fusion_result: Optional[Dict[str, Any]] = None

    prediction: Any = None

class BatchRequest(BaseModel):
    aws_api_key: Optional[str] = ""
    mapbox_api_key: Optional[str] = ""
    gemini_api_key: Optional[str] = ""
    sentinel_api_key: Optional[str] = ""

    run_osm: bool = False
    run_database: bool = False
    run_classification: bool = False
    run_object_detection: bool = False
    run_agent: bool = False
    run_heat_radiation: bool = False

    classification_model: str = "convnext_large"
    detection_model: str = "yolo11"

    fusion_weights: Optional[Dict[str, float]] = None
