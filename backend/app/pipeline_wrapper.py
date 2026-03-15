from typing import Dict, Any


def run_single_pipeline(
        lat: float,
        lon: float,
        buffer: float,
        parameters: Dict[str, Any],
        fusion_engine,
        fetch_osm_task=None,
        generate_map_html=None,
        fetch_aws_task=None,
        fetch_mapbox_image=None,
        classifier=None,
        detector=None,
        classifier_lock=None,
        detector_lock=None,
        reverse_geocode_fn=None,
):

    run_osm = parameters.get("run_osm", False)
    run_database = parameters.get("run_database", False)
    run_classification = parameters.get("run_classification", False)
    run_object_detection = parameters.get("run_object_detection", False)

    weights = parameters.get("fusion_weights", {
        "image": 0.4,
        "osm": 0.25,
        "database": 0.25,
        "agent": 0.10
    })

    active_modalities = {
        "image": run_classification or run_object_detection,
        "osm": run_osm,
        "database": run_database,
        "agent": False
    }

    osm_features = []
    osm_class_counts = {}
    db_records = []
    classification_results = []
    detection_results = []
    classifier_classes = []

    shared_image = None

    # OSM
    if run_osm and fetch_osm_task and generate_map_html:
        data, _ = fetch_osm_task(lat, lon, buffer)
        if data:
            _, osm_class_counts, osm_features = generate_map_html(lat, lon, buffer, data)

    # DB
    if run_database and fetch_aws_task:
        db_records, _ = fetch_aws_task(
            lat,
            lon,
            buffer,
            parameters.get("aws_api_key", "")
        )

    # Image
    if (run_classification or run_object_detection) and fetch_mapbox_image:
        shared_image = fetch_mapbox_image(
            lat,
            lon,
            buffer,
            parameters.get("mapbox_api_key", "")
        )

    if run_classification and classifier and classifier_lock and shared_image is not None:
        with classifier_lock:
            results, _, _, _ = classifier.predict(
                shared_image.copy(),
                parameters.get("classification_model", "convnext_large")
            )
        classification_results = results or []
        classifier_classes = [r["label"] for r in classification_results]
    if run_object_detection and detector and detector_lock and shared_image is not None:
        with detector_lock:
            det, _ = detector.predict(
                shared_image.copy(),
                parameters.get("detection_model", "yolo11")
            )
        detection_results = det or []

    # data fusion
    fusion_result = fusion_engine.run(
        lat=lat,
        lon=lon,
        classifier_classes=classifier_classes,
        classification=classification_results,
        detections=detection_results,
        osm_features=osm_features,
        db_records=db_records,
        agent_prediction=None,
        agent_confidence=0.0,
        weights=weights,
        active_modalities=active_modalities,
    )

    address = ""
    if reverse_geocode_fn:
        try:
            address = reverse_geocode_fn(lat, lon)
        except Exception:
            address = ""

    return {
        "latitude": lat,
        "longitude": lon,
        "buffer": buffer,
        "address": address,
        "fusion_result": fusion_result,
        "osm_features": osm_features,
        "osm_class_counts": osm_class_counts,
        "database_records": db_records,
        "classification_results": classification_results,
        "detection_results": detection_results,
    }