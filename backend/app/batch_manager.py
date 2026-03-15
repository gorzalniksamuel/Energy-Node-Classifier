import csv
import uuid
import os
import traceback
from typing import Dict, Any, List
from datetime import datetime

BATCH_DIR = "batch_outputs"
os.makedirs(BATCH_DIR, exist_ok=True)

def _sanitize_parameters(parameters):
    blocked = {
        "aws_api_key",
        "mapbox_api_key",
        "gemini_api_key",
        "google_maps_api_key",
        "sentinel_api_key",
    }
    return {k: v for k, v in parameters.items() if k not in blocked}

class BatchManager:

    def __init__(self):
        self.jobs = {}

    def create_job(self):
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "total": 0,
            "file": None,
            "error": None
        }
        return job_id

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def run_batch(self, job_id, rows, pipeline_callable, parameters, fusion_engine):
        output_path = os.path.join(BATCH_DIR, f"{job_id}.csv")
        self.jobs[job_id]["file"] = output_path
        self.jobs[job_id]["total"] = len(rows)

        safe_parameters = _sanitize_parameters(parameters)

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "latitude",
                    "longitude",
                    "address",
                    "fusion_class",
                    "fusion_score",
                    "fusion_classes",
                    "classification_raw",
                    "detection_raw",
                    "osm_raw",
                    "database_raw",
                    "parameters",
                    "timestamp"
                ])

                import time

                for idx, row in enumerate(rows):

                    print(f"[BATCH] Starting row {idx+1}/{len(rows)}")
                    t0 = time.time()

                    try:
                        result = pipeline_callable(
                            lat=row["latitude"],
                            lon=row["longitude"],
                            buffer=row["buffer"],
                            parameters=parameters,
                            fusion_engine=fusion_engine
                        )

                        print(f"[BATCH] Row {idx+1} finished in {time.time() - t0:.2f}s")

                        fusion_scores = result.get("fusion_result", {}).get("scores", {})
                        fusion_classes = result.get("fusion_result", {}).get("classes", [])

                        classification_raw = str(result.get("classification_results", []))
                        detection_raw = str(result.get("detection_results", []))
                        osm_raw = str(result.get("osm_features", []))
                        db_raw = str(result.get("database_records", []))

                        for label, score in fusion_scores.items():
                            writer.writerow([
                                row["latitude"],
                                row["longitude"],
                                result.get("address", ""),
                                label,
                                round(score, 4),
                                ",".join(fusion_classes),
                                classification_raw,
                                detection_raw,
                                osm_raw,
                                db_raw,
                                str(safe_parameters),
                                datetime.utcnow().isoformat()
                            ])

                    except Exception as row_error:
                        writer.writerow([
                            row["latitude"],
                            row["longitude"],
                            "",
                            "ERROR",
                            0,
                            "",
                            "",
                            "",
                            "",
                            "",
                            str(safe_parameters),
                            datetime.utcnow().isoformat()
                        ])
                        print("Row failed:", row_error)
                        traceback.print_exc()

                    self.jobs[job_id]["progress"] = idx + 1

            self.jobs[job_id]["status"] = "finished"

        except Exception:
            self.jobs[job_id]["status"] = "error"
            self.jobs[job_id]["error"] = traceback.format_exc()