"""FastAPI Backend Service for FireGuard AI (Render Cloud + Local REST API).

Uses lightweight ONNX Runtime inference (`models/best.onnx`) when available so
the API comfortably fits within Render Free's 512 MB RAM limit without loading
PyTorch, Ultralytics, Matplotlib, or Plotly. Falls back automatically to
PyTorch (`models/best.pt`) when ONNX weights are absent.
"""

from __future__ import annotations

import base64
import gc
import importlib.util
import io
import json
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402
from src.database import IncidentDatabase  # noqa: E402
from src.detector import FireSmokeDetector, ModelNotFoundError  # noqa: E402
from src.image_processor import InvalidImageError, process_image_bytes  # noqa: E402
from src.onnx_detector import OnnxFireSmokeDetector  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4"}
STATUS_READY = "MODEL READY"
STATUS_MISSING = "MODEL MISSING"
STATUS_INFERENCE_FAILED = "MODEL INFERENCE FAILED"
STATUS_CLASS_MISMATCH = "MODEL CLASS MISMATCH"

_STATE_LOCK = threading.RLock()
_DETECTOR: OnnxFireSmokeDetector | FireSmokeDetector | None = None
_HEALTH_CACHE: dict[str, Any] | None = None
_DATABASE: IncidentDatabase | None = None


def _configure_runtime_limits() -> None:
    """Keep CPU thread usage bounded for Render container memory stability."""
    config.ensure_directories()
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def get_database() -> IncidentDatabase:
    global _DATABASE
    with _STATE_LOCK:
        if _DATABASE is None:
            _DATABASE = IncidentDatabase(config.DB_PATH)
        return _DATABASE


def _can_use_onnx() -> bool:
    force_pt = os.environ.get("FIREGUARD_FORCE_PYTORCH", "").strip() == "1"
    if force_pt:
        return False
    onnx_path = config.MODELS_DIR / "best.onnx"
    return onnx_path.is_file() and importlib.util.find_spec("onnxruntime") is not None


def get_detector() -> OnnxFireSmokeDetector | FireSmokeDetector:
    """Return a single shared detector instance (ONNX preferred for low RAM)."""
    global _DETECTOR
    with _STATE_LOCK:
        if _DETECTOR is None:
            if _can_use_onnx():
                threads = int(os.environ.get("FIREGUARD_TORCH_THREADS", "2"))
                _DETECTOR = OnnxFireSmokeDetector(
                    model_path=config.MODELS_DIR / "best.onnx",
                    intra_threads=threads,
                )
            else:
                _DETECTOR = FireSmokeDetector(model_path=config.MODEL_PATH)
        return _DETECTOR


def _run_shared_health_check() -> dict[str, Any]:
    """Run startup diagnostics on the single shared detector (no duplicate model copy)."""
    detector = get_detector()
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    meta = _read_json_safe(config.MODELS_DIR / "model_metadata.json")
    meta_dict = meta if isinstance(meta, dict) else {}
    model_tier = str(meta_dict.get("model_tier", "UNKNOWN"))
    training_note = str(meta_dict.get("note", ""))

    target = detector.model_path
    if not target.is_file() or target.stat().st_size == 0:
        return {
            "status": STATUS_MISSING,
            "model_ready": False,
            "model_status_label": "Model unavailable",
            "model_tier": "NONE",
            "architecture": "Unavailable",
            "class_names": {},
            "execution_device": "cpu",
            "inference_latency_ms": None,
            "training_note": "No trained model weights found on disk.",
            "error_message": f"Trained weights not found at '{target}'.",
            "checked_at": checked_at,
        }

    is_onnx = isinstance(detector, OnnxFireSmokeDetector)
    arch = "yolo11n (onnxruntime, 640x640)" if is_onnx else "yolo11n (pytorch)"

    try:
        detector.load_model()
        classes = detector.class_names
    except Exception as exc:
        return {
            "status": STATUS_INFERENCE_FAILED,
            "model_ready": False,
            "model_status_label": "Model unavailable",
            "model_tier": model_tier,
            "architecture": arch,
            "class_names": {},
            "execution_device": "cpu",
            "inference_latency_ms": None,
            "training_note": training_note,
            "error_message": f"Failed to load model weights: {exc}",
            "checked_at": checked_at,
        }

    expected_set = set(config.TARGET_CLASSES)
    actual_set = set(classes.values())
    if not expected_set.issubset(actual_set):
        return {
            "status": STATUS_CLASS_MISMATCH,
            "model_ready": False,
            "model_status_label": "Model unavailable",
            "model_tier": model_tier,
            "architecture": arch,
            "class_names": classes,
            "execution_device": "cpu",
            "inference_latency_ms": None,
            "training_note": training_note,
            "error_message": f"Loaded model classes {classes} do not match fire/smoke.",
            "checked_at": checked_at,
        }

    probe_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    probe_frame[80:180, 90:190] = (30, 90, 220)
    try:
        t0 = time.perf_counter()
        detector.predict_frame(probe_frame, conf_threshold=0.25, imgsz=config.DEFAULT_IMAGE_SIZE)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    except Exception as exc:
        return {
            "status": STATUS_INFERENCE_FAILED,
            "model_ready": False,
            "model_status_label": "Model unavailable",
            "model_tier": model_tier,
            "architecture": arch,
            "class_names": classes,
            "execution_device": "cpu",
            "inference_latency_ms": None,
            "training_note": training_note,
            "error_message": f"Model probe inference failed: {exc}",
            "checked_at": checked_at,
        }

    gc.collect()
    return {
        "status": STATUS_READY,
        "model_ready": True,
        "model_status_label": "Model ready",
        "model_tier": model_tier,
        "architecture": arch,
        "class_names": {str(k): v for k, v in classes.items()},
        "execution_device": "cpu (onnxruntime)" if is_onnx else "cpu",
        "inference_latency_ms": latency_ms,
        "training_note": training_note,
        "error_message": None,
        "checked_at": checked_at,
    }


def get_health_report(force: bool = False) -> dict[str, Any]:
    global _HEALTH_CACHE
    with _STATE_LOCK:
        if force or _HEALTH_CACHE is None:
            _HEALTH_CACHE = _run_shared_health_check()
        return dict(_HEALTH_CACHE)


def _format_final_result_label(has_fire: bool, has_smoke: bool) -> str:
    if has_fire and has_smoke:
        return "Fire and Smoke Detected"
    if has_fire:
        return "Fire Detected"
    if has_smoke:
        return "Smoke Detected"
    return "No Fire or Smoke Detected"


def _format_confidence_display(confidence: float, has_detection: bool) -> str:
    if not has_detection or float(confidence) <= 0.0:
        return "N/A"
    return f"{float(confidence):.1%}"


def _verify_model_classes(detector: Any) -> None:
    names = detector.class_names
    normalized = {str(v).strip().lower() for v in names.values()}
    if not {"fire", "smoke"}.issubset(normalized):
        raise ValueError(
            f"Loaded model classes {names} do not match expected fire and smoke classes."
        )


def _encode_rgb_png_data_uri(rgb_arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb_arr).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _encode_png_bytes_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _read_json_safe(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _allowed_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _configure_runtime_limits()
    get_database()
    get_health_report(force=False)
    yield


app = FastAPI(
    title="FireGuard AI Backend API",
    description="YOLO11n Fire and Smoke Detection API designed for Render backend + Vercel frontend.",
    version=config.PROJECT_VERSION,
    lifespan=lifespan,
)

origins = _allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=(origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": config.PROJECT_NAME,
        "subtitle": "AI-Based Fire and Smoke Detection API",
        "version": config.PROJECT_VERSION,
        "status": "running",
        "health_endpoint": "/api/health",
    }


@app.get("/api/health")
def health(refresh: bool = False) -> dict[str, Any]:
    return get_health_report(force=refresh)


@app.post("/api/analyze")
async def analyze_media(file: UploadFile = File(...)) -> JSONResponse:
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    if ext not in (IMAGE_EXTS | VIDEO_EXTS):
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "final_result": "Analysis Failed",
                "visual_severity": "Analysis Failed",
                "error_message": "Unsupported file format. Please upload JPG, JPEG, PNG, or MP4.",
            },
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise InvalidImageError("Uploaded file is empty (0 bytes).")

        detector = get_detector()
        _verify_model_classes(detector)
        db = get_database()
        session_id = f"API-{uuid.uuid4().hex[:8].upper()}"

        if ext in IMAGE_EXTS:
            analysis = process_image_bytes(
                image_bytes=file_bytes,
                filename=filename,
                detector=detector,  # type: ignore[arg-type]
                conf_threshold=float(config.DEFAULT_CONFIDENCE_THRESHOLD),
                iou_threshold=float(config.DEFAULT_IOU_THRESHOLD),
                imgsz=int(config.DEFAULT_IMAGE_SIZE),
                min_box_area_ratio=0.0,
            )

            det_res = analysis.detection_result
            sev = analysis.severity
            has_fire = det_res.fire_count > 0
            has_smoke = det_res.smoke_count > 0
            has_any_det = bool(has_fire or has_smoke)
            final_label = _format_final_result_label(has_fire, has_smoke)

            db.insert_incident(
                file_name=analysis.filename,
                file_type="image",
                detection_type=final_label,
                confidence=det_res.max_confidence,
                fire_coverage=sev.fire_coverage_pct,
                smoke_coverage=sev.smoke_coverage_pct,
                visual_severity=sev.visual_severity,
                alert_level=analysis.alert_event.alert_level,
                output_path=str(analysis.output_path),
                original_path=str(analysis.original_path),
                session_id=session_id,
            )

            original_uri = _encode_rgb_png_data_uri(analysis.original_rgb)
            analyzed_uri = _encode_png_bytes_data_uri(analysis.annotated_bytes)
            gc.collect()

            return JSONResponse(
                status_code=200,
                content={
                    "status": "completed",
                    "media_type": "image",
                    "file_name": analysis.filename,
                    "original_media_url": original_uri,
                    "analyzed_media_url": analyzed_uri,
                    "has_fire": has_fire,
                    "has_smoke": has_smoke,
                    "fire_pct": round(float(sev.fire_coverage_pct), 2),
                    "smoke_pct": round(float(sev.smoke_coverage_pct), 2),
                    "combined_pct": round(float(sev.combined_coverage_pct), 2),
                    "confidence": float(det_res.max_confidence),
                    "confidence_display": _format_confidence_display(det_res.max_confidence, has_any_det),
                    "fire_confidence": float(det_res.fire_max_confidence),
                    "smoke_confidence": float(det_res.smoke_max_confidence),
                    "visual_severity": (
                        "No detection" if sev.visual_severity.lower() == "no detection" else sev.visual_severity
                    ),
                    "final_result": final_label,
                    "inference_time_ms": round(float(det_res.inference_time_ms), 2),
                    "detections": [
                        {
                            "class_name": d.class_name,
                            "confidence": d.confidence,
                            "bbox_xyxy": list(d.bbox_xyxy),
                            "bbox_normalized": list(d.bbox_normalized),
                        }
                        for d in det_res.detections
                    ],
                },
            )

        # MP4 Video processing flow (lazy import so image-only requests stay ultra-light)
        from src.video_processor import (
            InvalidVideoError,
            process_video_file,
            save_uploaded_video_bytes,
            validate_video_file,
        )

        try:
            saved_path = save_uploaded_video_bytes(file_bytes, filename)
            meta = validate_video_file(saved_path)
            fps = float(meta["fps"])
            sample_step = max(1, int(round(fps / 5.0))) if fps > 8 else 1

            vid_res = process_video_file(
                video_path=saved_path,
                detector=detector,  # type: ignore[arg-type]
                conf_threshold=float(config.DEFAULT_CONFIDENCE_THRESHOLD),
                iou_threshold=float(config.DEFAULT_IOU_THRESHOLD),
                frame_sample_rate=sample_step,
                temporal_enabled=False,
                min_box_area_ratio=0.0,
                source_filename=filename,
            )
        except InvalidVideoError as exc:
            raise ValueError(str(exc)) from exc

        fire_series = list(vid_res.growth_analysis.raw_fire_series)
        smoke_series = list(vid_res.growth_analysis.raw_smoke_series)
        max_fire = round(max(fire_series, default=0.0), 2)
        avg_fire = round(sum(fire_series) / len(fire_series), 2) if fire_series else 0.0
        max_smoke = round(max(smoke_series, default=0.0), 2)
        avg_smoke = round(sum(smoke_series) / len(smoke_series), 2) if smoke_series else 0.0
        max_comb = round(float(vid_res.peak_severity.combined_coverage_pct), 2)
        has_fire = bool(vid_res.first_fire_timestamp_sec is not None or max_fire > 0.0)
        has_smoke = bool(vid_res.first_smoke_timestamp_sec is not None or max_smoke > 0.0)
        has_any_det = bool(has_fire or has_smoke)
        final_label = _format_final_result_label(has_fire, has_smoke)

        peak_fire_conf = round(float(vid_res.peak_fire_confidence), 4)
        peak_smoke_conf = round(float(vid_res.peak_smoke_confidence), 4)

        sev_txt = (
            "No detection"
            if vid_res.peak_severity.visual_severity.lower() == "no detection"
            else vid_res.peak_severity.visual_severity
        )

        db.insert_incident(
            file_name=filename,
            file_type="video",
            detection_type=final_label,
            confidence=vid_res.max_confidence,
            fire_coverage=max_fire,
            smoke_coverage=max_smoke,
            visual_severity=vid_res.peak_severity.visual_severity,
            alert_level=vid_res.highest_alert_event.alert_level if vid_res.highest_alert_event else "GREEN",
            output_path=str(vid_res.output_video_path),
            original_path=str(saved_path),
            session_id=session_id,
            notes=f"Avg Fire: {avg_fire:.2f}%, Avg Smoke: {avg_smoke:.2f}% across {vid_res.total_frames} frames.",
        )

        snapshot_uri = None
        if vid_res.snapshot_path and vid_res.snapshot_path.is_file():
            snapshot_uri = _encode_png_bytes_data_uri(vid_res.snapshot_path.read_bytes())

        gc.collect()
        return JSONResponse(
            status_code=200,
            content={
                "status": "completed",
                "media_type": "video",
                "file_name": filename,
                "original_media_url": f"/api/media/uploads/{saved_path.name}",
                "analyzed_media_url": f"/api/media/outputs/{vid_res.output_video_path.name}",
                "snapshot_url": snapshot_uri,
                "has_fire": has_fire,
                "has_smoke": has_smoke,
                "fire_pct": max_fire,
                "avg_fire_pct": avg_fire,
                "smoke_pct": max_smoke,
                "avg_smoke_pct": avg_smoke,
                "combined_pct": max_comb,
                "confidence": float(vid_res.max_confidence),
                "confidence_display": _format_confidence_display(vid_res.max_confidence, has_any_det),
                "fire_confidence": peak_fire_conf,
                "smoke_confidence": peak_smoke_conf,
                "visual_severity": sev_txt,
                "final_result": final_label,
                "frames_processed": vid_res.total_frames,
                "duration_sec": vid_res.duration_sec,
                "processing_time_sec": vid_res.processing_time_sec,
            },
        )
    except ModelNotFoundError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "final_result": "Analysis Failed",
                "visual_severity": "Analysis Failed",
                "error_message": str(exc),
            },
        )
    except (InvalidImageError, ValueError) as exc:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "final_result": "Analysis Failed",
                "visual_severity": "Analysis Failed",
                "error_message": str(exc),
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "final_result": "Analysis Failed",
                "visual_severity": "Analysis Failed",
                "error_message": f"Could not complete detection: {exc}",
            },
        )


@app.get("/api/media/{folder}/{filename}")
def get_media(folder: str, filename: str):
    if folder not in {"uploads", "outputs"}:
        return JSONResponse(status_code=404, content={"error": "Media folder not found."})
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        return JSONResponse(status_code=400, content={"error": "Invalid media filename."})

    base_dir = config.UPLOADS_DIR if folder == "uploads" else config.OUTPUTS_DIR
    target = (base_dir / safe_name).resolve()
    if not str(target).startswith(str(base_dir.resolve())) or not target.is_file():
        return JSONResponse(status_code=404, content={"error": "Media file not found."})

    ext = target.suffix.lower()
    media_type = "video/mp4" if ext == ".mp4" else "image/png"
    return FileResponse(target, media_type=media_type, filename=safe_name)


@app.get("/api/history")
def list_history(limit: int = 200) -> dict[str, Any]:
    db = get_database()
    records = db.get_incidents(limit=max(1, min(limit, 500)))
    formatted = [
        {
            "id": r.get("incident_id") or r.get("id"),
            "created_at": r["created_at"],
            "file_name": r["file_name"],
            "file_type": r["file_type"],
            "fire_coverage": round(float(r["fire_coverage"]), 2),
            "smoke_coverage": round(float(r["smoke_coverage"]), 2),
            "confidence": float(r["confidence"]),
            "visual_severity": (
                "No detection" if str(r["visual_severity"]).lower() == "no detection" else r["visual_severity"]
            ),
            "detection_type": r["detection_type"],
        }
        for r in records
    ]
    return {"records": formatted, "total": len(formatted)}


@app.delete("/api/history")
def clear_history() -> dict[str, int]:
    db = get_database()
    deleted = db.clear_incidents()
    return {"deleted": deleted}


@app.get("/api/about")
def about_project() -> dict[str, Any]:
    meta = _read_json_safe(config.MODELS_DIR / "model_metadata.json") or {}
    eval_data = _read_json_safe(config.MODELS_DIR / "evaluation_results.json") or {}
    primary_eval = eval_data.get("primary_evaluation", {}) if isinstance(eval_data, dict) else {}
    return {
        "project_name": config.PROJECT_NAME,
        "subtitle": "AI-Based Fire and Smoke Detection",
        "version": config.PROJECT_VERSION,
        "disclaimer": config.ACADEMIC_DISCLAIMER,
        "architecture": {
            "frontend": "React + Vite (Vercel)",
            "backend": "FastAPI + Uvicorn (Render)",
            "model": "Ultralytics YOLO11n Object Detector (classes: 0=smoke, 1=fire)",
            "dataset": "D-Fire Dataset (DFireDataset)",
            "database": "SQLite Incident History",
        },
        "confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD,
        "image_size": config.DEFAULT_IMAGE_SIZE,
        "model_metadata": meta,
        "evaluation_summary": {
            "test_images_evaluated": primary_eval.get("test_images_evaluated"),
            "precision": primary_eval.get("precision"),
            "recall": primary_eval.get("recall"),
            "f1_score": primary_eval.get("f1_score"),
            "map50": primary_eval.get("map50"),
            "map50_95": primary_eval.get("map50_95"),
            "per_class": primary_eval.get("per_class"),
        },
    }
