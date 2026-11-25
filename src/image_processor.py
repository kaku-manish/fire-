"""Phase 5 — Image-Based Fire and Smoke Detection Processor.

Validates uploaded images (JPG, JPEG, PNG, WEBP), runs YOLO11 inference, draws
red boxes for fire and orange boxes for smoke, computes approximate visual
severity, evaluates software alerts, and prepares downloadable outputs.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

import config
from src.alert_manager import AlertEvent, AlertManager
from src.detector import (
    Detection,
    DetectionResult,
    FireSmokeDetector,
    draw_detections,
    summarize_detections,
)
from src.severity import SeverityAssessment, estimate_visual_severity
from src.temporal_filter import filter_by_geometry_and_confidence


class InvalidImageError(ValueError):
    """Raised when an uploaded image is empty, unsupported, oversized, or corrupted."""


@dataclass
class ImageAnalysisResult:
    """Complete result bundle for a processed image."""

    filename: str
    original_path: Path
    output_path: Path
    original_rgb: np.ndarray
    annotated_rgb: np.ndarray
    annotated_bytes: bytes
    detection_result: DetectionResult
    severity: SeverityAssessment
    alert_event: AlertEvent


def validate_image_bytes(
    image_bytes: bytes,
    filename: str,
    max_mb: int = config.MAX_UPLOAD_IMAGE_MB,
) -> np.ndarray:
    """Validate uploaded image bytes and decode into an OpenCV BGR array."""
    if not filename:
        raise InvalidImageError("No filename provided for image upload.")
    ext = Path(filename).suffix.lower()
    if ext not in config.SUPPORTED_IMAGE_EXTENSIONS:
        allowed = ", ".join(config.SUPPORTED_IMAGE_EXTENSIONS)
        raise InvalidImageError(
            f"Unsupported image format '{ext or 'unknown'}'. Allowed formats: {allowed}."
        )
    if not image_bytes:
        raise InvalidImageError("Uploaded image file is empty (0 bytes).")
    size_mb = len(image_bytes) / (1024.0 * 1024.0)
    if size_mb > max_mb:
        raise InvalidImageError(
            f"Image size ({size_mb:.2f} MB) exceeds the {max_mb} MB limit."
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            pil_img.verify()
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            rgb_img = pil_img.convert("RGB")
            rgb_arr = np.asarray(rgb_img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError, ModuleNotFoundError) as exc:
        raise InvalidImageError(f"Corrupted or unreadable image file: {exc}") from exc

    if rgb_arr.size == 0 or rgb_arr.shape[0] <= 0 or rgb_arr.shape[1] <= 0:
        raise InvalidImageError("Decoded image has invalid dimensions.")

    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


def encode_bgr_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    """Encode an OpenCV BGR image into PNG bytes for UI display and download."""
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode annotated image as PNG.")
    return encoded.tobytes()


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return cleaned[:48] or "image"


def process_image_bytes(
    image_bytes: bytes,
    filename: str,
    detector: FireSmokeDetector,
    conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
    imgsz: int = config.DEFAULT_IMAGE_SIZE,
    min_box_area_ratio: float = 0.0,
    alert_manager: AlertManager | None = None,
    uploads_dir: Path | None = None,
    outputs_dir: Path | None = None,
) -> ImageAnalysisResult:
    """Validate, detect, assess visual severity, save outputs, and evaluate alerts."""
    frame_bgr = validate_image_bytes(image_bytes, filename)
    height, width = frame_bgr.shape[:2]

    raw_result = detector.predict_frame(
        frame_bgr,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )

    filtered_detections: list[Detection] = raw_result.detections
    if min_box_area_ratio > 0.0:
        filtered_detections = filter_by_geometry_and_confidence(
            raw_result.detections,
            frame_width=width,
            frame_height=height,
            fire_conf_threshold=conf_threshold,
            smoke_conf_threshold=conf_threshold,
            min_box_area_ratio=min_box_area_ratio,
        )

    final_result = summarize_detections(
        detections=filtered_detections,
        image_width=width,
        image_height=height,
        inference_time_ms=raw_result.inference_time_ms,
        model_path=raw_result.model_path,
    )

    severity = estimate_visual_severity(filtered_detections, width, height)
    annotated_bgr = draw_detections(frame_bgr, filtered_detections)

    up_dir = uploads_dir or config.UPLOADS_DIR
    out_dir = outputs_dir or config.OUTPUTS_DIR
    up_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    uid = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    stem = _safe_stem(filename)
    ext = Path(filename).suffix.lower()

    original_path = up_dir / f"{stem}_{uid}{ext}"
    original_path.write_bytes(image_bytes)

    output_path = out_dir / f"{stem}_{uid}_detected.png"
    annotated_bytes = encode_bgr_to_png_bytes(annotated_bgr)
    output_path.write_bytes(annotated_bytes)

    manager = alert_manager or AlertManager()
    alert_event = manager.evaluate(
        source_filename=filename,
        has_fire=final_result.fire_count > 0,
        has_smoke=final_result.smoke_count > 0,
        confidence=final_result.max_confidence,
        fire_coverage_pct=severity.fire_coverage_pct,
        smoke_coverage_pct=severity.smoke_coverage_pct,
        visual_severity=severity.visual_severity,
        snapshot_path=str(output_path),
        fire_persistent=final_result.fire_count > 0,
        smoke_persistent=final_result.smoke_count > 0,
        suppress_duplicates=False,
    )

    return ImageAnalysisResult(
        filename=filename,
        original_path=original_path,
        output_path=output_path,
        original_rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        annotated_rgb=cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
        annotated_bytes=annotated_bytes,
        detection_result=final_result,
        severity=severity,
        alert_event=alert_event,
    )


def process_image_file(
    file_path: Path,
    detector: FireSmokeDetector,
    **kwargs: object,
) -> ImageAnalysisResult:
    """Convenience wrapper to process an image stored on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise InvalidImageError(f"Image file not found: {path}")
    return process_image_bytes(path.read_bytes(), path.name, detector, **kwargs)  # type: ignore[arg-type]
