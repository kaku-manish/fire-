"""Phase 5 — Core YOLO11 Fire and Smoke Detection Engine.

Provides model loading, standardized bounding-box inference, status derivation,
and OpenCV bounding-box visualization (Red for fire, Orange for smoke).
Never returns hardcoded or simulated detections in production inference.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config

# Ensure Ultralytics uses a writable settings directory in restricted environments
_DEFAULT_ULTRALYTICS_DIR = Path.home() / "AppData" / "Roaming" / "Ultralytics"
if "YOLO_CONFIG_DIR" not in os.environ:
    try:
        _DEFAULT_ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)
        _probe = _DEFAULT_ULTRALYTICS_DIR / ".write_probe"
        _probe.write_text("ok", encoding="utf-8")
        _probe.unlink(missing_ok=True)
    except OSError:
        fallback_dir = Path(tempfile.gettempdir()) / "fireguard_ultralytics"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(fallback_dir)


class ModelNotFoundError(FileNotFoundError):
    """Raised when trained YOLO model weights cannot be located on disk."""


@dataclass
class Detection:
    """Represents a single localized fire or smoke detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_normalized: tuple[float, float, float, float]
    area_px: float


@dataclass
class DetectionResult:
    """Structured inference result for a single image or video frame."""

    detections: list[Detection] = field(default_factory=list)
    status: str = "No fire or smoke detected."
    inference_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    fire_count: int = 0
    smoke_count: int = 0
    max_confidence: float = 0.0
    avg_confidence: float = 0.0
    fire_max_confidence: float = 0.0
    smoke_max_confidence: float = 0.0
    model_path: str = ""
    negative_safety_note: str = (
        "A negative prediction indicates no visible fire or smoke was detected above the "
        "configured threshold; it is NOT proof that the environment is safe."
    )


def normalize_class_label(raw_label: str) -> str:
    """Normalize raw YOLO class names to canonical 'fire' or 'smoke' labels."""
    lowered = str(raw_label).strip().lower()
    if lowered in {"fire", "flame", "flames"}:
        return "fire"
    if lowered in {"smoke", "plume"}:
        return "smoke"
    return lowered


def compute_detection_status(detections: list[Detection]) -> str:
    """Determine standardized detection status string from detection list."""
    classes = {det.class_name for det in detections}
    has_fire = "fire" in classes
    has_smoke = "smoke" in classes
    if has_fire and has_smoke:
        return "Fire and smoke detected."
    if has_fire:
        return "Fire detected."
    if has_smoke:
        return "Smoke detected."
    return "No fire or smoke detected."


def summarize_detections(
    detections: list[Detection],
    image_width: int,
    image_height: int,
    inference_time_ms: float = 0.0,
    model_path: str = "",
) -> DetectionResult:
    """Build a complete DetectionResult from a list of Detection objects."""
    fire_confs = [d.confidence for d in detections if d.class_name == "fire"]
    smoke_confs = [d.confidence for d in detections if d.class_name == "smoke"]
    fire_count = len(fire_confs)
    smoke_count = len(smoke_confs)
    confidences = [d.confidence for d in detections]
    max_conf = max(confidences) if confidences else 0.0
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    fire_max_conf = max(fire_confs) if fire_confs else 0.0
    smoke_max_conf = max(smoke_confs) if smoke_confs else 0.0
    return DetectionResult(
        detections=detections,
        status=compute_detection_status(detections),
        inference_time_ms=round(inference_time_ms, 2),
        image_width=image_width,
        image_height=image_height,
        fire_count=fire_count,
        smoke_count=smoke_count,
        max_confidence=round(max_conf, 4),
        avg_confidence=round(avg_conf, 4),
        fire_max_confidence=round(fire_max_conf, 4),
        smoke_max_confidence=round(smoke_max_conf, 4),
        model_path=model_path,
    )


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    line_thickness: int | None = None,
) -> np.ndarray:
    """Draw red bounding boxes for fire and orange bounding boxes for smoke."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Cannot draw detections on an empty frame.")
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]
    thickness = line_thickness or max(2, int(round(0.0025 * (h + w) / 2)))
    font_scale = max(0.5, min(0.9, thickness * 0.25))

    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = config.CLASS_COLORS_BGR.get(det.class_name, (0, 255, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.class_name.upper()} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        label_top = max(0, y1 - text_h - baseline - 6)
        label_bottom = label_top + text_h + baseline + 6
        label_right = min(w - 1, x1 + text_w + 8)

        cv2.rectangle(annotated, (x1, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return annotated


class FireSmokeDetector:
    """Wrapper around Ultralytics YOLO11 weights for fire and smoke inference."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else config.MODEL_PATH
        self._model: Any = None
        self._class_names: dict[int, str] = {}

    def ensure_model_exists(self) -> Path:
        """Verify that trained weights exist at self.model_path."""
        if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            raise ModelNotFoundError(
                f"Trained model weights not found at '{self.model_path}'. "
                "Train a model in Phase 4 (notebooks/model_training.ipynb or "
                "training/train.py) and place best.pt inside the models/ directory."
            )
        return self.model_path

    def load_model(self) -> Any:
        """Lazy-load Ultralytics YOLO model weights into memory."""
        if self._model is not None:
            return self._model
        self.ensure_model_exists()
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))
        raw_names = getattr(self._model, "names", {})
        if isinstance(raw_names, dict):
            self._class_names = {int(k): normalize_class_label(str(v)) for k, v in raw_names.items()}
        elif isinstance(raw_names, (list, tuple)):
            self._class_names = {idx: normalize_class_label(str(v)) for idx, v in enumerate(raw_names)}
        return self._model

    @property
    def class_names(self) -> dict[int, str]:
        """Return normalized class ID -> class name mapping from loaded model."""
        if not self._class_names:
            self.load_model()
        return dict(self._class_names)

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
        imgsz: int = config.DEFAULT_IMAGE_SIZE,
    ) -> DetectionResult:
        """Run YOLO inference on a single BGR image/frame and return structured detections."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame is empty or invalid.")

        model = self.load_model()
        height, width = frame_bgr.shape[:2]
        start_time = time.perf_counter()
        results = model.predict(
            source=frame_bgr,
            conf=float(conf_threshold),
            iou=float(iou_threshold),
            imgsz=int(imgsz),
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        detections: list[Detection] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)

                for coords, conf_val, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
                    x1 = float(np.clip(coords[0], 0.0, float(width)))
                    y1 = float(np.clip(coords[1], 0.0, float(height)))
                    x2 = float(np.clip(coords[2], 0.0, float(width)))
                    y2 = float(np.clip(coords[3], 0.0, float(height)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cls_name = self._class_names.get(int(cls_id), f"class_{cls_id}")
                    area_px = (x2 - x1) * (y2 - y1)
                    norm_box = (x1 / width, y1 / height, x2 / width, y2 / height)
                    detections.append(
                        Detection(
                            class_id=int(cls_id),
                            class_name=cls_name,
                            confidence=round(float(conf_val), 4),
                            bbox_xyxy=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                            bbox_normalized=tuple(round(v, 5) for v in norm_box),  # type: ignore[arg-type]
                            area_px=round(area_px, 2),
                        )
                    )

        return summarize_detections(
            detections=detections,
            image_width=width,
            image_height=height,
            inference_time_ms=elapsed_ms,
            model_path=str(self.model_path),
        )
