"""Step 3 — Model Health Check & Diagnostics Service.

Inspects model file presence, size, load status, architecture summary,
normalized class names ('fire', 'smoke'), PyTorch/CUDA execution device,
and real probe inference latency.

Status codes:
- MODEL MISSING
- MODEL LOADING
- MODEL READY
- MODEL INFERENCE FAILED
- MODEL CLASS MISMATCH
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
from src.detector import FireSmokeDetector, ModelNotFoundError, normalize_class_label

STATUS_MISSING = "MODEL MISSING"
STATUS_LOADING = "MODEL LOADING"
STATUS_READY = "MODEL READY"
STATUS_INFERENCE_FAILED = "MODEL INFERENCE FAILED"
STATUS_CLASS_MISMATCH = "MODEL CLASS MISMATCH"


@dataclass
class ModelHealthReport:
    """Structured diagnostic result returned by run_model_health_check."""

    status: str
    model_path: str
    file_exists: bool
    file_size_bytes: int
    file_size_mb: float
    load_succeeded: bool
    architecture: str
    class_names: dict[int, str]
    classes_verified: bool
    pytorch_available: bool
    pytorch_version: str
    cuda_available: bool
    execution_device: str
    inference_succeeded: bool
    avg_inference_latency_ms: float | None
    model_tier: str
    training_note: str
    error_message: str | None
    checked_at: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostic report into plain dictionary."""
        return asdict(self)


def read_model_metadata(model_path: Path) -> dict[str, Any]:
    """Read optional models/model_metadata.json describing training tier & provenance."""
    meta_path = model_path.parent / "model_metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _describe_architecture(yolo_model: Any, model_path: Path) -> str:
    """Extract readable model architecture name and parameter/layer count."""
    try:
        task = getattr(yolo_model, "task", "detect")
        inner = getattr(yolo_model, "model", None)
        yaml_cfg = getattr(inner, "yaml", {}) or {}
        yaml_file = Path(str(yaml_cfg.get("yaml_file", model_path.stem))).stem
        n_params = sum(p.numel() for p in inner.parameters()) if inner is not None else 0
        if n_params > 0:
            return f"{yaml_file} ({task}, {n_params / 1e6:.2f}M params)"
        return f"{yaml_file} ({task})"
    except Exception:
        return model_path.stem


def run_model_health_check(
    model_path: Path | str | None = None,
    probe_runs: int = 2,
) -> ModelHealthReport:
    """Execute end-to-end diagnostics on trained YOLO weights and probe inference."""
    target = Path(model_path) if model_path is not None else config.MODEL_PATH
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    cuda_ok = bool(torch.cuda.is_available())
    if cuda_ok:
        device_desc = f"cuda:0 ({torch.cuda.get_device_name(0)})"
    else:
        device_desc = "cpu"

    meta = read_model_metadata(target)
    model_tier = str(meta.get("model_tier", "UNKNOWN"))
    training_note = str(meta.get("note", ""))

    file_exists = target.is_file() and target.stat().st_size > 0
    file_size_bytes = target.stat().st_size if target.is_file() else 0
    file_size_mb = round(file_size_bytes / (1024.0 * 1024.0), 3)

    if not file_exists:
        return ModelHealthReport(
            status=STATUS_MISSING,
            model_path=str(target),
            file_exists=False,
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            load_succeeded=False,
            architecture="Unavailable",
            class_names={},
            classes_verified=False,
            pytorch_available=True,
            pytorch_version=torch.__version__,
            cuda_available=cuda_ok,
            execution_device=device_desc,
            inference_succeeded=False,
            avg_inference_latency_ms=None,
            model_tier="NONE",
            training_note="No trained model weights found on disk.",
            error_message=(
                f"Trained weights not found at '{target}'. Train a YOLO11 fire/smoke "
                "model (local or Google Colab) and place best.pt inside models/."
            ),
            checked_at=checked_at,
        )

    detector = FireSmokeDetector(model_path=target)
    try:
        yolo_model = detector.load_model()
    except (ModelNotFoundError, Exception) as exc:
        return ModelHealthReport(
            status=STATUS_INFERENCE_FAILED,
            model_path=str(target),
            file_exists=True,
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            load_succeeded=False,
            architecture="Load Failed",
            class_names={},
            classes_verified=False,
            pytorch_available=True,
            pytorch_version=torch.__version__,
            cuda_available=cuda_ok,
            execution_device=device_desc,
            inference_succeeded=False,
            avg_inference_latency_ms=None,
            model_tier=model_tier,
            training_note=training_note,
            error_message=f"Failed to load YOLO weights: {exc}",
            checked_at=checked_at,
        )

    raw_names = getattr(yolo_model, "names", {}) or {}
    if isinstance(raw_names, list):
        norm_names = {idx: normalize_class_label(str(name)) for idx, name in enumerate(raw_names)}
    else:
        norm_names = {int(k): normalize_class_label(str(v)) for k, v in raw_names.items()}

    arch_str = _describe_architecture(yolo_model, target)
    expected_set = set(config.TARGET_CLASSES)
    actual_set = set(norm_names.values())
    classes_ok = expected_set.issubset(actual_set) and len(actual_set) <= 4

    if not classes_ok:
        return ModelHealthReport(
            status=STATUS_CLASS_MISMATCH,
            model_path=str(target),
            file_exists=True,
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            load_succeeded=True,
            architecture=arch_str,
            class_names=norm_names,
            classes_verified=False,
            pytorch_available=True,
            pytorch_version=torch.__version__,
            cuda_available=cuda_ok,
            execution_device=device_desc,
            inference_succeeded=False,
            avg_inference_latency_ms=None,
            model_tier=model_tier,
            training_note=training_note,
            error_message=(
                f"Model class mismatch: expected {sorted(expected_set)}, "
                f"found {len(actual_set)} class(es) starting with {sorted(actual_set)[:6]}. "
                "Generic COCO weights cannot be used as models/best.pt."
            ),
            checked_at=checked_at,
        )

    # Probe real forward inference pass using a deterministic synthetic frame
    probe_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    probe_frame[80:180, 90:190] = (30, 90, 220)
    latencies: list[float] = []
    try:
        runs = max(1, int(probe_runs))
        for _ in range(runs):
            t0 = time.perf_counter()
            detector.predict_frame(probe_frame, conf_threshold=0.25, imgsz=320)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    except Exception as exc:
        return ModelHealthReport(
            status=STATUS_INFERENCE_FAILED,
            model_path=str(target),
            file_exists=True,
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            load_succeeded=True,
            architecture=arch_str,
            class_names=norm_names,
            classes_verified=True,
            pytorch_available=True,
            pytorch_version=torch.__version__,
            cuda_available=cuda_ok,
            execution_device=device_desc,
            inference_succeeded=False,
            avg_inference_latency_ms=None,
            model_tier=model_tier,
            training_note=training_note,
            error_message=f"Model inference test failed: {exc}",
            checked_at=checked_at,
        )

    avg_ms = round(sum(latencies) / len(latencies), 2)
    return ModelHealthReport(
        status=STATUS_READY,
        model_path=str(target),
        file_exists=True,
        file_size_bytes=file_size_bytes,
        file_size_mb=file_size_mb,
        load_succeeded=True,
        architecture=arch_str,
        class_names=norm_names,
        classes_verified=True,
        pytorch_available=True,
        pytorch_version=torch.__version__,
        cuda_available=cuda_ok,
        execution_device=device_desc,
        inference_succeeded=True,
        avg_inference_latency_ms=avg_ms,
        model_tier=model_tier,
        training_note=training_note,
        error_message=None,
        checked_at=checked_at,
    )
