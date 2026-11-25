"""Central configuration module for FireGuard AI.

Defines filesystem paths, dataset and model defaults, inference parameters,
visual severity thresholds, temporal filtering defaults, alert thresholds,
and academic prototype metadata shared across all modules.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# Ensure Ultralytics settings directory is writable (falls back to temp dir in restricted shells)
if "YOLO_CONFIG_DIR" not in os.environ:
    _default_yolo_dir = Path.home() / "AppData" / "Roaming" / "Ultralytics"
    try:
        _default_yolo_dir.mkdir(parents=True, exist_ok=True)
        _probe = _default_yolo_dir / ".write_probe"
        _probe.write_text("ok", encoding="utf-8")
        _probe.unlink(missing_ok=True)
    except OSError:
        _fallback_yolo = Path(tempfile.gettempdir()) / "fireguard_ultralytics"
        _fallback_yolo.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(_fallback_yolo)

# ---------------------------------------------------------------------------
# Project Identity & Academic Prototype Notice
# ---------------------------------------------------------------------------
PROJECT_NAME: str = "FireGuard AI"
PROJECT_SUBTITLE: str = (
    "Intelligent Fire and Smoke Detection with Early Warning and "
    "Visual Severity Analysis Using Deep Learning"
)
PROJECT_VERSION: str = "1.0.0"

ACADEMIC_DISCLAIMER: str = (
    "Academic Research Prototype: FireGuard AI is a software-only educational "
    "prototype developed for academic demonstration and evaluation. It is NOT a "
    "certified fire detection, life-safety, or emergency notification system, and "
    "must not be relied upon to protect life or property or to replace certified "
    "fire alarms and emergency services."
)

# ---------------------------------------------------------------------------
# Directory & File Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent

DATASET_DIR: Path = BASE_DIR / "dataset"
TRAIN_IMAGES_DIR: Path = DATASET_DIR / "train" / "images"
TRAIN_LABELS_DIR: Path = DATASET_DIR / "train" / "labels"
VALID_IMAGES_DIR: Path = DATASET_DIR / "valid" / "images"
VALID_LABELS_DIR: Path = DATASET_DIR / "valid" / "labels"
TEST_IMAGES_DIR: Path = DATASET_DIR / "test" / "images"
TEST_LABELS_DIR: Path = DATASET_DIR / "test" / "labels"
DATA_YAML_PATH: Path = DATASET_DIR / "data.yaml"

MODELS_DIR: Path = BASE_DIR / "models"
MODEL_PATH: Path = MODELS_DIR / "best.pt"

NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
TRAINING_DIR: Path = BASE_DIR / "training"
SRC_DIR: Path = BASE_DIR / "src"

ASSETS_DIR: Path = BASE_DIR / "assets"
STYLES_PATH: Path = ASSETS_DIR / "styles.css"

UPLOADS_DIR: Path = BASE_DIR / "uploads"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
REPORTS_DIR: Path = BASE_DIR / "reports"

DATABASE_DIR: Path = BASE_DIR / "database"
DB_PATH: Path = DATABASE_DIR / "fireguard.db"

TESTS_DIR: Path = BASE_DIR / "tests"

# ---------------------------------------------------------------------------
# Dataset Split & Training Defaults
# ---------------------------------------------------------------------------
SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "valid": 0.20,
    "test": 0.10,
}

# Initial model and training hyperparameters (subject to empirical tuning)
DEFAULT_BASE_MODEL: str = "yolo11n.pt"
COMPARISON_MODELS: tuple[str, ...] = ("yolo11n.pt", "yolo11s.pt", "yolov8n.pt")
DEFAULT_IMAGE_SIZE: int = 640
DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_OPTIMIZER: str = "auto"
DEFAULT_PATIENCE: int = 15
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Class Definitions & Rendering Colors
# ---------------------------------------------------------------------------
# Canonical class names used by the application regardless of dataset class ID order.
TARGET_CLASSES: tuple[str, ...] = ("fire", "smoke")

# OpenCV uses BGR ordering: Red for fire, Orange for smoke.
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "fire": (0, 0, 255),
    "smoke": (0, 140, 255),
}

CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "fire": (239, 68, 68),
    "smoke": (249, 115, 22),
}

CLASS_COLORS_HEX: dict[str, str] = {
    "fire": "#EF4444",
    "smoke": "#F97316",
}

# ---------------------------------------------------------------------------
# Supported Input Formats & Inference Settings
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov")

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
MIN_CONFIDENCE_THRESHOLD: float = 0.05
MAX_CONFIDENCE_THRESHOLD: float = 0.95
DEFAULT_IOU_THRESHOLD: float = 0.45
DEFAULT_FRAME_SAMPLE_RATE: int = 1

MAX_UPLOAD_IMAGE_MB: int = 25
MAX_UPLOAD_VIDEO_MB: int = 500

# ---------------------------------------------------------------------------
# Visual Severity Estimation Defaults (Approximate Bounding-Box Coverage)
# ---------------------------------------------------------------------------
COVERAGE_THRESHOLDS: dict[str, float] = {
    "none": 0.0,
    "low_max": 5.0,
    "medium_max": 20.0,
}

SEVERITY_LEVELS: tuple[str, ...] = ("No Detection", "Low", "Medium", "High")

# ---------------------------------------------------------------------------
# False Alarm Reduction & Temporal Verification Defaults
# ---------------------------------------------------------------------------
DEFAULT_TEMPORAL_WINDOW: int = 5
DEFAULT_MIN_CONSECUTIVE_FRAMES: int = 3
DEFAULT_MIN_PERSISTENCE_RATIO: float = 0.60
DEFAULT_EMA_ALPHA: float = 0.40
DEFAULT_MIN_BOX_AREA_RATIO: float = 0.0005

# ---------------------------------------------------------------------------
# Fire Growth Analysis Defaults
# ---------------------------------------------------------------------------
GROWTH_SMOOTHING_WINDOW: int = 5
GROWTH_MIN_FRAMES: int = 4
GROWTH_SLOPE_THRESHOLD: float = 0.15

GROWTH_CLASSIFICATIONS: tuple[str, ...] = (
    "Increasing",
    "Decreasing",
    "Relatively stable",
    "Insufficient data",
)

# ---------------------------------------------------------------------------
# Software Alert System Defaults
# ---------------------------------------------------------------------------
ALERT_LEVELS: tuple[str, ...] = ("GREEN", "YELLOW", "ORANGE", "RED")

ALERT_COLORS: dict[str, str] = {
    "GREEN": "#22C55E",
    "YELLOW": "#EAB308",
    "ORANGE": "#F97316",
    "RED": "#EF4444",
}

ALERT_THRESHOLDS: dict[str, float | int] = {
    "high_fire_coverage_pct": 20.0,
    "rapid_growth_rate_pct_per_sec": 1.5,
    "persistent_smoke_frames": 3,
    "persistent_fire_frames": 3,
    "cooldown_seconds": 10.0,
}


def get_required_directories() -> list[Path]:
    """Return all directories required by the FireGuard AI workspace."""
    return [
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VALID_IMAGES_DIR,
        VALID_LABELS_DIR,
        TEST_IMAGES_DIR,
        TEST_LABELS_DIR,
        MODELS_DIR,
        NOTEBOOKS_DIR,
        TRAINING_DIR,
        SRC_DIR,
        ASSETS_DIR,
        UPLOADS_DIR,
        OUTPUTS_DIR,
        REPORTS_DIR,
        DATABASE_DIR,
        TESTS_DIR,
    ]


def ensure_directories() -> list[Path]:
    """Create all required project directories if they do not exist."""
    created_or_verified: list[Path] = []
    for directory in get_required_directories():
        directory.mkdir(parents=True, exist_ok=True)
        created_or_verified.append(directory)
    return created_or_verified


def verify_project_structure() -> dict[str, bool]:
    """Check whether every required project directory exists."""
    return {
        str(directory.relative_to(BASE_DIR)): directory.is_dir()
        for directory in get_required_directories()
    }


def is_model_available(model_path: Path | None = None) -> bool:
    """Return True if a trained YOLO weights file exists and is non-empty."""
    target = model_path if model_path is not None else MODEL_PATH
    return target.is_file() and target.stat().st_size > 0


def get_config_summary() -> dict[str, Any]:
    """Return a structured summary of key paths and status flags."""
    return {
        "project_name": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "base_dir": str(BASE_DIR),
        "model_path": str(MODEL_PATH),
        "model_available": is_model_available(),
        "database_path": str(DB_PATH),
        "data_yaml_path": str(DATA_YAML_PATH),
        "default_confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "supported_image_extensions": SUPPORTED_IMAGE_EXTENSIONS,
        "supported_video_extensions": SUPPORTED_VIDEO_EXTENSIONS,
        "directories": verify_project_structure(),
    }


if __name__ == "__main__":
    ensure_directories()
    summary = get_config_summary()
    print(f"{summary['project_name']} v{summary['version']}")
    print(f"Base Directory : {summary['base_dir']}")
    print(f"Model Weights  : {summary['model_path']} (Present: {summary['model_available']})")
    print(f"Database Path  : {summary['database_path']}")
    print(f"Dataset Config : {summary['data_yaml_path']}")
    print("\nDirectory Verification:")
    for rel_path, exists in summary["directories"].items():
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {rel_path}")
    if not summary["model_available"]:
        print(
            "\nNote: No trained weights found at models/best.pt yet. "
            "Train your model in Phase 4 (Google Colab or training/train.py) "
            "and place best.pt in the models/ directory."
        )
