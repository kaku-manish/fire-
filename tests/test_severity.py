"""Phase 15 — Unit Tests for Visual Severity, Overlapping Union Geometry,
Fire Growth Analysis, and Dataset Preparation / Leakage-Free Splitting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.detector import Detection  # noqa: E402
from src.growth_analysis import (  # noqa: E402
    FrameCoverageMetric,
    analyze_fire_growth,
    create_growth_plotly_figure,
)
from src.severity import (  # noqa: E402
    classify_visual_severity,
    compute_union_area_xyxy,
    estimate_segmentation_coverage,
    estimate_visual_severity,
)
from training.prepare_dataset import (  # noqa: E402
    create_leakage_free_splits,
    extract_sequence_group,
    parse_and_validate_yolo_label,
)
from training.validate_dataset import (  # noqa: E402
    create_mosaic_4up,
    flip_horizontal_with_boxes,
    letterbox_image,
)


def _det(cls_name: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.85) -> Detection:
    return Detection(
        class_id=1 if cls_name == "fire" else 0,
        class_name=cls_name,
        confidence=conf,
        bbox_xyxy=(x1, y1, x2, y2),
        bbox_normalized=(x1 / 100.0, y1 / 100.0, x2 / 100.0, y2 / 100.0),
        area_px=(x2 - x1) * (y2 - y1),
    )


def test_scenario_21_overlapping_bounding_boxes_union() -> None:
    """Scenario 21: Overlapping fire boxes use union area, not naive area sum."""
    # Box A: [0,0,20,20] (400 px^2), Box B: [10,0,30,20] (400 px^2)
    # Overlap: [10,0,20,20] (200 px^2) -> Exact Union = 600 px^2 in 100x100 = 6.0%
    dets = [_det("fire", 0, 0, 20, 20), _det("fire", 10, 0, 30, 20)]
    union_area = compute_union_area_xyxy([d.bbox_xyxy for d in dets], 100, 100)
    assert union_area == pytest.approx(600.0)

    sev = estimate_visual_severity(dets, 100, 100)
    assert sev.fire_coverage_pct == pytest.approx(6.0)
    assert sev.combined_coverage_pct == pytest.approx(6.0)
    assert sev.visual_severity == "Medium"


def test_severity_threshold_boundaries() -> None:
    """Verify 0%, <5%, 5-<20%, and >=20% severity threshold mappings."""
    assert classify_visual_severity(0.0) == "No Detection"
    assert classify_visual_severity(0.01) == "Low"
    assert classify_visual_severity(4.99) == "Low"
    assert classify_visual_severity(5.0) == "Medium"
    assert classify_visual_severity(19.99) == "Medium"
    assert classify_visual_severity(20.0) == "High"


def test_optional_segmentation_mask_union() -> None:
    """Optional segmentation mask mode computes exact pixel union without double counting."""
    m1 = np.zeros((10, 10), dtype=bool)
    m2 = np.zeros((10, 10), dtype=bool)
    m1[0:4, 0:5] = True  # 20 pixels
    m2[0:4, 2:7] = True  # 20 pixels, 12 overlap -> 28 union pixels = 28%
    sev = estimate_segmentation_coverage([m1, m2], [], 10, 10, max_confidence=0.9)
    assert sev.fire_coverage_pct == pytest.approx(28.0)
    assert sev.visual_severity == "High"
    assert sev.measurement_method == "segmentation_mask_union"


def test_scenario_22_insufficient_frames_for_growth_analysis() -> None:
    """Scenario 22: Fewer than minimum required active frames yields 'Insufficient data'."""
    short_series = [
        FrameCoverageMetric(0, 0.0, 2.0, 0.0),
        FrameCoverageMetric(1, 0.5, 4.0, 0.0),
    ]
    growth = analyze_fire_growth(short_series, min_frames=4)
    assert growth.growth_classification == "Insufficient data"


def test_growth_trends_and_early_smoke_lead_time() -> None:
    """Verify Increasing, Decreasing, Stable growth classification and Plotly chart creation."""
    increasing = [
        FrameCoverageMetric(i, i * 1.0, 2.0 + i * 3.0, 5.0 if i >= 0 else 0.0)
        for i in range(6)
    ]
    res_inc = analyze_fire_growth(increasing)
    assert res_inc.growth_classification == "Increasing"
    fig = create_growth_plotly_figure(res_inc)
    assert len(fig.data) == 4

    decreasing = [
        FrameCoverageMetric(i, i * 1.0, 22.0 - i * 3.5, 4.0)
        for i in range(6)
    ]
    res_dec = analyze_fire_growth(decreasing)
    assert res_dec.growth_classification == "Decreasing"

    # Smoke appears at 1.0s before fire appears at 4.0s
    early_smoke = [
        FrameCoverageMetric(0, 0.0, 0.0, 0.0),
        FrameCoverageMetric(1, 1.0, 0.0, 4.0),
        FrameCoverageMetric(2, 2.0, 0.0, 6.0),
        FrameCoverageMetric(3, 3.0, 0.0, 7.0),
        FrameCoverageMetric(4, 4.0, 5.0, 8.0),
        FrameCoverageMetric(5, 5.0, 5.1, 8.0),
        FrameCoverageMetric(6, 6.0, 5.0, 8.0),
        FrameCoverageMetric(7, 7.0, 5.1, 8.0),
    ]
    res_early = analyze_fire_growth(early_smoke)
    assert res_early.smoke_before_fire is True
    assert res_early.smoke_lead_time_sec == pytest.approx(3.0)
    assert res_early.growth_classification == "Relatively stable"


def test_dataset_leakage_free_split_and_preprocessing(tmp_path: Path) -> None:
    """Verify sequence grouping prevents frame leakage across train/valid/test splits."""
    assert extract_sequence_group(Path("cam01_frame_0042.jpg")) == "cam01"
    assert extract_sequence_group(Path("1720453551_-02400.jpg")) == "1720453551"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    records = []
    for seq_idx in range(6):
        for frame_idx in range(2):
            img_p = raw_dir / f"seq{seq_idx}_frame_{frame_idx:03d}.png"
            lbl_p = raw_dir / f"seq{seq_idx}_frame_{frame_idx:03d}.txt"
            img_p.write_bytes(b"img")
            lbl_p.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            boxes, errs = parse_and_validate_yolo_label(lbl_p)
            assert not errs and len(boxes) == 1
            records.append(
                {
                    "image_path": str(img_p),
                    "label_path": str(lbl_p),
                    "group_id": extract_sequence_group(img_p),
                }
            )

    split_counts = create_leakage_free_splits(records, tmp_path / "yolo_ds", seed=42)
    assert sum(split_counts.values()) == len(records)

    # Verify preprocessing transformations keep bounding boxes normalized
    sample_img = np.zeros((120, 160, 3), dtype=np.uint8)
    lb_img, _, _ = letterbox_image(sample_img, (640, 640))
    assert lb_img.shape == (640, 640, 3)

    _, flipped_boxes = flip_horizontal_with_boxes(sample_img, [(1, 0.25, 0.5, 0.2, 0.2)])
    assert flipped_boxes[0][1] == pytest.approx(0.75)

    mosaic_img, mosaic_boxes = create_mosaic_4up([(sample_img, [(0, 0.5, 0.5, 0.4, 0.4)])] * 4, output_size=320)
    assert mosaic_img.shape == (320, 320, 3)
    assert len(mosaic_boxes) == 4
