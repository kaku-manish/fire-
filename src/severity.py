"""Phase 7 — Visual Severity Estimation Module.

Computes approximate visible fire and smoke coverage percentages relative to
the total frame area using exact bounding-box union geometry (preventing
double-counting of overlapping boxes). Also supports optional segmentation
mask union when properly annotated segmentation masks are available.

Disclaimer: Bounding-box coverage is an approximate visual measurement and does
NOT measure actual burned area, temperature, heat release rate, or physical danger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

import config
from src.detector import Detection

COVERAGE_DISCLAIMER: str = (
    "Visual coverage is an approximate 2D bounding-box area percentage in the camera "
    "frame. It does NOT represent actual burned area, temperature, fire intensity, "
    "or physical danger."
)


@dataclass
class SeverityAssessment:
    """Visual coverage and severity classification for a frame or image."""

    fire_coverage_pct: float
    smoke_coverage_pct: float
    combined_coverage_pct: float
    visual_severity: str
    fire_severity: str
    smoke_severity: str
    fire_box_count: int
    smoke_box_count: int
    max_confidence: float
    visual_indicator_color: str
    measurement_method: str = "bounding_box_union"
    disclaimer: str = COVERAGE_DISCLAIMER


def compute_union_area_xyxy(
    boxes: Iterable[tuple[float, float, float, float]],
    frame_width: int,
    frame_height: int,
) -> float:
    """Compute exact union area (in pixels^2) of multiple axis-aligned rectangles.

    Uses a 1D vertical sweep-line across sorted X-coordinates so overlapping
    bounding boxes are never counted more than once.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame width and height must be positive integers.")

    clamped: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in boxes:
        cx1 = max(0.0, min(float(frame_width), float(x1)))
        cy1 = max(0.0, min(float(frame_height), float(y1)))
        cx2 = max(0.0, min(float(frame_width), float(x2)))
        cy2 = max(0.0, min(float(frame_height), float(y2)))
        if cx2 > cx1 and cy2 > cy1:
            clamped.append((cx1, cy1, cx2, cy2))

    if not clamped:
        return 0.0

    xs = sorted({b[0] for b in clamped} | {b[2] for b in clamped})
    total_area = 0.0

    for i in range(len(xs) - 1):
        x_left, x_right = xs[i], xs[i + 1]
        dx = x_right - x_left
        if dx <= 0.0:
            continue

        y_intervals = [(b[1], b[3]) for b in clamped if b[0] <= x_left and b[2] >= x_right]
        if not y_intervals:
            continue

        y_intervals.sort(key=lambda pair: pair[0])
        merged_y_len = 0.0
        cur_start, cur_end = y_intervals[0]
        for y_start, y_end in y_intervals[1:]:
            if y_start <= cur_end:
                cur_end = max(cur_end, y_end)
            else:
                merged_y_len += cur_end - cur_start
                cur_start, cur_end = y_start, y_end
        merged_y_len += cur_end - cur_start
        total_area += dx * merged_y_len

    return total_area


def classify_visual_severity(
    coverage_pct: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Map visual coverage percentage to severity category."""
    limits = thresholds or config.COVERAGE_THRESHOLDS
    low_max = float(limits.get("low_max", 5.0))
    medium_max = float(limits.get("medium_max", 20.0))

    if coverage_pct <= 0.0:
        return "No Detection"
    if coverage_pct < low_max:
        return "Low"
    if coverage_pct < medium_max:
        return "Medium"
    return "High"


def severity_indicator_color(severity_label: str) -> str:
    """Return UI indicator hex color for a visual severity category."""
    return {
        "No Detection": "#22C55E",
        "Low": "#EAB308",
        "Medium": "#F97316",
        "High": "#EF4444",
    }.get(severity_label, "#94A3B8")


def estimate_visual_severity(
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
    thresholds: dict[str, float] | None = None,
) -> SeverityAssessment:
    """Compute union fire, smoke, and combined coverage percentages and severity."""
    total_area = float(frame_width) * float(frame_height)
    if total_area <= 0.0:
        raise ValueError("Total frame area must be positive.")

    fire_boxes = [d.bbox_xyxy for d in detections if d.class_name == "fire"]
    smoke_boxes = [d.bbox_xyxy for d in detections if d.class_name == "smoke"]
    all_boxes = fire_boxes + smoke_boxes

    fire_area = compute_union_area_xyxy(fire_boxes, frame_width, frame_height)
    smoke_area = compute_union_area_xyxy(smoke_boxes, frame_width, frame_height)
    combined_area = compute_union_area_xyxy(all_boxes, frame_width, frame_height)

    fire_pct = min(100.0, (fire_area / total_area) * 100.0)
    smoke_pct = min(100.0, (smoke_area / total_area) * 100.0)
    combined_pct = min(100.0, (combined_area / total_area) * 100.0)

    overall_sev = classify_visual_severity(combined_pct, thresholds)
    fire_sev = classify_visual_severity(fire_pct, thresholds)
    smoke_sev = classify_visual_severity(smoke_pct, thresholds)
    max_conf = max((d.confidence for d in detections), default=0.0)

    return SeverityAssessment(
        fire_coverage_pct=round(fire_pct, 2),
        smoke_coverage_pct=round(smoke_pct, 2),
        combined_coverage_pct=round(combined_pct, 2),
        visual_severity=overall_sev,
        fire_severity=fire_sev,
        smoke_severity=smoke_sev,
        fire_box_count=len(fire_boxes),
        smoke_box_count=len(smoke_boxes),
        max_confidence=round(max_conf, 4),
        visual_indicator_color=severity_indicator_color(overall_sev),
        measurement_method="bounding_box_union",
    )


def estimate_segmentation_coverage(
    fire_masks: list[np.ndarray],
    smoke_masks: list[np.ndarray],
    frame_width: int,
    frame_height: int,
    max_confidence: float = 0.0,
    thresholds: dict[str, float] | None = None,
) -> SeverityAssessment:
    """Optional pixel-level union coverage from segmentation masks when available."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame width and height must be positive.")
    total_pixels = float(frame_width * frame_height)

    def _union_mask(masks: list[np.ndarray]) -> np.ndarray:
        canvas = np.zeros((frame_height, frame_width), dtype=bool)
        for m in masks:
            arr = np.asarray(m, dtype=bool)
            if arr.shape != (frame_height, frame_width):
                raise ValueError(
                    f"Mask shape {arr.shape} does not match frame {(frame_height, frame_width)}."
                )
            canvas |= arr
        return canvas

    fire_union = _union_mask(fire_masks)
    smoke_union = _union_mask(smoke_masks)
    combined_union = fire_union | smoke_union

    fire_pct = round(float(fire_union.sum()) / total_pixels * 100.0, 2)
    smoke_pct = round(float(smoke_union.sum()) / total_pixels * 100.0, 2)
    combined_pct = round(float(combined_union.sum()) / total_pixels * 100.0, 2)
    overall_sev = classify_visual_severity(combined_pct, thresholds)

    return SeverityAssessment(
        fire_coverage_pct=fire_pct,
        smoke_coverage_pct=smoke_pct,
        combined_coverage_pct=combined_pct,
        visual_severity=overall_sev,
        fire_severity=classify_visual_severity(fire_pct, thresholds),
        smoke_severity=classify_visual_severity(smoke_pct, thresholds),
        fire_box_count=len(fire_masks),
        smoke_box_count=len(smoke_masks),
        max_confidence=round(max_confidence, 4),
        visual_indicator_color=severity_indicator_color(overall_sev),
        measurement_method="segmentation_mask_union",
    )
