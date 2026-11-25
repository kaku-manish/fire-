"""Phase 8 — False Alarm Reduction & Temporal Verification Module.

Reduces false positives from visually similar conditions (sunsets, sunlight,
artificial lighting, fog, steam, clouds, headlights, reflections) using:
1. Calibrated per-class confidence thresholds and minimum area geometry filtering.
2. Sliding-window and consecutive-frame temporal persistence verification (for video).
3. Exponential Moving Average (EMA) confidence smoothing across frames.

Does NOT rely exclusively on brittle RGB color rules. Temporal verification is
optional for single images and enabled by default for video streams.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
from src.detector import Detection


@dataclass
class TemporalFilterOutput:
    """Result returned after filtering detections for a single video frame."""

    confirmed_detections: list[Detection] = field(default_factory=list)
    raw_count: int = 0
    suppressed_count: int = 0
    fire_persistent: bool = False
    smoke_persistent: bool = False
    smoothed_fire_confidence: float = 0.0
    smoothed_smoke_confidence: float = 0.0
    fire_consecutive_frames: int = 0
    smoke_consecutive_frames: int = 0


def filter_by_geometry_and_confidence(
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
    fire_conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    smoke_conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    min_box_area_ratio: float = config.DEFAULT_MIN_BOX_AREA_RATIO,
) -> list[Detection]:
    """Filter raw detections by calibrated confidence and minimum relative box area."""
    total_area = float(max(1, frame_width * frame_height))
    kept: list[Detection] = []
    for det in detections:
        threshold = fire_conf_threshold if det.class_name == "fire" else smoke_conf_threshold
        if det.confidence < threshold:
            continue
        if (det.area_px / total_area) < min_box_area_ratio:
            continue
        kept.append(det)
    return kept


class TemporalFalseAlarmFilter:
    """Stateful multi-frame verifier and EMA confidence smoother for video streams."""

    def __init__(
        self,
        window_size: int = config.DEFAULT_TEMPORAL_WINDOW,
        min_consecutive_frames: int = config.DEFAULT_MIN_CONSECUTIVE_FRAMES,
        min_persistence_ratio: float = config.DEFAULT_MIN_PERSISTENCE_RATIO,
        ema_alpha: float = config.DEFAULT_EMA_ALPHA,
        fire_conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        smoke_conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        min_box_area_ratio: float = config.DEFAULT_MIN_BOX_AREA_RATIO,
        enabled: bool = True,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if min_consecutive_frames < 1:
            raise ValueError("min_consecutive_frames must be >= 1")
        self.window_size = int(window_size)
        self.min_consecutive_frames = int(min_consecutive_frames)
        self.min_persistence_ratio = float(min_persistence_ratio)
        self.ema_alpha = float(np.clip(ema_alpha, 0.05, 1.0))
        self.fire_conf_threshold = float(fire_conf_threshold)
        self.smoke_conf_threshold = float(smoke_conf_threshold)
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.enabled = bool(enabled)

        self._history: dict[str, deque[bool]] = {
            "fire": deque(maxlen=self.window_size),
            "smoke": deque(maxlen=self.window_size),
        }
        self._streaks: dict[str, int] = {"fire": 0, "smoke": 0}
        self._ema_conf: dict[str, float] = {"fire": 0.0, "smoke": 0.0}
        self.total_raw_detections: int = 0
        self.total_suppressed_detections: int = 0

    def reset(self) -> None:
        """Clear temporal buffers before processing a new video."""
        self._history["fire"].clear()
        self._history["smoke"].clear()
        self._streaks = {"fire": 0, "smoke": 0}
        self._ema_conf = {"fire": 0.0, "smoke": 0.0}
        self.total_raw_detections = 0
        self.total_suppressed_detections = 0

    def _is_class_persistent(self, cls_name: str) -> bool:
        hist = self._history[cls_name]
        if not hist:
            return False
        consecutive_ok = self._streaks[cls_name] >= self.min_consecutive_frames
        ratio_ok = (len(hist) >= self.min_consecutive_frames) and (
            sum(hist) / len(hist) >= self.min_persistence_ratio
        )
        return consecutive_ok or ratio_ok

    def update(
        self,
        raw_detections: list[Detection],
        frame_width: int,
        frame_height: int,
    ) -> TemporalFilterOutput:
        """Process one video frame's raw detections and emit temporally verified ones."""
        raw_count = len(raw_detections)
        self.total_raw_detections += raw_count

        pre_filtered = filter_by_geometry_and_confidence(
            raw_detections,
            frame_width=frame_width,
            frame_height=frame_height,
            fire_conf_threshold=self.fire_conf_threshold,
            smoke_conf_threshold=self.smoke_conf_threshold,
            min_box_area_ratio=self.min_box_area_ratio,
        )

        if not self.enabled:
            suppressed = raw_count - len(pre_filtered)
            self.total_suppressed_detections += suppressed
            fire_confs = [d.confidence for d in pre_filtered if d.class_name == "fire"]
            smoke_confs = [d.confidence for d in pre_filtered if d.class_name == "smoke"]
            return TemporalFilterOutput(
                confirmed_detections=pre_filtered,
                raw_count=raw_count,
                suppressed_count=suppressed,
                fire_persistent=bool(fire_confs),
                smoke_persistent=bool(smoke_confs),
                smoothed_fire_confidence=round(max(fire_confs, default=0.0), 4),
                smoothed_smoke_confidence=round(max(smoke_confs, default=0.0), 4),
                fire_consecutive_frames=1 if fire_confs else 0,
                smoke_consecutive_frames=1 if smoke_confs else 0,
            )

        for cls_name in ("fire", "smoke"):
            cls_confs = [d.confidence for d in pre_filtered if d.class_name == cls_name]
            present = len(cls_confs) > 0
            self._history[cls_name].append(present)
            if present:
                self._streaks[cls_name] += 1
                frame_max = max(cls_confs)
                prev = self._ema_conf[cls_name]
                self._ema_conf[cls_name] = (
                    frame_max
                    if prev == 0.0
                    else (self.ema_alpha * frame_max + (1.0 - self.ema_alpha) * prev)
                )
            else:
                self._streaks[cls_name] = 0
                self._ema_conf[cls_name] = (1.0 - self.ema_alpha) * self._ema_conf[cls_name]

        fire_ok = self._is_class_persistent("fire")
        smoke_ok = self._is_class_persistent("smoke")

        confirmed: list[Detection] = []
        for det in pre_filtered:
            if det.class_name == "fire" and fire_ok:
                confirmed.append(det)
            elif det.class_name == "smoke" and smoke_ok:
                confirmed.append(det)
            elif det.class_name not in {"fire", "smoke"}:
                confirmed.append(det)

        suppressed = raw_count - len(confirmed)
        self.total_suppressed_detections += suppressed

        return TemporalFilterOutput(
            confirmed_detections=confirmed,
            raw_count=raw_count,
            suppressed_count=suppressed,
            fire_persistent=fire_ok,
            smoke_persistent=smoke_ok,
            smoothed_fire_confidence=round(self._ema_conf["fire"], 4),
            smoothed_smoke_confidence=round(self._ema_conf["smoke"], 4),
            fire_consecutive_frames=self._streaks["fire"],
            smoke_consecutive_frames=self._streaks["smoke"],
        )


def calibrate_confidence_thresholds(
    validation_records: list[dict[str, Any]],
    candidates: list[float] | None = None,
) -> dict[str, Any]:
    """Sweep confidence thresholds on validation predictions to balance FPR and F1.

    Each record expects:
        {"has_ground_truth": bool, "max_confidence": float}
    """
    thresholds = candidates or [round(0.15 + 0.05 * i, 2) for i in range(15)]
    sweep: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    for th in thresholds:
        tp = fp = tn = fn = 0
        for rec in validation_records:
            gt = bool(rec["has_ground_truth"])
            pred = float(rec["max_confidence"]) >= th
            if gt and pred:
                tp += 1
            elif not gt and pred:
                fp += 1
            elif not gt and not pred:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        row = {
            "threshold": th,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "missed_detections": fn,
        }
        sweep.append(row)
        if best_row is None or row["f1_score"] > best_row["f1_score"]:
            best_row = row

    return {
        "recommended_threshold": best_row["threshold"] if best_row else config.DEFAULT_CONFIDENCE_THRESHOLD,
        "best_metrics": best_row,
        "threshold_sweep": sweep,
    }
