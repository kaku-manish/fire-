"""Phase 10 — Software-Based Intelligent Alert System.

Implements 4-tier visual alert classification (GREEN, YELLOW, ORANGE, RED),
configurable thresholds, duplicate suppression, cooldown periods, snapshot
metadata, and optional synthesized WAV alert chimes.

Notice: FireGuard AI is an academic prototype. It generates software UI alerts
only and NEVER contacts emergency services or claims certified life-safety protection.
"""

from __future__ import annotations

import io
import math
import struct
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import config

ALERT_PRIORITY: dict[str, int] = {
    "GREEN": 0,
    "YELLOW": 1,
    "ORANGE": 2,
    "RED": 3,
}

MONITORING_ALERT_MAP: dict[str, str] = {
    "GREEN": "NORMAL",
    "YELLOW": "WARNING",
    "ORANGE": "HIGH",
    "RED": "CRITICAL",
}

MONITORING_ALERT_COLORS: dict[str, str] = {
    "NORMAL": "#22C55E",
    "WARNING": "#EAB308",
    "HIGH": "#F97316",
    "CRITICAL": "#EF4444",
}

ALERT_DESCRIPTIONS: dict[str, str] = {
    "GREEN": "No visible fire or smoke detected above threshold.",
    "YELLOW": "Persistent smoke detected without visible flames (early warning).",
    "ORANGE": "Visible fire detected.",
    "RED": "Persistent fire with high or rapidly increasing visual coverage.",
}


def to_monitoring_level(alert_level: str) -> str:
    """Map GREEN/YELLOW/ORANGE/RED alert codes to NORMAL/WARNING/HIGH/CRITICAL."""
    return MONITORING_ALERT_MAP.get(str(alert_level).upper(), str(alert_level).upper())


@dataclass
class AlertEvent:
    """Structured software warning event produced by AlertManager."""

    alert_id: str
    alert_level: str
    alert_type: str
    timestamp: str
    source_filename: str
    confidence: float
    fire_coverage: float
    smoke_coverage: float
    visual_severity: str
    snapshot_path: str | None = None
    video_timestamp_sec: float | None = None
    message: str = ""
    suppressed: bool = False
    disclaimer: str = (
        "Academic software alert only — does not notify emergency services."
    )

    @property
    def monitoring_level(self) -> str:
        """Return industrial-style monitoring badge (NORMAL, WARNING, HIGH, CRITICAL)."""
        return to_monitoring_level(self.alert_level)


class AlertManager:
    """Configurable 4-tier alert evaluator with duplicate suppression and cooldown."""

    def __init__(self, thresholds: dict[str, float | int] | None = None) -> None:
        merged = dict(config.ALERT_THRESHOLDS)
        if thresholds:
            merged.update(thresholds)
        self.high_fire_coverage_pct = float(merged["high_fire_coverage_pct"])
        self.rapid_growth_rate_pct_per_sec = float(merged["rapid_growth_rate_pct_per_sec"])
        self.cooldown_seconds = float(merged["cooldown_seconds"])

        self._last_emitted_level: str | None = None
        self._last_emitted_source: str | None = None
        self._last_emitted_time_sec: float | None = None
        self.history: list[AlertEvent] = []

    def reset(self) -> None:
        """Reset cooldown and alert history for a fresh session or video."""
        self._last_emitted_level = None
        self._last_emitted_source = None
        self._last_emitted_time_sec = None
        self.history.clear()

    def classify_alert_level(
        self,
        has_fire: bool,
        has_smoke: bool,
        fire_coverage_pct: float,
        fire_persistent: bool = True,
        smoke_persistent: bool = True,
        fire_growth_rate_pct_per_sec: float = 0.0,
        growth_classification: str = "Insufficient data",
    ) -> tuple[str, str]:
        """Classify alert category (GREEN, YELLOW, ORANGE, RED) and descriptive type."""
        if not has_fire and not has_smoke:
            return "GREEN", "Normal / No Visible Detection"

        if has_fire:
            high_coverage = fire_coverage_pct >= self.high_fire_coverage_pct
            rapid_growth = (
                fire_growth_rate_pct_per_sec >= self.rapid_growth_rate_pct_per_sec
                or (
                    growth_classification == "Increasing"
                    and fire_coverage_pct >= (self.high_fire_coverage_pct * 0.5)
                )
            )
            if fire_persistent and (high_coverage or rapid_growth):
                return "RED", "Critical Persistent High/Growing Fire Alert"
            return "ORANGE", "Visible Fire Alert"

        if has_smoke and smoke_persistent:
            return "YELLOW", "Early Smoke Warning"
        return "YELLOW", "Smoke Observation Warning"

    def evaluate(
        self,
        source_filename: str,
        has_fire: bool,
        has_smoke: bool,
        confidence: float,
        fire_coverage_pct: float,
        smoke_coverage_pct: float,
        visual_severity: str,
        snapshot_path: str | None = None,
        video_timestamp_sec: float | None = None,
        event_clock_sec: float | None = None,
        fire_persistent: bool = True,
        smoke_persistent: bool = True,
        fire_growth_rate_pct_per_sec: float = 0.0,
        growth_classification: str = "Insufficient data",
        suppress_duplicates: bool = True,
    ) -> AlertEvent:
        """Evaluate detection metrics, apply cooldown suppression, and return AlertEvent."""
        level, alert_type = self.classify_alert_level(
            has_fire=has_fire,
            has_smoke=has_smoke,
            fire_coverage_pct=fire_coverage_pct,
            fire_persistent=fire_persistent,
            smoke_persistent=smoke_persistent,
            fire_growth_rate_pct_per_sec=fire_growth_rate_pct_per_sec,
            growth_classification=growth_classification,
        )

        now_utc = datetime.now(timezone.utc)
        clock = float(event_clock_sec) if event_clock_sec is not None else now_utc.timestamp()

        suppressed = False
        if (
            suppress_duplicates
            and self._last_emitted_level is not None
            and self._last_emitted_time_sec is not None
            and self._last_emitted_source == source_filename
        ):
            elapsed = clock - self._last_emitted_time_sec
            same_or_lower = ALERT_PRIORITY[level] <= ALERT_PRIORITY[self._last_emitted_level]
            if elapsed < self.cooldown_seconds and same_or_lower:
                suppressed = True

        alert_id = f"ALT-{now_utc.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        event = AlertEvent(
            alert_id=alert_id,
            alert_level=level,
            alert_type=alert_type,
            timestamp=now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            source_filename=source_filename,
            confidence=round(float(confidence), 4),
            fire_coverage=round(float(fire_coverage_pct), 2),
            smoke_coverage=round(float(smoke_coverage_pct), 2),
            visual_severity=visual_severity,
            snapshot_path=snapshot_path,
            video_timestamp_sec=(
                round(float(video_timestamp_sec), 2) if video_timestamp_sec is not None else None
            ),
            message=ALERT_DESCRIPTIONS[level],
            suppressed=suppressed,
        )

        if not suppressed:
            self._last_emitted_level = level
            self._last_emitted_source = source_filename
            self._last_emitted_time_sec = clock
            self.history.append(event)

        return event


def generate_alert_tone_wav_bytes(alert_level: str) -> bytes | None:
    """Synthesize a gentle browser-playable WAV notification chime for non-GREEN alerts."""
    if alert_level == "GREEN":
        return None

    pattern_map: dict[str, list[tuple[float, float]]] = {
        "YELLOW": [(660.0, 0.16)],
        "ORANGE": [(784.0, 0.14), (784.0, 0.14)],
        "RED": [(987.77, 0.14), (784.0, 0.12), (987.77, 0.18)],
    }
    tones = pattern_map.get(alert_level, [(660.0, 0.15)])
    sample_rate = 22050
    gap_samples = int(0.06 * sample_rate)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        frames = bytearray()
        for freq, dur in tones:
            n_samples = int(dur * sample_rate)
            for i in range(n_samples):
                env = min(1.0, i / 200.0) * min(1.0, (n_samples - i) / 200.0)
                val = int(12000 * env * math.sin(2.0 * math.pi * freq * i / sample_rate))
                frames.extend(struct.pack("<h", val))
            frames.extend(b"\x00\x00" * gap_samples)
        wf.writeframes(bytes(frames))

    return buf.getvalue()
