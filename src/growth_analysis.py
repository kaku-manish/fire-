"""Phase 9 — Temporal Fire Growth & Coverage Progression Analysis.

Tracks visible fire and smoke coverage percentages across prerecorded video
timestamps, applies rolling smoothing, identifies early smoke warnings prior
to flame visibility, classifies growth trends, and builds interactive Plotly
visualizations plus static chart exports for PDF reports.

Disclaimer: Bounding-box coverage growth reflects observed visual changes in
the recorded video only and does NOT predict physical fire spread or future
affected locations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

import config

if TYPE_CHECKING:
    import plotly.graph_objects as go

GROWTH_DISCLAIMER: str = (
    "Visual growth analysis measures changes in 2D bounding-box coverage across video "
    "frames. It does NOT predict physical fire spread, heat intensity, or future "
    "affected locations."
)


@dataclass
class FrameCoverageMetric:
    """Per-frame coverage and confidence observation for temporal analysis."""

    frame_index: int
    timestamp_sec: float
    fire_coverage_pct: float
    smoke_coverage_pct: float
    fire_confidence: float = 0.0
    smoke_confidence: float = 0.0


@dataclass
class GrowthAnalysisResult:
    """Summary of fire and smoke coverage progression across a video."""

    initial_fire_timestamp_sec: float | None
    initial_smoke_timestamp_sec: float | None
    smoke_before_fire: bool
    smoke_lead_time_sec: float | None
    max_fire_coverage_pct: float
    avg_fire_coverage_pct: float
    max_smoke_coverage_pct: float
    avg_smoke_coverage_pct: float
    fire_coverage_change_pct: float
    fire_growth_rate_pct_per_sec: float
    growth_classification: str
    frame_count_analyzed: int
    timestamps_sec: list[float] = field(default_factory=list)
    raw_fire_series: list[float] = field(default_factory=list)
    smoothed_fire_series: list[float] = field(default_factory=list)
    raw_smoke_series: list[float] = field(default_factory=list)
    smoothed_smoke_series: list[float] = field(default_factory=list)
    summary_text: str = ""
    disclaimer: str = GROWTH_DISCLAIMER


def smooth_series(values: Sequence[float], window: int = config.GROWTH_SMOOTHING_WINDOW) -> list[float]:
    """Apply centered rolling mean smoothing to suppress single-frame jitter."""
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return []
    eff_window = max(1, min(int(window), n))
    if eff_window == 1:
        return [round(float(v), 3) for v in arr]
    left = eff_window // 2
    right = (eff_window - 1) // 2
    return [
        round(float(np.mean(arr[max(0, i - left) : min(n, i + right + 1)])), 3)
        for i in range(n)
    ]


def analyze_fire_growth(
    frame_metrics: Sequence[FrameCoverageMetric],
    smoothing_window: int = config.GROWTH_SMOOTHING_WINDOW,
    min_frames: int = config.GROWTH_MIN_FRAMES,
    slope_threshold: float = config.GROWTH_SLOPE_THRESHOLD,
) -> GrowthAnalysisResult:
    """Analyze visible fire and smoke progression across successive video frames."""
    if not frame_metrics:
        return GrowthAnalysisResult(
            initial_fire_timestamp_sec=None,
            initial_smoke_timestamp_sec=None,
            smoke_before_fire=False,
            smoke_lead_time_sec=None,
            max_fire_coverage_pct=0.0,
            avg_fire_coverage_pct=0.0,
            max_smoke_coverage_pct=0.0,
            avg_smoke_coverage_pct=0.0,
            fire_coverage_change_pct=0.0,
            fire_growth_rate_pct_per_sec=0.0,
            growth_classification="Insufficient data",
            frame_count_analyzed=0,
            summary_text="No frames available for temporal growth analysis.",
        )

    ordered = sorted(frame_metrics, key=lambda m: (m.timestamp_sec, m.frame_index))
    timestamps = [round(float(m.timestamp_sec), 3) for m in ordered]
    raw_fire = [round(float(m.fire_coverage_pct), 3) for m in ordered]
    raw_smoke = [round(float(m.smoke_coverage_pct), 3) for m in ordered]

    smoothed_fire = smooth_series(raw_fire, window=smoothing_window)
    smoothed_smoke = smooth_series(raw_smoke, window=smoothing_window)

    initial_fire_ts = next((m.timestamp_sec for m in ordered if m.fire_coverage_pct > 0.0), None)
    initial_smoke_ts = next((m.timestamp_sec for m in ordered if m.smoke_coverage_pct > 0.0), None)

    smoke_before_fire = False
    smoke_lead_time: float | None = None
    if initial_smoke_ts is not None:
        if initial_fire_ts is None:
            smoke_before_fire = True
        elif initial_smoke_ts < initial_fire_ts:
            smoke_before_fire = True
            smoke_lead_time = round(initial_fire_ts - initial_smoke_ts, 2)

    active_fire = [v for v in raw_fire if v > 0.0]
    active_smoke = [v for v in raw_smoke if v > 0.0]

    max_fire = round(max(raw_fire), 2) if raw_fire else 0.0
    avg_fire = round(float(np.mean(active_fire)), 2) if active_fire else 0.0
    max_smoke = round(max(raw_smoke), 2) if raw_smoke else 0.0
    avg_smoke = round(float(np.mean(active_smoke)), 2) if active_smoke else 0.0

    fire_indices = [idx for idx, val in enumerate(raw_fire) if val > 0.0]
    if len(ordered) < min_frames or len(fire_indices) < min_frames:
        classification = "Insufficient data"
        slope = 0.0
        delta_pct = (
            round(smoothed_fire[fire_indices[-1]] - smoothed_fire[fire_indices[0]], 2)
            if len(fire_indices) >= 2
            else 0.0
        )
    else:
        start_idx = fire_indices[0]
        end_idx = fire_indices[-1]
        t_slice = np.asarray(timestamps[start_idx : end_idx + 1], dtype=float)
        active_smoothed = smooth_series(raw_fire[start_idx : end_idx + 1], window=smoothing_window)
        y_slice = np.asarray(active_smoothed, dtype=float)
        delta_pct = round(float(y_slice[-1] - y_slice[0]), 2)

        duration = float(t_slice[-1] - t_slice[0])
        if duration > 1e-6 and len(t_slice) >= 2:
            slope = float(np.polyfit(t_slice, y_slice, 1)[0])
        else:
            slope = float((y_slice[-1] - y_slice[0]) / max(1, len(y_slice) - 1))

        if slope >= slope_threshold or delta_pct >= 2.5:
            classification = "Increasing"
        elif slope <= -slope_threshold or delta_pct <= -2.5:
            classification = "Decreasing"
        else:
            classification = "Relatively stable"

    lead_msg = ""
    if smoke_before_fire and smoke_lead_time is not None:
        lead_msg = f" Smoke appeared {smoke_lead_time:.2f}s before visible flames."
    elif smoke_before_fire:
        lead_msg = " Smoke was detected without visible flames (early warning)."

    summary = (
        f"Growth trend: {classification}. Max fire coverage: {max_fire:.2f}%, "
        f"Avg fire coverage (active frames): {avg_fire:.2f}%, "
        f"Change: {delta_pct:+.2f} percentage points ({slope:+.3f}%/s).{lead_msg}"
    )

    return GrowthAnalysisResult(
        initial_fire_timestamp_sec=round(initial_fire_ts, 2) if initial_fire_ts is not None else None,
        initial_smoke_timestamp_sec=round(initial_smoke_ts, 2) if initial_smoke_ts is not None else None,
        smoke_before_fire=smoke_before_fire,
        smoke_lead_time_sec=smoke_lead_time,
        max_fire_coverage_pct=max_fire,
        avg_fire_coverage_pct=avg_fire,
        max_smoke_coverage_pct=max_smoke,
        avg_smoke_coverage_pct=avg_smoke,
        fire_coverage_change_pct=delta_pct,
        fire_growth_rate_pct_per_sec=round(slope, 4),
        growth_classification=classification,
        frame_count_analyzed=len(ordered),
        timestamps_sec=timestamps,
        raw_fire_series=raw_fire,
        smoothed_fire_series=smoothed_fire,
        raw_smoke_series=raw_smoke,
        smoothed_smoke_series=smoothed_smoke,
        summary_text=summary,
    )


def create_growth_plotly_figure(growth: GrowthAnalysisResult, title: str = "Fire and smoke visual coverage over time") -> go.Figure:
    """Create an interactive dark-themed Plotly line chart of coverage vs time."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=growth.timestamps_sec,
            y=growth.raw_fire_series,
            mode="lines",
            name="Fire Coverage (Raw %)",
            line={"color": "rgba(239, 68, 68, 0.35)", "width": 1.5, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=growth.timestamps_sec,
            y=growth.smoothed_fire_series,
            mode="lines",
            name="Fire Coverage (Smoothed %)",
            line={"color": "#EF4444", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=growth.timestamps_sec,
            y=growth.raw_smoke_series,
            mode="lines",
            name="Smoke Coverage (Raw %)",
            line={"color": "rgba(249, 115, 22, 0.35)", "width": 1.5, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=growth.timestamps_sec,
            y=growth.smoothed_smoke_series,
            mode="lines",
            name="Smoke Coverage (Smoothed %)",
            line={"color": "#F97316", "width": 3},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Visual Coverage (%)",
        template="plotly_dark",
        paper_bgcolor="#171B21",
        plot_bgcolor="#12151A",
        font={"color": "#E2E8F0"},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 45, "r": 25, "t": 55, "b": 45},
    )
    return fig


def export_growth_chart_png(growth: GrowthAnalysisResult, output_path: Path) -> Path:
    """Save a static Matplotlib PNG of the coverage progression chart for PDF reports."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor("#111418")
    ax.set_facecolor("#181C22")

    ax.plot(
        growth.timestamps_sec,
        growth.smoothed_fire_series,
        color="#EF4444",
        linewidth=2.4,
        label="Fire Coverage (Smoothed %)",
    )
    ax.plot(
        growth.timestamps_sec,
        growth.smoothed_smoke_series,
        color="#F97316",
        linewidth=2.4,
        label="Smoke Coverage (Smoothed %)",
    )
    ax.set_title(f"Temporal Visual Coverage ({growth.growth_classification})", color="#F8FAFC")
    ax.set_xlabel("Time (seconds)", color="#CBD5E1")
    ax.set_ylabel("Visual Coverage (%)", color="#CBD5E1")
    ax.tick_params(colors="#CBD5E1")
    ax.grid(color="#2A303A", linestyle="--", linewidth=0.6)
    ax.legend(facecolor="#181C22", labelcolor="#F8FAFC")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
