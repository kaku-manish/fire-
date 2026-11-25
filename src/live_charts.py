"""Dynamic Plotly chart builders for the Live Audit monitoring dashboard."""

from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go


def _apply_dark_layout(fig: go.Figure, title: str, x_title: str, y_title: str, y_range: list[float] | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_dark",
        paper_bgcolor="#171B21",
        plot_bgcolor="#12151A",
        font={"color": "#E2E8F0", "size": 11},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 18, "t": 48, "b": 38},
        height=265,
    )
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def build_confidence_time_chart(
    timestamps: Sequence[float],
    fire_confs: Sequence[float],
    smoke_confs: Sequence[float],
) -> go.Figure:
    """Plot Fire and Smoke prediction confidence percentages over video time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=[round(c * 100.0, 2) for c in fire_confs],
            mode="lines+markers",
            name="Fire Conf %",
            line={"color": "#EF4444", "width": 2.4},
            marker={"size": 4},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=[round(c * 100.0, 2) for c in smoke_confs],
            mode="lines+markers",
            name="Smoke Conf %",
            line={"color": "#F97316", "width": 2.4},
            marker={"size": 4},
        )
    )
    return _apply_dark_layout(fig, "Fire & Smoke Confidence Over Time", "Time (s)", "Confidence (%)", [0.0, 105.0])


def build_coverage_time_chart(
    timestamps: Sequence[float],
    fire_covs: Sequence[float],
    smoke_covs: Sequence[float],
) -> go.Figure:
    """Plot approximate 2D visual fire and smoke coverage over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(fire_covs),
            mode="lines",
            name="Fire Coverage %",
            fill="tozeroy",
            line={"color": "#EF4444", "width": 2.2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(smoke_covs),
            mode="lines",
            name="Smoke Coverage %",
            fill="tozeroy",
            line={"color": "#F97316", "width": 2.2},
        )
    )
    return _apply_dark_layout(fig, "Visual Coverage Over Time", "Time (s)", "Coverage (%)")


def build_latency_time_chart(
    timestamps: Sequence[float],
    latencies_ms: Sequence[float],
) -> go.Figure:
    """Plot per-frame model inference latency (ms) over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(latencies_ms),
            mode="lines",
            name="Latency (ms)",
            line={"color": "#38BDF8", "width": 2.2},
        )
    )
    return _apply_dark_layout(fig, "Inference Latency Over Time", "Time (s)", "Latency (ms)")


def build_detection_frequency_chart(
    fire_frames: int,
    smoke_frames: int,
    both_frames: int,
    clear_frames: int,
) -> go.Figure:
    """Plot frame-level detection frequency counts."""
    labels = ["Fire Only", "Smoke Only", "Fire + Smoke", "No Detection"]
    values = [
        max(0, fire_frames - both_frames),
        max(0, smoke_frames - both_frames),
        both_frames,
        clear_frames,
    ]
    colors = ["#EF4444", "#F97316", "#EAB308", "#22C55E"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=values,
                textposition="auto",
            )
        ]
    )
    return _apply_dark_layout(fig, "Frame Detection Frequency", "Category", "Frames")
