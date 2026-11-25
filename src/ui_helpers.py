"""Shared Streamlit helpers, circular SVG gauges, and simple status headers."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import streamlit as st

import config
from src.database import IncidentDatabase
from src.detector import FireSmokeDetector
from src.model_health import STATUS_READY, ModelHealthReport, run_model_health_check


@st.cache_resource
def get_database() -> IncidentDatabase:
    """Return a cached SQLite IncidentDatabase instance."""
    return IncidentDatabase(config.DB_PATH)


@st.cache_resource
def get_detector(model_path_str: str) -> FireSmokeDetector:
    """Return a cached FireSmokeDetector wrapper for the target weights path."""
    return FireSmokeDetector(model_path=Path(model_path_str))


def get_model_health(force: bool = False) -> ModelHealthReport:
    """Check and cache model readiness cleanly without cluttering the UI."""
    if force or "latest_health_report" not in st.session_state or st.session_state.latest_health_report is None:
        st.session_state.latest_health_report = run_model_health_check(config.MODEL_PATH, probe_runs=1)
    return st.session_state.latest_health_report


def inject_custom_css() -> None:
    """Load assets/styles.css into the active Streamlit page."""
    if config.STYLES_PATH.is_file():
        css = config.STYLES_PATH.read_text(encoding="utf-8")
        st.html(f"<style>{css}</style>")


def render_top_header(health: ModelHealthReport) -> None:
    """Render clean brand header with compact Model Ready / Model Unavailable pill."""
    is_ready = health.status == STATUS_READY
    pill_cls = "fg-model-ready" if is_ready else "fg-model-unavailable"
    pill_txt = "Model ready" if is_ready else "Model unavailable"

    st.html(
        f"""
        <div class="fg-topbar">
            <div>
                <div class="fg-brand-title">&#128293; {html.escape(config.PROJECT_NAME)}</div>
                <div class="fg-brand-sub">AI-Based Fire and Smoke Detection</div>
            </div>
            <div class="fg-model-pill {pill_cls}">{pill_txt}</div>
        </div>
        """
    )

    if is_ready and health.model_tier in {"SMOKE_TEST", "SUBSET_CPU_16EP"}:
        st.html(
            '<div class="fg-smoke-note">'
            "Note: Balanced D-Fire subset CPU YOLO11n weights are active for academic demonstration. "
            "Run notebooks/model_training.ipynb on Google Colab GPU (all 21,527 D-Fire images) for full production accuracy."
            "</div>"
        )
    elif not is_ready:
        st.error(
            health.error_message
            or "Trained fire and smoke model weights were not found at models/best.pt.",
            title="Model Unavailable",
        )


def render_circular_gauge(
    title: str,
    percentage: float,
    accent_color: str,
    subtitle: str = "Approximate visible region",
) -> None:
    """Render a reliable percentage coverage indicator card using native Streamlit components."""
    del accent_color  # Kept for signature compatibility; uses native Streamlit widgets reliably.
    pct = max(0.0, min(100.0, float(percentage)))
    display_pct = f"{pct:.2f}%"
    with st.container(border=True):
        st.metric(label=title, value=display_pct)
        st.progress(pct / 100.0)
        if subtitle:
            st.caption(subtitle)


def format_confidence_display(confidence: float, has_detection: bool = True) -> str:
    """Format confidence score cleanly, returning 'N/A' when no detections occurred."""
    if not has_detection or float(confidence) <= 0.0:
        return "N/A"
    return f"{float(confidence):.1%}"


def severity_css_class(severity: str) -> str:
    """Map visual severity category to color-coded banner CSS class."""
    key = severity.strip().lower()
    if key == "low":
        return "fg-sev-yellow"
    if key == "medium":
        return "fg-sev-orange"
    if key in {"high", "analysis failed", "failed"}:
        return "fg-sev-red"
    return "fg-sev-green"


def format_final_result_label(has_fire: bool, has_smoke: bool) -> str:
    """Return user-friendly final detection result string."""
    if has_fire and has_smoke:
        return "Fire and Smoke Detected"
    if has_fire:
        return "Fire Detected"
    if has_smoke:
        return "Smoke Detected"
    return "No Fire or Smoke Detected"


def render_final_result_banner(
    final_result: str,
    visual_severity: str,
    confidence: float,
    fire_confidence: float | None = None,
    smoke_confidence: float | None = None,
) -> None:
    """Render color-coded detection summary banner (green/yellow/orange/red)."""
    is_failed = final_result.strip().lower() == "analysis failed"
    sev_cls = "fg-sev-red" if is_failed else severity_css_class(visual_severity)
    sev_display = "No detection" if visual_severity.lower() == "no detection" else visual_severity
    conf_text = format_confidence_display(confidence, (not is_failed) and float(confidence) > 0.0)
    conf_parts = [f"Detection confidence: <strong>{html.escape(conf_text)}</strong>"]
    if not is_failed and fire_confidence is not None and fire_confidence > 0.0:
        conf_parts.append(f"Fire: <strong>{fire_confidence:.1%}</strong>")
    if not is_failed and smoke_confidence is not None and smoke_confidence > 0.0:
        conf_parts.append(f"Smoke: <strong>{smoke_confidence:.1%}</strong>")
    conf_summary = " &nbsp;|&nbsp; ".join(conf_parts)
    st.html(
        f"""
        <div class="fg-result-banner {sev_cls}">
            <div style="font-size:1.08rem; font-weight:700; margin-bottom:0.2rem;">
                Final result: {html.escape(final_result)}
            </div>
            <div style="font-size:0.9rem;">
                Visual severity: <strong>{html.escape(sev_display)}</strong> &nbsp;|&nbsp;
                {conf_summary}
            </div>
        </div>
        """
    )


def load_json_if_exists(path: Path) -> dict[str, Any] | list[Any] | None:
    """Safely parse a JSON report file if present on disk."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
