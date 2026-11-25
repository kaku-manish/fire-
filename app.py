"""FireGuard AI — Simplified 4-Page Streamlit Application Entry Point."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import streamlit as st

import config
from src.model_health import STATUS_READY
from src.ui_helpers import (
    get_database,
    get_model_health,
    inject_custom_css,
    render_top_header,
)

st.set_page_config(
    page_title="FireGuard AI — Fire & Smoke Detection",
    page_icon=":material/local_fire_department:",
    layout="wide",
    initial_sidebar_state="expanded",
)

config.ensure_directories()
inject_custom_css()

if "session_id" not in st.session_state:
    st.session_state.session_id = (
        f"SES-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    )
if "latest_health_report" not in st.session_state:
    st.session_state.latest_health_report = None
if "live_audit_result" not in st.session_state:
    st.session_state.live_audit_result = None
if "last_uploaded_file_sig" not in st.session_state:
    st.session_state.last_uploaded_file_sig = None
if "stop_video_audit" not in st.session_state:
    st.session_state.stop_video_audit = False

_db = get_database()
health = get_model_health(force=False)

# Strictly 4 clean pages in sidebar
pages = [
    st.Page(
        "app_pages/home.py",
        title="Home",
        icon=":material/home:",
        default=True,
        help="Project overview and quick start.",
    ),
    st.Page(
        "app_pages/live_audit.py",
        title="Live Audit",
        icon=":material/radar:",
        help="Upload an image or MP4 video to detect fire and smoke.",
    ),
    st.Page(
        "app_pages/detection_history.py",
        title="Detection History",
        icon=":material/history:",
        help="View and manage saved detection results.",
    ),
    st.Page(
        "app_pages/about_project.py",
        title="About Project",
        icon=":material/info:",
        help="Project architecture, dataset, model, and limitations.",
    ),
]

current_page = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown("### FireGuard AI")
    st.caption("AI-Based Fire & Smoke Detection")
    st.divider()
    if health.status == STATUS_READY:
        st.success("Model Ready", icon=":material/check_circle:")
    else:
        st.error("Model Unavailable", icon=":material/warning:")
    st.caption("Software-only academic prototype")

render_top_header(health)
current_page.run()
