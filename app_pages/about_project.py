"""Page 4 — About Project (Student & Evaluator Friendly Overview)."""

from __future__ import annotations

import streamlit as st

import config

st.header("About Project")

st.subheader("Project Objective")
st.markdown(
    "FireGuard AI is a software-only academic deep learning prototype designed to detect **fire** "
    "and **smoke** from uploaded images and prerecorded videos, calculate approximate affected "
    "visual coverage percentages, classify visual severity, and store detection history."
)

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("**Problem Statement**")
        st.markdown(
            "Traditional physical fire sensors require heat or smoke particles to reach a mounted "
            "device, while basic color-thresholding software frequently triggers false alarms on "
            "sunsets, bright lights, and orange objects."
        )
with c2:
    with st.container(border=True):
        st.markdown("**Proposed Solution**")
        st.markdown(
            "FireGuard AI uses a trained **YOLO11n** object detection model to locate fire and smoke "
            "separately or together, compute bounding-box union coverage without double-counting "
            "overlapping regions, and present results in a clean web interface."
        )

st.subheader("Core Technologies & Dataset")
t1, t2 = st.columns(2)
with t1:
    with st.container(border=True):
        st.markdown("- **D-Fire Dataset:** Labeled image collection covering fire, smoke, joint fire + smoke, and negative background scenes (`0: smoke`, `1: fire`).")
        st.markdown("- **YOLO11n Model:** Lightweight Ultralytics single-stage object detector fine-tuned via transfer learning.")
with t2:
    with st.container(border=True):
        st.markdown("- **Python & OpenCV:** Image validation, video frame extraction, and red/orange bounding-box drawing.")
        st.markdown("- **Streamlit & SQLite:** Interactive 4-page web dashboard and local incident database (`fireguard.db`).")

st.subheader("System Architecture")
st.html(
    """
    <div class="fg-arch-flow">
        Dataset &rarr; Model Training &rarr; Trained YOLO Model &rarr; Uploaded Image or Video
        &rarr; Fire and Smoke Detection &rarr; Percentage Calculation &rarr; Result
    </div>
    """
)

st.subheader("System Workflow")
st.markdown(
    "1. **Upload Media:** Select a JPG, JPEG, PNG image or MP4 video on the **Live Audit** page.\n"
    "2. **Model Inference:** The trained YOLO11n model detects fire (red boxes) and smoke (orange boxes) along with confidence scores.\n"
    "3. **Union Coverage Calculation:** Fire coverage and smoke coverage are calculated separately using bounding-box union areas so overlapping boxes are never double-counted.\n"
    "4. **Visual Severity & Save:** Combined coverage classifies visual severity (`No detection`, `Low <5%`, `Medium 5–20%`, `High >=20%`) and saves the record to SQLite."
)

st.subheader("Project Limitations")
st.warning(config.ACADEMIC_DISCLAIMER, title="Academic Prototype Limitations")
st.markdown(
    "- **Approximate visual measurement:** Bounding-box percentages represent 2D frame coverage, not physical fire size, temperature, or property damage.\n"
    "- **Software-only prototype:** Does not connect to cameras, physical alarms, sensors, or emergency services.\n"
    "- **Negative predictions:** A negative result means no fire or smoke was detected above threshold—not proof that an environment is completely safe."
)
