"""Page 1 — FireGuard AI Simple Home Page."""

from __future__ import annotations

import streamlit as st

st.header("FireGuard AI")
st.subheader("AI-Based Fire and Smoke Detection")

st.markdown(
    "FireGuard AI is a software-based deep learning prototype that identifies visible **fire** "
    "and **smoke** from uploaded images and videos. Using a custom-trained **YOLO11n** object "
    "detection model, the system highlights affected areas with bounding boxes, estimates the "
    "approximate percentage of the frame affected, and logs each detection into local storage."
)

c1, c2, c3 = st.columns(3)
with c1:
    st.html(
        """
        <div class="fg-feature-card fg-feature-fire">
            <h4 style="margin-top:0; color:#F8FAFC;">Fire Detection</h4>
            <p style="color:#CBD5E1; font-size:0.9rem; margin-bottom:0;">
                Detects visible flames, marks them with red bounding boxes, and calculates
                the approximate percentage of the scene covered by fire.
            </p>
        </div>
        """
    )
with c2:
    st.html(
        """
        <div class="fg-feature-card fg-feature-smoke">
            <h4 style="margin-top:0; color:#F8FAFC;">Smoke Detection</h4>
            <p style="color:#CBD5E1; font-size:0.9rem; margin-bottom:0;">
                Detects smoke plumes independently with orange bounding boxes, supporting early
                smoke identification even before open flames appear.
            </p>
        </div>
        """
    )
with c3:
    st.html(
        """
        <div class="fg-feature-card fg-feature-soft">
            <h4 style="margin-top:0; color:#F8FAFC;">Software-Only Detection</h4>
            <p style="color:#CBD5E1; font-size:0.9rem; margin-bottom:0;">
                Works completely in software on uploaded JPG, PNG, JPEG, and MP4 files—no
                physical cameras, sensors, or extra hardware required.
            </p>
        </div>
        """
    )

st.write("")
if st.button("Start Detection", type="primary", icon=":material/radar:", width="content"):
    st.switch_page("app_pages/live_audit.py")
