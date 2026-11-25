"""Page 3 — Simple Detection History Table & Clear History Control."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui_helpers import get_database

st.header("Detection History")
st.caption("Automatically saved results from completed image and video analyses.")

db = get_database()
records = db.get_incidents()

if not records:
    st.info("No detections recorded yet. Go to **Live Audit** to analyze an image or video.")
else:
    rows = [
        {
            "Date and time": r["created_at"],
            "Uploaded file name": r["file_name"],
            "Fire coverage (%)": round(float(r["fire_coverage"]), 2),
            "Smoke coverage (%)": round(float(r["smoke_coverage"]), 2),
            "Visual severity": (
                "No detection" if str(r["visual_severity"]).lower() == "no detection" else r["visual_severity"]
            ),
            "Detection result": r["detection_type"],
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, alt="Saved FireGuard AI detection history records")

    st.write("")
    c1, c2 = st.columns([2, 1])
    with c1:
        confirm_clear = st.checkbox(
            "Confirm clearing all saved detection records",
            key="confirm_clear_history",
        )
    with c2:
        if st.button(
            "Clear History",
            icon=":material/delete:",
            disabled=not confirm_clear,
            width="stretch",
        ):
            deleted = db.clear_incidents()
            st.success(f"Cleared {deleted} detection record(s).")
            st.rerun()
