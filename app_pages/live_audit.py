"""Page 2 — Live Audit (Unified Image & Video Fire and Smoke Detection)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

import config
from src.detector import Detection, ModelNotFoundError, draw_detections
from src.image_processor import InvalidImageError, process_image_bytes
from src.model_health import STATUS_READY
from src.severity import classify_visual_severity, estimate_visual_severity
from src.ui_helpers import (
    format_confidence_display,
    format_final_result_label,
    get_database,
    get_detector,
    get_model_health,
    render_circular_gauge,
    render_final_result_banner,
)
from src.video_processor import (
    InvalidVideoError,
    _transcode_to_h264_if_available,
    save_uploaded_video_bytes,
    validate_video_file,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4"}

st.header("Live Audit")
st.caption("Upload an image or video to detect fire and smoke.")

health = get_model_health(force=False)
model_ready = health.status == STATUS_READY
db = get_database()

uploaded_file = st.file_uploader(
    "Drag and drop image or video",
    type=["jpg", "jpeg", "png", "mp4"],
    help="Supported formats: JPG, JPEG, PNG, MP4",
)

# Clear stale analysis output immediately whenever the user changes or removes the file
current_file_sig = (uploaded_file.name, uploaded_file.size) if uploaded_file is not None else None
if st.session_state.get("last_uploaded_file_sig") != current_file_sig:
    st.session_state.live_audit_result = None
    st.session_state.stop_video_audit = False
    st.session_state.last_uploaded_file_sig = current_file_sig

btn_col1, btn_col2 = st.columns([4, 1])
with btn_col1:
    analyze_clicked = st.button(
        "Analyze Fire & Smoke",
        type="primary",
        disabled=(uploaded_file is None or not model_ready),
        width="stretch",
    )
with btn_col2:
    stop_clicked = st.button(
        "Stop",
        disabled=(uploaded_file is None or Path(uploaded_file.name).suffix.lower() not in VIDEO_EXTS),
        width="stretch",
    )

if stop_clicked:
    st.session_state.stop_video_audit = True


def _verify_model_classes(detector: Any) -> None:
    """Inspect loaded model class mappings and verify fire & smoke support."""
    names = detector.class_names
    normalized = {str(v).strip().lower() for v in names.values()}
    if not {"fire", "smoke"}.issubset(normalized):
        raise ValueError(
            f"Loaded model classes {names} do not match expected fire and smoke classes."
        )


def _run_image_analysis(file_bytes: bytes, filename: str) -> None:
    detector = get_detector(str(config.MODEL_PATH))
    _verify_model_classes(detector)
    with st.spinner("Analyzing image with YOLO11n..."):
        analysis = process_image_bytes(
            image_bytes=file_bytes,
            filename=filename,
            detector=detector,
            conf_threshold=float(config.DEFAULT_CONFIDENCE_THRESHOLD),
            iou_threshold=float(config.DEFAULT_IOU_THRESHOLD),
            min_box_area_ratio=0.0,
        )

    det_res = analysis.detection_result
    sev = analysis.severity
    has_fire = det_res.fire_count > 0
    has_smoke = det_res.smoke_count > 0
    final_label = format_final_result_label(has_fire, has_smoke)

    db.insert_incident(
        file_name=analysis.filename,
        file_type="image",
        detection_type=final_label,
        confidence=det_res.max_confidence,
        fire_coverage=sev.fire_coverage_pct,
        smoke_coverage=sev.smoke_coverage_pct,
        visual_severity=sev.visual_severity,
        alert_level=analysis.alert_event.alert_level,
        output_path=str(analysis.output_path),
        original_path=str(analysis.original_path),
        session_id=st.session_state.session_id,
    )

    st.session_state.live_audit_result = {
        "status": "completed",
        "media_type": "image",
        "file_name": analysis.filename,
        "original_rgb": analysis.original_rgb,
        "analyzed_rgb": analysis.annotated_rgb,
        "has_fire": has_fire,
        "has_smoke": has_smoke,
        "fire_pct": sev.fire_coverage_pct,
        "smoke_pct": sev.smoke_coverage_pct,
        "combined_pct": sev.combined_coverage_pct,
        "confidence": det_res.max_confidence,
        "fire_confidence": det_res.fire_max_confidence,
        "smoke_confidence": det_res.smoke_max_confidence,
        "visual_severity": sev.visual_severity,
        "final_result": final_label,
    }


def _run_video_analysis(file_bytes: bytes, filename: str) -> None:
    st.session_state.stop_video_audit = False
    detector = get_detector(str(config.MODEL_PATH))
    _verify_model_classes(detector)

    saved_path = save_uploaded_video_bytes(file_bytes, filename)
    meta = validate_video_file(saved_path)
    width = int(meta["width"])
    height = int(meta["height"])
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])

    # Stream smoothly on CPU while annotating all frames in output MP4
    sample_step = max(1, int(round(fps / 5.0))) if fps > 8 else 1
    uid = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    stem = Path(filename).stem
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_out = config.OUTPUTS_DIR / f"{stem}_{uid}_raw.mp4"
    final_out = config.OUTPUTS_DIR / f"{stem}_{uid}_analyzed.mp4"

    writer = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise InvalidVideoError("Failed to initialize output MP4 encoder.")

    progress_bar = st.progress(0, text="Processing video frames...")
    media_col1, media_col2 = st.columns(2)
    with media_col1:
        orig_slot = st.empty()
    with media_col2:
        pred_slot = st.empty()
    gauge_col1, gauge_col2 = st.columns(2)
    with gauge_col1:
        fire_gauge_slot = st.empty()
    with gauge_col2:
        smoke_gauge_slot = st.empty()
    status_slot = st.empty()

    fire_covs: list[float] = []
    smoke_covs: list[float] = []
    comb_covs: list[float] = []
    all_confs: list[float] = []
    fire_confs: list[float] = []
    smoke_confs: list[float] = []
    last_dets: list[Detection] = []
    last_orig_rgb = None
    last_pred_rgb = None
    any_fire = False
    any_smoke = False
    frame_idx = 0

    cap = cv2.VideoCapture(str(saved_path))
    try:
        while True:
            if st.session_state.get("stop_video_audit", False):
                break
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                res = detector.predict_frame(
                    frame_bgr,
                    conf_threshold=float(config.DEFAULT_CONFIDENCE_THRESHOLD),
                    iou_threshold=float(config.DEFAULT_IOU_THRESHOLD),
                )
                last_dets = res.detections
                sev = estimate_visual_severity(last_dets, width, height)

                fire_covs.append(sev.fire_coverage_pct)
                smoke_covs.append(sev.smoke_coverage_pct)
                comb_covs.append(sev.combined_coverage_pct)
                if res.max_confidence > 0.0:
                    all_confs.append(res.max_confidence)
                if res.fire_max_confidence > 0.0:
                    fire_confs.append(res.fire_max_confidence)
                if res.smoke_max_confidence > 0.0:
                    smoke_confs.append(res.smoke_max_confidence)
                if res.fire_count > 0:
                    any_fire = True
                if res.smoke_count > 0:
                    any_smoke = True

                annotated = draw_detections(frame_bgr, last_dets)
                last_orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                last_pred_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                orig_slot.image(last_orig_rgb, caption="Original video", alt="Original video frame", width="stretch")
                pred_slot.image(last_pred_rgb, caption="AI analyzed video", alt="Analyzed video frame", width="stretch")

                with fire_gauge_slot.container():
                    render_circular_gauge("Fire affected region", sev.fire_coverage_pct, "#EF4444", "Current frame")
                with smoke_gauge_slot.container():
                    render_circular_gauge("Smoke affected region", sev.smoke_coverage_pct, "#F97316", "Current frame")

                cur_has_det = res.fire_count > 0 or res.smoke_count > 0
                cur_label = format_final_result_label(res.fire_count > 0, res.smoke_count > 0)
                cur_conf = format_confidence_display(res.max_confidence, cur_has_det)
                status_slot.info(
                    f"Frame {frame_idx + 1}/{total_frames} — Current detection: {cur_label} "
                    f"(Confidence: {cur_conf})"
                )
            else:
                annotated = draw_detections(frame_bgr, last_dets)

            writer.write(annotated)
            frame_idx += 1
            progress_bar.progress(
                min(100, int((frame_idx / max(1, total_frames)) * 100)),
                text=f"Processed frame {frame_idx} of {total_frames}...",
            )
    finally:
        cap.release()
        writer.release()

    output_video_path = _transcode_to_h264_if_available(raw_out, final_out)
    progress_bar.progress(100, text="Video analysis complete.")

    max_fire = round(max(fire_covs, default=0.0), 2)
    avg_fire = round(sum(fire_covs) / len(fire_covs), 2) if fire_covs else 0.0
    max_smoke = round(max(smoke_covs, default=0.0), 2)
    avg_smoke = round(sum(smoke_covs) / len(smoke_covs), 2) if smoke_covs else 0.0
    max_comb = round(max(comb_covs, default=0.0), 2)
    peak_conf = round(max(all_confs, default=0.0), 4)
    peak_fire_conf = round(max(fire_confs, default=0.0), 4)
    peak_smoke_conf = round(max(smoke_confs, default=0.0), 4)
    overall_sev = classify_visual_severity(max_comb)
    final_label = format_final_result_label(any_fire, any_smoke)

    alert_lvl = "GREEN"
    if any_fire and max_comb >= 20.0:
        alert_lvl = "RED"
    elif any_fire:
        alert_lvl = "ORANGE"
    elif any_smoke:
        alert_lvl = "YELLOW"

    db.insert_incident(
        file_name=filename,
        file_type="video",
        detection_type=final_label,
        confidence=peak_conf,
        fire_coverage=max_fire,
        smoke_coverage=max_smoke,
        visual_severity=overall_sev,
        alert_level=alert_lvl,
        output_path=str(output_video_path),
        original_path=str(saved_path),
        session_id=st.session_state.session_id,
        notes=f"Avg Fire: {avg_fire:.2f}%, Avg Smoke: {avg_smoke:.2f}% across {frame_idx} frames.",
    )

    st.session_state.live_audit_result = {
        "status": "completed",
        "media_type": "video",
        "file_name": filename,
        "original_video_path": str(saved_path),
        "analyzed_video_path": str(output_video_path),
        "last_orig_rgb": last_orig_rgb,
        "last_pred_rgb": last_pred_rgb,
        "has_fire": any_fire,
        "has_smoke": any_smoke,
        "fire_pct": max_fire,
        "avg_fire_pct": avg_fire,
        "smoke_pct": max_smoke,
        "avg_smoke_pct": avg_smoke,
        "combined_pct": max_comb,
        "confidence": peak_conf,
        "fire_confidence": peak_fire_conf,
        "smoke_confidence": peak_smoke_conf,
        "visual_severity": overall_sev,
        "final_result": final_label,
        "frames_processed": frame_idx,
    }


if uploaded_file is not None and analyze_clicked:
    ext = Path(uploaded_file.name).suffix.lower()
    try:
        if ext in IMAGE_EXTS:
            _run_image_analysis(uploaded_file.getvalue(), uploaded_file.name)
        elif ext in VIDEO_EXTS:
            _run_video_analysis(uploaded_file.getvalue(), uploaded_file.name)
        else:
            st.session_state.live_audit_result = {
                "status": "failed",
                "file_name": uploaded_file.name,
                "error_message": "Unsupported file format. Please upload JPG, JPEG, PNG, or MP4.",
                "final_result": "Analysis Failed",
                "visual_severity": "Analysis Failed",
            }
    except (ModelNotFoundError, InvalidImageError, InvalidVideoError, ValueError) as exc:
        st.session_state.live_audit_result = {
            "status": "failed",
            "file_name": uploaded_file.name,
            "error_message": str(exc),
            "final_result": "Analysis Failed",
            "visual_severity": "Analysis Failed",
        }
    except Exception as exc:
        st.session_state.live_audit_result = {
            "status": "failed",
            "file_name": uploaded_file.name,
            "error_message": f"Could not complete detection: {exc}",
            "final_result": "Analysis Failed",
            "visual_severity": "Analysis Failed",
        }

# Render latest completed or failed analysis result cleanly
res = st.session_state.live_audit_result
if res is not None:
    if res.get("status") == "failed":
        st.error(res.get("error_message", "Analysis could not be completed."), title="Analysis Failed")
        render_final_result_banner("Analysis Failed", "Analysis Failed", 0.0)
    else:
        if res["media_type"] == "image":
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.image(res["original_rgb"], caption="Original image", alt="Original uploaded image", width="stretch")
            with m_col2:
                st.image(
                    res["analyzed_rgb"],
                    caption="AI analyzed image (Red: Fire | Orange: Smoke)",
                    alt="AI analyzed image with fire and smoke bounding boxes",
                    width="stretch",
                )

            g1, g2 = st.columns(2)
            with g1:
                render_circular_gauge(
                    "Fire affected region",
                    res["fire_pct"],
                    "#EF4444",
                    f"Combined visible region: {res.get('combined_pct', 0.0):.2f}%",
                )
            with g2:
                render_circular_gauge(
                    "Smoke affected region",
                    res["smoke_pct"],
                    "#F97316",
                    "Approximate 2D visible area",
                )

        elif res["media_type"] == "video":
            v_col1, v_col2 = st.columns(2)
            orig_vp = Path(res["original_video_path"])
            pred_vp = Path(res["analyzed_video_path"])
            with v_col1:
                if orig_vp.is_file():
                    st.video(orig_vp.read_bytes(), alt="Original uploaded video")
                st.caption("Original video")
            with v_col2:
                if pred_vp.is_file():
                    st.video(pred_vp.read_bytes(), alt="AI analyzed video with bounding boxes")
                elif res.get("last_pred_rgb") is not None:
                    st.image(res["last_pred_rgb"], caption="Final analyzed frame", alt="Final analyzed frame", width="stretch")
                st.caption("AI analyzed video")

            g1, g2 = st.columns(2)
            with g1:
                render_circular_gauge(
                    "Fire affected region (Max)",
                    res["fire_pct"],
                    "#EF4444",
                    f"Average across frames: {res['avg_fire_pct']:.2f}% | Combined max: {res.get('combined_pct', 0.0):.2f}%",
                )
            with g2:
                render_circular_gauge(
                    "Smoke affected region (Max)",
                    res["smoke_pct"],
                    "#F97316",
                    f"Average across frames: {res['avg_smoke_pct']:.2f}%",
                )

        has_any_det = bool(res["has_fire"] or res["has_smoke"])
        conf_display = format_confidence_display(res["confidence"], has_any_det)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Fire detected", "Yes" if res["has_fire"] else "No", border=True)
        s2.metric("Smoke detected", "Yes" if res["has_smoke"] else "No", border=True)
        s3.metric("Detection confidence", conf_display, border=True)
        sev_txt = "No detection" if res["visual_severity"].lower() == "no detection" else res["visual_severity"]
        s4.metric("Visual severity", sev_txt, border=True)

        render_final_result_banner(
            res["final_result"],
            res["visual_severity"],
            res["confidence"],
            fire_confidence=res.get("fire_confidence"),
            smoke_confidence=res.get("smoke_confidence"),
        )

        st.caption(
            "Note: Fire and smoke affected region percentages are approximate 2D bounding-box visual "
            "measurements relative to the frame area, not actual physical damage, heat, or safety risk."
        )
