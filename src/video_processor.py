"""Phase 6 — Memory-Efficient Video-Based Fire and Smoke Detection Processor.

Streams prerecorded videos (MP4, AVI, MOV) frame-by-frame using OpenCV without
loading all frames into RAM, applies temporal false-alarm filtering, computes
per-frame visual severity, performs temporal growth analysis, logs alert events,
and exports a playable annotated video plus peak detection snapshot.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2

import config
from src.alert_manager import ALERT_PRIORITY, AlertEvent, AlertManager
from src.detector import (
    Detection,
    FireSmokeDetector,
    compute_detection_status,
    draw_detections,
)
from src.growth_analysis import (
    FrameCoverageMetric,
    GrowthAnalysisResult,
    analyze_fire_growth,
    export_growth_chart_png,
)
from src.severity import SeverityAssessment, estimate_visual_severity
from src.temporal_filter import TemporalFalseAlarmFilter


class InvalidVideoError(ValueError):
    """Raised when a video file is empty, unsupported, corrupted, or unreadable."""


@dataclass
class VideoTimelineEvent:
    """Significant detection or alert event timestamp in a video."""

    timestamp_sec: float
    frame_index: int
    event_type: str
    detection_status: str
    confidence: float
    fire_coverage_pct: float
    smoke_coverage_pct: float
    alert_level: str


@dataclass
class VideoAnalysisResult:
    """Complete analytics, output paths, and timeline for a processed video."""

    filename: str
    original_path: Path
    output_video_path: Path
    snapshot_path: Path | None
    growth_chart_path: Path | None
    duration_sec: float
    source_fps: float
    total_frames: int
    processed_frames: int
    frame_sample_rate: int
    processing_time_sec: float
    processing_fps: float
    overall_status: str
    max_confidence: float
    avg_confidence: float
    first_smoke_timestamp_sec: float | None
    first_fire_timestamp_sec: float | None
    peak_coverage_timestamp_sec: float | None
    peak_severity: SeverityAssessment
    growth_analysis: GrowthAnalysisResult
    raw_detections_total: int
    suppressed_detections_total: int
    peak_fire_confidence: float = 0.0
    peak_smoke_confidence: float = 0.0
    alert_timeline: list[AlertEvent] = field(default_factory=list)
    highest_alert_event: AlertEvent | None = None
    timeline_events: list[VideoTimelineEvent] = field(default_factory=list)


def validate_video_file(
    video_path: Path,
    max_mb: int = config.MAX_UPLOAD_VIDEO_MB,
) -> dict[str, float | int]:
    """Verify video extension, size, and OpenCV VideoCapture readability."""
    if not video_path.is_file():
        raise InvalidVideoError(f"Video file does not exist: {video_path}")
    ext = video_path.suffix.lower()
    if ext not in config.SUPPORTED_VIDEO_EXTENSIONS:
        allowed = ", ".join(config.SUPPORTED_VIDEO_EXTENSIONS)
        raise InvalidVideoError(
            f"Unsupported video format '{ext or 'unknown'}'. Allowed formats: {allowed}."
        )
    file_size = video_path.stat().st_size
    if file_size == 0:
        raise InvalidVideoError("Uploaded video file is empty (0 bytes).")
    size_mb = file_size / (1024.0 * 1024.0)
    if size_mb > max_mb:
        raise InvalidVideoError(f"Video size ({size_mb:.2f} MB) exceeds {max_mb} MB limit.")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise InvalidVideoError(
                f"Failed to open video '{video_path.name}'. Codec may be unsupported or file corrupted."
            )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        if not ok or width <= 0 or height <= 0:
            raise InvalidVideoError("Video stream contains no readable frames or invalid dimensions.")
        safe_fps = fps if fps > 0.1 else 25.0
        return {
            "width": width,
            "height": height,
            "fps": safe_fps,
            "total_frames": max(1, total_frames),
        }
    finally:
        cap.release()


def _transcode_to_h264_if_available(input_mp4: Path, target_mp4: Path) -> Path:
    """Transcode OpenCV mp4v video to browser-compatible H.264 if ffmpeg is present."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return input_mp4

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_mp4),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target_mp4),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
        if completed.returncode == 0 and target_mp4.is_file() and target_mp4.stat().st_size > 0:
            input_mp4.unlink(missing_ok=True)
            return target_mp4
    except (OSError, subprocess.SubprocessError):
        pass
    return input_mp4


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return cleaned[:48] or "video"


def save_uploaded_video_bytes(video_bytes: bytes, filename: str, uploads_dir: Path | None = None) -> Path:
    """Persist uploaded video bytes cleanly under uploads/ and validate before processing."""
    if not filename:
        raise InvalidVideoError("No filename provided for video upload.")
    ext = Path(filename).suffix.lower()
    if ext not in config.SUPPORTED_VIDEO_EXTENSIONS:
        allowed = ", ".join(config.SUPPORTED_VIDEO_EXTENSIONS)
        raise InvalidVideoError(
            f"Unsupported video format '{ext or 'unknown'}'. Allowed formats: {allowed}."
        )
    if not video_bytes:
        raise InvalidVideoError("Uploaded video file is empty (0 bytes).")

    up_dir = uploads_dir or config.UPLOADS_DIR
    up_dir.mkdir(parents=True, exist_ok=True)
    uid = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    target = up_dir / f"{_safe_stem(filename)}_{uid}{ext}"
    target.write_bytes(video_bytes)
    validate_video_file(target)
    return target


def process_video_file(
    video_path: Path,
    detector: FireSmokeDetector,
    conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
    frame_sample_rate: int = config.DEFAULT_FRAME_SAMPLE_RATE,
    temporal_enabled: bool = True,
    temporal_window: int = config.DEFAULT_TEMPORAL_WINDOW,
    min_consecutive_frames: int = config.DEFAULT_MIN_CONSECUTIVE_FRAMES,
    min_box_area_ratio: float = config.DEFAULT_MIN_BOX_AREA_RATIO,
    alert_cooldown_sec: float = float(config.ALERT_THRESHOLDS["cooldown_seconds"]),
    outputs_dir: Path | None = None,
    progress_callback: Callable[[float, int, int], None] | None = None,
    source_filename: str | None = None,
) -> VideoAnalysisResult:
    """Process a prerecorded video frame-by-frame with temporal filtering & growth tracking."""
    meta = validate_video_file(video_path)
    width = int(meta["width"])
    height = int(meta["height"])
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])
    sample_rate = max(1, int(frame_sample_rate))
    display_name = source_filename or video_path.name

    out_dir = outputs_dir or config.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    uid = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    stem = _safe_stem(display_name)
    raw_out_mp4 = out_dir / f"{stem}_{uid}_raw_annotated.mp4"
    final_out_mp4 = out_dir / f"{stem}_{uid}_detected.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_out_mp4), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise InvalidVideoError("OpenCV VideoWriter failed to initialize output MP4 encoder.")

    temporal_filter = TemporalFalseAlarmFilter(
        window_size=temporal_window,
        min_consecutive_frames=min_consecutive_frames,
        fire_conf_threshold=conf_threshold,
        smoke_conf_threshold=conf_threshold,
        min_box_area_ratio=min_box_area_ratio,
        enabled=temporal_enabled,
    )
    alert_manager = AlertManager({"cooldown_seconds": alert_cooldown_sec})

    frame_metrics: list[FrameCoverageMetric] = []
    timeline_events: list[VideoTimelineEvent] = []
    alert_timeline: list[AlertEvent] = []

    first_smoke_ts: float | None = None
    first_fire_ts: float | None = None
    peak_combined_cov = -1.0
    peak_frame_bgr = None
    peak_ts: float | None = None
    peak_severity = estimate_visual_severity([], width, height)

    last_confirmed_detections: list[Detection] = []
    all_confidences: list[float] = []
    all_fire_confidences: list[float] = []
    all_smoke_confidences: list[float] = []
    any_fire_seen = False
    any_smoke_seen = False

    cap = cv2.VideoCapture(str(video_path))
    start_clock = time.perf_counter()
    frame_idx = 0
    inferred_frames = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            timestamp_sec = frame_idx / fps
            if frame_idx % sample_rate == 0:
                inferred_frames += 1
                raw_res = detector.predict_frame(
                    frame_bgr,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
                t_out = temporal_filter.update(raw_res.detections, width, height)
                last_confirmed_detections = t_out.confirmed_detections

                sev = estimate_visual_severity(last_confirmed_detections, width, height)
                fire_confs = [d.confidence for d in last_confirmed_detections if d.class_name == "fire"]
                smoke_confs = [d.confidence for d in last_confirmed_detections if d.class_name == "smoke"]
                all_confidences.extend(d.confidence for d in last_confirmed_detections)
                all_fire_confidences.extend(fire_confs)
                all_smoke_confidences.extend(smoke_confs)

                if fire_confs:
                    any_fire_seen = True
                    if first_fire_ts is None:
                        first_fire_ts = round(timestamp_sec, 2)
                        timeline_events.append(
                            VideoTimelineEvent(
                                timestamp_sec=first_fire_ts,
                                frame_index=frame_idx,
                                event_type="First Fire Detection",
                                detection_status=compute_detection_status(last_confirmed_detections),
                                confidence=max(fire_confs),
                                fire_coverage_pct=sev.fire_coverage_pct,
                                smoke_coverage_pct=sev.smoke_coverage_pct,
                                alert_level="ORANGE",
                            )
                        )

                if smoke_confs:
                    any_smoke_seen = True
                    if first_smoke_ts is None:
                        first_smoke_ts = round(timestamp_sec, 2)
                        timeline_events.append(
                            VideoTimelineEvent(
                                timestamp_sec=first_smoke_ts,
                                frame_index=frame_idx,
                                event_type="First Smoke Detection",
                                detection_status=compute_detection_status(last_confirmed_detections),
                                confidence=max(smoke_confs),
                                fire_coverage_pct=sev.fire_coverage_pct,
                                smoke_coverage_pct=sev.smoke_coverage_pct,
                                alert_level="YELLOW",
                            )
                        )

                frame_metrics.append(
                    FrameCoverageMetric(
                        frame_index=frame_idx,
                        timestamp_sec=round(timestamp_sec, 3),
                        fire_coverage_pct=sev.fire_coverage_pct,
                        smoke_coverage_pct=sev.smoke_coverage_pct,
                        fire_confidence=max(fire_confs, default=0.0),
                        smoke_confidence=max(smoke_confs, default=0.0),
                    )
                )

                annotated_frame = draw_detections(frame_bgr, last_confirmed_detections)
                if sev.combined_coverage_pct > peak_combined_cov:
                    peak_combined_cov = sev.combined_coverage_pct
                    peak_frame_bgr = annotated_frame.copy()
                    peak_ts = round(timestamp_sec, 2)
                    peak_severity = sev

                if last_confirmed_detections:
                    alert_ev = alert_manager.evaluate(
                        source_filename=display_name,
                        has_fire=bool(fire_confs),
                        has_smoke=bool(smoke_confs),
                        confidence=max((d.confidence for d in last_confirmed_detections), default=0.0),
                        fire_coverage_pct=sev.fire_coverage_pct,
                        smoke_coverage_pct=sev.smoke_coverage_pct,
                        visual_severity=sev.visual_severity,
                        video_timestamp_sec=timestamp_sec,
                        event_clock_sec=timestamp_sec,
                        fire_persistent=t_out.fire_persistent,
                        smoke_persistent=t_out.smoke_persistent,
                        suppress_duplicates=True,
                    )
                    if not alert_ev.suppressed:
                        alert_timeline.append(alert_ev)
            else:
                annotated_frame = draw_detections(frame_bgr, last_confirmed_detections)

            writer.write(annotated_frame)
            frame_idx += 1

            if progress_callback is not None and (frame_idx % 5 == 0 or frame_idx >= total_frames):
                progress_callback(min(1.0, frame_idx / max(1, total_frames)), frame_idx, total_frames)
    finally:
        cap.release()
        writer.release()

    elapsed_sec = max(1e-6, time.perf_counter() - start_clock)
    processing_fps = frame_idx / elapsed_sec

    output_video_path = _transcode_to_h264_if_available(raw_out_mp4, final_out_mp4)

    snapshot_path: Path | None = None
    if peak_frame_bgr is not None:
        snapshot_path = out_dir / f"{stem}_{uid}_snapshot.png"
        cv2.imwrite(str(snapshot_path), peak_frame_bgr)
        for ev in alert_timeline:
            ev.snapshot_path = str(snapshot_path)

    growth_result = analyze_fire_growth(frame_metrics)
    growth_chart_path: Path | None = None
    if frame_metrics:
        growth_chart_path = export_growth_chart_png(
            growth_result,
            out_dir / f"{stem}_{uid}_growth.png",
        )

    # Re-evaluate peak / overall alert considering final growth trend
    if any_fire_seen or any_smoke_seen:
        final_alert = alert_manager.evaluate(
            source_filename=display_name,
            has_fire=any_fire_seen,
            has_smoke=any_smoke_seen,
            confidence=max(all_confidences, default=0.0),
            fire_coverage_pct=peak_severity.fire_coverage_pct,
            smoke_coverage_pct=peak_severity.smoke_coverage_pct,
            visual_severity=peak_severity.visual_severity,
            snapshot_path=str(snapshot_path) if snapshot_path else None,
            video_timestamp_sec=peak_ts,
            event_clock_sec=(frame_idx / fps) + alert_cooldown_sec + 1.0,
            fire_persistent=any_fire_seen,
            smoke_persistent=any_smoke_seen,
            fire_growth_rate_pct_per_sec=growth_result.fire_growth_rate_pct_per_sec,
            growth_classification=growth_result.growth_classification,
            suppress_duplicates=False,
        )
        if not alert_timeline or ALERT_PRIORITY[final_alert.alert_level] > max(
            ALERT_PRIORITY[a.alert_level] for a in alert_timeline
        ):
            alert_timeline.append(final_alert)
        highest_alert = max(alert_timeline, key=lambda a: ALERT_PRIORITY[a.alert_level])
    else:
        highest_alert = alert_manager.evaluate(
            source_filename=display_name,
            has_fire=False,
            has_smoke=False,
            confidence=0.0,
            fire_coverage_pct=0.0,
            smoke_coverage_pct=0.0,
            visual_severity="No Detection",
            snapshot_path=str(snapshot_path) if snapshot_path else None,
            video_timestamp_sec=0.0,
            suppress_duplicates=False,
        )
        alert_timeline.append(highest_alert)

    if any_fire_seen and any_smoke_seen:
        overall_status = "Fire and smoke detected."
    elif any_fire_seen:
        overall_status = "Fire detected."
    elif any_smoke_seen:
        overall_status = "Smoke detected."
    else:
        overall_status = "No fire or smoke detected."

    return VideoAnalysisResult(
        filename=display_name,
        original_path=video_path,
        output_video_path=output_video_path,
        snapshot_path=snapshot_path,
        growth_chart_path=growth_chart_path,
        duration_sec=round(frame_idx / fps, 2),
        source_fps=round(fps, 2),
        total_frames=frame_idx,
        processed_frames=inferred_frames,
        frame_sample_rate=sample_rate,
        processing_time_sec=round(elapsed_sec, 2),
        processing_fps=round(processing_fps, 2),
        overall_status=overall_status,
        max_confidence=round(max(all_confidences, default=0.0), 4),
        avg_confidence=round(
            sum(all_confidences) / len(all_confidences) if all_confidences else 0.0, 4
        ),
        first_smoke_timestamp_sec=first_smoke_ts,
        first_fire_timestamp_sec=first_fire_ts,
        peak_coverage_timestamp_sec=peak_ts,
        peak_severity=peak_severity,
        growth_analysis=growth_result,
        raw_detections_total=temporal_filter.total_raw_detections,
        suppressed_detections_total=temporal_filter.total_suppressed_detections,
        peak_fire_confidence=round(max(all_fire_confidences, default=0.0), 4),
        peak_smoke_confidence=round(max(all_smoke_confidences, default=0.0), 4),
        alert_timeline=alert_timeline,
        highest_alert_event=highest_alert,
        timeline_events=timeline_events,
    )
