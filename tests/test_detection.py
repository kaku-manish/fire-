"""Phase 15 — Automated Detection, Video Processing, Edge-Case & Streamlit UI Tests.

Uses controlled mock YOLO detector outputs ONLY inside software unit/integration
tests to verify downstream detection, filtering, image/video pipelines, and
error handling deterministically without fabricating real model accuracy.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402
from src.detector import (  # noqa: E402
    Detection,
    DetectionResult,
    FireSmokeDetector,
    ModelNotFoundError,
    summarize_detections,
)
from src.image_processor import InvalidImageError, process_image_bytes  # noqa: E402
from src.video_processor import InvalidVideoError, process_video_file  # noqa: E402


class MockUnitTestDetector(FireSmokeDetector):
    """Deterministic test double returning pre-scripted Detection objects per frame."""

    def __init__(self, rule_fn: Callable[[np.ndarray, int], list[Detection]]) -> None:
        super().__init__(model_path=Path("mock_unit_test_weights.pt"))
        self._rule_fn = rule_fn
        self._call_idx = 0

    def ensure_model_exists(self) -> Path:
        return self.model_path

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
        imgsz: int = config.DEFAULT_IMAGE_SIZE,
    ) -> DetectionResult:
        h, w = frame_bgr.shape[:2]
        dets = [d for d in self._rule_fn(frame_bgr, self._call_idx) if d.confidence >= conf_threshold]
        self._call_idx += 1
        return summarize_detections(dets, w, h, inference_time_ms=4.2, model_path="mock_unit_test")


def _encode_png_bytes(width: int = 200, height: int = 200, bgr: tuple[int, int, int] = (40, 40, 40)) -> bytes:
    img = np.full((height, width, 3), bgr, dtype=np.uint8)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _make_box(cls_name: str, conf: float, x1: float, y1: float, x2: float, y2: float, w: int = 200, h: int = 200) -> Detection:
    cls_id = 1 if cls_name == "fire" else 0
    return Detection(
        class_id=cls_id,
        class_name=cls_name,
        confidence=conf,
        bbox_xyxy=(x1, y1, x2, y2),
        bbox_normalized=(x1 / w, y1 / h, x2 / w, y2 / h),
        area_px=(x2 - x1) * (y2 - y1),
    )


def _write_test_video(path: Path, frame_count: int = 12, fps: float = 10.0, size: tuple[int, int] = (160, 120)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for i in range(frame_count):
        frame = np.full((size[1], size[0], 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(frame, f"F{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        writer.write(frame)
    writer.release()
    return path


def test_scenario_01_image_fire_only(tmp_path: Path) -> None:
    """Scenario 1: Image containing fire only draws red box and returns 'Fire detected.'"""
    det = MockUnitTestDetector(lambda _f, _i: [_make_box("fire", 0.88, 40, 40, 100, 100)])
    res = process_image_bytes(
        _encode_png_bytes(), "fire_only.png", det, uploads_dir=tmp_path / "up", outputs_dir=tmp_path / "out"
    )
    assert res.detection_result.status == "Fire detected."
    assert res.detection_result.fire_count == 1
    assert res.detection_result.smoke_count == 0
    assert res.alert_event.alert_level == "ORANGE"
    # Verify red border pixel drawn at (40, 40) in RGB output -> (255, 0, 0)
    assert tuple(res.annotated_rgb[40, 40]) == (255, 0, 0)


def test_scenario_02_image_smoke_only(tmp_path: Path) -> None:
    """Scenario 2: Image containing smoke only draws orange box and triggers YELLOW warning."""
    det = MockUnitTestDetector(lambda _f, _i: [_make_box("smoke", 0.76, 30, 30, 110, 90)])
    res = process_image_bytes(
        _encode_png_bytes(), "smoke_only.png", det, uploads_dir=tmp_path / "up", outputs_dir=tmp_path / "out"
    )
    assert res.detection_result.status == "Smoke detected."
    assert res.detection_result.smoke_count == 1
    assert res.alert_event.alert_level == "YELLOW"
    # Orange BGR (0, 140, 255) -> RGB (255, 140, 0)
    assert tuple(res.annotated_rgb[30, 30]) == (255, 140, 0)


def test_scenario_03_image_fire_and_smoke(tmp_path: Path) -> None:
    """Scenario 3: Image containing both fire and smoke returns 'Fire and smoke detected.'"""
    det = MockUnitTestDetector(
        lambda _f, _i: [
            _make_box("smoke", 0.72, 20, 10, 120, 80),
            _make_box("fire", 0.91, 50, 80, 110, 140),
        ]
    )
    res = process_image_bytes(
        _encode_png_bytes(), "both.png", det, uploads_dir=tmp_path / "up", outputs_dir=tmp_path / "out"
    )
    assert res.detection_result.status == "Fire and smoke detected."
    assert res.detection_result.fire_count == 1
    assert res.detection_result.smoke_count == 1


def test_scenario_04_normal_image(tmp_path: Path) -> None:
    """Scenario 4: Normal environment returns negative status, GREEN alert, and safety caveat."""
    det = MockUnitTestDetector(lambda _f, _i: [])
    res = process_image_bytes(
        _encode_png_bytes(), "normal.png", det, uploads_dir=tmp_path / "up", outputs_dir=tmp_path / "out"
    )
    assert res.detection_result.status == "No fire or smoke detected."
    assert res.severity.visual_severity == "No Detection"
    assert res.alert_event.alert_level == "GREEN"
    assert "NOT proof" in res.detection_result.negative_safety_note


def test_scenario_05_foggy_environment_filtered(tmp_path: Path) -> None:
    """Scenario 5: Low-confidence fog detection suppressed by calibrated confidence threshold."""
    det = MockUnitTestDetector(lambda _f, _i: [_make_box("smoke", 0.22, 10, 10, 180, 180)])
    res = process_image_bytes(
        _encode_png_bytes(),
        "foggy_road.jpg",
        det,
        conf_threshold=0.35,
        uploads_dir=tmp_path / "up",
        outputs_dir=tmp_path / "out",
    )
    assert res.detection_result.status == "No fire or smoke detected."
    assert res.alert_event.alert_level == "GREEN"


def test_scenario_06_sunset_image_filtered(tmp_path: Path) -> None:
    """Scenario 6: Sub-threshold sunset glow rejected cleanly."""
    det = MockUnitTestDetector(lambda _f, _i: [_make_box("fire", 0.28, 0, 20, 190, 90)])
    res = process_image_bytes(
        _encode_png_bytes(bgr=(40, 110, 235)),
        "sunset.png",
        det,
        conf_threshold=0.35,
        uploads_dir=tmp_path / "up",
        outputs_dir=tmp_path / "out",
    )
    assert res.detection_result.status == "No fire or smoke detected."


def test_scenario_07_artificial_lighting_tiny_box_filtered(tmp_path: Path) -> None:
    """Scenario 7: Tiny artificial bulb highlight filtered by minimum area constraint."""
    det = MockUnitTestDetector(lambda _f, _i: [_make_box("fire", 0.65, 98, 98, 100, 100)])
    res = process_image_bytes(
        _encode_png_bytes(),
        "lamp.png",
        det,
        conf_threshold=0.35,
        min_box_area_ratio=0.0005,
        uploads_dir=tmp_path / "up",
        outputs_dir=tmp_path / "out",
    )
    assert res.detection_result.status == "No fire or smoke detected."


def test_scenario_08_low_light_fire_image(tmp_path: Path) -> None:
    """Scenario 8: Genuine low-light fire detected accurately."""
    det = MockUnitTestDetector(lambda _f, _i: [_make_box("fire", 0.89, 70, 70, 120, 130)])
    res = process_image_bytes(
        _encode_png_bytes(bgr=(10, 10, 10)),
        "night_fire.png",
        det,
        uploads_dir=tmp_path / "up",
        outputs_dir=tmp_path / "out",
    )
    assert res.detection_result.status == "Fire detected."
    assert res.severity.fire_coverage_pct == pytest.approx(7.5, abs=0.1)


def test_scenario_09_small_visible_fire_region(tmp_path: Path) -> None:
    """Scenario 9: Small visible fire region (<5% coverage) classified as Low severity."""
    # 25x25 box in 200x200 frame = 625 / 40000 = 1.56%
    det = MockUnitTestDetector(lambda _f, _i: [_make_box("fire", 0.82, 80, 80, 105, 105)])
    res = process_image_bytes(
        _encode_png_bytes(), "small_fire.png", det, uploads_dir=tmp_path / "up", outputs_dir=tmp_path / "out"
    )
    assert res.severity.fire_coverage_pct == pytest.approx(1.56, abs=0.05)
    assert res.severity.visual_severity == "Low"


def test_scenario_10_large_visible_fire_region(tmp_path: Path) -> None:
    """Scenario 10: Large visible fire region (>=20% coverage) classified High & RED alert."""
    # 120x100 box in 200x200 = 12000 / 40000 = 30%
    det = MockUnitTestDetector(lambda _f, _i: [_make_box("fire", 0.94, 40, 40, 160, 140)])
    res = process_image_bytes(
        _encode_png_bytes(), "large_fire.png", det, uploads_dir=tmp_path / "up", outputs_dir=tmp_path / "out"
    )
    assert res.severity.fire_coverage_pct == pytest.approx(30.0, abs=0.1)
    assert res.severity.visual_severity == "High"
    assert res.alert_event.alert_level == "RED"


def test_scenario_11_video_increasing_fire_coverage(tmp_path: Path) -> None:
    """Scenario 11: Video with growing fire region classifies growth as 'Increasing'."""
    vid_path = _write_test_video(tmp_path / "growing.mp4", frame_count=12, fps=4.0, size=(160, 120))

    def _growing_rule(_frame: np.ndarray, idx: int) -> list[Detection]:
        side = min(110, 20 + idx * 8)
        return [_make_box("fire", 0.85, 10, 10, 10 + side, 10 + side, w=160, h=120)]

    det = MockUnitTestDetector(_growing_rule)
    res = process_video_file(
        vid_path,
        det,
        temporal_enabled=True,
        temporal_window=4,
        min_consecutive_frames=2,
        outputs_dir=tmp_path / "vout",
    )
    assert res.overall_status == "Fire detected."
    assert res.output_video_path.is_file()
    assert res.growth_analysis.growth_classification == "Increasing"
    assert res.first_fire_timestamp_sec is not None


def test_scenario_12_video_no_fire_or_smoke(tmp_path: Path) -> None:
    """Scenario 12: Clean normal video produces no detections and GREEN alert."""
    vid_path = _write_test_video(tmp_path / "normal.mp4", frame_count=8, fps=5.0)
    # Inject a 1-frame transient false-positive spike on frame 2 to verify temporal filter suppresses it!
    det = MockUnitTestDetector(
        lambda _f, i: [_make_box("smoke", 0.60, 20, 20, 60, 60, w=160, h=120)] if i == 2 else []
    )
    res = process_video_file(
        vid_path,
        det,
        temporal_enabled=True,
        temporal_window=5,
        min_consecutive_frames=3,
        outputs_dir=tmp_path / "vout2",
    )
    assert res.overall_status == "No fire or smoke detected."
    assert res.suppressed_detections_total == 1
    assert res.highest_alert_event is not None
    assert res.highest_alert_event.alert_level == "GREEN"


def test_scenario_13_corrupted_image(tmp_path: Path) -> None:
    """Scenario 13: Corrupted image bytes raise InvalidImageError."""
    det = MockUnitTestDetector(lambda _f, _i: [])
    with pytest.raises(InvalidImageError, match="Corrupted"):
        process_image_bytes(b"not-a-valid-png", "broken.png", det, uploads_dir=tmp_path, outputs_dir=tmp_path)


def test_scenario_14_unsupported_file_format(tmp_path: Path) -> None:
    """Scenario 14: Unsupported image/video extensions raise validation errors."""
    det = MockUnitTestDetector(lambda _f, _i: [])
    with pytest.raises(InvalidImageError, match="Unsupported"):
        process_image_bytes(b"abc", "document.pdf", det, uploads_dir=tmp_path, outputs_dir=tmp_path)
    bad_vid = tmp_path / "clip.mkv"
    bad_vid.write_bytes(b"dummy")
    with pytest.raises(InvalidVideoError, match="Unsupported"):
        process_video_file(bad_vid, det, outputs_dir=tmp_path)


def test_scenario_15_missing_model_weights(tmp_path: Path) -> None:
    """Scenario 15: Attempting inference with missing best.pt raises ModelNotFoundError."""
    detector = FireSmokeDetector(model_path=tmp_path / "nonexistent_best.pt")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ModelNotFoundError, match="not found"):
        detector.predict_frame(frame)


def test_scenario_16_empty_upload(tmp_path: Path) -> None:
    """Scenario 16: Empty 0-byte image and video uploads are rejected cleanly."""
    det = MockUnitTestDetector(lambda _f, _i: [])
    with pytest.raises(InvalidImageError, match="empty"):
        process_image_bytes(b"", "empty.jpg", det, uploads_dir=tmp_path, outputs_dir=tmp_path)
    empty_vid = tmp_path / "empty.mp4"
    empty_vid.write_bytes(b"")
    with pytest.raises(InvalidVideoError, match="empty"):
        process_video_file(empty_vid, det, outputs_dir=tmp_path)


def test_scenario_20_video_processing_failure(tmp_path: Path) -> None:
    """Scenario 20: Corrupted/unreadable MP4 file raises InvalidVideoError cleanly."""
    det = MockUnitTestDetector(lambda _f, _i: [])
    corrupted_vid = tmp_path / "corrupt.mp4"
    corrupted_vid.write_bytes(b"corrupted-mp4-header-bytes")
    with pytest.raises(InvalidVideoError):
        process_video_file(corrupted_vid, det, outputs_dir=tmp_path)


def test_streamlit_pages_compile_and_assets() -> None:
    """Verify entrypoint, all 4 simplified page scripts, theme TOML, and CSS compile cleanly."""
    import py_compile
    import tomllib

    py_compile.compile(str(BASE_DIR / "app.py"), doraise=True)
    expected_pages = [
        "home.py",
        "live_audit.py",
        "detection_history.py",
        "about_project.py",
    ]
    for page_name in expected_pages:
        page_file = BASE_DIR / "app_pages" / page_name
        assert page_file.is_file()
        src_text = page_file.read_text(encoding="utf-8")
        assert "use_container_width" not in src_text
        py_compile.compile(str(page_file), doraise=True)

    theme_cfg = tomllib.loads((BASE_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert theme_cfg["theme"]["base"] == "dark"
    assert (BASE_DIR / "assets" / "styles.css").is_file()


def test_streamlit_all_pages_smoke() -> None:
    """Verify the main Streamlit app and all 4 pages render without uncaught exceptions."""
    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=15).run()
    assert not at.exception

    for page_rel in [
        "app_pages/live_audit.py",
        "app_pages/detection_history.py",
        "app_pages/about_project.py",
        "app_pages/home.py",
    ]:
        at.switch_page(page_rel).run()
        assert not at.exception, f"Exception on page {page_rel}: {at.exception}"


def test_per_class_confidence_and_native_percentage_helpers() -> None:
    """Verify per-class confidences, N/A formatting, and native non-SVG coverage cards."""
    from src.ui_helpers import format_confidence_display

    fire_box = _make_box("fire", 0.81, 10, 10, 80, 80)
    smoke_box = _make_box("smoke", 0.64, 90, 20, 160, 110)
    summary = summarize_detections([fire_box, smoke_box], 200, 200)
    assert summary.max_confidence == 0.81
    assert summary.fire_max_confidence == 0.81
    assert summary.smoke_max_confidence == 0.64

    empty_summary = summarize_detections([], 200, 200)
    assert empty_summary.max_confidence == 0.0
    assert empty_summary.fire_max_confidence == 0.0
    assert empty_summary.smoke_max_confidence == 0.0

    assert format_confidence_display(0.0, False) == "N/A"
    assert format_confidence_display(0.0, True) == "N/A"
    assert format_confidence_display(0.752, True) == "75.2%"

    probe_code = """
import streamlit as st
from src.ui_helpers import render_circular_gauge, render_final_result_banner
render_circular_gauge("Fire affected region", 25.0, "#EF4444", "Sub")
render_circular_gauge("Smoke affected region", 0.0, "#F97316", "Sub")
render_final_result_banner("No Fire or Smoke Detected", "No Detection", 0.0)
render_final_result_banner("Fire and Smoke Detected", "High", 0.81, fire_confidence=0.81, smoke_confidence=0.64)
render_final_result_banner("Analysis Failed", "Analysis Failed", 0.0)
"""
    at = AppTest.from_string(probe_code).run()
    assert not at.exception
    metric_vals = [(m.label, m.value) for m in at.metric]
    assert ("Fire affected region", "25.00%") in metric_vals
    assert ("Smoke affected region", "0.00%") in metric_vals
    assert [round(p.proto.value, 2) for p in at.progress] == [25, 0]
    html_bodies = [h.body for h in at.html]
    assert all("<svg" not in b.lower() for b in html_bodies)
    assert "Detection confidence: <strong>N/A</strong>" in html_bodies[0]
    assert "Fire: <strong>81.0%</strong>" in html_bodies[1]
    assert "Smoke: <strong>64.0%</strong>" in html_bodies[1]
    assert "Final result: Analysis Failed" in html_bodies[2]


def test_live_audit_clears_stale_result_and_renders_failure_state() -> None:
    """Verify Live Audit clears stale output on file change and shows Analysis Failed cleanly."""
    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=15)
    at.session_state["live_audit_result"] = {
        "status": "completed",
        "media_type": "image",
        "file_name": "stale.jpg",
        "original_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "analyzed_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "has_fire": True,
        "has_smoke": False,
        "fire_pct": 12.5,
        "smoke_pct": 0.0,
        "combined_pct": 12.5,
        "confidence": 0.88,
        "fire_confidence": 0.88,
        "smoke_confidence": 0.0,
        "visual_severity": "Medium",
        "final_result": "Fire Detected",
    }
    at.session_state["last_uploaded_file_sig"] = ("stale.jpg", 1234)
    at.run()
    at.switch_page("app_pages/live_audit.py").run()
    assert not at.exception
    # Because uploaded_file is None while last_uploaded_file_sig was ('stale.jpg', 1234), stale result must be cleared
    assert at.session_state["live_audit_result"] is None

    # Now verify explicit Analysis Failed state renders cleanly instead of 'No detection'
    at.session_state["live_audit_result"] = {
        "status": "failed",
        "file_name": "broken.jpg",
        "error_message": "Simulated decode failure",
        "final_result": "Analysis Failed",
        "visual_severity": "Analysis Failed",
    }
    at.run()
    assert not at.exception
    assert any("Simulated decode failure" in err.value for err in at.error)
    assert any("Final result: Analysis Failed" in h.body for h in at.html)
