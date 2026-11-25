"""Step 15 — Unit & Integration Tests for FireGuard AI Upgrade Modules:
- Model Health Check diagnostics (MISSING, INFERENCE_FAILED, CLASS_MISMATCH, READY)
- Bounded LiveAuditLogger filtering & CSV export
- Class-aware IoU matching (PASS / FAIL / NOT EVALUATED)
- Extended SQLite audit tables & Audit PDF report generation
- Sample media discovery and NORMAL/WARNING/HIGH/CRITICAL alert mapping
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402
from src.alert_manager import AlertManager, to_monitoring_level  # noqa: E402
from src.audit_engine import (  # noqa: E402
    GroundTruthBox,
    compute_iou_xyxy,
    discover_audit_images,
    discover_sample_videos,
    match_predictions_to_ground_truth,
)
from src.audit_logger import LiveAuditLogger  # noqa: E402
from src.database import IncidentDatabase  # noqa: E402
from src.detector import Detection  # noqa: E402
from src.model_health import (  # noqa: E402
    STATUS_CLASS_MISMATCH,
    STATUS_INFERENCE_FAILED,
    STATUS_MISSING,
    STATUS_READY,
    run_model_health_check,
)
from src.report_generator import generate_audit_pdf_report  # noqa: E402


def _det(cls_name: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.8) -> Detection:
    return Detection(
        class_id=1 if cls_name == "fire" else 0,
        class_name=cls_name,
        confidence=conf,
        bbox_xyxy=(x1, y1, x2, y2),
        bbox_normalized=(x1 / 100.0, y1 / 100.0, x2 / 100.0, y2 / 100.0),
        area_px=(x2 - x1) * (y2 - y1),
    )


def test_model_health_statuses(tmp_path: Path) -> None:
    """Verify MODEL MISSING, MODEL INFERENCE FAILED, and MODEL CLASS MISMATCH checks."""
    missing = run_model_health_check(tmp_path / "does_not_exist.pt")
    assert missing.status == STATUS_MISSING
    assert missing.file_exists is False

    corrupted = tmp_path / "corrupted.pt"
    corrupted.write_bytes(b"not-a-valid-pytorch-checkpoint")
    failed = run_model_health_check(corrupted)
    assert failed.status == STATUS_INFERENCE_FAILED
    assert failed.load_succeeded is False

    # Check if generic COCO yolo11n.pt exists at workspace root -> must flag CLASS MISMATCH!
    coco_pt = BASE_DIR / "yolo11n.pt"
    if coco_pt.is_file():
        mismatch = run_model_health_check(coco_pt)
        assert mismatch.status == STATUS_CLASS_MISMATCH
        assert mismatch.classes_verified is False

    # If trained models/best.pt exists, verify MODEL READY and real inference
    if config.is_model_available():
        ready = run_model_health_check(config.MODEL_PATH)
        assert ready.status == STATUS_READY
        assert ready.classes_verified is True
        assert ready.inference_succeeded is True
        assert ready.avg_inference_latency_ms is not None and ready.avg_inference_latency_ms > 0.0


def test_audit_logger_bounded_filtering_and_csv() -> None:
    """Verify bounded circular retention, search/level/event filtering, and CSV export."""
    logger = LiveAuditLogger(max_entries=55)
    for i in range(70):
        logger.log("INFO", "Frame processed", source_item=f"Frame {i:03d}", inference_latency_ms=12.5)
    logger.log("WARNING", "Smoke detected", source_item="Frame 071", detection_class="smoke", confidence=0.72, alert_level="WARNING")
    logger.log("CRITICAL", "Alert triggered", source_item="Frame 072", detection_class="fire", confidence=0.91, alert_level="CRITICAL")

    assert len(logger) == 55
    crit = logger.get_entries(level="CRITICAL")
    assert len(crit) == 1 and crit[0]["source_item"] == "Frame 072"
    smoke_q = logger.get_entries(search_text="smoke")
    assert len(smoke_q) == 1
    csv_bytes = logger.export_csv_bytes()
    assert b"Alert triggered" in csv_bytes

    logger.clear()
    assert len(logger) == 0


def test_class_aware_iou_matching() -> None:
    """Verify class-aware IoU matching returns PASS, FAIL, or NOT EVALUATED accurately."""
    assert compute_iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert compute_iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

    gt = [GroundTruthBox(1, "fire", (10, 10, 50, 50))]
    pred_good = [_det("fire", 12, 12, 48, 48, conf=0.88)]
    res_pass = match_predictions_to_ground_truth(pred_good, gt, iou_threshold=0.5)
    assert res_pass.true_positives == 1
    assert res_pass.false_positives == 0
    assert res_pass.false_negatives == 0
    assert res_pass.audit_status == "PASS"

    # Wrong class ('smoke' overlapping 'fire' box) must yield FP=1, FN=1, FAIL!
    pred_wrong_cls = [_det("smoke", 10, 10, 50, 50, conf=0.9)]
    res_fail = match_predictions_to_ground_truth(pred_wrong_cls, gt, iou_threshold=0.5)
    assert res_fail.true_positives == 0
    assert res_fail.false_positives == 1
    assert res_fail.false_negatives == 1
    assert res_fail.audit_status == "FAIL"

    # Missing ground truth -> NOT EVALUATED
    res_unlabeled = match_predictions_to_ground_truth(pred_good, None, iou_threshold=0.5)
    assert res_unlabeled.audit_status == "NOT EVALUATED"


def test_extended_audit_database_and_pdf_report(tmp_path: Path) -> None:
    """Verify all 6 extended SQLite audit tables and Step 14 Audit PDF Report generation."""
    db = IncidentDatabase(tmp_path / "audit_upgrade.db")
    sid = db.start_audit_session("best.pt", "automated_dataset_audit", "test_batch")
    chk_id = db.log_model_health_check("models/best.pt", STATUS_READY, "yolo11n", '{"0":"smoke","1":"fire"}', "cpu", 45.2, "SMOKE_TEST")
    assert chk_id.startswith("CHK-")

    snap = tmp_path / "sample_cmp.png"
    cv2.imwrite(str(snap), np.full((60, 120, 3), (30, 80, 210), dtype=np.uint8))

    inserted = db.insert_dataset_audit_results(
        sid,
        [
            {
                "image_name": "img1.jpg",
                "category": "Fire only",
                "ground_truth_class": "Fire",
                "predicted_class": "Fire",
                "matched_detections": 1,
                "false_positives": 0,
                "false_negatives": 0,
                "confidence": 0.86,
                "iou": 0.78,
                "inference_time_ms": 44.0,
                "audit_status": "PASS",
                "annotated_path": str(snap),
            }
        ],
    )
    assert inserted == 1
    db.insert_detection_event(sid, "img1.jpg", "Fire", 0.86, fire_coverage=12.0, inference_latency_ms=44.0)
    db.insert_alert_incident(sid, "img1.jpg", "Fire", 0.86, "HIGH", fire_coverage=12.0, snapshot_path=str(snap))
    db.finish_audit_session(sid, 1, 1, 0, 44.0, "Precision=1.0, Recall=1.0", "COMPLETED")
    db.insert_model_evaluation_summary("best.pt", "test", 1, 1.0, 1.0, 1.0, map50=0.8, avg_inference_ms=44.0, avg_fps=22.7)

    sessions = db.get_audit_sessions()
    assert len(sessions) == 1 and sessions[0]["status"] == "COMPLETED"
    assert len(db.get_model_health_checks()) == 1
    assert len(db.get_dataset_audit_results(sid)) == 1
    assert len(db.get_alert_incidents()) == 1
    assert b"automated_dataset_audit" in db.export_audit_sessions_csv()

    pdf_bytes, pdf_path = generate_audit_pdf_report(
        audit_session=sessions[0],
        health_report={"status": STATUS_READY, "model_tier": "SMOKE_TEST", "architecture": "yolo11n", "execution_device": "cpu"},
        metrics={"evaluated_sample_count": 1, "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "avg_inference_ms": 44.0},
        detection_examples=[snap],
        alert_history=db.get_alert_incidents(),
        output_path=tmp_path / "audit_report.pdf",
    )
    assert pdf_path.is_file() and pdf_bytes.startswith(b"%PDF-")


def test_monitoring_alert_level_mapping_and_samples() -> None:
    """Verify NORMAL/WARNING/HIGH/CRITICAL mapping and built-in sample media discovery."""
    assert to_monitoring_level("GREEN") == "NORMAL"
    assert to_monitoring_level("YELLOW") == "WARNING"
    assert to_monitoring_level("ORANGE") == "HIGH"
    assert to_monitoring_level("RED") == "CRITICAL"

    mgr = AlertManager()
    ev = mgr.evaluate("clip.mp4", False, True, 0.72, 0.0, 3.0, "Low", suppress_duplicates=False)
    assert ev.monitoring_level == "WARNING"

    sample_imgs = discover_audit_images(split="test", include_samples=True)
    sample_vids = discover_sample_videos()
    assert len(sample_imgs) >= 25
    assert len(sample_vids) >= 1
