"""Phase 15 — Unit & Integration Tests for SQLite Database, Alert Suppression,
and ReportLab PDF / CSV Report Generation.
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

from src.alert_manager import AlertManager, generate_alert_tone_wav_bytes  # noqa: E402
from src.database import DatabaseOperationError, IncidentDatabase  # noqa: E402
from src.report_generator import generate_csv_report, generate_pdf_report  # noqa: E402


def test_database_crud_filtering_and_dashboard_metrics(tmp_path: Path) -> None:
    """Verify SQLite insert, search, type/severity/date filtering, delete, and KPIs."""
    db = IncidentDatabase(tmp_path / "fireguard_test.db")
    inc1 = db.insert_incident(
        file_name="warehouse_blaze.jpg",
        file_type="image",
        detection_type="Fire detected.",
        confidence=0.92,
        fire_coverage=15.4,
        smoke_coverage=0.0,
        visual_severity="Medium",
        alert_level="ORANGE",
        created_at="2026-11-20 10:00:00",
    )
    inc2 = db.insert_incident(
        file_name="forest_plume.mp4",
        file_type="video",
        detection_type="Smoke detected.",
        confidence=0.81,
        fire_coverage=0.0,
        smoke_coverage=8.2,
        visual_severity="Medium",
        alert_level="YELLOW",
        video_timestamp=4.5,
        created_at="2026-11-22 14:30:00",
    )
    inc3 = db.insert_incident(
        file_name="sunset_harbor.png",
        file_type="image",
        detection_type="No fire or smoke detected.",
        confidence=0.0,
        fire_coverage=0.0,
        smoke_coverage=0.0,
        visual_severity="No Detection",
        alert_level="GREEN",
        created_at="2026-11-24 18:10:00",
    )

    assert len(db.get_incidents()) == 3
    assert len(db.get_incidents(detection_type="Smoke detected.")) == 1
    assert len(db.get_incidents(severity="Medium")) == 2
    assert len(db.get_incidents(search_query="warehouse")) == 1
    assert len(db.get_incidents(start_date="2026-11-22", end_date="2026-11-24")) == 2

    metrics = db.get_dashboard_metrics()
    assert metrics["total_analyzed_files"] == 3
    assert metrics["total_fire_incidents"] == 1
    assert metrics["total_smoke_incidents"] == 1
    assert metrics["total_negative_predictions"] == 1
    assert metrics["average_confidence"] == pytest.approx(0.865, abs=0.001)

    deleted = db.delete_incidents([inc1])
    assert deleted == 1
    assert db.get_incident_by_id(inc1) is None
    assert db.get_incident_by_id(inc2) is not None
    assert db.get_incident_by_id(inc3) is not None


def test_scenario_17_database_insertion_failure(tmp_path: Path) -> None:
    """Scenario 17: Missing required fields or unwritable DB raises DatabaseOperationError."""
    db = IncidentDatabase(tmp_path / "fail_test.db")
    with pytest.raises(DatabaseOperationError):
        db.insert_incident(
            file_name="",
            file_type="image",
            detection_type="Fire detected.",
            confidence=0.8,
            fire_coverage=10.0,
            smoke_coverage=0.0,
            visual_severity="Medium",
            alert_level="ORANGE",
        )

    # Point DB path at a directory instead of a file to trigger SQLite connection failure
    bad_dir = tmp_path / "is_a_directory"
    bad_dir.mkdir()
    with pytest.raises(DatabaseOperationError):
        IncidentDatabase(bad_dir)


def test_scenario_18_report_generation(tmp_path: Path) -> None:
    """Scenario 18: Generate valid ReportLab PDF (with images) and CSV exports."""
    orig = tmp_path / "orig.png"
    snap = tmp_path / "snap.png"
    chart = tmp_path / "growth.png"
    dummy = np.full((80, 120, 3), (30, 60, 200), dtype=np.uint8)
    cv2.imwrite(str(orig), dummy)
    cv2.imwrite(str(snap), dummy)
    cv2.imwrite(str(chart), dummy)

    record = {
        "incident_id": "INC-TEST-001",
        "session_id": "SES-TEST-001",
        "file_name": "factory_test.png",
        "file_type": "image",
        "detection_type": "Fire and smoke detected.",
        "confidence": 0.91,
        "fire_coverage": 12.5,
        "smoke_coverage": 18.0,
        "visual_severity": "High",
        "alert_level": "RED",
        "video_timestamp": 3.2,
        "created_at": "2026-11-25 18:00:00",
        "output_path": str(snap),
        "original_path": str(orig),
        "growth_trend": "Increasing",
        "notes": "Smoke preceded flames by 2.0s.",
    }

    pdf_bytes, pdf_path = generate_pdf_report(
        record,
        output_path=tmp_path / "INC-TEST-001_report.pdf",
        growth_chart_path=chart,
    )
    assert pdf_path.is_file()
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000

    csv_bytes, csv_path = generate_csv_report([record], output_path=tmp_path / "history.csv")
    assert csv_path.is_file()
    assert b"INC-TEST-001" in csv_bytes
    assert b"factory_test.png" in csv_bytes


def test_scenario_19_duplicate_alert_prevention() -> None:
    """Scenario 19: Suppress duplicate alerts within cooldown while allowing escalation."""
    mgr = AlertManager({"cooldown_seconds": 10.0, "high_fire_coverage_pct": 20.0})

    # t=0s: Smoke detected -> YELLOW emitted
    ev1 = mgr.evaluate(
        source_filename="demo.mp4",
        has_fire=False,
        has_smoke=True,
        confidence=0.75,
        fire_coverage_pct=0.0,
        smoke_coverage_pct=4.0,
        visual_severity="Low",
        event_clock_sec=0.0,
    )
    assert ev1.alert_level == "YELLOW"
    assert ev1.suppressed is False

    # t=3s (within 10s cooldown): Same YELLOW smoke -> suppressed!
    ev2 = mgr.evaluate(
        source_filename="demo.mp4",
        has_fire=False,
        has_smoke=True,
        confidence=0.78,
        fire_coverage_pct=0.0,
        smoke_coverage_pct=4.5,
        visual_severity="Low",
        event_clock_sec=3.0,
    )
    assert ev2.alert_level == "YELLOW"
    assert ev2.suppressed is True

    # t=4s (still within cooldown, but escalates YELLOW -> ORANGE fire): allowed immediately!
    ev3 = mgr.evaluate(
        source_filename="demo.mp4",
        has_fire=True,
        has_smoke=True,
        confidence=0.88,
        fire_coverage_pct=8.0,
        smoke_coverage_pct=5.0,
        visual_severity="Medium",
        event_clock_sec=4.0,
    )
    assert ev3.alert_level == "ORANGE"
    assert ev3.suppressed is False

    # t=16s (after cooldown expires): Same ORANGE allowed again
    ev4 = mgr.evaluate(
        source_filename="demo.mp4",
        has_fire=True,
        has_smoke=True,
        confidence=0.90,
        fire_coverage_pct=9.0,
        smoke_coverage_pct=5.0,
        visual_severity="Medium",
        event_clock_sec=16.0,
    )
    assert ev4.suppressed is False

    # Verify synthesized WAV tone generation
    wav = generate_alert_tone_wav_bytes("RED")
    assert wav is not None and wav[:4] == b"RIFF"
    assert generate_alert_tone_wav_bytes("GREEN") is None
