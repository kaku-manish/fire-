"""Integration tests for the Render FastAPI backend API (`backend/main.py`)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import backend.main as backend_main  # noqa: E402
import config  # noqa: E402
from src.database import IncidentDatabase  # noqa: E402


def _make_blank_png_bytes(width: int = 128, height: int = 128) -> bytes:
    img = np.full((height, width, 3), (35, 45, 50), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_backend_health_about_and_history_lifecycle(tmp_path: Path, monkeypatch) -> None:
    """Verify /api/health, /api/about, and /api/history CRUD using an isolated temp SQLite DB."""
    temp_db = IncidentDatabase(tmp_path / "test_api.db")
    monkeypatch.setattr(backend_main, "_DATABASE", temp_db)

    with TestClient(backend_main.app) as client:
        root_resp = client.get("/")
        assert root_resp.status_code == 200
        assert root_resp.json()["status"] == "running"

        health_resp = client.get("/api/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["model_ready"] is True
        assert set(health_data["class_names"].values()) == {"smoke", "fire"}

        about_resp = client.get("/api/about")
        assert about_resp.status_code == 200
        about_data = about_resp.json()
        assert about_data["architecture"]["frontend"] == "React + Vite (Vercel)"
        assert about_data["architecture"]["backend"] == "FastAPI + Uvicorn (Render)"

        hist_resp = client.get("/api/history")
        assert hist_resp.status_code == 200
        assert hist_resp.json()["total"] == 0


def test_backend_analyze_image_real_and_error_cases(tmp_path: Path, monkeypatch) -> None:
    """Verify /api/analyze on real fire+smoke sample, normal sample, and invalid uploads."""
    temp_db = IncidentDatabase(tmp_path / "test_img_api.db")
    monkeypatch.setattr(backend_main, "_DATABASE", temp_db)
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")

    sample_fire_smoke = BASE_DIR / "samples" / "images" / "fire_and_smoke" / "AoF07781.jpg"
    assert sample_fire_smoke.is_file()

    with TestClient(backend_main.app) as client:
        # 1. Real fire + smoke sample image
        resp = client.post(
            "/api/analyze",
            files={"file": ("AoF07781.jpg", sample_fire_smoke.read_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["media_type"] == "image"
        assert data["original_media_url"].startswith("data:image/png;base64,")
        assert data["analyzed_media_url"].startswith("data:image/png;base64,")
        assert data["has_fire"] is True
        assert data["fire_pct"] > 0.0
        assert data["confidence_display"] != "N/A"
        assert data["final_result"] in {"Fire Detected", "Fire and Smoke Detected"}

        # 2. Clean blank image -> 0 detections -> N/A confidence & 0.0% coverages
        blank_resp = client.post(
            "/api/analyze",
            files={"file": ("blank.png", _make_blank_png_bytes(), "image/png")},
        )
        assert blank_resp.status_code == 200
        bdata = blank_resp.json()
        assert bdata["fire_pct"] == 0.0
        assert bdata["smoke_pct"] == 0.0
        assert bdata["confidence_display"] == "N/A"
        assert bdata["visual_severity"] == "No detection"
        assert bdata["final_result"] == "No Fire or Smoke Detected"

        # 3. History captured 2 analyses, then clears cleanly
        hist = client.get("/api/history").json()
        assert hist["total"] == 2
        cleared = client.delete("/api/history").json()
        assert cleared["deleted"] == 2
        assert client.get("/api/history").json()["total"] == 0

        # 4. Unsupported extension & corrupted bytes return Analysis Failed cleanly
        bad_ext = client.post(
            "/api/analyze",
            files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert bad_ext.status_code == 400
        assert bad_ext.json()["final_result"] == "Analysis Failed"

        corrupted = client.post(
            "/api/analyze",
            files={"file": ("broken.png", b"not-an-image", "image/png")},
        )
        assert corrupted.status_code == 400
        assert corrupted.json()["final_result"] == "Analysis Failed"


def test_backend_analyze_mp4_and_media_serving(tmp_path: Path, monkeypatch) -> None:
    """Verify /api/analyze on short MP4 video and /api/media download protection."""
    temp_db = IncidentDatabase(tmp_path / "test_vid_api.db")
    monkeypatch.setattr(backend_main, "_DATABASE", temp_db)
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")

    vid_file = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(vid_file), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (128, 96))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((96, 128, 3), dtype=np.uint8))
    writer.release()

    with TestClient(backend_main.app) as client:
        resp = client.post(
            "/api/analyze",
            files={"file": ("tiny.mp4", vid_file.read_bytes(), "video/mp4")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["media_type"] == "video"
        assert data["frames_processed"] == 4
        assert data["analyzed_media_url"].startswith("/api/media/outputs/")

        media_resp = client.get(data["analyzed_media_url"])
        assert media_resp.status_code == 200
        assert len(media_resp.content) > 0

        traversal = client.get("/api/media/outputs/..secret.mp4")
        assert traversal.status_code in {400, 404}
