# FireGuard AI — Existing Project Audit (`PROJECT_AUDIT.md`)

**Audit Date:** 2026-11-25  
**Project Root:** `c:\Users\kakum\Desktop\fire`  
**Audit Scope:** Full codebase, `.venv` Python environment, dependencies, datasets, model weights, SQLite database, Streamlit UI, and test suite.

---

## 1. Environment & Dependency Audit

| Component | Verified Value | Status |
| :--- | :--- | :--- |
| **Virtual Environment** | `c:\Users\kakum\Desktop\fire\.venv` | Working |
| **Python Version** | `3.13.5` | Working |
| **PyTorch** | `2.15.0+cpu` | Working (CPU-only) |
| **CUDA / GPU Acceleration** | `torch.cuda.is_available() == False` (no NVIDIA GPU detected) | CPU Fallback Only |
| **Ultralytics YOLO** | `8.4.218` | Working |
| **OpenCV** | `5.1.0` | Working |
| **Streamlit** | `1.67.0` | Working |
| **Plotly / ReportLab / Pandas / Scikit-learn / PyYAML / Pytest** | All installed in `.venv` | Working |
| **System `ffmpeg`** | Not present on `PATH` | Optional (OpenCV `mp4v` fallback active) |

### Note on `requirements.txt`
`requirements.txt` specifies compatible lower/upper bounds for all core AI, web, SQLite/CSV, ReportLab PDF, and Pytest packages. No missing Python dependencies were found.

---

## 2. File & Architecture Audit

### Working Components (Verified & Preserved)
1. **Configuration & Styling:** `config.py` (central paths, thresholds, academic disclaimers), `.streamlit/config.toml` (dark charcoal theme), `assets/styles.css` (custom card & alert styling).
2. **Streamlit Application (`app.py` & `app_pages/`):**
   - Clean 7-page navigation (`Home`, `Image detection`, `Video detection`, `Analytics dashboard`, `Model performance`, `Incident history`, `About project`).
   - Verified via `AppTest` and headless server probe (`http://localhost:8501` / Agent API `ready` on all 7 pages).
3. **Backend Services (`src/`):**
   - `detector.py` — `FireSmokeDetector` lazy-loads Ultralytics YOLO weights, normalizes class labels (`fire`, `smoke`), computes status, and draws red/orange bounding boxes.
   - `image_processor.py` — Validates `.jpg/.jpeg/.png/.webp`, runs detection, computes severity & alerts, outputs downloadable PNGs.
   - `video_processor.py` — Streams `.mp4/.avi/.mov` frame-by-frame via `cv2.VideoCapture` / `cv2.VideoWriter` with sampling rate, temporal filter, growth tracking, and peak snapshot export.
   - `severity.py` — Sweep-line bounding-box union area calculation + optional segmentation mask coverage.
   - `temporal_filter.py` — Geometry/confidence filtering, sliding-window consecutive-frame persistence, EMA smoothing, and threshold sweep calibration.
   - `growth_analysis.py` — Rolling smoothing, linear slope trend (`Increasing`, `Decreasing`, `Relatively stable`, `Insufficient data`), smoke lead-time, and Plotly figures.
   - `alert_manager.py` — Cooldown-managed alerts and synthesized WAV tones.
   - `database.py` — Parameterized SQLite storage at `database/fireguard.db` (`sessions` and `incidents` tables).
   - `report_generator.py` — ReportLab PDF & CSV exports.
4. **Training & Testing Scripts (`training/`, `notebooks/`, `tests/`):**
   - `training/prepare_dataset.py`, `training/validate_dataset.py`, `training/train.py`, `training/evaluate.py`, and `notebooks/model_training.ipynb`.
   - `tests/test_detection.py`, `tests/test_severity.py`, `tests/test_database.py` — 29 tests passing cleanly.

---

## 3. Identified Gaps, Blockers & Missing Components

### Critical Blocker: Missing Model Weights (`models/best.pt`) & Empty Dataset Splits
- **Inspection Result:** `models/` contains only `.gitkeep` — **no trained `.pt` weights exist anywhere in `c:\Users\kakum\Desktop\fire`**.
- **Dataset Inspection Result:** `dataset/train/images/`, `dataset/valid/images/`, and `dataset/test/images/` contain only `.gitkeep`. `dataset/data.yaml` exists and maps `{0: smoke, 1: fire}`.
- **Hardware Constraint:** This local machine runs `PyTorch 2.15.0+cpu` with **no CUDA GPU** (`CUDA: False`). Full 50-epoch training on the ~21,000-image D-Fire dataset is impractical on CPU alone and requires either **Google Colab GPU** (via `notebooks/model_training.ipynb`) or downloading an open-licensed D-Fire subset/trained fire-smoke weights, while **never** renaming COCO pretrained `yolo11n.pt` weights as `best.pt`.

### Missing / Upgrade-Target Features (Steps 3–15)
1. **Model Health Check Diagnostics Service (`Step 3`):** No dedicated diagnostics service yet verifying file existence, size, load status, architecture, class names (`fire`, `smoke`), device, CUDA status, test inference pass, average latency, and explicit status badges (`MODEL MISSING`, `MODEL LOADING`, `MODEL READY`, `MODEL INFERENCE FAILED`, `MODEL CLASS MISMATCH`).
2. **Live Audit Page & Navigation Order (`Step 4`):** `Live Audit` page is missing and must be placed second in sidebar navigation right after `Home`, featuring:
   - **Mode A:** Simulated Live Video Monitoring (`Start`, `Pause`, `Resume`, `Stop`, `Restart`, `Loop`, playback speed, frame sampling, live KPIs, dynamic charts, `"SIMULATED LIVE STREAM — PRERECORDED VIDEO"` notice).
   - **Mode B:** Dataset Live Audit (category filter, class-aware IoU matching at configurable `0.5` IoU threshold, TP/FP/FN, transition speed, auto-run to sample limit).
   - **Mode C:** Manual Model Validation (upload image + optional YOLO `.txt` ground truth comparison).
3. **Live Audit Logs (`Step 7`):** Bounded in-memory monitoring console with levels (`INFO`, `SUCCESS`, `WARNING`, `ERROR`, `CRITICAL`), search, event filter, clear, and CSV export.
4. **Automated Model Audit (`Step 8`) & Expanded Evaluation (`Step 9`):** Batch dataset audit (`Quick: 20`, `Standard: 100`, `Extended: 500`, `Full test`) with `PASS` / `FAIL` / `NOT EVALUATED` table, PR curve, confidence distribution, per-class chart, and success/failure galleries.
5. **Alert Terminology Alignment (`Step 10`):** Align live audit monitoring alert levels cleanly with `NORMAL`, `WARNING`, `HIGH`, `CRITICAL` (mapping backward-compatibly with `database.py` / `alert_manager.py`).
6. **Built-in Demonstration Samples (`Step 11`):** `samples/` directory (`images/{fire,smoke,fire_and_smoke,normal,hard_negatives}` and `videos/`) plus an **Auto Demo** runner.
7. **Extended Database Schema & PDF Audit Reports (`Steps 13–14`):** Add tables for `audit_sessions`, `model_health_checks`, `dataset_audit_results`, `detection_events`, `alert_incidents`, and `model_evaluation_summaries`, plus a comprehensive **Generate Audit Report** PDF generator distinguishing software verification, AI evaluation, and lack of fire-safety certification.

---

## 4. Compatibility & Path Findings
- All existing imports (`config`, `src.*`, `training.*`) resolve cleanly when executed from `c:\Users\kakum\Desktop\fire`.
- `src/detector.py` automatically falls back `YOLO_CONFIG_DIR` to `%TEMP%\fireguard_ultralytics` when `%APPDATA%\Ultralytics` is read-only inside restricted shells.
- All Streamlit pages use modern `width="stretch"` / `width="content"` parameters (no deprecated `use_container_width`), and will preserve full backward compatibility throughout the upgrade.
