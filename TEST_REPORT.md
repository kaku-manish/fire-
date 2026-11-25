# FireGuard AI — Software Verification & Model Test Report (`TEST_REPORT.md`)

**Report Date:** 2026-11-25  
**Environment:** Windows 11 · Python `3.13.5` (`.venv`) · PyTorch `2.15.0+cpu` (`CUDA: False`) · Ultralytics `8.4.218` · OpenCV `5.1.0` · Streamlit `1.67.0`

---

## 1. Executive Summary

- **Software Verification Status:** **PASSED (34/34 automated tests + live 8-page Streamlit runtime check)**
- **Model Availability Status:** **`MODEL READY` (`SMOKE_TEST` tier)** — A genuine 5-epoch YOLO11n smoke-test model trained on 240 real D-Fire images is installed at `models/best.pt` (backed up at `models/smoke_test_best.pt` and tagged in `models/model_metadata.json`).
- **Hardware Blocker for Final Full-Scale Training:** This machine has **no CUDA GPU** (`torch.cuda.is_available() == False`). Full 50-epoch training across all 21,527 D-Fire images should be executed on **Google Colab T4 GPU** using [notebooks/model_training.ipynb](file:///c:/Users/kakum/Desktop/fire/notebooks/model_training.ipynb) before replacing `models/best.pt` for final academic accuracy reporting.

---

## 2. Dataset & Sample Media Verification

| Split / Directory | Contents | Verification Result |
| :--- | :--- | :--- |
| `dataset/train/` | 240 D-Fire images (`60` fire-only, `60` smoke-only, `60` fire+smoke, `60` normal/hard-negative) | 0 corrupted, 0 invalid label lines, imbalance ratio `1.73` |
| `dataset/valid/` | 60 D-Fire images (`15` per category) | 0 corrupted, 0 invalid label lines, imbalance ratio `1.87` |
| `dataset/test/` | 100 D-Fire images (`25` per category) | 0 corrupted, 0 invalid label lines, imbalance ratio `1.98` |
| `samples/images/` | 25 held-out D-Fire test samples (`5` each in `fire`, `smoke`, `fire_and_smoke`, `normal`, `hard_negatives`) | Verified via `discover_audit_images` |
| `samples/videos/` | `wildfire_dfire_author.mp4` (131 frames, 1280×720, MIT License) + `normal_environment_demo.mp4` (80 frames, 640×480) | Verified via OpenCV `VideoCapture` and `process_video_file` |

---

## 3. Smoke-Test Model Training & Independent Test Evaluation

### Smoke-Test Training Run (`training/train.py --model yolo11n.pt --epochs 5 --imgsz 640 --batch 8 --smoke-test`)
- **Duration:** 5 epochs completed in `0.084 hours` on CPU (`Intel Core Ultra 5 125H`).
- **Checkpoint Output:** `models/best.pt` (`5.5 MB`), `models/smoke_test_best.pt`, `models/model_metadata.json` (`model_tier: "SMOKE_TEST"`).
- **Validation Split (60 images, 86 instances):** `Precision = 0.305`, `Recall = 0.280`, `mAP@50 = 0.203`, `mAP@50–95 = 0.106`.

### Independent Test Split Evaluation (`training/evaluate.py --weights models/best.pt`)
Evaluated on **100 held-out D-Fire test images** (161 ground-truth instances: 54 smoke, 107 fire; 25 negative background scenes):

| Metric | Observed Value | Notes |
| :--- | :--- | :--- |
| **Overall Precision** | `0.3345` | Smoke: `0.278` · Fire: `0.391` |
| **Overall Recall** | `0.1725` | Smoke: `0.0926` · Fire: `0.252` |
| **F1-Score** | `0.2276` | Computed directly from test Precision & Recall |
| **mAP@50** | `0.0929` | Smoke: `0.029` · Fire: `0.157` |
| **mAP@50–95** | `0.0354` | Smoke: `0.0135` · Fire: `0.0572` |
| **Avg Inference Latency / FPS** | `37.85 ms` (`26.42 FPS`) | Measured per test image on CPU |
| **Baseline vs Enhanced Negative FPR** | `20.0%` (`5/25`) → `12.0%` (`3/25`) | Enhanced (`conf=0.35`, min area filter) reduced false alarms by 40% relative; positive image recall trade-off `52.0% → 34.67%` |

> **Honest Interpretation:** Because these weights came from a 5-epoch CPU smoke test on 240 images, missed detections and occasional false positives occur (as seen in both the test metrics above and sample inference). This validates the end-to-end training, evaluation, and false-alarm comparison pipeline while clearly motivating the full 50-epoch Colab GPU run on all 21,527 D-Fire images.

---

## 4. Automated Test Execution Log

### A. Unit & Integration Pytest Suite (`33 passed in 11.39s`)
Command: `.venv\Scripts\python.exe -m pytest -k "not test_streamlit_all_pages_smoke" -v`

- **`tests/test_audit_upgrade.py` (5/5 PASSED):**
  - `test_model_health_statuses` — verified `MODEL MISSING`, `MODEL INFERENCE FAILED`, `MODEL CLASS MISMATCH` (on generic COCO `yolo11n.pt`), and `MODEL READY` (on trained `models/best.pt`).
  - `test_audit_logger_bounded_filtering_and_csv` — verified bounded `maxlen` retention, search/level filtering, clear, and CSV export.
  - `test_class_aware_iou_matching` — verified exact IoU calculation, class-aware matching (`PASS`, `FAIL` on class mismatch, `NOT EVALUATED` when unlabeled).
  - `test_extended_audit_database_and_pdf_report` — verified all 6 new audit tables (`audit_sessions`, `model_health_checks`, `dataset_audit_results`, `detection_events`, `alert_incidents`, `model_evaluation_summaries`) and Audit PDF generation.
  - `test_monitoring_alert_level_mapping_and_samples` — verified `NORMAL`/`WARNING`/`HIGH`/`CRITICAL` mapping and `samples/` discovery.
- **`tests/test_detection.py` (18/18 unit/integration tests PASSED):** Scenarios 1–16 & 20 (fire-only, smoke-only, fire+smoke, normal, fog, sunset, artificial light tiny-box filter, low-light fire, small/large coverage, increasing fire video, clean video with temporal spike suppression, corrupted image after `ultralytics` PIL patch, unsupported formats, missing weights, empty upload, corrupted video, and 8-page compilation/theme TOML check). *(Uses controlled mock detector outputs only inside unit tests.)*
- **`tests/test_severity.py` (6/6 PASSED):** Overlapping box union (no double-counting), threshold boundaries, optional segmentation mask union, insufficient frames, increasing/decreasing/stable trends + early smoke lead time, sequence-grouped leakage-free split & mosaic transform.
- **`tests/test_database.py` (4/4 PASSED):** SQLite CRUD & filtering, insertion failure handling, ReportLab PDF + CSV export, duplicate alert cooldown suppression & immediate escalation.

### B. Streamlit `AppTest` & Live Headless Server Verification (`1 passed in 8.09s` + Live HTTP Probe)
Command: `.venv\Scripts\python.exe -m pytest tests\test_detection.py -k test_streamlit_all_pages_smoke -v` (executed outside Windows sandbox due to loopback socket restriction inside sandbox)
- `test_streamlit_all_pages_smoke`: **PASSED** (0 exceptions, 0 deprecation warnings across `app.py` and all 8 `app_pages/*.py`).
- **Live Headless Server (`streamlit run app.py --server.enableAgentApi=true`):**
  - `/_stcore/health`: `ok`
  - All 8 pages (`Home`, `Live audit`, `Image detection`, `Video detection`, `Analytics dashboard`, `Model performance`, `Incident history`, `About project`): status `ready`
  - Interactive `"Run Model Health Check"` button triggered via Agent API: `ready`
  - Server console output: **Clean** (zero tracebacks, zero Arrow serialization warnings, zero Pandas deprecation warnings).

### C. Sample Video Inference Pipeline Check
- `samples/videos/wildfire_dfire_author.mp4` (`131` frames): streamed cleanly at `21.02 FPS`, produced playable MP4 output in `outputs/`.
- `samples/videos/normal_environment_demo.mp4` (`80` frames): streamed cleanly, classified `No fire or smoke detected.`, alert level `NORMAL`.

---

## 5. Unresolved Items / Next Steps for Final Academic Evaluation

1. **Full 50-Epoch GPU Training:** Execute [notebooks/model_training.ipynb](file:///c:/Users/kakum/Desktop/fire/notebooks/model_training.ipynb) in Google Colab (T4 GPU) on the complete 21,527-image D-Fire dataset to replace the 5-epoch CPU `SMOKE_TEST` weights at `models/best.pt` with a full `FULL_EVALUATED` model.
2. **Optional `ffmpeg` Installation:** Adding `ffmpeg` to Windows `PATH` enables automatic `mp4v → H.264` transcoding for inline browser playback of exported processed videos (already gracefully falls back to downloadable `mp4v` MP4).
3. **Sandbox Loopback Note:** In restricted Windows sandboxes blocking TCP loopback `socketpair()`, `AppTest` and `streamlit run` require running in normal Command Prompt (`cmd.exe`) or with network/port permission.
