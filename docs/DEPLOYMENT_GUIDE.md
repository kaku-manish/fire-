# Phase 16 — FireGuard AI Deployment & Operations Guide

> **Academic Prototype Notice:** FireGuard AI is an educational research prototype. It is not a certified fire safety system and must not be deployed for life-safety protection or automated emergency dispatch.

---

## 1. Local Deployment (Windows, Linux, macOS)

### Step 1: Clone or Open the Project Directory
```bash
cd /path/to/FireGuard-AI
```
*(On this Windows workspace: `cd /d c:\Users\kakum\Desktop\fire`)*

### Step 2: Create and Activate a Virtual Environment

**Windows Command Prompt (`cmd.exe`):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Trained Model Weights (`models/best.pt`)
1. **Local CPU Smoke-Test Weights (Quick Validation):**
   ```cmd
   python training\prepare_dataset.py --fetch-dfire-subset
   python training\validate_dataset.py
   python training\train.py --model yolo11n.pt --epochs 5 --imgsz 640 --smoke-test
   ```
   This saves explicitly tagged `SMOKE_TEST` weights (`models/best.pt`, `models/smoke_test_best.pt`, `models/model_metadata.json`) so Live Audit and all pipeline tests run end-to-end.
2. **Full 50-Epoch GPU Training (Final Evaluated Model):**
   Upload `notebooks/model_training.ipynb` to **Google Colab (T4 GPU)** to train on the complete 21,527-image D-Fire dataset for 50 epochs (`seed=42`, `patience=15`), download `best.pt`, and overwrite `models/best.pt`.
3. **Evaluate on Independent Test Split:**
   ```cmd
   python training\evaluate.py --weights models\best.pt
   ```
   This populates `models/evaluation_results.json` and `outputs/evaluation/` charts for the **Model Performance** and **Live Audit** pages.

### Step 5: Initialize Directories & SQLite Database
```bash
python config.py
python -c "from src.database import IncidentDatabase; IncidentDatabase()"
```
This creates `database/fireguard.db` with normalized `sessions`, `incidents`, `audit_sessions`, `model_health_checks`, `dataset_audit_results`, `detection_events`, `alert_incidents`, and `model_evaluation_summaries` tables.

### Step 6: Start the Streamlit Application
```bash
streamlit run app.py
```
Open the URL printed in the terminal (default: `http://localhost:8501`).

---

## 2. Cloud Deployment Options (Vercel + Render / Free Demos)

### Option A: Vercel (Frontend) + Render (FastAPI Backend) — Recommended Split
1. **Backend on Render (`backend/` + `render.yaml`):**
   - Connect your GitHub repository as a **Web Service** (or Blueprint reading `render.yaml`).
   - Build Command: `pip install --upgrade pip && pip install -r backend/requirements-render.txt` (uses CPU-only PyTorch wheels + `opencv-python-headless` to fit cleanly inside cloud containers).
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - Health Check Path: `/api/health`
   - Environment Variables: `PYTHON_VERSION=3.11.11`, `YOLO_CONFIG_DIR=/tmp/ultralytics`, `FIREGUARD_TORCH_THREADS=2`, `ALLOWED_ORIGINS=https://your-vercel-app.vercel.app`.
   - **Memory note:** YOLO11n CPU inference uses ~371 MB RSS on single images; keep `--workers 1` and short MP4 clips on Render Free (`512 MB RAM`), or select Render Starter for longer HD video uploads.
2. **Frontend on Vercel (`frontend/`):**
   - Import the same GitHub repository in Vercel, set **Root Directory** to `frontend`, and Framework Preset to **Vite**.
   - Add Environment Variable `VITE_API_URL=https://<your-render-service>.onrender.com` (no trailing slash).
   - `frontend/vercel.json` handles SPA routing rewrites automatically.

### Option B: Streamlit Community Cloud
1. Push your source code (without `dataset/` images, `runs/`, `uploads/`, `outputs/`, `reports/`, or `database/*.db`) to GitHub. Keep `.gitignore` intact.
2. Add a `packages.txt` at the repo root containing Linux system libraries required by OpenCV & video codecs:
   ```text
   libgl1
   libglib2.0-0
   ffmpeg
   ```
3. In `requirements.txt`, switch `opencv-python` to `opencv-python-headless>=4.9.0` for headless Linux containers.
4. Ensure `models/best.pt` is small enough for GitHub (<100 MB; YOLO11n `.pt` is typically ~5–6 MB) or downloaded cleanly at startup from a public model artifact URL into `models/best.pt`.
5. **Resource Limits:** Streamlit Community Cloud free tier typically provides ~1 GB RAM and shared CPU; keep `imgsz=640`, use `YOLO11n` (lightweight), limit video sizes (`<50 MB`), and set `frame_sample_rate >= 2` for smoother execution.

### Option B: Hugging Face Spaces (Streamlit SDK)
1. Create a new Space selecting **Streamlit** SDK and **CPU basic (16 GB RAM, 2 vCPU)** free tier.
2. Push `app.py`, `app_pages/`, `src/`, `assets/`, `.streamlit/`, `config.py`, `requirements.txt`, `packages.txt` (`ffmpeg`, `libgl1`), and `models/best.pt` (via Git LFS if >10 MB).
3. Note that Hugging Face Spaces container storage resets on Space restart unless persistent storage is attached; treat `database/fireguard.db` as session-demo storage and encourage downloading PDF/CSV reports.

---

## 3. Security & Repository Hygiene Checklist
- Never hardcode API keys, tokens, or credentials in Python files.
- Never commit `dataset/train/images/*`, `uploads/*`, `outputs/*`, `reports/*`, or `database/*.db` (already enforced by `.gitignore`).
- Keep `enableAgentApi` disabled in `.streamlit/config.toml` (only use it temporarily as a CLI flag during testing).

---

## 4. Troubleshooting Common Errors

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Model weights not found at models/best.pt` | Model not yet trained/copied | Train via Phase 4 notebook/script and copy `best.pt` into `models/best.pt`. |
| `streamlit` command not recognized | `.venv` not activated | Activate `.venv\Scripts\activate.bat` (CMD) or `source .venv/bin/activate` (Linux). |
| Video doesn't play inline in browser | OpenCV wrote `mp4v` without `ffmpeg` | Install `ffmpeg` so `video_processor.py` auto-transcodes to H.264, or download the processed MP4 to play in VLC. |
| Slow CPU video analysis | Every frame inferred on CPU | Increase **Process every Nth frame** slider (e.g. `2` or `3`) on the Video Detection page. |
| SQLite lock / permission error | Directory read-only | Ensure write permission on `database/` and `uploads/`, `outputs/`, `reports/`. |
