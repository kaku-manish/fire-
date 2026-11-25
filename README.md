# FireGuard AI — AI-Based Fire and Smoke Detection

A clean, software-only academic deep learning prototype built with **Python**, **Ultralytics YOLO11n**, **OpenCV**, **FastAPI**, **React + Vite**, **SQLite**, and **Streamlit** to detect **fire** and **smoke** from uploaded images and videos.

> **Academic Prototype Disclaimer:** FireGuard AI is an educational research prototype. It is **not** a certified fire safety system and does not connect to physical cameras, sensors, or emergency services.

---

## Cloud & Local Architecture

- **`frontend/` (Vercel — React + Vite):** Dark-themed 4-page SPA (**Home**, **Live Audit**, **Detection History**, **About Project**) configured with `vercel.json` SPA rewrites and `VITE_API_URL`.
- **`backend/` (Render — FastAPI + Uvicorn):** REST API (`GET /api/health`, `POST /api/analyze`, `GET/DELETE /api/history`, `GET /api/about`) powered by [`src/`](src/) and `models/best.pt`, configured cleanly with [`render.yaml`](render.yaml) and [`backend/requirements-render.txt`](backend/requirements-render.txt).
- **`app.py` + `app_pages/` (Local Streamlit):** Preserved for direct single-command local academic demos (`streamlit run app.py`).

---

## Deploying to Render (Backend) + Vercel (Frontend)

### 1. Deploy Backend on Render
1. Push this repository to GitHub.
2. In **Render**, create a new **Web Service** connected to your repo (or use **New Blueprint** to auto-read `render.yaml`).
3. Confirm settings:
   - **Build Command:** `pip install --upgrade pip && pip install -r backend/requirements-render.txt`
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Health Check Path:** `/api/health`
   - **Environment Variables:** `PYTHON_VERSION=3.11.11`, `YOLO_CONFIG_DIR=/tmp/ultralytics`, `ALLOWED_ORIGINS=*` (set to your Vercel URL after Step 2).
4. Copy your Render URL (e.g. `https://fireguard-ai-backend.onrender.com`).

### 2. Deploy Frontend on Vercel
1. In **Vercel**, import the same GitHub repository.
2. Set **Root Directory** to `frontend` and Framework Preset to **Vite**.
3. Add Environment Variable:
   - `VITE_API_URL` = `https://fireguard-ai-backend.onrender.com` *(no trailing slash)*
4. Click **Deploy**.

> **Render Memory Note:** PyTorch + YOLO11n inference measures ~371 MB RSS on single images; Render's Free tier (`512 MB RAM`) works best with images and short clips (`--workers 1`), while Render Starter (`2 GB RAM`) is recommended for larger MP4 videos. Also note Render Free spins down after inactivity (~30–50s cold start).

---

## Quick Installation & Startup (Windows Command Prompt)

```cmd
cd /d c:\Users\kakum\Desktop\fire
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 2-Minute Academic Demonstration Flow

1. Start on **Home** and verify the top-right status indicator shows **Model ready**.
2. Click **Start Detection** to open **Live Audit**.
3. Drag and drop a sample image from `samples/images/fire/` or `samples/images/fire_and_smoke/` (or an `.mp4` from `samples/videos/`) and click **Analyze Fire & Smoke**.
4. Show the side-by-side bounding boxes (Red = Fire, Orange = Smoke), circular fire/smoke affected region indicators, detection confidence %, visual severity, and final result banner.
5. Open **Detection History** to verify the saved record in `database/fireguard.db`, then use **About Project** to walk through the architecture diagram.

---

## Training & Evaluation Scripts (Preserved in `training/` & `notebooks/`)

- **Prepare D-Fire dataset:** `python training\prepare_dataset.py --fetch-dfire-subset`
- **Validate & generate `dataset/data.yaml`:** `python training\validate_dataset.py`
- **Full 50-epoch GPU training:** Run [notebooks/model_training.ipynb](file:///c:/Users/kakum/Desktop/fire/notebooks/model_training.ipynb) in Google Colab and place `best.pt` at `models/best.pt`.
- **Independent test evaluation:** `python training\evaluate.py --weights models\best.pt`
- **Run automated software tests:** `pytest -v`
