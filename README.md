# FireGuard AI — AI-Based Fire and Smoke Detection

A clean, software-only academic deep learning prototype built with **Python**, **Ultralytics YOLO11n**, **OpenCV**, **SQLite**, and **Streamlit** to detect **fire** and **smoke** from uploaded images and videos.

> **Academic Prototype Disclaimer:** FireGuard AI is an educational research prototype. It is **not** a certified fire safety system and does not connect to physical cameras, sensors, or emergency services.

---

## Simple 4-Page Interface

1. **Home** — Project overview, Fire Detection / Smoke Detection / Software-Only highlights, and **Start Detection** button.
2. **Live Audit** — Upload a `JPG`, `JPEG`, `PNG`, or `MP4` file, click **Analyze Fire & Smoke**, view original vs. AI-annotated media side by side (red boxes for fire, orange boxes for smoke), inspect circular fire & smoke coverage indicators, confidence %, visual severity (`No detection`, `Low <5%`, `Medium 5–20%`, `High >=20%`), and auto-save to SQLite.
3. **Detection History** — Simple table of saved detections (`Date and time`, `Uploaded file name`, `Fire coverage (%)`, `Smoke coverage (%)`, `Visual severity`, `Detection result`) with a confirmed **Clear History** button.
4. **About Project** — Beginner-friendly explanation of the problem, D-Fire dataset, YOLO11n model, architecture flow, and prototype limitations.

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
