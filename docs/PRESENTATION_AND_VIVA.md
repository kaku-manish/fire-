# Phase 18 — 15-Slide Project Presentation & 50 Viva Q&A

---

## PART A: 15-Slide Project Presentation Outline

### Slide 1 — Project Title
- **Title:** FireGuard AI — Intelligent Fire and Smoke Detection with Early Warning and Visual Severity Analysis Using Deep Learning
- **Subtitle:** Software-Only Academic Computer Vision Prototype
- **Key Stack:** Python · PyTorch · Ultralytics YOLO11 · OpenCV · Streamlit · SQLite · Plotly · ReportLab

### Slide 2 — Introduction
- Early detection of smoke and fire minimizes response delays and situational blind spots.
- Smoke frequently appears prior to open flames, enabling earlier warning cues.
- Computer vision enables localization, visual coverage analysis, and logging using standard imagery and prerecorded videos—without specialized hardware.

### Slide 3 — Problem Statement
- Conventional physical sensors require smoke/heat to physically reach the sensor.
- Flame-only vision models miss early smoldering smoke stages.
- Naive color thresholding fails on sunsets, artificial lighting, headlights, fog, and clouds.
- Raw detections lack severity sizing, temporal trend tracking, and structured reporting.

### Slide 4 — Existing System & Limitations
- **Physical Sensors:** No spatial localization, coverage insight, or visual evidence; hardware-dependent.
- **RGB/HSV Heuristics:** High false-alarm rate in warm lighting or sunsets.
- **Basic Image Classifiers:** Frame-level label only; cannot separate overlapping fire & smoke areas or track growth.

### Slide 5 — Proposed System (FireGuard AI)
- Software-only end-to-end pipeline operating on uploaded images and prerecorded videos.
- Dual-class **YOLO11** localization (`smoke` and `fire`).
- Temporal false-alarm filtering + hard-negative curation.
- Bounding-box union severity estimation (`Low`, `Medium`, `High`).
- Temporal fire growth analysis + 4-tier alerts (`GREEN` / `YELLOW` / `ORANGE` / `RED`).
- SQLite incident database + ReportLab PDF & CSV reporting inside a 7-page Streamlit dashboard.

### Slide 6 — Project Objectives
- Detect fire alone, smoke alone, fire + smoke together, and normal scenes.
- Quantify approximate visual frame coverage without double-counting overlapping boxes.
- Reduce false alarms from visually deceptive scenes without brittle RGB-only rules.
- Track fire growth trends in prerecorded videos and generate downloadable incident reports.

### Slide 7 — Technology Stack
- **AI / CV:** Python, PyTorch, Ultralytics YOLO11, OpenCV, NumPy, Pillow, Scikit-learn, Pandas
- **Dashboard & Charts:** Streamlit, Custom CSS/Theme TOML, Plotly
- **Storage & Reporting:** SQLite (`fireguard.db`), ReportLab (PDF), CSV
- **Training & Testing:** Google Colab (GPU), Jupyter Notebook, Pytest

### Slide 8 — System Architecture
- **Pipeline Flow:** Upload -> Validation (`PIL`/`OpenCV`) -> `FireSmokeDetector` (`YOLO11`) -> `TemporalFalseAlarmFilter` -> `SeverityEstimator` & `FireGrowthAnalyzer` -> `AlertManager` -> `IncidentDatabase` (`SQLite`) -> `ReportGenerator` (`PDF`/`CSV`) -> Streamlit 7-Page UI.

### Slide 9 — Dataset & Preprocessing
- **Primary Dataset:** D-Fire (`gaia-solutions-on-demand/DFireDataset`) — 21,000+ images across fire, smoke, joint fire+smoke, and normal/hard-negative backgrounds.
- **External Smoke Dataset:** HPWREN FIgLib wildfire ignition sequences.
- **Preparation (`prepare_dataset.py` & `validate_dataset.py`):** Corrupt/duplicate cleanup (SHA-256 + dHash), leakage-free sequence-grouped 70/20/10 split, letterboxing (`640×640`), and training-only conservative augmentations (flip, brightness/contrast, scale, mosaic, mild noise).

### Slide 10 — Model Training Methodology
- Transfer learning starting from COCO-pretrained **YOLO11n** (and optional comparison with **YOLO11s** and **YOLOv8n**).
- Initial config: `640×640`, `epochs=50–100`, `batch=16`, `optimizer=auto`, `patience=15`, `seed=42`.
- Automated GPU detection in `notebooks/model_training.ipynb` (Colab) and `training/train.py`.
- Epoch validation, early stopping, best checkpoint export (`models/best.pt`), and ONNX export.

### Slide 11 — Key Proposed Enhancements
1. **Early Smoke Warnings:** Independent `smoke` detection triggers `YELLOW` alerts prior to visible flames.
2. **False Alarm Reduction:** Hard-negative training + calibrated confidence + minimum area + consecutive-frame temporal verification + EMA smoothing.
3. **Exact Union Severity:** Sweep-line rectangle union prevents inflated coverage when boxes overlap.
4. **Growth & Alerts:** Rolling-smoothed growth slope classifies `Increasing`, `Decreasing`, `Relatively stable`, or `Insufficient data`.

### Slide 12 — Application Demonstration (7 Dashboard Pages)
- **Home:** System status, workflow, readiness check, disclaimer.
- **Image Detection:** Drag-and-drop image, confidence slider, side-by-side preview, coverage & PDF download.
- **Video Detection:** Progress bar, processed playback, timestamps, Plotly growth chart, alert timeline.
- **Analytics Dashboard:** Live KPIs and distributions from `fireguard.db`.
- **Model Performance:** Real training/test metrics & confusion matrix (or honest unavailable notice).
- **Incident History:** Search/filter/delete records + CSV & PDF export.
- **About Project:** Architecture, datasets, methodology, limitations.

### Slide 13 — Experimental Evaluation Framework
- Independent test split evaluation (`training/evaluate.py`): Precision, Recall, F1-score, mAP@50, mAP@50–95, inference latency (ms), and FPS.
- Baseline vs Enhanced false-alarm comparison on negative test images + smoke onset delay measurement on video sequences.
- *(Presenter inserts actual observed Colab/local numbers from `models/evaluation_results.json` after training.)*

### Slide 14 — Limitations & Future Scope
- **Limitations:** Academic prototype only; no hardware sensors or emergency dispatch; 2D coverage does not measure heat/burned area; negative predictions do not guarantee safety; visual growth does not forecast physical fire spread.
- **Future Scope:** Pixel-level instance segmentation, ONNX/TensorRT optimization, active learning on hard negatives, multi-camera correlation.

### Slide 15 — Conclusion
- Successfully integrates YOLO11 detection, temporal filtering, union visual severity, video growth tracking, SQLite incident management, and ReportLab PDF reporting into one cohesive software platform.
- Ready for academic demonstration, evaluation, and further research extensions.

---

## PART B: 50 Important Viva Questions & Answers

1. **What is FireGuard AI?**
   FireGuard AI is a software-only academic deep learning prototype that detects fire and smoke in uploaded images and prerecorded videos, estimates approximate visual coverage, suppresses false alarms, analyzes video growth trends, stores incident history in SQLite, and generates PDF/CSV reports.
2. **Why was YOLO11 selected over older detectors or two-stage detectors like Faster R-CNN?**
   YOLO11 is a fast, modern single-stage object detector with an anchor-free decoupled head and efficient backbone (`YOLO11n`), enabling strong localization accuracy at interactive laptop/Colab speeds compared to slower two-stage region-proposal networks.
3. **Why start with `YOLO11n` instead of a huge model like `YOLO11x`?**
   `YOLO11n` (nano) has low parameter count and memory footprint, allowing fast training on free Google Colab GPUs and smooth inference on standard student laptops without dedicated GPUs, while still serving as a clean baseline against `YOLO11s` and `YOLOv8n`.
4. **How are fire and smoke detected independently?**
   The model is trained on two separate classes in `dataset/data.yaml` (`0: smoke` and `1: fire`). Each predicted bounding box carries its own class ID and confidence score, so the system can flag smoke alone, fire alone, or both simultaneously.
5. **Why is detecting smoke separately from fire important?**
   In many real-world incidents, smoldering materials emit smoke before visible flames appear. Detecting smoke independently allows `YELLOW` early-warning alerts before full flame escalation.
6. **How do YOLO bounding boxes work?**
   YOLO predicts four coordinates per object—either normalized center/width/height `(cx, cy, w, h)` in labels or corner coordinates `(x1, y1, x2, y2)` during inference—along with a confidence score and class probability, filtered via Non-Maximum Suppression (NMS).
7. **What is Non-Maximum Suppression (NMS)?**
   NMS removes redundant overlapping bounding boxes predicting the same object by keeping the highest-confidence box and suppressing nearby boxes whose Intersection over Union (IoU) exceeds the `iou_threshold`.
8. **How does transfer learning work in FireGuard AI?**
   Instead of initializing weights randomly, we load `yolo11n.pt` pretrained on COCO (which already learned general edges, textures, and shapes) and fine-tune the network on the D-Fire dataset for our 2 target classes (`smoke`, `fire`).
9. **What is `dataset/data.yaml` and why is it needed?**
   `data.yaml` tells Ultralytics where the `train`, `valid`, and `test` image folders live, the number of classes (`nc: 2`), and the exact mapping from integer class IDs (`0`, `1`) to class names (`smoke`, `fire`).
10. **What class IDs does the D-Fire dataset use?**
    In `gaia-solutions-on-demand/DFireDataset`, class `0` is `smoke` and class `1` is `fire`. Empty `.txt` label files represent background/negative scenes.
11. **How does `prepare_dataset.py` prevent data leakage between splits?**
    Consecutive frames from the same video or camera sequence share filename prefixes. We extract a `group_id` for each frame and use `GroupShuffleSplit` so all frames from a sequence land exclusively in either train (70%), valid (20%), or test (10%).
12. **How are duplicate images detected and handled?**
    `prepare_dataset.py` computes cryptographic SHA-256 hashes to remove exact byte duplicates and perceptual difference hashes (dHash) to detect near-identical frames.
13. **Why are augmentations applied only to the training set?**
    Validation and test splits must reflect clean, unaltered real-world inputs so evaluation metrics measure true generalization rather than synthetic artifacts.
14. **Why avoid aggressive color/hue augmentations for fire and smoke?**
    Extreme hue shifts can turn normal blue skies or green foliage into orange/red flame-like blobs or convert flames into unnatural colors, teaching false visual cues.
15. **What is letterboxing and why does YOLO use it?**
    Letterboxing resizes an image while keeping its original aspect ratio and pads the remaining border with gray `(114, 114, 114)` to reach `640 × 640`, avoiding geometric distortion of flame/smoke shapes.
16. **How does 4-up mosaic augmentation help?**
    Mosaic combines four training images into one `640 × 640` canvas and shifts their bounding boxes accordingly, teaching the detector to recognize smaller and partially occluded fire/smoke regions at varied scales.
17. **What causes false alarms in visual fire detection?**
    Visually similar phenomena such as sunsets, bright sunlight, warm artificial lamps, vehicle headlights, orange objects, fog, clouds, steam, and shiny reflections.
18. **Why shouldn't a fire detector rely only on RGB/HSV color rules?**
    Flame colors vary from pale yellow to deep red depending on fuel/lighting, smoke varies from white to dark gray, and non-fire objects (sunsets, orange shirts, streetlights) share identical RGB values.
19. **How does FireGuard AI reduce false alarms?**
    Through three layers: (1) negative/hard-negative training examples, (2) calibrated confidence thresholds and minimum relative box area filtering, and (3) multi-frame temporal persistence + EMA confidence smoothing on videos.
20. **How does temporal consistency verification work?**
    `TemporalFalseAlarmFilter` maintains a sliding window and consecutive-frame counter per class. A detection is confirmed only after appearing consistently across `min_consecutive_frames` (e.g., 3 frames) or exceeding `min_persistence_ratio` inside the window.
21. **What is EMA confidence smoothing?**
    Exponential Moving Average (`ema = alpha * current + (1 - alpha) * previous`) smooths frame-to-frame confidence jitter so a single noisy frame spike cannot abruptly distort confidence metrics.
22. **What is the trade-off of false alarm reduction?**
    Stricter thresholds and multi-frame persistence reduce false-positive rate (FPR), but can slightly increase missed detections on faint targets and introduce a small verification delay (`(min_consecutive_frames - 1) / fps` seconds).
23. **How is visual severity estimated in `src/severity.py`?**
    We compute `Visual Coverage (%) = (Union Area of Detected Boxes / Total Frame Area) * 100` and classify `0%` as `No Detection`, `<5%` as `Low`, `5% to <20%` as `Medium`, and `>=20%` as `High`.
24. **Why do we compute the union of bounding boxes instead of summing their areas?**
    Multiple overlapping fire or smoke boxes cover shared pixels. Simply summing box areas could double-count overlaps and even exceed 100%, whereas a sweep-line rectangle union counts every covered pixel at most once.
25. **Does visual coverage % measure burned area or fire temperature?**
    No. It is strictly an approximate 2D screen-area metric that depends on camera angle and distance, not physical heat release, temperature, or real-world hazard.
26. **How could segmentation improve visual severity estimation?**
    Bounding boxes include background corners around irregular flames/plumes. Pixel-level segmentation masks (`estimate_segmentation_coverage`) measure exact visible flame/smoke pixels when segmentation annotations are available.
27. **How does `src/growth_analysis.py` analyze fire growth in videos?**
    It records per-frame fire and smoke coverage percentages over timestamps, applies centered rolling-mean smoothing, fits a linear regression slope across active frames, and classifies the pattern as `Increasing`, `Decreasing`, `Relatively stable`, or `Insufficient data`.
28. **Can bounding-box growth predict physical fire spread to new rooms/locations?**
    No. Bounding-box growth only describes observed visual changes within the recorded video frames (which can also vary with camera zoom or smoke occlusion), not physical fire spread modeling.
29. **How does FireGuard AI handle videos without running out of memory?**
    `src/video_processor.py` reads and writes one frame at a time using `cv2.VideoCapture` and `cv2.VideoWriter` streaming loops instead of storing an array of all decoded frames in RAM.
30. **What is frame sampling (`frame_sample_rate`) in video detection?**
    Running YOLO inference every Nth frame reduces CPU load on long videos while propagating the latest confirmed bounding boxes onto intermediate frames so output playback stays smooth.
31. **What are the four software alert categories in FireGuard AI?**
    `GREEN` (no detection), `YELLOW` (persistent smoke / early warning), `ORANGE` (visible fire), and `RED` (persistent fire with high `>=20%` visual coverage or rapid growth rate).
32. **How does duplicate alert suppression work?**
    `AlertManager` tracks the last emitted alert level and timestamp per file. Repeated alerts of the same or lower severity within `cooldown_seconds` are marked suppressed, while escalations (e.g., `YELLOW -> ORANGE -> RED`) pass immediately.
33. **How does this project work without any hardware sensors, Raspberry Pi, or Arduino?**
    It operates entirely in software on digital media (uploaded images and prerecorded videos) processed by deep learning and computer vision libraries on a standard PC/laptop.
34. **Why is a negative prediction NOT proof that an environment is safe?**
    A fire could be hidden behind obstacles, outside the camera's field of view, obscured by darkness, or below the detection confidence threshold.
35. **Why does FireGuard AI log one summary row per video instead of one row per frame?**
    Logging every frame (e.g., 30 rows per second) would bloat `fireguard.db`, slow dashboard queries, and clutter incident history. Instead, we log one episode summary with peak coverage, first detection timestamp, and growth trend.
36. **How does `src/database.py` prevent SQL injection?**
    All queries (`INSERT`, `SELECT`, `DELETE`) use SQLite parameterized `?` placeholders rather than string formatting or concatenation.
37. **What happens if `models/best.pt` is missing when someone opens the dashboard?**
    `FireSmokeDetector` detects the missing file and raises `ModelNotFoundError`, which the Streamlit UI catches and displays cleanly with training instructions—without crashing or faking predictions.
38. **Why does the Model Performance page show "Unavailable" before training?**
    To preserve academic honesty: metrics (`Precision`, `Recall`, `F1`, `mAP@50`, `mAP@50-95`) are loaded exclusively from real JSON outputs (`models/evaluation_results.json`) generated after actual training and testing.
39. **How are PDF reports generated?**
    `src/report_generator.py` uses **ReportLab** (`SimpleDocTemplate`, `Table`, `Paragraph`, `Image`) to compile incident metadata, original/annotated snapshots, video growth graphs, summaries, and prototype disclaimers into a downloadable `.pdf`.
40. **What is the difference between `Precision`, `Recall`, and `mAP@50`?**
    `Precision` measures how many predicted detections were actually fire/smoke (low false alarms). `Recall` measures how many real fire/smoke instances were successfully caught (low missed detections). `mAP@50` averages per-class Average Precision at IoU threshold `0.50`.
41. **What does `mAP@50–95` measure compared to `mAP@50`?**
    `mAP@50–95` averages AP across 10 stricter IoU thresholds (`0.50` to `0.95` in `0.05` steps), testing both classification accuracy and tight bounding-box localization precision.
42. **What is early stopping (`patience=15`) during training?**
    If validation performance fails to improve for 15 consecutive epochs, Ultralytics halts training automatically to prevent overfitting and save compute time, keeping the best checkpoint in `best.pt`.
43. **Why set a fixed random seed (`seed=42`)?**
    Fixing the random seed makes dataset splits, training initialization, and augmentation previews reproducible across experiments.
44. **What is the difference between `ImageProcessor` validation and `VideoProcessor` validation?**
    `ImageProcessor` validates `.jpg/.jpeg/.png/.webp` bytes via Pillow `verify()` and OpenCV decode; `VideoProcessor` validates `.mp4/.avi/.mov` files, size, codec readability (`VideoCapture.isOpened()`), and non-zero frame dimensions.
45. **How does the project handle browser compatibility for processed videos?**
    OpenCV writes `mp4v` reliably; if `ffmpeg` is available on the system, `video_processor.py` automatically transcodes the file to `libx264` (`yuv420p`, `+faststart`) for universal HTML5 `<video>` playback while providing a download button either way.
46. **What is the difference between mock detections in tests vs real inference?**
    In `tests/test_detection.py`, a mock detector is used solely to test downstream logic (box drawing, severity math, temporal filter, DB logging, error paths) deterministically. In `app.py` and `src/`, mock predictions are never allowed.
47. **How does Streamlit manage shared resources like the YOLO model and SQLite DB without reloading on every click?**
    Using `@st.cache_resource` in `src/ui_helpers.py`, the model wrapper and `IncidentDatabase` instance are cached across reruns, while per-user state lives in `st.session_state`.
48. **What happens if someone uploads a corrupted image or unsupported `.txt` file?**
    `validate_image_bytes` raises `InvalidImageError`, which the UI catches and displays via `st.error(...)` without writing corrupt rows to SQLite.
49. **How is FireGuard AI different from a basic YOLO fire detection script?**
    Basic scripts only run `model.predict()` on an image. FireGuard AI adds leakage-free dataset auditing, dual fire/smoke early warning, temporal false-alarm filtering, union severity estimation, video growth trend classification, cooldown-managed alerts, SQLite persistence, automated PDF/CSV reports, 22+ automated test cases, and a 7-page interactive dashboard.
50. **Why is FireGuard AI labeled an academic prototype rather than a certified fire alarm?**
    Certified life-safety systems undergo rigorous hardware redundancy, UL/EN regulatory testing, environmental calibration, and failsafe alarm routing. FireGuard AI is an educational software-only computer vision demonstration designed for academic evaluation.
