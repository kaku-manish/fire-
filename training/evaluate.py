"""Phase 14 — Independent Test Evaluation, False Alarm Comparison, and Latency Benchmarking.

Evaluates trained YOLO weights on dataset/test/ without fabricating scores,
measures per-image inference latency and FPS, computes false-positive rate (FPR)
before and after false-alarm reduction, estimates smoke detection delay on
annotated video sequences, and compares YOLO11n, YOLO11s, and YOLOv8n.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from src.audit_engine import discover_audit_images, run_automated_dataset_audit  # noqa: E402
from src.database import IncidentDatabase  # noqa: E402
from src.detector import FireSmokeDetector  # noqa: E402
from src.temporal_filter import TemporalFalseAlarmFilter, filter_by_geometry_and_confidence  # noqa: E402
from training.prepare_dataset import VALID_IMAGE_EXTS, parse_and_validate_yolo_label  # noqa: E402


def _generate_per_class_chart(per_class: dict[str, dict[str, float]], out_path: Path) -> Path | None:
    if not per_class:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    classes = list(per_class.keys())
    p_vals = [per_class[c].get("precision", 0.0) for c in classes]
    r_vals = [per_class[c].get("recall", 0.0) for c in classes]
    ap_vals = [per_class[c].get("map50", 0.0) for c in classes]

    x = np.arange(len(classes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor("#111418")
    ax.set_facecolor("#181C22")
    ax.bar(x - w, p_vals, w, label="Precision", color="#22C55E")
    ax.bar(x, r_vals, w, label="Recall", color="#EAB308")
    ax.bar(x + w, ap_vals, w, label="mAP@50", color="#F97316")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], color="#F8FAFC")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", color="#CBD5E1")
    ax.set_title("Per-Class Detection Performance", color="#F8FAFC")
    ax.tick_params(colors="#CBD5E1")
    ax.legend(facecolor="#181C22", labelcolor="#F8FAFC")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def evaluate_yolo_on_test_split(
    weights_path: Path,
    data_yaml: Path,
    imgsz: int = config.DEFAULT_IMAGE_SIZE,
    conf: float = 0.25,
    iou: float = 0.45,
) -> dict[str, Any]:
    """Run Ultralytics validation on the independent 'test' split and benchmark speed."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    data_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root_path = Path(data_cfg.get("path", data_yaml.parent))
    test_images_dir = root_path / str(data_cfg.get("test", "test/images"))
    test_labels_dir = test_images_dir.parent / "labels"

    test_images = (
        sorted(p for p in test_images_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS)
        if test_images_dir.is_dir()
        else []
    )
    if not test_images:
        raise ValueError(f"No test images found under {test_images_dir}")

    model = YOLO(str(weights_path))
    val_res = model.val(data=str(data_yaml), split="test", imgsz=imgsz, conf=conf, iou=iou, plots=True)

    box_metrics = val_res.box
    precision = float(getattr(box_metrics, "mp", 0.0))
    recall = float(getattr(box_metrics, "mr", 0.0))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    map50 = float(getattr(box_metrics, "map50", 0.0))
    map50_95 = float(getattr(box_metrics, "map", 0.0))

    names = val_res.names if hasattr(val_res, "names") else model.names
    per_class: dict[str, dict[str, float]] = {}
    p_arr = list(getattr(box_metrics, "p", []))
    r_arr = list(getattr(box_metrics, "r", []))
    ap50_arr = list(getattr(box_metrics, "ap50", []))
    ap_arr = list(getattr(box_metrics, "ap", []))
    for idx, cls_id in enumerate(sorted(names.keys())):
        cls_name = str(names[cls_id])
        c_p = float(p_arr[idx]) if idx < len(p_arr) else 0.0
        c_r = float(r_arr[idx]) if idx < len(r_arr) else 0.0
        c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
        per_class[cls_name] = {
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "f1_score": round(c_f1, 4),
            "map50": round(float(ap50_arr[idx]) if idx < len(ap50_arr) else 0.0, 4),
            "map50_95": round(float(ap_arr[idx]) if idx < len(ap_arr) else 0.0, 4),
        }

    # Benchmark per-image inference latency, FPS, and negative-image FPR
    detector = FireSmokeDetector(model_path=weights_path)
    latencies_ms: list[float] = []
    negative_total = 0
    baseline_false_positives = 0
    enhanced_false_positives = 0
    positive_total = 0
    baseline_true_positives = 0
    enhanced_true_positives = 0

    for img_path in test_images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        lbl_path = test_labels_dir / f"{img_path.stem}.txt"
        gt_boxes, _ = parse_and_validate_yolo_label(lbl_path)
        is_negative = len(gt_boxes) == 0

        t0 = time.perf_counter()
        raw_res = detector.predict_frame(frame, conf_threshold=0.25, iou_threshold=iou)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        filtered_dets = filter_by_geometry_and_confidence(
            raw_res.detections,
            frame_width=w,
            frame_height=h,
            fire_conf_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
            smoke_conf_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
            min_box_area_ratio=config.DEFAULT_MIN_BOX_AREA_RATIO,
        )

        if is_negative:
            negative_total += 1
            if raw_res.detections:
                baseline_false_positives += 1
            if filtered_dets:
                enhanced_false_positives += 1
        else:
            positive_total += 1
            if raw_res.detections:
                baseline_true_positives += 1
            if filtered_dets:
                enhanced_true_positives += 1

    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0
    baseline_fpr = (baseline_false_positives / negative_total) if negative_total > 0 else None
    enhanced_fpr = (enhanced_false_positives / negative_total) if negative_total > 0 else None
    baseline_img_recall = (baseline_true_positives / positive_total) if positive_total > 0 else None
    enhanced_img_recall = (enhanced_true_positives / positive_total) if positive_total > 0 else None

    save_dir = Path(val_res.save_dir)
    pr_candidates = [save_dir / "BoxPR_curve.png", save_dir / "PR_curve.png"]
    pr_curve_path = next((str(p) for p in pr_candidates if p.is_file()), None)

    eval_out_dir = config.OUTPUTS_DIR / "evaluation"
    per_class_chart = _generate_per_class_chart(per_class, eval_out_dir / "per_class_chart.png")

    audit_items = discover_audit_images(dataset_dir=root_path, split="test", include_samples=False)
    audit_summary = run_automated_dataset_audit(
        image_items=audit_items,
        detector=detector,
        conf_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold=0.5,
        save_dir=eval_out_dir,
    ) if audit_items else {"artifacts": {}}
    arts = audit_summary.get("artifacts", {})

    return {
        "model_name": weights_path.name,
        "weights_path": str(weights_path),
        "test_images_evaluated": len(latencies_ms),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "per_class": per_class,
        "avg_inference_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "false_alarm_evaluation": {
            "negative_images": negative_total,
            "positive_images": positive_total,
            "baseline_false_positives": baseline_false_positives,
            "enhanced_false_positives": enhanced_false_positives,
            "baseline_fpr": round(baseline_fpr, 4) if baseline_fpr is not None else None,
            "enhanced_fpr": round(enhanced_fpr, 4) if enhanced_fpr is not None else None,
            "baseline_image_recall": round(baseline_img_recall, 4) if baseline_img_recall is not None else None,
            "enhanced_image_recall": round(enhanced_img_recall, 4) if enhanced_img_recall is not None else None,
            "note": (
                "Single-image filtering evaluates calibrated confidence and minimum bounding-box "
                "area; temporal persistence filtering adds multi-frame verification on video streams."
            ),
        },
        "confusion_matrix_path": str(save_dir / "confusion_matrix.png"),
        "pr_curve_path": pr_curve_path,
        "per_class_chart_path": str(per_class_chart) if per_class_chart else None,
        "confidence_dist_path": arts.get("confidence_distribution"),
        "success_gallery_path": arts.get("success_gallery"),
        "failure_gallery_path": arts.get("failure_gallery"),
    }


def measure_video_smoke_delay(
    weights_path: Path,
    video_path: Path,
    smoke_onset_sec: float,
    conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Measure smoke detection delay before and after temporal filtering on an annotated video."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    detector = FireSmokeDetector(model_path=weights_path)
    temporal = TemporalFalseAlarmFilter()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    baseline_first_sec: float | None = None
    filtered_first_sec: float | None = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = frame_idx / fps
            res = detector.predict_frame(frame, conf_threshold=conf_threshold)
            raw_smoke = [d for d in res.detections if d.class_name == "smoke"]
            if baseline_first_sec is None and raw_smoke and ts >= smoke_onset_sec:
                baseline_first_sec = ts

            filtered = temporal.update(res.detections, frame.shape[1], frame.shape[0])
            filt_smoke = [d for d in filtered.confirmed_detections if d.class_name == "smoke"]
            if filtered_first_sec is None and filt_smoke and ts >= smoke_onset_sec:
                filtered_first_sec = ts
            frame_idx += 1
    finally:
        cap.release()

    return {
        "video": video_path.name,
        "smoke_onset_sec": smoke_onset_sec,
        "baseline_first_detection_sec": round(baseline_first_sec, 3) if baseline_first_sec is not None else None,
        "filtered_first_detection_sec": round(filtered_first_sec, 3) if filtered_first_sec is not None else None,
        "baseline_delay_sec": (
            round(baseline_first_sec - smoke_onset_sec, 3) if baseline_first_sec is not None else None
        ),
        "filtered_delay_sec": (
            round(filtered_first_sec - smoke_onset_sec, 3) if filtered_first_sec is not None else None
        ),
        "temporal_tradeoff_note": (
            f"Requiring {temporal.min_consecutive_frames} consecutive frames reduces transient false "
            f"alarms while adding ~{(temporal.min_consecutive_frames - 1) / fps:.2f}s minimum verification delay."
        ),
    }


def generate_comparison_chart(evaluations: list[dict[str, Any]], output_path: Path) -> Path:
    """Plot actual evaluation metric comparison across evaluated models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [e["model_name"] for e in evaluations]
    metrics = ["precision", "recall", "f1_score", "map50", "map50_95"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#111418")
    ax.set_facecolor("#181C22")

    x = np.arange(len(names))
    width = 0.15
    colors = ["#22C55E", "#EAB308", "#F97316", "#38BDF8", "#EF4444"]
    for idx, metric in enumerate(metrics):
        vals = [e.get(metric, 0.0) for e in evaluations]
        ax.bar(x + (idx - 2) * width, vals, width, label=metric.upper(), color=colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(names, color="#F8FAFC")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", color="#CBD5E1")
    ax.set_title("Model Evaluation Comparison (Independent Test Set)", color="#F8FAFC")
    ax.tick_params(colors="#CBD5E1")
    ax.legend(facecolor="#181C22", labelcolor="#F8FAFC")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="FireGuard AI — Phase 14 Test Evaluation")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=Path,
        default=[config.MODEL_PATH],
        help="One or more trained .pt weight files to evaluate",
    )
    parser.add_argument("--data", type=Path, default=config.DATA_YAML_PATH, help="Path to data.yaml")
    parser.add_argument("--video", type=Path, default=None, help="Optional annotated test video for smoke delay")
    parser.add_argument("--smoke-onset", type=float, default=0.0, help="Ground-truth smoke onset seconds in --video")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for weight_file in args.weights:
        res = evaluate_yolo_on_test_split(weight_file, args.data)
        if args.video is not None:
            res["video_smoke_delay"] = measure_video_smoke_delay(weight_file, args.video, args.smoke_onset)
        results.append(res)

    chart_path = generate_comparison_chart(results, config.OUTPUTS_DIR / "model_comparison.png")
    payload = {
        "primary_evaluation": results[0],
        "model_comparisons": results,
        "comparison_chart_path": str(chart_path),
    }
    eval_file = config.MODELS_DIR / "evaluation_results.json"
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    db = IncidentDatabase(config.DB_PATH)
    primary = results[0]
    db.insert_model_evaluation_summary(
        model_identifier=primary["model_name"],
        dataset_split="test",
        sample_count=primary["test_images_evaluated"],
        precision=primary["precision"],
        recall=primary["recall"],
        f1_score=primary["f1_score"],
        map50=primary["map50"],
        map50_95=primary["map50_95"],
        false_alarm_rate=primary["false_alarm_evaluation"].get("enhanced_fpr"),
        avg_inference_ms=primary["avg_inference_ms"],
        avg_fps=primary["fps"],
        details_json=json.dumps(primary.get("per_class", {})),
    )
    print(f"Saved actual evaluation outputs to: {eval_file}")


if __name__ == "__main__":
    main()
