"""Steps 4, 8, 9, 11 — Live Audit Engine, Class-Aware IoU Validation & Visual Galleries.

Provides:
- Class-aware bounding-box IoU matching against YOLO ground-truth annotations
- Dataset & built-in sample discovery across 5 scenario categories
- Single-image and batch automated dataset audits (PASS / FAIL / NOT EVALUATED)
- Standardized evaluation chart generation (PR curve, confidence distribution,
  per-class precision/recall, ground-truth vs prediction gallery, success/failure galleries)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.detector import Detection, FireSmokeDetector, draw_detections, normalize_class_label
from src.severity import SeverityAssessment, estimate_visual_severity
from training.prepare_dataset import DFIRE_CLASS_NAMES, VALID_IMAGE_EXTS, parse_and_validate_yolo_label
from training.validate_dataset import yolo_to_xyxy

AUDIT_CATEGORIES: tuple[str, ...] = (
    "All",
    "Fire only",
    "Smoke only",
    "Fire and smoke",
    "Normal environment",
    "Hard-negative examples",
)

AUDIT_PRESETS: dict[str, int | None] = {
    "Quick audit (up to 20 images)": 20,
    "Standard audit (up to 100 images)": 100,
    "Extended audit (up to 500 images)": 500,
    "Full test dataset": None,
}


@dataclass
class GroundTruthBox:
    """Pixel-space ground-truth bounding box parsed from YOLO .txt labels."""

    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class IoUMatchResult:
    """Result of greedy class-aware IoU matching between predictions and ground truth."""

    true_positives: int
    false_positives: int
    false_negatives: int
    mean_matched_iou: float
    max_matched_iou: float
    matched_pairs: list[tuple[Detection, GroundTruthBox, float]] = field(default_factory=list)
    audit_status: str = "NOT EVALUATED"


@dataclass
class ImageAuditRow:
    """Complete single-image audit record for Mode B, Mode C, Automated Audit & Auto Demo."""

    image_path: str
    image_name: str
    category: str
    ground_truth_class: str
    predicted_class: str
    matched_detections: int
    false_positives: int
    false_negatives: int
    confidence: float
    iou: float
    inference_time_ms: float
    fire_coverage_pct: float
    smoke_coverage_pct: float
    visual_severity: str
    audit_status: str
    gt_count: int
    pred_count: int
    annotated_path: str = ""
    comparison_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_iou_xyxy(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union (IoU) between two pixel (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0.0 else 0.0


def load_ground_truth_boxes(
    label_path: Path | None,
    img_width: int,
    img_height: int,
    class_names: dict[int, str] | None = None,
) -> list[GroundTruthBox] | None:
    """Load YOLO ground-truth boxes in pixel xyxy space; returns None if label is unavailable."""
    if label_path is None or not label_path.is_file():
        return None
    names = class_names or DFIRE_CLASS_NAMES
    raw_boxes, errors = parse_and_validate_yolo_label(label_path, valid_class_ids=set(names.keys()))
    if errors:
        return None

    gt_boxes: list[GroundTruthBox] = []
    for box in raw_boxes:
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box, img_width, img_height)
        cls_name = normalize_class_label(names.get(cls_id, f"class_{cls_id}"))
        gt_boxes.append(
            GroundTruthBox(
                class_id=cls_id,
                class_name=cls_name,
                bbox_xyxy=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
            )
        )
    return gt_boxes


def match_predictions_to_ground_truth(
    predictions: list[Detection],
    ground_truth: list[GroundTruthBox] | None,
    iou_threshold: float = 0.5,
) -> IoUMatchResult:
    """Perform greedy highest-confidence class-aware IoU matching."""
    if ground_truth is None:
        return IoUMatchResult(
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            mean_matched_iou=0.0,
            max_matched_iou=0.0,
            matched_pairs=[],
            audit_status="NOT EVALUATED",
        )

    sorted_preds = sorted(predictions, key=lambda d: d.confidence, reverse=True)
    matched_gt_indices: set[int] = set()
    matched_pairs: list[tuple[Detection, GroundTruthBox, float]] = []
    ious: list[float] = []
    false_positives = 0

    for pred in sorted_preds:
        best_iou = -1.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt_indices:
                continue
            if pred.class_name != gt.class_name:
                continue
            iou_val = compute_iou_xyxy(pred.bbox_xyxy, gt.bbox_xyxy)
            if iou_val >= iou_threshold and iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matched_gt_indices.add(best_gt_idx)
            matched_pairs.append((pred, ground_truth[best_gt_idx], round(best_iou, 4)))
            ious.append(best_iou)
        else:
            false_positives += 1

    true_positives = len(matched_pairs)
    false_negatives = len(ground_truth) - true_positives
    mean_iou = round(float(sum(ious) / len(ious)), 4) if ious else 0.0
    max_iou = round(float(max(ious)), 4) if ious else 0.0

    audit_status = "PASS" if (false_positives == 0 and false_negatives == 0) else "FAIL"
    return IoUMatchResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        mean_matched_iou=mean_iou,
        max_matched_iou=max_iou,
        matched_pairs=matched_pairs,
        audit_status=audit_status,
    )


def _summarize_classes(classes: list[str]) -> str:
    unique = set(classes)
    if not unique:
        return "None"
    if unique == {"fire"}:
        return "Fire"
    if unique == {"smoke"}:
        return "Smoke"
    if {"fire", "smoke"}.issubset(unique):
        return "Fire + Smoke"
    return ", ".join(sorted(unique))


def infer_image_category(image_path: Path, gt_boxes: list[GroundTruthBox] | None) -> str:
    """Categorize an image into one of the 5 standard audit categories."""
    parent_name = image_path.parent.name.lower()
    if parent_name == "hard_negatives":
        return "Hard-negative examples"
    if parent_name == "normal":
        return "Normal environment"
    if parent_name == "fire":
        return "Fire only"
    if parent_name == "smoke":
        return "Smoke only"
    if parent_name == "fire_and_smoke":
        return "Fire and smoke"

    if gt_boxes is None:
        return "Unlabeled"
    cls_set = {b.class_name for b in gt_boxes}
    if not cls_set:
        if image_path.name.startswith("WEB"):
            return "Hard-negative examples"
        return "Normal environment"
    if cls_set == {"fire"}:
        return "Fire only"
    if cls_set == {"smoke"}:
        return "Smoke only"
    return "Fire and smoke"


def find_matching_label_file(image_path: Path) -> Path | None:
    """Resolve YOLO .txt label file in sibling path or parallel labels/ folder."""
    candidates = [
        image_path.with_suffix(".txt"),
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def discover_audit_images(
    dataset_dir: Path | None = None,
    samples_dir: Path | None = None,
    split: str = "test",
    category: str = "All",
    include_samples: bool = True,
) -> list[dict[str, Any]]:
    """Discover labeled and sample images filtered by audit category."""
    ds_root = dataset_dir or config.DATASET_DIR
    smp_root = samples_dir or (config.BASE_DIR / "samples")
    items: list[dict[str, Any]] = []
    seen_resolved: set[str] = set()

    search_roots: list[Path] = []
    if split == "all_splits":
        for sp in ("test", "valid", "train"):
            search_roots.append(ds_root / sp / "images")
    else:
        search_roots.append(ds_root / split / "images")

    if include_samples and (smp_root / "images").is_dir():
        search_roots.append(smp_root / "images")

    for root in search_roots:
        if not root.is_dir():
            continue
        for img_path in sorted(root.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in VALID_IMAGE_EXTS:
                continue
            key = str(img_path.resolve())
            if key in seen_resolved:
                continue
            seen_resolved.add(key)

            lbl_path = find_matching_label_file(img_path)
            gt_boxes: list[GroundTruthBox] | None = None
            if lbl_path is not None:
                img = cv2.imread(str(img_path))
                if img is not None:
                    gt_boxes = load_ground_truth_boxes(lbl_path, img.shape[1], img.shape[0])
            cat = infer_image_category(img_path, gt_boxes)
            if category != "All" and cat != category:
                continue
            items.append(
                {
                    "image_path": img_path,
                    "label_path": lbl_path,
                    "category": cat,
                }
            )
    return items


def discover_sample_videos(samples_dir: Path | None = None) -> list[Path]:
    """List prerecorded sample videos available under samples/videos/."""
    smp_root = (samples_dir or (config.BASE_DIR / "samples")) / "videos"
    if not smp_root.is_dir():
        return []
    return sorted(
        p for p in smp_root.iterdir() if p.is_file() and p.suffix.lower() in config.SUPPORTED_VIDEO_EXTENSIONS
    )


def draw_ground_truth_and_predictions(
    frame_bgr: np.ndarray,
    ground_truth: list[GroundTruthBox] | None,
    predictions: list[Detection],
) -> np.ndarray:
    """Create a side-by-side comparison canvas: Ground Truth (left) vs Predictions (right)."""
    h, w = frame_bgr.shape[:2]
    left = frame_bgr.copy()
    if ground_truth is not None:
        for gt in ground_truth:
            x1, y1, x2, y2 = (int(round(v)) for v in gt.bbox_xyxy)
            cv2.rectangle(left, (x1, y1), (x2, y2), (34, 197, 94), 2)
            cv2.putText(
                left,
                f"GT:{gt.class_name.upper()}",
                (max(0, x1), max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (34, 197, 94),
                2,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(left, "No GT Labels", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    right = draw_detections(frame_bgr, predictions)
    cv2.putText(left, "Ground Truth", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(right, "YOLO11 Prediction", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return np.hstack([left, right])


def audit_single_image(
    image_path: Path,
    detector: FireSmokeDetector,
    label_path: Path | None = None,
    conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = 0.5,
    category: str | None = None,
    save_dir: Path | None = None,
) -> tuple[ImageAuditRow, np.ndarray, np.ndarray, SeverityAssessment, list[Detection], list[GroundTruthBox] | None]:
    """Run real YOLO inference on a single image and evaluate against GT if available."""
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError(f"Failed to read image '{image_path}'.")
    h, w = frame_bgr.shape[:2]

    resolved_lbl = label_path if label_path is not None else find_matching_label_file(image_path)
    gt_boxes = load_ground_truth_boxes(resolved_lbl, w, h)
    resolved_cat = category or infer_image_category(image_path, gt_boxes)

    det_res = detector.predict_frame(frame_bgr, conf_threshold=conf_threshold)
    severity = estimate_visual_severity(det_res.detections, w, h)
    match_res = match_predictions_to_ground_truth(det_res.detections, gt_boxes, iou_threshold=iou_threshold)

    annotated_bgr = draw_detections(frame_bgr, det_res.detections)
    comparison_bgr = draw_ground_truth_and_predictions(frame_bgr, gt_boxes, det_res.detections)

    annotated_path_str = ""
    comparison_path_str = ""
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        uid = uuid.uuid4().hex[:6]
        ann_file = save_dir / f"{image_path.stem}_{uid}_pred.png"
        cmp_file = save_dir / f"{image_path.stem}_{uid}_gt_vs_pred.png"
        cv2.imwrite(str(ann_file), annotated_bgr)
        cv2.imwrite(str(cmp_file), comparison_bgr)
        annotated_path_str = str(ann_file)
        comparison_path_str = str(cmp_file)

    gt_label_str = "Not Available" if gt_boxes is None else _summarize_classes([g.class_name for g in gt_boxes])
    pred_label_str = _summarize_classes([d.class_name for d in det_res.detections])

    row = ImageAuditRow(
        image_path=str(image_path),
        image_name=image_path.name,
        category=resolved_cat,
        ground_truth_class=gt_label_str,
        predicted_class=pred_label_str,
        matched_detections=match_res.true_positives,
        false_positives=match_res.false_positives,
        false_negatives=match_res.false_negatives,
        confidence=det_res.max_confidence,
        iou=match_res.max_matched_iou if match_res.max_matched_iou > 0 else match_res.mean_matched_iou,
        inference_time_ms=det_res.inference_time_ms,
        fire_coverage_pct=severity.fire_coverage_pct,
        smoke_coverage_pct=severity.smoke_coverage_pct,
        visual_severity=severity.visual_severity,
        audit_status=match_res.audit_status,
        gt_count=len(gt_boxes) if gt_boxes is not None else 0,
        pred_count=len(det_res.detections),
        annotated_path=annotated_path_str,
        comparison_path=comparison_path_str,
    )
    return row, annotated_bgr, comparison_bgr, severity, det_res.detections, gt_boxes


def run_automated_dataset_audit(
    image_items: list[dict[str, Any]],
    detector: FireSmokeDetector,
    conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = 0.5,
    max_images: int | None = None,
    save_dir: Path | None = None,
    progress_callback: Callable[[float, int, int, ImageAuditRow], None] | None = None,
) -> dict[str, Any]:
    """Audit a selected subset of labeled images and compile aggregate metrics & galleries."""
    selected = image_items[:max_images] if max_images is not None else image_items
    total = len(selected)
    out_dir = save_dir or (config.OUTPUTS_DIR / "audit_samples")
    rows: list[ImageAuditRow] = []

    for idx, item in enumerate(selected):
        img_path = Path(item["image_path"])
        lbl_path = Path(item["label_path"]) if item.get("label_path") else None
        row, _, _, _, _, _ = audit_single_image(
            image_path=img_path,
            detector=detector,
            label_path=lbl_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            category=item.get("category"),
            save_dir=out_dir if idx < 40 else None,
        )
        rows.append(row)
        if progress_callback is not None:
            progress_callback((idx + 1) / max(1, total), idx + 1, total, row)

    evaluated = [r for r in rows if r.audit_status != "NOT EVALUATED"]
    total_tp = sum(r.matched_detections for r in evaluated)
    total_fp = sum(r.false_positives for r in evaluated)
    total_fn = sum(r.false_negatives for r in evaluated)

    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    neg_eval = [r for r in evaluated if r.gt_count == 0]
    neg_fp_images = sum(1 for r in neg_eval if r.pred_count > 0)
    neg_fpr = (neg_fp_images / len(neg_eval)) if neg_eval else None

    avg_ms = (sum(r.inference_time_ms for r in rows) / len(rows)) if rows else 0.0
    avg_fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
    pass_count = sum(1 for r in evaluated if r.audit_status == "PASS")
    fail_count = sum(1 for r in evaluated if r.audit_status == "FAIL")

    charts = generate_audit_visual_artifacts(rows, out_dir)
    summary = {
        "evaluated_sample_count": len(rows),
        "labeled_sample_count": len(evaluated),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "negative_images_evaluated": len(neg_eval),
        "negative_false_alarm_rate": round(neg_fpr, 4) if neg_fpr is not None else None,
        "avg_inference_ms": round(avg_ms, 2),
        "avg_inference_fps": round(avg_fps, 2),
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "rows": [r.to_dict() for r in rows],
        "artifacts": charts,
    }
    (out_dir / "latest_dataset_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def generate_audit_visual_artifacts(rows: list[ImageAuditRow], out_dir: Path) -> dict[str, str]:
    """Generate confidence distribution, per-class chart, and success/failure galleries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if not rows:
        return artifacts

    # 1. Confidence distribution histogram
    confs = [r.confidence for r in rows if r.pred_count > 0]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    fig.patch.set_facecolor("#111418")
    ax.set_facecolor("#181C22")
    if confs:
        ax.hist(confs, bins=10, range=(0.0, 1.0), color="#F97316", edgecolor="#0F1216")
    else:
        ax.text(0.5, 0.5, "No Positive Detections", ha="center", va="center", color="#94A3B8")
    ax.set_title("Prediction Confidence Distribution", color="#F8FAFC")
    ax.set_xlabel("Confidence", color="#CBD5E1")
    ax.set_ylabel("Images", color="#CBD5E1")
    ax.tick_params(colors="#CBD5E1")
    conf_path = out_dir / "confidence_distribution.png"
    plt.tight_layout()
    fig.savefig(conf_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    artifacts["confidence_distribution"] = str(conf_path)

    # 2. Success & Failure galleries from annotated comparison images
    for label, status_target, filename in [
        ("success_gallery", "PASS", "success_gallery.png"),
        ("failure_gallery", "FAIL", "failure_gallery.png"),
    ]:
        candidates = [
            Path(r.comparison_path)
            for r in rows
            if r.audit_status == status_target and r.comparison_path and Path(r.comparison_path).is_file()
        ][:4]
        if not candidates:
            continue
        fig_g, axes_g = plt.subplots(len(candidates), 1, figsize=(8.5, 2.8 * len(candidates)))
        fig_g.patch.set_facecolor("#111418")
        axes_list = [axes_g] if len(candidates) == 1 else list(axes_g)
        for ax_g, img_p in zip(axes_list, candidates):
            bgr = cv2.imread(str(img_p))
            if bgr is not None:
                ax_g.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ax_g.set_title(img_p.stem, color="#F8FAFC", fontsize=9)
            ax_g.axis("off")
        gal_path = out_dir / filename
        plt.tight_layout()
        fig_g.savefig(gal_path, dpi=130, facecolor=fig_g.get_facecolor())
        plt.close(fig_g)
        artifacts[label] = str(gal_path)

    return artifacts
