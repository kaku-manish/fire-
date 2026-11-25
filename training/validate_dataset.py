"""Phase 3 — Data Preprocessing, Training-Only Augmentations, and data.yaml Generation.

Implements image validation, YOLO-compatible letterboxing, pixel normalization,
bounding-box coordinate conversions, conservative training-only augmentations
(horizontal flip, brightness/contrast variation, scaling, 4-up mosaic, controlled
Gaussian noise), class imbalance analysis, preview visualizations, and data.yaml export.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from training.prepare_dataset import (  # noqa: E402
    DFIRE_CLASS_NAMES,
    VALID_IMAGE_EXTS,
    parse_and_validate_yolo_label,
    validate_single_image,
)

YoloBox = tuple[int, float, float, float, float]


def letterbox_image(
    image_bgr: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize an image preserving aspect ratio and pad to target_size (YOLO letterbox)."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot letterbox an empty image.")

    orig_h, orig_w = image_bgr.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return canvas, scale, (pad_w, pad_h)


def normalize_image(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 image to RGB float32 normalized to [0.0, 1.0]."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot normalize an empty image.")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def yolo_to_xyxy(box: YoloBox, img_w: int, img_h: int) -> tuple[int, float, float, float, float]:
    """Convert normalized YOLO (cls, cx, cy, w, h) to pixel (cls, x1, y1, x2, y2)."""
    cls_id, cx, cy, bw, bh = box
    x1 = max(0.0, (cx - bw / 2.0) * img_w)
    y1 = max(0.0, (cy - bh / 2.0) * img_h)
    x2 = min(float(img_w), (cx + bw / 2.0) * img_w)
    y2 = min(float(img_h), (cy + bh / 2.0) * img_h)
    return cls_id, x1, y1, x2, y2


def xyxy_to_yolo(
    cls_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    min_box_px: float = 2.0,
) -> YoloBox | None:
    """Convert pixel (cls, x1, y1, x2, y2) back to clamped normalized YOLO format."""
    cx1 = max(0.0, min(float(img_w), x1))
    cy1 = max(0.0, min(float(img_h), y1))
    cx2 = max(0.0, min(float(img_w), x2))
    cy2 = max(0.0, min(float(img_h), y2))
    bw_px = cx2 - cx1
    bh_px = cy2 - cy1
    if bw_px < min_box_px or bh_px < min_box_px:
        return None
    cx = ((cx1 + cx2) / 2.0) / img_w
    cy = ((cy1 + cy2) / 2.0) / img_h
    bw = bw_px / img_w
    bh = bh_px / img_h
    return (
        int(cls_id),
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip(bw, 1e-4, 1.0)),
        float(np.clip(bh, 1e-4, 1.0)),
    )


def flip_horizontal_with_boxes(
    image_bgr: np.ndarray, boxes: list[YoloBox]
) -> tuple[np.ndarray, list[YoloBox]]:
    """Horizontally flip image and mirror normalized X-center bounding-box coordinates."""
    flipped = cv2.flip(image_bgr, 1)
    flipped_boxes: list[YoloBox] = [
        (cls_id, float(np.clip(1.0 - cx, 0.0, 1.0)), cy, bw, bh)
        for cls_id, cx, cy, bw, bh in boxes
    ]
    return flipped, flipped_boxes


def adjust_brightness_contrast(
    image_bgr: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """Apply conservative brightness (beta) and contrast (alpha) variation."""
    alpha_clamped = float(np.clip(alpha, 0.75, 1.25))
    beta_clamped = float(np.clip(beta, -25.0, 25.0))
    return cv2.convertScaleAbs(image_bgr, alpha=alpha_clamped, beta=beta_clamped)


def add_controlled_noise(
    image_bgr: np.ndarray,
    sigma: float = 8.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add mild Gaussian sensor noise without distorting flame/smoke colors."""
    generator = rng or np.random.default_rng(config.RANDOM_SEED)
    noise = generator.normal(0.0, min(max(sigma, 0.0), 20.0), image_bgr.shape)
    noisy = np.clip(image_bgr.astype(np.float32) + noise, 0.0, 255.0)
    return noisy.astype(np.uint8)


def scale_image_with_boxes(
    image_bgr: np.ndarray,
    boxes: list[YoloBox],
    scale_factor: float = 1.15,
) -> tuple[np.ndarray, list[YoloBox]]:
    """Scale image around its center and remap YOLO bounding boxes accurately."""
    h, w = image_bgr.shape[:2]
    scale = float(np.clip(scale_factor, 0.70, 1.35))
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((h, w, 3), (114, 114, 114), dtype=np.uint8)
    offset_x = (w - new_w) // 2
    offset_y = (h - new_h) // 2

    if scale >= 1.0:
        crop_x = -offset_x
        crop_y = -offset_y
        canvas = resized[crop_y : crop_y + h, crop_x : crop_x + w].copy()
    else:
        canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

    new_boxes: list[YoloBox] = []
    for box in boxes:
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
        tx1 = x1 * scale + offset_x
        ty1 = y1 * scale + offset_y
        tx2 = x2 * scale + offset_x
        ty2 = y2 * scale + offset_y
        mapped = xyxy_to_yolo(cls_id, tx1, ty1, tx2, ty2, w, h)
        if mapped is not None:
            new_boxes.append(mapped)
    return canvas, new_boxes


def create_mosaic_4up(
    samples: list[tuple[np.ndarray, list[YoloBox]]],
    output_size: int = 640,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[YoloBox]]:
    """Build a 4-image YOLO mosaic and transform all bounding boxes into the canvas."""
    if len(samples) != 4:
        raise ValueError("Mosaic augmentation requires exactly 4 (image, boxes) samples.")
    generator = rng or np.random.default_rng(config.RANDOM_SEED)
    mosaic_img = np.full((output_size, output_size, 3), (114, 114, 114), dtype=np.uint8)
    xc = int(generator.integers(int(output_size * 0.35), int(output_size * 0.65)))
    yc = int(generator.integers(int(output_size * 0.35), int(output_size * 0.65)))

    quadrants = [
        (0, 0, xc, yc),
        (xc, 0, output_size, yc),
        (0, yc, xc, output_size),
        (xc, yc, output_size, output_size),
    ]
    mosaic_boxes: list[YoloBox] = []

    for (img, boxes), (qx1, qy1, qx2, qy2) in zip(samples, quadrants):
        qw = max(1, qx2 - qx1)
        qh = max(1, qy2 - qy1)
        tile = cv2.resize(img, (qw, qh), interpolation=cv2.INTER_LINEAR)
        mosaic_img[qy1:qy2, qx1:qx2] = tile

        for box in boxes:
            cls_id, bx1, by1, bx2, by2 = yolo_to_xyxy(box, qw, qh)
            mapped = xyxy_to_yolo(
                cls_id,
                bx1 + qx1,
                by1 + qy1,
                bx2 + qx1,
                by2 + qy1,
                output_size,
                output_size,
            )
            if mapped is not None:
                mosaic_boxes.append(mapped)

    return mosaic_img, mosaic_boxes


def augment_training_sample(
    image_bgr: np.ndarray,
    boxes: list[YoloBox],
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, list[YoloBox]]]:
    """Generate deterministic training-only augmented variants for a sample."""
    rng = np.random.default_rng(seed)
    letterboxed, _, _ = letterbox_image(image_bgr, (config.DEFAULT_IMAGE_SIZE, config.DEFAULT_IMAGE_SIZE))

    # Map boxes into letterboxed frame
    h, w = image_bgr.shape[:2]
    lb_img, scale, (pad_w, pad_h) = letterbox_image(
        image_bgr, (config.DEFAULT_IMAGE_SIZE, config.DEFAULT_IMAGE_SIZE)
    )
    lb_boxes: list[YoloBox] = []
    for box in boxes:
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
        mapped = xyxy_to_yolo(
            cls_id,
            x1 * scale + pad_w,
            y1 * scale + pad_h,
            x2 * scale + pad_w,
            y2 * scale + pad_h,
            config.DEFAULT_IMAGE_SIZE,
            config.DEFAULT_IMAGE_SIZE,
        )
        if mapped is not None:
            lb_boxes.append(mapped)

    flipped_img, flipped_boxes = flip_horizontal_with_boxes(lb_img, lb_boxes)
    bc_img = adjust_brightness_contrast(
        lb_img,
        alpha=float(rng.uniform(0.85, 1.15)),
        beta=float(rng.uniform(-16.0, 16.0)),
    )
    scaled_img, scaled_boxes = scale_image_with_boxes(lb_img, lb_boxes, scale_factor=1.15)
    noisy_img = add_controlled_noise(bc_img, sigma=7.0, rng=rng)

    return {
        "letterboxed": (lb_img, lb_boxes),
        "horizontal_flip": (flipped_img, flipped_boxes),
        "brightness_contrast_noise": (noisy_img, lb_boxes),
        "scaled_zoom": (scaled_img, scaled_boxes),
    }


def analyze_class_imbalance(
    dataset_dir: Path,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Analyze class instance counts and background ratio across train/valid/test splits."""
    classes = class_names or DFIRE_CLASS_NAMES
    valid_ids = set(classes.keys())
    summary: dict[str, Any] = {}

    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.is_dir():
            summary[split] = {"images": 0, "instances": {}, "background_images": 0}
            continue

        images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
        instance_counts: Counter[str] = Counter()
        bg_count = 0
        invalid_boxes = 0

        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            boxes, errs = parse_and_validate_yolo_label(lbl_path, valid_class_ids=valid_ids)
            if errs:
                invalid_boxes += len(errs)
                continue
            if not boxes:
                bg_count += 1
            for box in boxes:
                instance_counts[classes[box[0]]] += 1

        max_inst = max(instance_counts.values()) if instance_counts else 0
        min_inst = min(instance_counts.values()) if instance_counts else 0
        imbalance_ratio = round(max_inst / min_inst, 2) if min_inst > 0 else None

        summary[split] = {
            "images": len(images),
            "background_images": bg_count,
            "positive_images": len(images) - bg_count,
            "instances": {name: int(instance_counts.get(name, 0)) for name in classes.values()},
            "imbalance_ratio": imbalance_ratio,
            "invalid_label_lines": invalid_boxes,
        }
    return summary


def generate_data_yaml(
    dataset_dir: Path,
    class_names: dict[int, str] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create a YOLO-compatible data.yaml file reflecting actual class mappings."""
    classes = class_names or DFIRE_CLASS_NAMES
    yaml_path = output_path or (dataset_dir / "data.yaml")
    payload = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(classes),
        "names": {int(k): str(v) for k, v in sorted(classes.items())},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def _draw_yolo_boxes_preview(
    image_bgr: np.ndarray,
    boxes: list[YoloBox],
    class_names: dict[int, str],
) -> np.ndarray:
    """Draw labeled boxes onto an image copy for visual verification."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    for box in boxes:
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        color = config.CLASS_COLORS_BGR.get(cls_name.lower(), (0, 255, 255))
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            canvas,
            cls_name,
            (int(x1), max(16, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def generate_preprocessing_preview(
    samples: list[tuple[np.ndarray, list[YoloBox]]],
    output_path: Path,
    class_names: dict[int, str] | None = None,
) -> Path:
    """Render side-by-side original and training-augmented sample visualizations."""
    if not samples:
        raise ValueError("Need at least one image sample to build preprocessing preview.")
    classes = class_names or DFIRE_CLASS_NAMES
    base_img, base_boxes = samples[0]
    augmented = augment_training_sample(base_img, base_boxes, seed=config.RANDOM_SEED)

    panels: list[tuple[str, np.ndarray]] = [
        ("Original", _draw_yolo_boxes_preview(base_img, base_boxes, classes)),
        ("Letterboxed 640x640", _draw_yolo_boxes_preview(*augmented["letterboxed"], classes)),
        ("Horizontal Flip", _draw_yolo_boxes_preview(*augmented["horizontal_flip"], classes)),
        ("Brightness/Contrast/Noise", _draw_yolo_boxes_preview(*augmented["brightness_contrast_noise"], classes)),
        ("Scaled Zoom", _draw_yolo_boxes_preview(*augmented["scaled_zoom"], classes)),
    ]

    if len(samples) >= 4:
        mosaic_img, mosaic_boxes = create_mosaic_4up(samples[:4], output_size=config.DEFAULT_IMAGE_SIZE)
        panels.append(("4-Up Mosaic", _draw_yolo_boxes_preview(mosaic_img, mosaic_boxes, classes)))

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    fig.patch.set_facecolor("#111418")
    axes_flat = np.array(axes).reshape(-1)

    for ax in axes_flat:
        ax.axis("off")

    for ax, (title, img_bgr) in zip(axes_flat, panels):
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="#F8FAFC", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="FireGuard AI — Phase 3 Dataset Validation & Preprocessing")
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR, help="Root dataset directory")
    parser.add_argument(
        "--preview-out",
        type=Path,
        default=config.OUTPUTS_DIR / "preprocessing_augmentation_preview.png",
        help="Output path for augmentation preview figure",
    )
    args = parser.parse_args()

    yaml_path = generate_data_yaml(args.dataset_dir, DFIRE_CLASS_NAMES)
    imbalance = analyze_class_imbalance(args.dataset_dir, DFIRE_CLASS_NAMES)

    print(f"Generated data.yaml : {yaml_path}")
    print("Class imbalance analysis:")
    print(json.dumps(imbalance, indent=2))

    # Load up to 4 valid training samples for augmentation visualization
    train_img_dir = args.dataset_dir / "train" / "images"
    train_lbl_dir = args.dataset_dir / "train" / "labels"
    samples: list[tuple[np.ndarray, list[YoloBox]]] = []
    if train_img_dir.is_dir():
        for img_path in sorted(train_img_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_IMAGE_EXTS:
                continue
            ok, _, _ = validate_single_image(img_path)
            if not ok:
                continue
            boxes, errs = parse_and_validate_yolo_label(train_lbl_dir / f"{img_path.stem}.txt")
            if errs:
                continue
            img = cv2.imread(str(img_path))
            if img is not None:
                samples.append((img, boxes))
            if len(samples) == 4:
                break

    if samples:
        preview_file = generate_preprocessing_preview(samples, args.preview_out, DFIRE_CLASS_NAMES)
        print(f"Augmentation preview: {preview_file}")
    else:
        print("No training images found yet; download/prepare D-Fire in Phase 2 to render augmentation previews.")


if __name__ == "__main__":
    main()
