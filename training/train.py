"""Phase 4 — Deep Learning Model Training (YOLO11 Transfer Learning).

Trains YOLO11n (baseline) and optionally YOLO11s / YOLOv8n on the prepared
fire and smoke dataset using Ultralytics transfer learning.

Note: Hyperparameters in config.py represent initial configurations, not
guaranteed optimal settings. All reported metrics come strictly from actual
validation and test runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from training.prepare_dataset import VALID_IMAGE_EXTS  # noqa: E402


def detect_training_device() -> str:
    """Automatically select CUDA GPU when available, else CPU."""
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def select_batch_size(requested_batch: int, device: str) -> int:
    """Select an appropriate batch size based on CUDA memory or CPU availability."""
    if device != "cpu":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 6.0:
                return min(requested_batch, 8)
            if vram_gb < 10.0:
                return min(requested_batch, 16)
            return requested_batch
        except Exception:
            return min(requested_batch, 16)
    return min(requested_batch, 8)


def validate_training_prerequisites(data_yaml: Path) -> dict[str, Any]:
    """Verify data.yaml and non-empty train/valid image splits before training."""
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found at {data_yaml}. "
            "Run Phase 2/3 dataset preparation first."
        )
    data_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root_path = Path(data_cfg.get("path", data_yaml.parent))
    train_dir = root_path / str(data_cfg.get("train", "train/images"))
    val_dir = root_path / str(data_cfg.get("val", "valid/images"))

    train_count = (
        sum(1 for p in train_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS)
        if train_dir.is_dir()
        else 0
    )
    val_count = (
        sum(1 for p in val_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS)
        if val_dir.is_dir()
        else 0
    )
    if train_count == 0 or val_count == 0:
        raise ValueError(
            f"Dataset splits are empty (train={train_count}, valid={val_count}). "
            "Download and prepare D-Fire in dataset/ before starting training."
        )
    return {"train_images": train_count, "valid_images": val_count, "names": data_cfg.get("names", {})}


def plot_training_curves(results_csv: Path, output_path: Path) -> Path | None:
    """Plot actual training/validation losses and precision/recall/mAP curves."""
    if not results_csv.is_file():
        return None
    df = pd.read_csv(results_csv)
    df.columns = [col.strip() for col in df.columns]
    if df.empty or "epoch" not in df.columns:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor("#111418")

    # Panel 1: Box & classification loss
    for col, label, color in [
        ("train/box_loss", "Train Box Loss", "#EF4444"),
        ("val/box_loss", "Val Box Loss", "#F97316"),
        ("train/cls_loss", "Train Cls Loss", "#3B82F6"),
        ("val/cls_loss", "Val Cls Loss", "#A855F7"),
    ]:
        if col in df.columns:
            axes[0].plot(df["epoch"], df[col], label=label, color=color, linewidth=2)
    axes[0].set_title("Training & Validation Losses", color="#F8FAFC")
    axes[0].set_xlabel("Epoch", color="#CBD5E1")
    axes[0].set_ylabel("Loss", color="#CBD5E1")
    axes[0].legend(facecolor="#181C22", labelcolor="#F8FAFC")
    axes[0].set_facecolor("#181C22")
    axes[0].tick_params(colors="#CBD5E1")

    # Panel 2: Precision, Recall, mAP@50, mAP@50-95
    for col, label, color in [
        ("metrics/precision(B)", "Precision", "#22C55E"),
        ("metrics/recall(B)", "Recall", "#EAB308"),
        ("metrics/mAP50(B)", "mAP@50", "#38BDF8"),
        ("metrics/mAP50-95(B)", "mAP@50-95", "#F97316"),
    ]:
        if col in df.columns:
            axes[1].plot(df["epoch"], df[col], label=label, color=color, linewidth=2)
    axes[1].set_title("Validation Precision, Recall & mAP", color="#F8FAFC")
    axes[1].set_xlabel("Epoch", color="#CBD5E1")
    axes[1].set_ylabel("Score", color="#CBD5E1")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(facecolor="#181C22", labelcolor="#F8FAFC")
    axes[1].set_facecolor("#181C22")
    axes[1].tick_params(colors="#CBD5E1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def extract_best_epoch_metrics(results_csv: Path) -> dict[str, Any]:
    """Read real best-epoch metrics from Ultralytics results.csv."""
    if not results_csv.is_file():
        return {}
    df = pd.read_csv(results_csv)
    df.columns = [col.strip() for col in df.columns]
    if df.empty:
        return {}
    score_col = "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in df.columns else df.columns[-1]
    best_row = df.loc[df[score_col].idxmax()]
    precision = float(best_row.get("metrics/precision(B)", 0.0))
    recall = float(best_row.get("metrics/recall(B)", 0.0))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "best_epoch": int(best_row.get("epoch", 0)),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "map50": round(float(best_row.get("metrics/mAP50(B)", 0.0)), 4),
        "map50_95": round(float(best_row.get("metrics/mAP50-95(B)", 0.0)), 4),
    }


def train_single_model(
    base_model: str,
    data_yaml: Path,
    epochs: int = config.DEFAULT_EPOCHS,
    imgsz: int = config.DEFAULT_IMAGE_SIZE,
    batch: int = config.DEFAULT_BATCH_SIZE,
    optimizer: str = config.DEFAULT_OPTIMIZER,
    patience: int = config.DEFAULT_PATIENCE,
    seed: int = config.RANDOM_SEED,
    project_dir: Path | None = None,
    run_name: str | None = None,
    copy_as_best: bool = True,
    export_onnx: bool = False,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Run Ultralytics YOLO transfer learning and persist checkpoint & training logs."""
    dataset_info = validate_training_prerequisites(data_yaml)
    device = detect_training_device()
    effective_batch = select_batch_size(batch, device)
    out_project = project_dir or (config.BASE_DIR / "runs" / "detect")
    exp_name = run_name or (f"fireguard_{Path(base_model).stem}_smoke" if smoke_test else f"fireguard_{Path(base_model).stem}")

    warmup_epochs = round(min(3.0, max(0.5, epochs * 0.15)), 2)

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=effective_batch,
        optimizer=optimizer,
        patience=patience,
        warmup_epochs=warmup_epochs,
        cos_lr=True,
        seed=seed,
        device=device,
        val=True,
        save=True,
        plots=True,
        project=str(out_project),
        name=exp_name,
        exist_ok=True,
    )

    save_dir = Path(getattr(results, "save_dir", out_project / exp_name))
    best_ckpt = save_dir / "weights" / "best.pt"
    results_csv = save_dir / "results.csv"
    curves_path = plot_training_curves(results_csv, config.MODELS_DIR / f"{exp_name}_curves.png")
    best_metrics = extract_best_epoch_metrics(results_csv)

    copied_weights: str | None = None
    if copy_as_best and best_ckpt.is_file():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, config.MODEL_PATH)
        copied_weights = str(config.MODEL_PATH)
        if smoke_test:
            shutil.copy2(best_ckpt, config.MODELS_DIR / "smoke_test_best.pt")

        model_tier = "SMOKE_TEST" if smoke_test else "FULL_EVALUATED"
        tier_note = (
            f"CPU/subset smoke-test weights trained for {epochs} epochs on {dataset_info['train_images']} "
            "D-Fire training images for pipeline verification. Run notebooks/model_training.ipynb "
            "on Google Colab GPU (50 epochs, full D-Fire) for final production evaluation."
            if smoke_test
            else f"Full model trained for {epochs} epochs on {dataset_info['train_images']} training images."
        )
        metadata_payload = {
            "model_tier": model_tier,
            "base_model": base_model,
            "epochs": epochs,
            "imgsz": imgsz,
            "device": device,
            "train_images": dataset_info["train_images"],
            "valid_images": dataset_info["valid_images"],
            "classes": dataset_info["names"],
            "note": tier_note,
        }
        (config.MODELS_DIR / "model_metadata.json").write_text(
            json.dumps(metadata_payload, indent=2), encoding="utf-8"
        )

    onnx_path: str | None = None
    if export_onnx and best_ckpt.is_file():
        trained_model = YOLO(str(best_ckpt))
        exported = trained_model.export(format="onnx", imgsz=imgsz)
        onnx_path = str(exported)

    summary = {
        "base_model": base_model,
        "device": device,
        "epochs_configured": epochs,
        "imgsz": imgsz,
        "batch": effective_batch,
        "optimizer": optimizer,
        "patience": patience,
        "seed": seed,
        "dataset_info": dataset_info,
        "run_dir": str(save_dir),
        "best_checkpoint": str(best_ckpt),
        "installed_model_path": copied_weights,
        "results_csv": str(results_csv),
        "confusion_matrix_path": str(save_dir / "confusion_matrix.png"),
        "training_curves_path": str(curves_path) if curves_path else None,
        "onnx_export_path": onnx_path,
        "validation_metrics": best_metrics,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="FireGuard AI — Phase 4 YOLO11 Training")
    parser.add_argument("--model", type=str, default=config.DEFAULT_BASE_MODEL, help="Base weights (e.g. yolo11n.pt)")
    parser.add_argument("--data", type=Path, default=config.DATA_YAML_PATH, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Training epochs (50-100 recommended)")
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMAGE_SIZE, help="Input image size")
    parser.add_argument("--batch", type=int, default=config.DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--optimizer", type=str, default=config.DEFAULT_OPTIMIZER, help="Optimizer")
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    parser.add_argument("--run-name", type=str, default=None, help="Custom experiment run directory name")
    parser.add_argument("--compare", action="store_true", help="Train and compare YOLO11n, YOLO11s, and YOLOv8n")
    parser.add_argument("--export-onnx", action="store_true", help="Export best model to ONNX format")
    parser.add_argument("--smoke-test", action="store_true", help="Mark weights explicitly as smoke-test validation weights")
    parser.add_argument(
        "--no-copy-as-best",
        action="store_true",
        help="Keep best.pt inside runs/detect/<run> without overwriting models/best.pt until evaluated",
    )
    args = parser.parse_args()

    models_to_train = list(config.COMPARISON_MODELS) if args.compare else [args.model]
    summaries: list[dict[str, Any]] = []

    for idx, model_name in enumerate(models_to_train):
        print(f"\n=== Training {model_name} (seed={args.seed}, imgsz={args.imgsz}) ===")
        summary = train_single_model(
            base_model=model_name,
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer=args.optimizer,
            patience=args.patience,
            seed=args.seed,
            run_name=args.run_name if idx == 0 else None,
            copy_as_best=(idx == 0) and (not args.no_copy_as_best),
            export_onnx=args.export_onnx and (idx == 0) and (not args.no_copy_as_best),
            smoke_test=args.smoke_test,
        )
        summaries.append(summary)

    summary_file = config.MODELS_DIR / "training_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summaries if len(summaries) > 1 else summaries[0], indent=2), encoding="utf-8")
    print(f"\nSaved training summary to: {summary_file}")


if __name__ == "__main__":
    main()
