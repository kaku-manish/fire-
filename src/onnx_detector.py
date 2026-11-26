"""Lightweight ONNX Runtime detector matching FireSmokeDetector interface without PyTorch."""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from src.detector import (
    Detection,
    DetectionResult,
    ModelNotFoundError,
    normalize_class_label,
    summarize_detections,
)


def _letterbox_ultralytics(
    frame_bgr: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and pad an image identically to Ultralytics YOLO preprocessing."""
    orig_h, orig_w = frame_bgr.shape[:2]
    target_w, target_h = int(target_size[0]), int(target_size[1])
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = (
        cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if (orig_w, orig_h) != (new_w, new_h)
        else frame_bgr
    )
    dw = (target_w - new_w) / 2.0
    dh = (target_h - new_h) / 2.0
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )
    return padded, scale, (left, top)


class OnnxFireSmokeDetector:
    """CPU memory-efficient YOLO11 ONNX Runtime detector for cloud API inference."""

    def __init__(self, model_path: Path | str | None = None, intra_threads: int = 2) -> None:
        self.model_path = Path(model_path) if model_path is not None else (config.MODELS_DIR / "best.onnx")
        self._intra_threads = max(1, int(intra_threads))
        self._session: Any = None
        self._input_name: str = "images"
        self._output_name: str = "output0"
        self._input_hw: tuple[int, int] = (640, 640)
        self._class_names: dict[int, str] = {0: "smoke", 1: "fire"}

    def ensure_model_exists(self) -> Path:
        if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            raise ModelNotFoundError(
                f"Trained ONNX model weights not found at '{self.model_path}'."
            )
        return self.model_path

    def load_model(self) -> Any:
        if self._session is not None:
            return self._session
        self.ensure_model_exists()
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._intra_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        if len(inp.shape) == 4 and isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
            self._input_hw = (int(inp.shape[2]), int(inp.shape[3]))
        self._output_name = self._session.get_outputs()[0].name

        meta_map = self._session.get_modelmeta().custom_metadata_map or {}
        if "names" in meta_map:
            try:
                raw_names = ast.literal_eval(meta_map["names"])
                if isinstance(raw_names, dict):
                    self._class_names = {
                        int(k): normalize_class_label(str(v)) for k, v in raw_names.items()
                    }
                elif isinstance(raw_names, (list, tuple)):
                    self._class_names = {
                        idx: normalize_class_label(str(v)) for idx, v in enumerate(raw_names)
                    }
            except Exception:
                pass
        return self._session

    @property
    def class_names(self) -> dict[int, str]:
        if self._session is None:
            self.load_model()
        return dict(self._class_names)

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
        imgsz: int = config.DEFAULT_IMAGE_SIZE,
    ) -> DetectionResult:
        del imgsz  # Static ONNX input uses exported model dimensions (640x640)
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame is empty or invalid.")

        session = self.load_model()
        orig_h, orig_w = frame_bgr.shape[:2]
        in_h, in_w = self._input_hw

        t0 = time.perf_counter()
        letterboxed, scale, (pad_left, pad_top) = _letterbox_ultralytics(
            frame_bgr, target_size=(in_w, in_h)
        )
        rgb = letterboxed[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...], dtype=np.float32)

        raw_out = session.run([self._output_name], {self._input_name: blob})[0]
        preds = raw_out[0].T  # (8400, 4 + nc)

        class_scores = preds[:, 4:]
        cls_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(preds.shape[0]), cls_ids]
        keep = confs >= float(conf_threshold)

        detections: list[Detection] = []
        if np.any(keep):
            kept_preds = preds[keep]
            kept_cls = cls_ids[keep]
            kept_confs = confs[keep]

            # Convert center-wh in letterbox coordinates back to original xyxy
            cx = kept_preds[:, 0]
            cy = kept_preds[:, 1]
            bw = kept_preds[:, 2]
            bh = kept_preds[:, 3]
            x1_arr = np.clip((cx - bw * 0.5 - pad_left) / scale, 0.0, float(orig_w))
            y1_arr = np.clip((cy - bh * 0.5 - pad_top) / scale, 0.0, float(orig_h))
            x2_arr = np.clip((cx + bw * 0.5 - pad_left) / scale, 0.0, float(orig_w))
            y2_arr = np.clip((cy + bh * 0.5 - pad_top) / scale, 0.0, float(orig_h))

            # Class-aware NMS matching Ultralytics default behavior
            candidates: list[Detection] = []
            for cls_id in np.unique(kept_cls):
                cls_mask = kept_cls == cls_id
                boxes_xywh = np.stack(
                    [
                        x1_arr[cls_mask],
                        y1_arr[cls_mask],
                        np.maximum(0.0, x2_arr[cls_mask] - x1_arr[cls_mask]),
                        np.maximum(0.0, y2_arr[cls_mask] - y1_arr[cls_mask]),
                    ],
                    axis=1,
                ).tolist()
                cls_confs = kept_confs[cls_mask].tolist()
                indices = cv2.dnn.NMSBoxes(
                    boxes_xywh,
                    cls_confs,
                    float(conf_threshold),
                    float(iou_threshold),
                )
                if len(indices) == 0:
                    continue
                cls_indices = np.where(cls_mask)[0]
                for idx in np.asarray(indices).reshape(-1):
                    g_idx = int(cls_indices[int(idx)])
                    x1 = float(x1_arr[g_idx])
                    y1 = float(y1_arr[g_idx])
                    x2 = float(x2_arr[g_idx])
                    y2 = float(y2_arr[g_idx])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    c_id = int(kept_cls[g_idx])
                    c_name = self._class_names.get(c_id, f"class_{c_id}")
                    area_px = (x2 - x1) * (y2 - y1)
                    norm_box = (x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h)
                    candidates.append(
                        Detection(
                            class_id=c_id,
                            class_name=c_name,
                            confidence=round(float(kept_confs[g_idx]), 4),
                            bbox_xyxy=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                            bbox_normalized=tuple(round(v, 5) for v in norm_box),  # type: ignore[arg-type]
                            area_px=round(area_px, 2),
                        )
                    )
            detections = sorted(candidates, key=lambda d: d.confidence, reverse=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return summarize_detections(
            detections=detections,
            image_width=orig_w,
            image_height=orig_h,
            inference_time_ms=elapsed_ms,
            model_path=str(self.model_path),
        )
