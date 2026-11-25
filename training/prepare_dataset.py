"""Phase 2 — Dataset Collection, Auditing, Deduplication, and Leakage-Free Splitting.

Supports preparing the primary D-Fire dataset
(https://github.com/gaia-solutions-on-demand/DFireDataset) and optionally the
HPWREN FIgLib wildfire smoke dataset (https://www.hpwren.ucsd.edu/FIgLib/) into
the YOLO directory layout under dataset/{train,valid,test}/{images,labels}.

All dataset statistics are computed strictly from actual files on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import struct
import subprocess
import sys
import threading
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402

# Default D-Fire class mapping (verified against DFireDataset repository)
DFIRE_CLASS_NAMES: dict[int, str] = {0: "smoke", 1: "fire"}
VALID_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DATASET_LICENSES: dict[str, dict[str, str]] = {
    "D-Fire": {
        "source": "https://github.com/gaia-solutions-on-demand/DFireDataset",
        "license": "CC0 1.0 / Open Academic Dataset (verify upstream LICENSE before redistribution)",
        "citation": (
            "de Venancio, P. V. A. B., Lisboa, A. C., & Barbosa, A. V. (2022). "
            "An automatic fire detection system based on deep convolutional neural networks "
            "for low-power, resource-constrained devices. Neural Computing and Applications."
        ),
    },
    "FIgLib": {
        "source": "https://www.hpwren.ucsd.edu/FIgLib/",
        "license": "HPWREN Data Use Policy (as-is, explicit HPWREN/UCSD attribution required)",
        "citation": (
            "Dewangan, A., Pande, Y., Braun, H.-W., et al. (2022). "
            "FIgLib & SmokeyNet: Dataset and Deep Learning Model for Real-Time Wildland "
            "Fire Smoke Detection. Remote Sensing, 14(4), 1007."
        ),
    },
}


@dataclass
class AuditIssue:
    """Represents a single dataset integrity or annotation validation issue."""

    file_path: str
    issue_type: str
    details: str


def check_dataset_licensing() -> dict[str, dict[str, str]]:
    """Return licensing and attribution requirements for D-Fire and FIgLib."""
    return DATASET_LICENSES


def download_file_with_progress(url: str, destination: Path, timeout: int = 60) -> Path:
    """Stream-download a dataset archive without loading it completely into RAM."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total_bytes = int(response.headers.get("content-length", 0))
    with destination.open("wb") as handle, tqdm(
        total=total_bytes or None,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {destination.name}",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                handle.write(chunk)
                pbar.update(len(chunk))
    return destination


def clone_dfire_metadata(target_dir: Path) -> Path:
    """Clone the D-Fire GitHub repository for scripts, docs, and license verification."""
    repo_url = "https://github.com/gaia-solutions-on-demand/DFireDataset.git"
    if (target_dir / ".git").is_dir():
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True)
    return target_dir


def extract_archive(archive_path: Path, extract_to: Path) -> Path:
    """Extract a downloaded ZIP archive safely (preventing path traversal)."""
    extract_to.mkdir(parents=True, exist_ok=True)
    target_root = extract_to.resolve()
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            dest_path = (extract_to / member.filename).resolve()
            if not str(dest_path).startswith(str(target_root)):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(extract_to)
    return extract_to


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Return SHA-256 hex digest for exact duplicate detection."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> str | None:
    """Compute a 64-bit perceptual difference hash (dHash) for near-duplicate checks."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if val else "0" for val in diff.flatten())
    return f"{int(bits, 2):016x}"


def validate_single_image(image_path: Path) -> tuple[bool, str | None, tuple[int, int] | None]:
    """Verify that an image file is readable, non-empty, and uncorrupted."""
    if not image_path.is_file() or image_path.stat().st_size == 0:
        return False, "Missing or zero-byte image file", None
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            width, height = img.size
        if width <= 0 or height <= 0:
            return False, f"Invalid dimensions: {width}x{height}", None
        cv_img = cv2.imread(str(image_path))
        if cv_img is None:
            return False, "OpenCV failed to decode image", None
        return True, None, (width, height)
    except (UnidentifiedImageError, OSError, ValueError, ModuleNotFoundError) as exc:
        return False, f"Corrupted image: {exc}", None


def parse_and_validate_yolo_label(
    label_path: Path,
    valid_class_ids: set[int] | None = None,
) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Parse and validate a YOLO .txt annotation file.

    An empty label file is valid in YOLO and indicates a negative/background sample.
    """
    valid_ids = valid_class_ids if valid_class_ids is not None else set(DFIRE_CLASS_NAMES.keys())
    boxes: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []

    if not label_path.is_file():
        return boxes, [f"Missing label file: {label_path.name}"]

    raw_text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw_text:
        return boxes, errors

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {line_no}: expected 5 values, got {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"Line {line_no}: non-numeric annotation values '{line}'")
            continue

        if class_id not in valid_ids:
            errors.append(f"Line {line_no}: invalid class_id {class_id} (expected {sorted(valid_ids)})")
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            errors.append(f"Line {line_no}: center ({cx:.4f}, {cy:.4f}) outside [0, 1]")
            continue
        if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            errors.append(f"Line {line_no}: width/height ({bw:.4f}, {bh:.4f}) must be in (0, 1]")
            continue
        boxes.append((class_id, cx, cy, bw, bh))

    return boxes, errors


def extract_sequence_group(image_path: Path) -> str:
    """Infer a sequence/video group key from filename to prevent temporal leakage.

    Examples:
        'cam01_frame_0042.jpg' -> 'cam01'
        '1720453551_-02400.jpg' -> '1720453551'
        'WEB00123.jpg' -> 'WEB00123'
    """
    stem = image_path.stem
    # Match HPWREN FIgLib pattern: <origin_ts>_<offset>
    figlib_match = re.match(r"^(\d+)_[+-]?\d+$", stem)
    if figlib_match:
        return figlib_match.group(1)
    # Match generic video/frame naming: <prefix>_frame_001, <prefix>_frame001, or <prefix>-001
    frame_match = re.match(r"^(.+?)(?:[_-](?:frame|frm|img|f)?[_-]?\d+)$", stem, flags=re.IGNORECASE)
    if frame_match and len(frame_match.group(1)) >= 3:
        return frame_match.group(1)
    return stem


def discover_image_label_pairs(root_dir: Path) -> list[tuple[Path, Path | None]]:
    """Recursively discover images and locate matching YOLO .txt label files."""
    pairs: list[tuple[Path, Path | None]] = []
    if not root_dir.exists():
        return pairs

    image_files = sorted(
        p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    )
    for img_path in image_files:
        candidates = [
            img_path.with_suffix(".txt"),
            img_path.parent.parent / "labels" / f"{img_path.stem}.txt",
        ]
        matched_label = next((cand for cand in candidates if cand.is_file()), None)
        pairs.append((img_path, matched_label))
    return pairs


def audit_and_deduplicate(
    source_dir: Path,
    class_names: dict[int, str] | None = None,
    remove_near_duplicates: bool = False,
) -> dict[str, Any]:
    """Audit images/labels on disk and flag corrupted, missing, or duplicate samples."""
    classes = class_names or DFIRE_CLASS_NAMES
    valid_ids = set(classes.keys())
    pairs = discover_image_label_pairs(source_dir)

    seen_sha256: dict[str, Path] = {}
    seen_dhash: dict[str, Path] = {}
    clean_records: list[dict[str, Any]] = []
    issues: list[AuditIssue] = []

    class_instance_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    duplicate_exact = 0
    duplicate_near = 0
    corrupted_count = 0
    missing_label_count = 0

    for img_path, lbl_path in tqdm(pairs, desc="Auditing dataset", disable=len(pairs) < 20):
        ok, err_msg, dims = validate_single_image(img_path)
        if not ok:
            corrupted_count += 1
            issues.append(AuditIssue(str(img_path), "corrupted_image", err_msg or "Invalid image"))
            continue

        sha = compute_sha256(img_path)
        if sha in seen_sha256:
            duplicate_exact += 1
            issues.append(
                AuditIssue(
                    str(img_path),
                    "exact_duplicate",
                    f"Matches {seen_sha256[sha].name}",
                )
            )
            continue
        seen_sha256[sha] = img_path

        dhash = compute_dhash(img_path)
        if dhash is not None and dhash in seen_dhash:
            duplicate_near += 1
            issues.append(
                AuditIssue(
                    str(img_path),
                    "near_duplicate",
                    f"Perceptual dHash matches {seen_dhash[dhash].name}",
                )
            )
            if remove_near_duplicates:
                continue
        elif dhash is not None:
            seen_dhash[dhash] = img_path

        if lbl_path is None:
            missing_label_count += 1
            issues.append(AuditIssue(str(img_path), "missing_label", "No matching .txt file"))
            continue

        boxes, label_errors = parse_and_validate_yolo_label(lbl_path, valid_class_ids=valid_ids)
        if label_errors:
            issues.append(AuditIssue(str(lbl_path), "invalid_label", "; ".join(label_errors)))
            continue

        present_classes = {classes[b[0]] for b in boxes}
        for b in boxes:
            class_instance_counts[classes[b[0]]] += 1

        if not present_classes:
            category = "normal_negative"
        elif present_classes == {"fire"}:
            category = "fire_only"
        elif present_classes == {"smoke"}:
            category = "smoke_only"
        else:
            category = "fire_and_smoke"
        category_counts[category] += 1

        clean_records.append(
            {
                "image_path": str(img_path),
                "label_path": str(lbl_path),
                "width": dims[0] if dims else 0,
                "height": dims[1] if dims else 0,
                "group_id": extract_sequence_group(img_path),
                "category": category,
                "box_count": len(boxes),
                "boxes": boxes,
            }
        )

    return {
        "source_dir": str(source_dir),
        "total_discovered_images": len(pairs),
        "valid_clean_images": len(clean_records),
        "corrupted_images": corrupted_count,
        "missing_labels": missing_label_count,
        "exact_duplicates_removed": duplicate_exact,
        "near_duplicates_detected": duplicate_near,
        "class_instance_counts": dict(class_instance_counts),
        "image_category_counts": dict(category_counts),
        "issues": [asdict(issue) for issue in issues],
        "clean_records": clean_records,
    }


def create_leakage_free_splits(
    clean_records: list[dict[str, Any]],
    output_dataset_dir: Path,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
    copy_files: bool = True,
) -> dict[str, int]:
    """Split clean samples into train/valid/test while keeping video sequence groups isolated."""
    if not clean_records:
        raise ValueError("No clean dataset records available to split.")
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio:.4f})")

    groups = np.array([rec["group_id"] for rec in clean_records])
    indices = np.arange(len(clean_records))
    unique_groups = len(set(groups))

    if unique_groups >= 3 and len(clean_records) >= 5:
        holdout_ratio = valid_ratio + test_ratio
        gss_train = GroupShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=seed)
        train_idx, holdout_idx = next(gss_train.split(indices, groups=groups))

        holdout_groups = groups[holdout_idx]
        if len(set(holdout_groups)) >= 2 and len(holdout_idx) >= 2:
            rel_test = test_ratio / holdout_ratio
            gss_val_test = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
            rel_val_idx, rel_test_idx = next(gss_val_test.split(holdout_idx, groups=holdout_groups))
            valid_idx = holdout_idx[rel_val_idx]
            test_idx = holdout_idx[rel_test_idx]
        else:
            split_pt = max(1, int(round(len(holdout_idx) * (valid_ratio / holdout_ratio))))
            valid_idx = holdout_idx[:split_pt]
            test_idx = holdout_idx[split_pt:]
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        n_train = max(1, int(round(len(shuffled) * train_ratio)))
        n_valid = max(0, int(round(len(shuffled) * valid_ratio)))
        train_idx = shuffled[:n_train]
        valid_idx = shuffled[n_train : n_train + n_valid]
        test_idx = shuffled[n_train + n_valid :]

    split_map = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    counts: dict[str, int] = {}

    # Verify zero group leakage across splits
    split_groups = {
        name: {clean_records[i]["group_id"] for i in idxs} for name, idxs in split_map.items()
    }
    if unique_groups >= 3:
        assert split_groups["train"].isdisjoint(split_groups["valid"])
        assert split_groups["train"].isdisjoint(split_groups["test"])
        assert split_groups["valid"].isdisjoint(split_groups["test"])

    for split_name, idxs in split_map.items():
        img_out_dir = output_dataset_dir / split_name / "images"
        lbl_out_dir = output_dataset_dir / split_name / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for idx in idxs:
            rec = clean_records[int(idx)]
            src_img = Path(rec["image_path"])
            src_lbl = Path(rec["label_path"])
            dst_img = img_out_dir / src_img.name
            dst_lbl = lbl_out_dir / f"{src_img.stem}.txt"
            if dst_img.exists() and dst_img.resolve() != src_img.resolve():
                unique_stem = f"{src_img.stem}_{idx}"
                dst_img = img_out_dir / f"{unique_stem}{src_img.suffix}"
                dst_lbl = lbl_out_dir / f"{unique_stem}.txt"

            if copy_files:
                if src_img.resolve() != dst_img.resolve():
                    shutil.copy2(src_img, dst_img)
                if src_lbl.resolve() != dst_lbl.resolve():
                    shutil.copy2(src_lbl, dst_lbl)
        counts[split_name] = len(idxs)

    return counts


def index_figlib_sequences(figlib_dir: Path) -> list[dict[str, Any]]:
    """Index downloaded HPWREN FIgLib sequences for external smoke & negative testing.

    Parses filenames matching `<origin_ts>_<offset>.jpg` where offset < 0 precedes
    visible smoke plume appearance and offset >= 0 indicates smoke presence.
    """
    sequences: list[dict[str, Any]] = []
    if not figlib_dir.exists():
        return sequences

    pattern = re.compile(r"^(\d+)_([+-]?\d+)$")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for img_path in sorted(figlib_dir.rglob("*.jpg")):
        match = pattern.match(img_path.stem)
        if not match:
            continue
        origin_ts, offset_sec = match.group(1), int(match.group(2))
        seq_id = f"{img_path.parent.name}_{origin_ts}"
        grouped[seq_id].append(
            {
                "image_path": str(img_path),
                "offset_sec": offset_sec,
                "expected_smoke": offset_sec >= 0,
            }
        )

    for seq_id, frames in grouped.items():
        frames.sort(key=lambda item: item["offset_sec"])
        sequences.append(
            {
                "sequence_id": seq_id,
                "frame_count": len(frames),
                "pre_ignition_frames": sum(1 for f in frames if not f["expected_smoke"]),
                "smoke_frames": sum(1 for f in frames if f["expected_smoke"]),
                "frames": frames,
            }
        )
    return sequences


def generate_audit_charts(audit_report: dict[str, Any], split_counts: dict[str, int], out_path: Path) -> Path:
    """Generate dataset distribution summary charts from actual audited counts."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.patch.set_facecolor("#111418")

    # 1. Image category breakdown
    cat_counts = audit_report.get("image_category_counts", {})
    cat_labels = list(cat_counts.keys()) or ["No Data"]
    cat_vals = list(cat_counts.values()) or [0]
    axes[0].bar(cat_labels, cat_vals, color=["#EF4444", "#F97316", "#F59E0B", "#22C55E"][: len(cat_labels)])
    axes[0].set_title("Image Categories", color="#F8FAFC")
    axes[0].tick_params(axis="x", rotation=20, colors="#CBD5E1")
    axes[0].tick_params(axis="y", colors="#CBD5E1")
    axes[0].set_facecolor("#181C22")

    # 2. Bounding box instance counts
    cls_counts = audit_report.get("class_instance_counts", {})
    cls_labels = list(cls_counts.keys()) or ["No Annotations"]
    cls_vals = list(cls_counts.values()) or [0]
    axes[1].bar(cls_labels, cls_vals, color=["#F97316", "#EF4444"][: len(cls_labels)])
    axes[1].set_title("Bounding Box Instances", color="#F8FAFC")
    axes[1].tick_params(colors="#CBD5E1")
    axes[1].set_facecolor("#181C22")

    # 3. Train/Valid/Test split sizes
    s_labels = list(split_counts.keys()) or ["train", "valid", "test"]
    s_vals = list(split_counts.values()) or [0, 0, 0]
    axes[2].bar(s_labels, s_vals, color=["#3B82F6", "#A855F7", "#14B8A6"])
    axes[2].set_title("Leakage-Free Split Counts", color="#F8FAFC")
    axes[2].tick_params(colors="#CBD5E1")
    axes[2].set_facecolor("#181C22")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


class HttpRangeZipStream(io.RawIOBase):
    """Seekable HTTP Range reader allowing selective extraction from large remote ZIPs."""

    def __init__(self, url: str, timeout: int = 30) -> None:
        self.timeout = timeout
        head = requests.head(url, allow_redirects=True, timeout=timeout)
        head.raise_for_status()
        self.url = head.url
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self._session = requests.Session()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        self.pos = max(0, min(self.size, self.pos))
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, buffer: bytearray | memoryview) -> int:
        n = len(buffer)
        if self.pos >= self.size or n == 0:
            return 0
        end = min(self.size - 1, self.pos + n - 1)
        resp = self._session.get(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.content
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


DFIRE_HF_ZIP_URL = (
    "https://huggingface.co/datasets/Stevebirth/fire-ulm-dfire/resolve/main/dfire_yolo_20261016.zip"
)


def _classify_label_bytes(raw_text: str) -> str:
    """Return 'normal_negative', 'fire_only', 'smoke_only', or 'fire_and_smoke'."""
    classes = set()
    for line in raw_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                classes.add(int(float(parts[0])))
            except ValueError:
                continue
    if not classes:
        return "normal_negative"
    if classes == {1}:
        return "fire_only"
    if classes == {0}:
        return "smoke_only"
    return "fire_and_smoke"


def fetch_dfire_stratified_subset(
    output_dataset_dir: Path,
    train_per_cat: int = 80,
    val_per_cat: int = 20,
    test_per_cat: int = 30,
    seed: int = 42,
    zip_url: str = DFIRE_HF_ZIP_URL,
) -> dict[str, Any]:
    """Stream-extract a stratified balanced subset of D-Fire preserving official splits."""
    rng = np.random.default_rng(seed)
    stream_raw = HttpRangeZipStream(zip_url)
    stream = io.BufferedReader(stream_raw, buffer_size=65536)
    zf = zipfile.ZipFile(stream)

    manifest = json.loads(zf.read("dfire_yolo/file_manifest.json").decode("utf-8"))
    files = manifest.get("files", [])

    # Group image and label metadata by (split, stem)
    paired: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for entry in files:
        rel_path = str(entry["path"])
        split = str(entry["split"])
        stem = Path(rel_path).stem
        if "/images/" in rel_path:
            paired[(split, stem)]["image"] = entry
        elif "/labels/" in rel_path:
            paired[(split, stem)]["label"] = entry

    targets = {"train": train_per_cat, "val": val_per_cat, "test": test_per_cat}
    split_dir_map = {"train": "train", "val": "valid", "test": "test"}
    extracted_counts: dict[str, dict[str, int]] = {}

    for split_key, per_cat in targets.items():
        candidates = [
            pair
            for (sp, _), pair in paired.items()
            if sp == split_key and "image" in pair and "label" in pair
        ]
        # Prefer diverse sources by shuffling deterministically
        order = rng.permutation(len(candidates))
        neg_items = []
        pos_items = []
        for idx in order:
            item = candidates[int(idx)]
            if int(item["label"]["bytes"]) == 0:
                neg_items.append(item)
            else:
                pos_items.append(item)

        buckets: dict[str, list[tuple[dict[str, Any], bytes]]] = {
            "normal_negative": [],
            "fire_only": [],
            "smoke_only": [],
            "fire_and_smoke": [],
        }

        # Prioritize WEB hard negatives + AoF/Public negatives for normal_negative
        web_negs = [it for it in neg_items if Path(it["image"]["path"]).name.startswith("WEB")]
        other_negs = [it for it in neg_items if not Path(it["image"]["path"]).name.startswith("WEB")]
        ordered_negs = []
        for a, b in zip(web_negs, other_negs):
            ordered_negs.extend([a, b])
        ordered_negs.extend(web_negs[len(other_negs) :])
        ordered_negs.extend(other_negs[len(web_negs) :])

        for item in ordered_negs[:per_cat]:
            buckets["normal_negative"].append((item, b""))

        _tls = threading.local()

        def _get_session() -> requests.Session:
            if not hasattr(_tls, "session"):
                _tls.session = requests.Session()
            return _tls.session

        def _read_zip_entry(zip_entry_name: str) -> bytes:
            info = zf.getinfo(zip_entry_name)
            start = info.header_offset
            # Local header is 30 bytes + filename + variable extra field (usually <= 64 bytes)
            fetch_len = 30 + len(info.filename.encode("utf-8")) + 128 + info.compress_size
            end = min(stream_raw.size - 1, start + fetch_len - 1)
            resp = _get_session().get(
                stream_raw.url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=30,
            )
            resp.raise_for_status()
            buf = resp.content
            fname_len, extra_len = struct.unpack("<HH", buf[26:30])
            data_start = 30 + fname_len + extra_len
            payload = buf[data_start : data_start + info.compress_size]
            if info.compress_type == zipfile.ZIP_STORED:
                return payload
            if info.compress_type == zipfile.ZIP_DEFLATED:
                return zlib.decompress(payload, -zlib.MAX_WBITS)
            return zf.read(zip_entry_name)

        # Batch-scan small positive labels concurrently in waves of 160
        wave_size = 160
        pos_offset = 0
        with ThreadPoolExecutor(max_workers=16) as pool:
            while pos_offset < len(pos_items) and not all(
                len(buckets[c]) >= per_cat for c in ("fire_only", "smoke_only", "fire_and_smoke")
            ):
                wave = pos_items[pos_offset : pos_offset + wave_size]
                pos_offset += wave_size
                lbl_paths = [f"dfire_yolo/{it['label']['path']}" for it in wave]
                lbl_results = list(pool.map(_read_zip_entry, lbl_paths))
                for item, lbl_bytes in zip(wave, lbl_results):
                    cat = _classify_label_bytes(lbl_bytes.decode("utf-8", errors="replace"))
                    if cat != "normal_negative" and len(buckets[cat]) < per_cat:
                        buckets[cat].append((item, lbl_bytes))

        out_split = split_dir_map[split_key]
        img_out = output_dataset_dir / out_split / "images"
        lbl_out = output_dataset_dir / out_split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        cat_saved: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            for cat_name, items in buckets.items():
                img_paths = [f"dfire_yolo/{it['image']['path']}" for it, _ in items]
                imgs_bytes = list(pool.map(_read_zip_entry, img_paths))
                saved = 0
                for (item, lbl_bytes), img_bytes in zip(items, imgs_bytes):
                    img_name = Path(item["image"]["path"]).name
                    lbl_name = Path(item["label"]["path"]).name
                    (img_out / img_name).write_bytes(img_bytes)
                    (lbl_out / lbl_name).write_bytes(lbl_bytes)
                    saved += 1
                cat_saved[cat_name] = saved
        extracted_counts[out_split] = cat_saved

    return {
        "source": manifest.get("source"),
        "license": manifest.get("license"),
        "counts_by_split_and_category": extracted_counts,
    }


def populate_samples_from_dataset(dataset_dir: Path, samples_dir: Path, max_per_category: int = 5) -> dict[str, int]:
    """Copy representative held-out test images and labels into samples/images/*."""
    test_img_dir = dataset_dir / "test" / "images"
    test_lbl_dir = dataset_dir / "test" / "labels"
    categories = ("fire", "smoke", "fire_and_smoke", "normal", "hard_negatives")
    for cat in categories:
        (samples_dir / "images" / cat).mkdir(parents=True, exist_ok=True)
    (samples_dir / "videos").mkdir(parents=True, exist_ok=True)

    counts = {cat: 0 for cat in categories}
    if not test_img_dir.is_dir():
        return counts

    for img_path in sorted(test_img_dir.iterdir()):
        if img_path.suffix.lower() not in VALID_IMAGE_EXTS:
            continue
        lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
        raw_lbl = lbl_path.read_text(encoding="utf-8", errors="replace") if lbl_path.is_file() else ""
        cls_cat = _classify_label_bytes(raw_lbl)
        if cls_cat == "fire_only":
            target_cat = "fire"
        elif cls_cat == "smoke_only":
            target_cat = "smoke"
        elif cls_cat == "fire_and_smoke":
            target_cat = "fire_and_smoke"
        elif img_path.name.startswith("WEB"):
            target_cat = "hard_negatives"
        else:
            target_cat = "normal"

        if counts[target_cat] >= max_per_category:
            continue
        dest_dir = samples_dir / "images" / target_cat
        shutil.copy2(img_path, dest_dir / img_path.name)
        if lbl_path.is_file():
            shutil.copy2(lbl_path, dest_dir / lbl_path.name)
        counts[target_cat] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="FireGuard AI — Phase 2 Dataset Preparation & Audit")
    parser.add_argument("--source-dir", type=Path, default=None, help="Path to raw extracted D-Fire images/labels")
    parser.add_argument("--zip-path", type=Path, default=None, help="Optional path to local D-Fire ZIP archive")
    parser.add_argument("--download-url", type=str, default=None, help="Optional direct ZIP URL for D-Fire archive")
    parser.add_argument("--fetch-dfire-subset", action="store_true", help="Stream balanced stratified D-Fire subset from HuggingFace mirror")
    parser.add_argument("--train-per-cat", type=int, default=80, help="Images per category for train subset")
    parser.add_argument("--val-per-cat", type=int, default=20, help="Images per category for valid subset")
    parser.add_argument("--test-per-cat", type=int, default=30, help="Images per category for test subset")
    parser.add_argument("--clone-dfire-repo", action="store_true", help="Clone DFireDataset GitHub repo metadata")
    parser.add_argument("--figlib-dir", type=Path, default=None, help="Optional local HPWREN FIgLib directory")
    parser.add_argument("--output-dir", type=Path, default=config.DATASET_DIR, help="Output YOLO dataset root")
    parser.add_argument("--remove-near-duplicates", action="store_true", help="Drop perceptual dHash duplicates")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for split generation")
    args = parser.parse_args()

    print("=== Dataset Licensing & Attribution ===")
    for name, info in check_dataset_licensing().items():
        print(f"- {name}: {info['license']} ({info['source']})")

    raw_dir = config.DATASET_DIR / "raw"
    if args.clone_dfire_repo:
        repo_dir = clone_dfire_metadata(raw_dir / "DFireDataset_repo")
        print(f"Cloned D-Fire repository to: {repo_dir}")

    if args.fetch_dfire_subset:
        print("Streaming stratified D-Fire subset (preserving official train/val/test splits)...")
        subset_info = fetch_dfire_stratified_subset(
            args.output_dir,
            train_per_cat=args.train_per_cat,
            val_per_cat=args.val_per_cat,
            test_per_cat=args.test_per_cat,
            seed=args.seed,
        )
        print("Subset downloaded:", json.dumps(subset_info, indent=2))
        sample_counts = populate_samples_from_dataset(args.output_dir, config.BASE_DIR / "samples")
        print("Populated samples/images counts:", sample_counts)

    if args.download_url:
        zip_target = raw_dir / "dfire_download.zip"
        download_file_with_progress(args.download_url, zip_target)
        args.zip_path = zip_target

    if args.zip_path:
        extracted = extract_archive(args.zip_path, raw_dir / "extracted")
        if args.source_dir is None:
            args.source_dir = extracted

    source_to_audit = args.source_dir if args.source_dir is not None else config.DATASET_DIR
    audit = audit_and_deduplicate(
        source_to_audit,
        class_names=DFIRE_CLASS_NAMES,
        remove_near_duplicates=args.remove_near_duplicates,
    )

    split_counts: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    if audit["clean_records"] and args.source_dir is not None:
        split_counts = create_leakage_free_splits(
            audit["clean_records"],
            args.output_dir,
            train_ratio=config.SPLIT_RATIOS["train"],
            valid_ratio=config.SPLIT_RATIOS["valid"],
            test_ratio=config.SPLIT_RATIOS["test"],
            seed=args.seed,
        )
    else:
        for split_name in ("train", "valid", "test"):
            split_imgs = args.output_dir / split_name / "images"
            if split_imgs.is_dir():
                split_counts[split_name] = sum(
                    1 for p in split_imgs.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS
                )

    figlib_summary: list[dict[str, Any]] = []
    if args.figlib_dir:
        figlib_summary = index_figlib_sequences(args.figlib_dir)

    report_payload = {
        "licensing": check_dataset_licensing(),
        "audit_summary": {k: v for k, v in audit.items() if k != "clean_records"},
        "split_counts": split_counts,
        "figlib_sequences_indexed": len(figlib_summary),
    }

    report_path = args.output_dir / "dataset_audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    chart_path = generate_audit_charts(audit, split_counts, args.output_dir / "dataset_summary_charts.png")

    print("\n=== Dataset Audit Complete ===")
    print(f"Discovered images : {audit['total_discovered_images']}")
    print(f"Valid clean images: {audit['valid_clean_images']}")
    print(f"Corrupted images  : {audit['corrupted_images']}")
    print(f"Exact duplicates  : {audit['exact_duplicates_removed']}")
    print(f"Near duplicates   : {audit['near_duplicates_detected']}")
    print(f"Split counts      : {split_counts}")
    print(f"Audit JSON saved  : {report_path}")
    print(f"Summary chart     : {chart_path}")


if __name__ == "__main__":
    main()
