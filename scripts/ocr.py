"""
Stage 1 — OCR: RVL-CDIP images → preprocessed OpenCV → pytesseract → text files + manifest CSVs.

Uses Hugging Face *streaming* (no full ~100+ GiB local dataset). Stops after enough
samples per class. Resumable: chunks of 100; checkpoint under data/processed/.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from datasets import load_dataset
from PIL import Image

RNG_SEED = 42
STREAM_SHUFFLE_BUFFER = 10_000
# Safety cap: stop scanning if stream never fills quotas (misconfigured labels / empty split)
MAX_STREAM_EXAMPLES = 5_000_000

# Only these RVL-CDIP label IDs
TARGET_LABELS: tuple[int, ...] = (2, 6, 7, 14)
LABEL_NAMES: dict[int, str] = {
    2: "email",
    6: "invoice",
    7: "letter",
    14: "scientific_report",
}

# output split name -> HuggingFace split name, samples per class
SPLIT_CONFIG: dict[str, dict[str, str | int]] = {
    "train": {"hf_split": "train", "per_class": 2000},
    "validation": {"hf_split": "val", "per_class": 500},
    "test": {"hf_split": "test", "per_class": 500},
}

CHUNK_SIZE = 100
CHECKPOINT_NAME = "ocr_checkpoint.json"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def ocr_worker_count() -> int:
    """Threads to use for OCR. Tesseract runs in a subprocess, so threads help."""
    cpu = os.cpu_count() or 4
    default = max(1, min(8, cpu))
    return max(1, _env_int("OCR_WORKERS", default))


def ocr_parallel_batch_size(workers: int) -> int:
    return max(8, _env_int("OCR_PARALLEL_BATCH", workers * 2))


def maybe_resize(pil_image: Image.Image) -> Image.Image:
    """Set OCR_MAX_EDGE=1800 (or similar) to shrink huge scans before OCR (faster)."""
    max_edge = _env_int("OCR_MAX_EDGE", 0)
    if max_edge <= 0:
        return pil_image
    w, h = pil_image.size
    m = max(w, h)
    if m <= max_edge:
        return pil_image
    scale = max_edge / m
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return pil_image.resize((nw, nh), resample)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_processed() -> Path:
    return project_root() / "data" / "processed"


def checkpoint_path() -> Path:
    return data_processed() / CHECKPOINT_NAME


def ensure_dirs() -> None:
    root = data_processed()
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_CONFIG:
        for lab in TARGET_LABELS:
            (root / split / str(lab)).mkdir(parents=True, exist_ok=True)


def pil_to_gray_uint8(img: Image.Image) -> np.ndarray:
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.uint8)


def deskew_angle(gray: np.ndarray) -> float:
    """Estimate skew angle (degrees) from binary foreground."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return 0.0
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return float(angle)


def deskew_if_needed(gray: np.ndarray, min_angle: float = 0.5) -> np.ndarray:
    if os.environ.get("OCR_SKIP_DESKEW", "").strip().lower() in ("1", "true", "yes"):
        return gray
    angle = deskew_angle(gray)
    if abs(angle) < min_angle:
        return gray
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    """grayscale → deskew (if needed) → Otsu threshold → PIL for Tesseract."""
    gray = pil_to_gray_uint8(pil_image)
    gray = deskew_if_needed(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def ocr_image(pil_image: Image.Image) -> str:
    pil_image = maybe_resize(pil_image)
    proc = preprocess_for_ocr(pil_image)
    cfg = os.environ.get("OCR_TESSERACT_CONFIG", "").strip()
    if cfg:
        return pytesseract.image_to_string(proc, config=cfg)
    return pytesseract.image_to_string(proc)


def _run_ocr_batch(images: list[Image.Image], workers: int) -> list[str]:
    if len(images) == 1 or workers <= 1:
        return [ocr_image(im) for im in images]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(ocr_image, images))


def expected_job_ids_for_split(split_name: str) -> list[str]:
    """Deterministic ids: {split}_{label}_{seq:05d} with seq in [0, per_class)."""
    per_class = int(SPLIT_CONFIG[split_name]["per_class"])
    ids: list[str] = []
    for lab in TARGET_LABELS:
        for seq in range(per_class):
            ids.append(f"{split_name}_{lab}_{seq:05d}")
    return ids


def all_expected_job_ids() -> list[str]:
    out: list[str] = []
    for split_name in SPLIT_CONFIG:
        out.extend(expected_job_ids_for_split(split_name))
    return out


def load_checkpoint() -> set[str]:
    path = checkpoint_path()
    if not path.is_file():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("completed_ids", []))


def save_checkpoint(completed: set[str]) -> None:
    ensure_dirs()
    with open(checkpoint_path(), "w", encoding="utf-8") as f:
        json.dump({"completed_ids": sorted(completed)}, f, indent=0)


def text_path_for_id(split_name: str, label: int, uid: str) -> Path:
    return data_processed() / split_name / str(label) / f"{uid}.txt"


def completed_from_txt_files() -> set[str]:
    """Any existing *.txt under data/processed/{split}/{label}/ counts as done."""
    done: set[str] = set()
    root = data_processed()
    for split_name in SPLIT_CONFIG:
        for lab in TARGET_LABELS:
            d = root / split_name / str(lab)
            if not d.is_dir():
                continue
            for p in d.glob("*.txt"):
                done.add(p.stem)
    return done


def rebuild_manifest_csv(split_name: str) -> None:
    """Rewrite data/processed/{split}.csv from all *.txt under that split."""
    root = data_processed() / split_name
    rows: list[dict] = []
    for lab in TARGET_LABELS:
        lab_dir = root / str(lab)
        if not lab_dir.is_dir():
            continue
        for p in sorted(lab_dir.glob("*.txt")):
            uid = p.stem
            text = p.read_text(encoding="utf-8")
            rows.append(
                {
                    "id": uid,
                    "label": lab,
                    "text": text,
                    "label_name": LABEL_NAMES[lab],
                }
            )
    rows.sort(key=lambda r: r["id"])
    out = data_processed() / f"{split_name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "text", "label_name"])
        w.writeheader()
        w.writerows(rows)


def rebuild_all_manifest_csvs() -> None:
    for split_name in SPLIT_CONFIG:
        rebuild_manifest_csv(split_name)


def split_fully_done(split_name: str, completed: set[str]) -> bool:
    return all(uid in completed for uid in expected_job_ids_for_split(split_name))


def process_split_streaming(split_name: str, completed: set[str]) -> set[str]:
    """
    Stream one HF split with shuffle(seed=42); take first per_class examples per target label.
    OCR each accepted row unless uid already in completed. Updates completed in place.
    """
    cfg = SPLIT_CONFIG[split_name]
    hf_split = str(cfg["hf_split"])
    per_class = int(cfg["per_class"])

    if split_fully_done(split_name, completed):
        print(f"  Split {split_name!r}: already complete, skipping.", file=sys.stderr)
        return completed

    workers = ocr_worker_count()
    batch_fill = ocr_parallel_batch_size(workers)
    print(
        f"  OCR parallelism: {workers} worker(s), batch flush ≥ {batch_fill} images "
        f"(OCR_WORKERS / OCR_PARALLEL_BATCH)",
        file=sys.stderr,
    )

    ds = load_dataset("chainyo/rvl-cdip", split=hf_split, streaming=True)
    ds = ds.shuffle(seed=RNG_SEED, buffer_size=STREAM_SHUFFLE_BUFFER)

    counts: dict[int, int] = {lab: 0 for lab in TARGET_LABELS}
    scanned = 0
    chunk_since_ckpt: list[tuple[str, str]] = []
    pending_rows: list[tuple[str, str, int, Image.Image]] = []

    def flush_pending() -> None:
        if not pending_rows:
            return
        images = [t[3] for t in pending_rows]
        texts = _run_ocr_batch(images, workers)
        for (uid, sp, lab, _), text in zip(pending_rows, texts):
            out_path = text_path_for_id(sp, lab, uid)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as tf:
                tf.write(text)
            completed.add(uid)
            chunk_since_ckpt.append((sp, uid))
            if len(chunk_since_ckpt) >= CHUNK_SIZE:
                save_checkpoint(completed)
                for s in {t[0] for t in chunk_since_ckpt}:
                    rebuild_manifest_csv(s)
                print(
                    f"  {split_name}: checkpoint ({len(completed)} total completed)",
                    file=sys.stderr,
                )
                chunk_since_ckpt.clear()
        pending_rows.clear()

    for ex in ds:
        scanned += 1
        if scanned > MAX_STREAM_EXAMPLES:
            raise RuntimeError(
                f"Stream scan exceeded {MAX_STREAM_EXAMPLES} examples for split {hf_split!r} "
                f"without filling quotas — check labels / dataset."
            )

        lab = int(ex["label"])
        if lab not in TARGET_LABELS:
            continue
        if counts[lab] >= per_class:
            continue

        seq = counts[lab]
        counts[lab] += 1
        uid = f"{split_name}_{lab}_{seq:05d}"

        if uid in completed:
            if all(counts[l] >= per_class for l in TARGET_LABELS):
                flush_pending()
                break
            continue

        image = ex["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        pending_rows.append((uid, split_name, lab, image))
        if len(pending_rows) >= batch_fill:
            flush_pending()

        if all(counts[l] >= per_class for l in TARGET_LABELS):
            flush_pending()
            break

    flush_pending()

    if chunk_since_ckpt:
        save_checkpoint(completed)
        for sp in {t[0] for t in chunk_since_ckpt}:
            rebuild_manifest_csv(sp)

    missing = [l for l in TARGET_LABELS if counts[l] < per_class]
    if missing:
        raise RuntimeError(
            f"Split {hf_split!r}: could not collect enough labels after scanning "
            f"{scanned} examples. Counts: {counts}"
        )

    print(f"  Split {split_name!r}: done (scanned ~{scanned} stream examples).", file=sys.stderr)
    return completed


def run_ocr_pipeline() -> None:
    ensure_dirs()
    completed = load_checkpoint() | completed_from_txt_files()
    expected = set(all_expected_job_ids())
    completed &= expected  # drop stray ids if any

    pending_n = len(expected - completed)
    if pending_n == 0:
        print("OCR already complete. Rebuilding CSVs.", file=sys.stderr)
        rebuild_all_manifest_csvs()
        return

    w = ocr_worker_count()
    print(
        f"Streaming mode (no full local dataset). Pending OCR: {pending_n} / {len(expected)}. "
        f"OCR_WORKERS={w} (see README for speed options).",
        file=sys.stderr,
    )

    for split_name in SPLIT_CONFIG:
        completed = process_split_streaming(split_name, completed)

    rebuild_all_manifest_csvs()
    print("OCR pipeline finished.", file=sys.stderr)


def main() -> None:
    run_ocr_pipeline()


if __name__ == "__main__":
    main()
