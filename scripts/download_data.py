"""
Download a balanced subset of RVL-CDIP document images from HuggingFace.

Classes used (same 4 as original project):
  2  → email
  6  → invoice
  7  → letter
  14 → scientific_report

Target counts per class:
  train: 2000
  validation: 500
  test: 500

Images are saved as PNG files under:
  data/raw/{split}/{label_name}/{index:05d}.png

Progress is checkpointed so runs can be safely interrupted and resumed.
"""

import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_ID = "aharley/rvl_cdip"
TARGET_LABELS = {2: "email", 6: "invoice", 7: "letter", 14: "scientific_report"}
TARGETS = {"train": 2000, "validation": 500, "test": 500}
SEED = 42

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHECKPOINT_FILE = ROOT / "data" / "download_checkpoint.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {split: {str(lbl): 0 for lbl in TARGET_LABELS} for split in TARGETS}


def save_checkpoint(ckpt: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2)


def count_existing(split: str, label_name: str) -> int:
    d = RAW_DIR / split / label_name
    if not d.exists():
        return 0
    return len(list(d.glob("*.png")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def download_split(split: str, ckpt: dict) -> None:
    target = TARGETS[split]
    needed = {
        lbl: target - count_existing(split, name)
        for lbl, name in TARGET_LABELS.items()
    }
    if all(v <= 0 for v in needed.values()):
        print(f"  [{split}] already complete, skipping.")
        return

    print(f"  [{split}] streaming dataset …")
    ds = load_dataset(DATASET_ID, split=split, streaming=True)

    counters = {lbl: count_existing(split, name) for lbl, name in TARGET_LABELS.items()}

    for example in ds:
        label = example["label"]
        if label not in TARGET_LABELS:
            continue
        if counters[label] >= target:
            continue

        label_name = TARGET_LABELS[label]
        out_dir = RAW_DIR / split / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        img: Image.Image = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        idx = counters[label]
        img.save(out_dir / f"{idx:05d}.png")
        counters[label] = idx + 1

        # checkpoint every 50 images per class
        if counters[label] % 50 == 0:
            for lbl, cnt in counters.items():
                ckpt[split][str(lbl)] = cnt
            save_checkpoint(ckpt)
            done = sum(min(c, target) for c in counters.values())
            total = len(TARGET_LABELS) * target
            print(f"    {split}: {done}/{total} images saved", end="\r", flush=True)

        if all(counters[lbl] >= target for lbl in TARGET_LABELS):
            break

    # final checkpoint
    for lbl, cnt in counters.items():
        ckpt[split][str(lbl)] = cnt
    save_checkpoint(ckpt)

    for lbl, name in TARGET_LABELS.items():
        n = count_existing(split, name)
        status = "OK" if n >= target else f"only {n}/{target}"
        print(f"    {split}/{name}: {status}          ")


def main() -> None:
    print("=== RVL-CDIP image downloader ===")
    print(f"Dataset : {DATASET_ID}")
    print(f"Classes : {list(TARGET_LABELS.values())}")
    print(f"Targets : {TARGETS}")
    print()

    ckpt = load_checkpoint()

    for split in TARGETS:
        print(f"Processing split: {split}")
        download_split(split, ckpt)

    print("\nDone. Images saved to data/raw/")
    print("Next step: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
