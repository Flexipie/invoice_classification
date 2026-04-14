"""
Preprocess raw document images and cache as .npy arrays for fast loading.

Reads from:  data/raw/{split}/{label_name}/*.png
Writes to:   data/processed/{split}_images.npy   shape (N, IMG_SIZE, IMG_SIZE) uint8
             data/processed/{split}_labels.npy   shape (N,) int64
             data/processed/label_map.json        {int → label_name}

Settings (override via env vars):
  IMG_SIZE   – resize target (default 128)
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = int(os.environ.get("IMG_SIZE", 128))

LABEL_NAMES = ["email", "invoice", "letter", "scientific_report"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_NAMES)}

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def load_split(split: str):
    images, labels = [], []
    split_dir = RAW_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"No data found at {split_dir}. Run scripts/download_data.py first."
        )

    for label_name in LABEL_NAMES:
        class_dir = split_dir / label_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found, skipping.")
            continue

        files = sorted(class_dir.glob("*.png"))
        print(f"  {split}/{label_name}: {len(files)} images")

        for fpath in files:
            img = Image.open(fpath).convert("L")  # grayscale
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            images.append(np.array(img, dtype=np.uint8))
            labels.append(LABEL_TO_IDX[label_name])

    return np.stack(images), np.array(labels, dtype=np.int64)


def main() -> None:
    print(f"=== Preprocessing images (size={IMG_SIZE}×{IMG_SIZE}) ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        print(f"\n[{split}]")
        images, labels = load_split(split)
        print(f"  → {images.shape[0]} images, labels distribution: {np.bincount(labels)}")

        np.save(OUT_DIR / f"{split}_images.npy", images)
        np.save(OUT_DIR / f"{split}_labels.npy", labels)
        print(f"  Saved to data/processed/{split}_images.npy")

    # save label map
    label_map = {str(i): name for i, name in enumerate(LABEL_NAMES)}
    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\nDone. Next step: python scripts/train_classical.py")


if __name__ == "__main__":
    main()
