"""
Orchestrate data prep without modifying the repo.

Strategy:
  - Import the repo's download_data.download_class function (don't call its main,
    which has a hardcoded 10k/1k/1k budget we don't want on a laptop).
  - Drive it with our own per-class budget (env-driven).
  - Then import and run preprocess.main() to regenerate .npy caches.

This also avoids the Windows cp1252 UnicodeEncodeError in the repo's print()
calls: we set sys.stdout to UTF-8 at import time so arrows/check-marks render.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Force UTF-8 stdout so the repo's unicode arrows don't crash on Windows cp1252.
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(SCRIPTS_DIR))

import download_data as dl   # noqa: E402
import preprocess  as pp     # noqa: E402

LABELS = ["email", "invoice", "letter", "scientific_report"]

TARGETS = {
    "train":      int(os.environ.get("PER_CLASS_TRAIN", 500)),
    "validation": int(os.environ.get("PER_CLASS_VAL",   100)),
    "test":       int(os.environ.get("PER_CLASS_TEST",  200)),
}


def download_small() -> None:
    """Drive dl.download_class() per split/class with our custom budget."""
    ckpt = dl.load_checkpoint()

    for split, per_class in TARGETS.items():
        print(f"\n[{split}] listing shards …")
        shards = dl.get_shard_files(split)
        print(f"  {len(shards)} shard files found")

        for label_id, label_name in dl.TARGET_LABELS.items():
            print(f"  -> {split}/{label_name}  (target {per_class})")
            dl.download_class(split, label_id, label_name, per_class, ckpt, shards)


def trim_split(split: str, per_class: int) -> None:
    """After download, if we over-collected in a previous run, trim."""
    split_dir = RAW_DIR / split
    if not split_dir.exists():
        print(f"  [trim] {split_dir} missing, skipping")
        return

    for label in LABELS:
        class_dir = split_dir / label
        if not class_dir.exists():
            continue
        files = sorted(class_dir.glob("*.png"))
        if len(files) <= per_class:
            continue
        for f in files[per_class:]:
            try:
                f.unlink()
            except OSError as e:
                print(f"    WARN: cannot remove {f}: {e}")
        print(f"  [trim] {split}/{label}: kept {per_class}, removed {len(files) - per_class}")


def clear_npy_cache() -> None:
    if not PROCESSED_DIR.exists():
        return
    n = 0
    for pattern in ("*_images.npy", "*_labels.npy"):
        for f in PROCESSED_DIR.glob(pattern):
            f.unlink()
            n += 1
    if n:
        print(f"  Cleared {n} stale .npy cache file(s)")


def main() -> None:
    print("=== 00_prepare_data ===")
    print(f"Budget per class: {TARGETS}")
    print(f"Root: {ROOT}")

    print("\n--- Downloading (skips classes that already meet budget) ---")
    download_small()

    print("\n--- Trimming (in case a previous run over-collected) ---")
    for split, per_class in TARGETS.items():
        trim_split(split, per_class)

    clear_npy_cache()

    print("\n--- Preprocessing to .npy ---")
    pp.main()

    print("\nDone. Next: python comparison/scripts/01_ocr_images.py")


if __name__ == "__main__":
    main()
