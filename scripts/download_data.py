"""
Download a balanced subset of RVL-CDIP document images from HuggingFace.

Strategy: download individual parquet shards directly (hf_hub_download) and
read only the rows belonging to our target labels. Since the dataset is sorted
by label, each class occupies a contiguous band of shards — we only need to
download ~2 shards per class instead of all 119.

Classes:
  2  → email          6  → invoice
  7  → letter        14  → scientific_report

Targets:  train 2000  |  validation 500  |  test 500
"""

import io
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ID       = "chainyo/rvl-cdip"
TARGET_LABELS = {2: "email", 6: "invoice", 7: "letter", 14: "scientific_report"}
NUM_CLASSES   = 16

TARGETS    = {"train": 2000, "validation": 500, "test": 500}
TOTAL_ROWS = {"train": 320_000, "validation": 40_000, "test": 40_000}
# HF split name differs from our key
HF_SPLIT   = {"train": "train", "validation": "val", "test": "test"}

ROOT            = Path(__file__).parent.parent
RAW_DIR         = ROOT / "data" / "raw"
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


def get_shard_files(split: str) -> list[str]:
    """List parquet shard paths in the HF repo for a given split."""
    hf_name = HF_SPLIT[split]
    all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
    shards = sorted(
        f for f in all_files
        if f.startswith(f"data/{hf_name}") and f.endswith(".parquet")
    )
    return shards


def img_bytes_from_row(row_dict: dict) -> bytes:
    """Extract raw image bytes from a pyarrow row dict."""
    img_field = row_dict["image"]
    if isinstance(img_field, dict):
        return img_field.get("bytes") or img_field["path"]
    if isinstance(img_field, (bytes, bytearray)):
        return bytes(img_field)
    # pyarrow scalar
    val = img_field.as_py() if hasattr(img_field, "as_py") else img_field
    if isinstance(val, dict):
        return val.get("bytes") or val["path"]
    return val


# ---------------------------------------------------------------------------
# Core: download one class from one split
# ---------------------------------------------------------------------------

def download_class(split: str, label: int, label_name: str,
                   target: int, ckpt: dict, shards: list[str]) -> None:
    already = count_existing(split, label_name)
    if already >= target:
        print(f"    {split}/{label_name}: already {already}/{target} ✓")
        return

    out_dir = RAW_DIR / split / label_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total      = TOTAL_ROWS[split]
    num_shards = len(shards)
    rows_per_shard = total / num_shards          # float, for offset maths
    rows_per_class = total // NUM_CLASSES

    # Estimate which shard contains the first row of this label
    label_start_row  = label * rows_per_class
    start_shard_idx  = max(0, int(label_start_row / rows_per_shard) - 1)

    collected = already
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    for shard_idx in range(start_shard_idx, num_shards):
        if collected >= target:
            break

        shard_path = shards[shard_idx]
        print(f"    [{split}/{label_name}] shard {shard_idx}/{num_shards-1} …", flush=True)

        local = hf_hub_download(REPO_ID, filename=shard_path, repo_type="dataset")
        table = pq.read_table(local, columns=["image", "label"])

        # Filter to this label
        mask    = pc.equal(table.column("label"), label)
        filtered = table.filter(mask)

        n_found = len(filtered)
        if n_found == 0:
            # If we're past the expected region, no more data for this label
            shard_center = (shard_idx + 0.5) * rows_per_shard
            if shard_center > (label + 1) * rows_per_class:
                print(f"      no more {label_name} rows, stopping.")
                break
            continue

        for i in range(n_found):
            if collected >= target:
                break
            row = {col: filtered.column(col)[i] for col in filtered.column_names}
            raw = img_bytes_from_row(row)
            img = Image.open(io.BytesIO(raw)).convert("L")
            img.save(out_dir / f"{collected:05d}.png")
            collected += 1

        print(f"      → {collected}/{target} saved", flush=True)

    ckpt[split][str(label)] = collected
    save_checkpoint(ckpt)
    status = "✓" if collected >= target else f"WARNING: only {collected}/{target}"
    print(f"    {split}/{label_name}: {collected}/{target} {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== RVL-CDIP image downloader (parquet shard mode) ===")
    print(f"Repo    : {REPO_ID}")
    print(f"Classes : {list(TARGET_LABELS.values())}")
    print(f"Targets : {TARGETS}\n")

    ckpt = load_checkpoint()

    for split in TARGETS:
        print(f"[{split}] listing shards …")
        shards = get_shard_files(split)
        print(f"  {len(shards)} shard files found")

        for label, label_name in TARGET_LABELS.items():
            download_class(split, label, label_name, TARGETS[split], ckpt, shards)

    print("\nDone. Images saved to data/raw/")
    print("Next step: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
