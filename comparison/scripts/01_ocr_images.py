"""
OCR every raw PNG in data/raw/{split}/{label}/*.png using Tesseract and cache
the text to data/processed/{split}_texts.json.

Output schema:
  [
    {"path": "data/raw/train/invoice/00001.png",
     "label": 1,
     "label_name": "invoice",
     "text": "INVOICE #123 ..."},
    ...
  ]

Idempotent: if the output JSON already contains entries for the same source
paths, those are reused and only missing images are OCR'd.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from PIL import Image
import pytesseract

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

LABELS = ["email", "invoice", "letter", "scientific_report"]
LABEL_TO_IDX = {n: i for i, n in enumerate(LABELS)}

MAX_CHARS = int(os.environ.get("OCR_MAX_CHARS", 2000))
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.name == "nt" and Path(TESSERACT_CMD).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


_WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    return _WS_RE.sub(" ", raw).strip()[:MAX_CHARS]


def ocr_one(path: Path) -> str:
    with Image.open(path) as im:
        im = im.convert("L")
        try:
            txt = pytesseract.image_to_string(im, lang="eng", config="--psm 6")
        except pytesseract.TesseractError as e:
            print(f"    TesseractError on {path}: {e}")
            return ""
    return clean_text(txt)


def load_existing(out_file: Path) -> dict[str, dict]:
    if not out_file.exists():
        return {}
    try:
        data = json.loads(out_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {row["path"]: row for row in data}


def process_split(split: str) -> None:
    split_dir = RAW_DIR / split
    if not split_dir.exists():
        print(f"  [{split}] {split_dir} missing, skipping")
        return

    out_file = PROCESSED_DIR / f"{split}_texts.json"
    cache = load_existing(out_file)
    if cache:
        print(f"  [{split}] reusing {len(cache)} cached entries")

    rows: list[dict] = []
    total = 0
    new_ocr = 0

    for label in LABELS:
        class_dir = split_dir / label
        if not class_dir.exists():
            continue
        files = sorted(class_dir.glob("*.png"))
        print(f"  [{split}/{label}] {len(files)} images", flush=True)

        for i, fpath in enumerate(files):
            rel = str(fpath.relative_to(ROOT)).replace("\\", "/")
            total += 1
            if rel in cache and cache[rel].get("text") is not None:
                rows.append(cache[rel])
                continue

            text = ocr_one(fpath)
            rows.append({
                "path":       rel,
                "label":      LABEL_TO_IDX[label],
                "label_name": label,
                "text":       text,
            })
            new_ocr += 1

            if new_ocr % 50 == 0:
                print(f"      OCR'd {new_ocr} new images …", flush=True)
                # incremental save for crash safety
                out_file.write_text(
                    json.dumps(rows, ensure_ascii=False), encoding="utf-8"
                )

    out_file.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"  [{split}] saved {len(rows)} rows → {out_file} (new OCR: {new_ocr})")


def main() -> None:
    print("=== 01_ocr_images ===")
    print(f"Tesseract: {pytesseract.pytesseract.tesseract_cmd}")
    try:
        version = pytesseract.get_tesseract_version()
        print(f"  version: {version}")
    except Exception as e:
        print(f"  WARN: cannot get Tesseract version: {e}")
        print("  Install: winget install --id UB-Mannheim.TesseractOCR")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        print(f"\n[{split}]")
        process_split(split)

    print("\nDone. Next: python comparison/scripts/02_train_distilbert.py")


if __name__ == "__main__":
    main()
