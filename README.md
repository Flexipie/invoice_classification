# Document Classification & Invoice Extraction

Traditional ML pipeline (no LLMs): OCR → TF-IDF + structural features → sklearn classifiers → regex/spaCy extraction for invoices.

**Git:** `data/processed/` (OCR text + CSVs, ~tens of MB) is **included** so clones can run feature extraction without re-OCR. To regenerate from scratch, delete that folder and run `python scripts/ocr.py`. `models/` and `reports/` stay empty until you train.

## Prerequisites

- **Python 3.9+** (for **Python 3.9**, `requirements.txt` pins **spaCy 3.7.x**; use **Python 3.10+** if you want newer spaCy without that pin)
- **Tesseract OCR** (system binary; required by `pytesseract`):
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt-get install tesseract-ocr`

## Setup (automated)

From the `document-classifier` folder:

```bash
bash setup_env.sh
source .venv/bin/activate
```

This creates `.venv`, installs everything in `requirements.txt`, downloads `en_core_web_sm`, and checks Hugging Face access to `chainyo/rvl-cdip`. Install **Tesseract** separately if the script warns (see Prerequisites).

Manual setup is the same as before: `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`.

## Stage 1 — OCR

Builds stratified samples from `chainyo/rvl-cdip` (labels 2, 6, 7, 14), runs OpenCV preprocessing and Tesseract, writes:

- `data/processed/{split}/{label}/{id}.txt`
- `data/processed/{split}.csv` (columns: `id`, `label`, `text`, `label_name`)

Splits: `train` (2000/class), `validation` (500/class), `test` (500/class). Random seed **42**.

Processing uses chunks of **100** images; progress is stored in `data/processed/ocr_checkpoint.json`. Existing `.txt` files are treated as done so runs can resume. CSVs are rebuilt from `.txt` files after each chunk.

**Disk space:** OCR uses Hugging Face **streaming** (no full ~100+ GiB local copy of RVL-CDIP). You only need space for OCR text output and the Hugging Face cache for shards as they stream (far smaller than materializing the whole dataset).

```bash
python scripts/ocr.py
```

The first run still **streams** many examples over the network until quotas are filled; it can take a long time but should not require hundreds of gigabytes free.

### Faster OCR (optional)

OCR is the slow part. Defaults use **several threads** (`OCR_WORKERS`, capped at 8) because Tesseract runs in a subprocess and work can overlap. Tune with environment variables:

| Variable | Effect |
|----------|--------|
| `OCR_WORKERS` | Thread count (default: up to 8, based on CPU). Set `1` to force single-threaded. |
| `OCR_PARALLEL_BATCH` | Flush parallel batches when this many images are queued (default: `max(8, 2 × OCR_WORKERS)`). |
| `OCR_MAX_EDGE` | If set (e.g. `1800`), shrink long page sides before OCR — **much faster**, slightly rougher text. |
| `OCR_SKIP_DESKEW` | Set to `1` to skip deskew — faster, worse on rotated scans. |
| `OCR_TESSERACT_CONFIG` | Extra Tesseract flags, e.g. `--oem 1 --psm 6` (see Tesseract docs). |

Example:

```bash
OCR_MAX_EDGE=1800 OCR_SKIP_DESKEW=1 python scripts/ocr.py
```

## Later stages (not yet implemented)

```bash
python scripts/features.py
python scripts/train.py
python scripts/extract.py
python app.py   # Flask on port 5000
```
