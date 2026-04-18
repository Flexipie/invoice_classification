# Transformer Comparison (OCR + DistilBERT)

This folder adds a **text-transformer** benchmark (OCR on raw document images -> DistilBERT classifier) alongside the repo's existing pure-CV pipeline (SVD+SVM, HOG+SVM, HOG+RF, CNN, Stacking).

Everything here is **local and untracked**. No file outside `comparison/` is modified.

## Why this comparison

The assignment (see `Session 12 - Group Work.pdf`) forbids generative AI but explicitly allows traditional ML, CV, and NLP. A DistilBERT **encoder** used for sequence classification is a discriminative model — not generative — and so is a legitimate "classical-ish" benchmark to compare against.

The interesting question we answer: does an off-the-shelf text transformer run on noisy OCR text beat a purpose-built CV stack on low-resolution scanned documents?

## Pipeline

```
data/raw/*.png  ->  Tesseract OCR  ->  data/processed/*_texts.json
                                   \
                                    ->  DistilBERT fine-tune -> evaluate -> reports/cm_distilbert.png
data/raw/*.png  ->  preprocess.py  ->  *_images.npy  ->  existing CV models -> reports/results.json
                                                                                      |
                                                             comparison/reports/comparison.{md,png,json}
```

## How to run (Windows PowerShell)

```powershell
# from repo root (two levels above this folder)

# 1. Set up env (once)
python -m venv invoice_classification\.venv
& invoice_classification\.venv\Scripts\python.exe -m pip install --upgrade pip
& invoice_classification\.venv\Scripts\python.exe -m pip install -r invoice_classification\requirements.txt
& invoice_classification\.venv\Scripts\python.exe -m pip install -r invoice_classification\comparison\requirements.txt
& invoice_classification\.venv\Scripts\python.exe -m spacy download en_core_web_sm

# 2. Full pipeline (small CPU-friendly budget: 500 train / 100 val / 200 test per class)
$env:PER_CLASS_TRAIN=500; $env:PER_CLASS_VAL=100; $env:PER_CLASS_TEST=200
$env:PYTHONIOENCODING="utf-8"; $env:MPLBACKEND="Agg"
& invoice_classification\.venv\Scripts\python.exe invoice_classification\comparison\scripts\00_prepare_data.py
& invoice_classification\.venv\Scripts\python.exe invoice_classification\comparison\scripts\01_ocr_images.py
& invoice_classification\.venv\Scripts\python.exe invoice_classification\comparison\scripts\02_train_distilbert.py
& invoice_classification\.venv\Scripts\python.exe invoice_classification\comparison\scripts\04_run_existing_cv.py
& invoice_classification\.venv\Scripts\python.exe invoice_classification\comparison\scripts\03_evaluate_distilbert.py
& invoice_classification\.venv\Scripts\python.exe invoice_classification\comparison\scripts\05_compare_all.py

# Note: 03 runs AFTER 04 because the repo's evaluate.py overwrites reports/results.json.
# 05 pulls DistilBERT metrics from its own dump anyway (comparison/reports/distilbert_test.json)
# so the order really only matters for the human-readable results.json.

# 3. Smoke test (50 per class, 1 epoch, for quick verification)
$env:PER_CLASS_TRAIN=50; $env:PER_CLASS_VAL=20; $env:PER_CLASS_TEST=30; $env:BERT_EPOCHS=1
& invoice_classification\.venv\Scripts\python.exe invoice_classification\comparison\scripts\00_prepare_data.py
& invoice_classification\.venv\Scripts\python.exe invoice_classification\comparison\scripts\01_ocr_images.py
& invoice_classification\.venv\Scripts\python.exe invoice_classification\comparison\scripts\02_train_distilbert.py
& invoice_classification\.venv\Scripts\python.exe invoice_classification\comparison\scripts\03_evaluate_distilbert.py
```

## Environment variables

| Var | Default | Meaning |
| --- | --- | --- |
| `PER_CLASS_TRAIN` | 500 | Train images per class kept after download |
| `PER_CLASS_VAL`   | 100 | Validation images per class |
| `PER_CLASS_TEST`  | 200 | Test images per class |
| `TESSERACT_CMD`   | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Path to the Tesseract executable (Windows only) |
| `BERT_MODEL`      | `distilbert-base-uncased` | HF model name. Fall back to `prajjwal1/bert-tiny` if CPU is too slow |
| `BERT_EPOCHS`     | 2 | Number of fine-tuning epochs |
| `BERT_BATCH_SIZE` | 8 | Per-device batch size |
| `BERT_MAX_LEN`    | 256 | Token truncation length |

## Outputs

- `comparison/models/distilbert_best/` — HF `save_pretrained` checkpoint (~260 MB)
- `reports/cm_distilbert.png` — confusion matrix of the transformer
- `reports/results.json` — updated with `"DistilBERT-OCR": <acc>`
- `comparison/reports/comparison.{md,png,json}` — side-by-side table, bar chart, merged metrics

## Notes

- CPU-only machines: expect OCR to take ~3-5 min per 1000 images and DistilBERT training to take ~30-60 min per epoch at the default budget. Use the smoke test config for a first run.
- Tesseract must be installed separately. On Windows: `winget install --id UB-Mannheim.TesseractOCR`.
