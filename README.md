# Document Classification & Invoice Extraction

Pure Computer Vision pipeline — no LLMs, no OCR at classification time.

**Categories:** email · invoice · letter · scientific_report  
**Dataset:** [RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) (12 000-image subset, balanced 4-class)

## Approach

Three models are trained and compared:

| Model | Features | Classifier |
|-------|----------|------------|
| A | Raw pixels → TruncatedSVD (200 components) | SVM (RBF) |
| B1 | HOG (orientations=9, cells=8×8) | SVM (linear) |
| B2 | HOG | Random Forest |
| C | End-to-end CNN (3 conv blocks) | Softmax |

Invoice extraction (fields: number, dates, issuer, recipient, total) runs only after a document is classified as `invoice`, using Tesseract OCR + regex + spaCy NER.

## Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt-get install tesseract-ocr`

## Setup

```bash
cd document-classifier
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Pipeline (run in order)

### 1 — Download images

```bash
python scripts/download_data.py
```

Streams RVL-CDIP from HuggingFace and saves:
```
data/raw/{train,validation,test}/{email,invoice,letter,scientific_report}/*.png
```
2 000 train + 500 val + 500 test images per class. Progress is checkpointed — safe to interrupt and resume.

### 2 — Preprocess

```bash
python scripts/preprocess.py
```

Resizes every image to 128×128 grayscale and saves `.npy` arrays:
```
data/processed/{train,validation,test}_{images,labels}.npy
```

Override the target size: `IMG_SIZE=64 python scripts/preprocess.py`

### 3a — Train classical models

```bash
python scripts/train_classical.py
```

Trains Models A, B1, B2. Saves `models/{svd_svm,hog_svm,hog_rf}.pkl`.  
Prints per-model validation accuracy and classification report.

### 3b — Train CNN

```bash
python scripts/train_cnn.py
```

Trains Model C for 20 epochs with cosine LR decay and light augmentation.  
Saves best checkpoint to `models/cnn_best.pth`.

Override hyperparameters:
```bash
EPOCHS=30 BATCH_SIZE=64 LR=5e-4 python scripts/train_cnn.py
```

### 4 — Evaluate all models

```bash
python scripts/evaluate.py
```

Runs every saved model on the test set, prints confusion matrices, saves:
- `reports/cm_*.png` — per-model confusion matrix plots
- `reports/results.json` — accuracy summary

### 5 — Single-image prediction

```bash
python scripts/predict.py path/to/document.png
python scripts/predict.py path/to/document.pdf --model hog_svm
```

Options:
- `--model cnn|svd_svm|hog_svm|hog_rf` (default: `cnn`)
- `--no-extract` — skip invoice extraction

### 6 — Invoice extraction (standalone)

```bash
python scripts/extract.py --image path/to/invoice.png
```

Returns JSON:
```json
{
  "invoice_number": "INV-2024-001",
  "invoice_date": "01/03/2024",
  "due_date": "31/03/2024",
  "issuer_name": "Acme Corp",
  "recipient_name": "John Doe",
  "total_amount": "$1,250.00"
}
```

### 7 — Flask API

```bash
python app.py          # default port 5000
PORT=8080 python app.py
```

**Endpoints:**

```
POST /classify        multipart file → {"label": "invoice", "confidence": 0.97}
POST /extract         multipart file → {"label": "invoice", "confidence": 0.97, "fields": {...}}
GET  /health          → {"status": "ok", "cnn_ready": true}
```

Example with curl:
```bash
curl -F "file=@invoice.png" http://localhost:5000/extract
```

## Directory structure

```
document-classifier/
├── app.py                    Flask API
├── requirements.txt
├── data/
│   ├── raw/                  Downloaded PNG images (git-ignored)
│   └── processed/            .npy arrays + label_map.json (git-ignored)
├── models/                   Trained model files (git-ignored)
├── reports/                  Confusion matrix plots + results.json
└── scripts/
    ├── download_data.py
    ├── preprocess.py
    ├── train_classical.py    Models A, B1, B2
    ├── train_cnn.py          Model C
    ├── evaluate.py
    ├── predict.py
    └── extract.py
```
