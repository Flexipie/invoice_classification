# Invoicinator 3000

Document classification and invoice field extraction pipeline built on the RVL-CDIP dataset. Combines classical ML, deep learning, and NLP into a stacking ensemble that classifies documents as **email**, **invoice**, **letter**, or **scientific report**. When an invoice is detected, key fields are automatically extracted with bounding box localization.

Everything runs locally — no cloud APIs or LLMs.

## Models

We train four base models, each extracting different features from the same document image, then combine them in a stacking ensemble.

| Model | Feature Extraction | Classifier | Accuracy |
|-------|-------------------|------------|----------|
| SVD + SVM | Flatten pixels (64×64) → TruncatedSVD to 200 components | RBF SVM (C=10) | 79.3% |
| HOG + SVM | Histogram of Oriented Gradients — 324-d descriptor capturing edges and layout | LinearSVC (C=1.0) | 75.1% |
| HOG + RF | Same HOG features | Random Forest (200 trees) | 77.2% |
| CNN | 3-block ConvNet (32→64→128 filters) with BatchNorm, learns hierarchical visual features | FC layers (32K→512→4) | 86.5% |
| DistilBERT | Tesseract OCR → tokenized text → fine-tuned DistilBERT embeddings (768-d) | Transformer classification head | Optional |

### Stacking Ensemble

Each base model outputs 4 class probabilities. These are concatenated into a 16-dimensional meta-feature vector and fed into a Logistic Regression meta-learner trained on the validation set (to avoid data leakage).

**Result: 87.5% accuracy** — outperforms every individual model.

### Invoice Field Extraction

When a document is classified as an invoice, we run Tesseract OCR with Otsu thresholding, then extract fields using regex patterns and spaCy NER:

- Invoice number, invoice date, due date, total amount
- Issuer (ORG entities via spaCy) and recipient (PERSON entities)
- Each field is mapped to word-level bounding boxes using pytesseract's `image_to_data`

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Classical ML | scikit-learn (SVM, Random Forest, TruncatedSVD), scikit-image (HOG) |
| Deep Learning | PyTorch (CNN with cosine annealing LR, augmentation) |
| NLP | HuggingFace Transformers (DistilBERT), spaCy (NER) |
| OCR | Tesseract via pytesseract |
| Backend | Flask |
| Frontend | HTML/CSS/JS, Chart.js |
| Dataset | RVL-CDIP via HuggingFace Datasets |

## Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` / `sudo apt-get install tesseract-ocr`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Training

Run scripts individually in order:

```bash
python scripts/download_data.py        # download RVL-CDIP from HuggingFace
python scripts/preprocess.py           # resize to 128x128 grayscale .npy
python scripts/train_classical.py      # SVD+SVM, HOG+SVM, HOG+RF
python scripts/train_cnn.py            # CNN (20 epochs, cosine LR)
python scripts/train_stacking.py       # stacking ensemble
python scripts/evaluate.py             # test set evaluation
```

Or use **`pipeline.ipynb`** — it runs the full pipeline in sequence and is the easiest way to train everything from a single notebook.

Optional (DistilBERT):
```bash
python scripts/ocr_images.py           # batch OCR all images
python scripts/train_distilbert.py     # fine-tune on OCR text
```

## Running the App

```bash
python app.py                          # starts on http://localhost:5000
```

Pages:
- `/` — home
- `/classify` — upload a document image or PDF to classify it; invoices get automatic field extraction with bounding boxes overlaid on the preview
- `/performance` — interactive dashboard with accuracy comparisons, per-class F1 scores, and confusion matrices for all models

## API

```bash
# classify a document
curl -F "file=@document.png" http://localhost:5000/classify

# classify + extract invoice fields (if invoice detected)
curl -F "file=@invoice.pdf" http://localhost:5000/extract

# health check
curl http://localhost:5000/health
```

## Project Structure

```
├── app.py                  Flask server + API endpoints
├── pipeline.ipynb          Full training pipeline notebook
├── requirements.txt        Python dependencies
├── scripts/
│   ├── download_data.py    Stream RVL-CDIP from HuggingFace
│   ├── preprocess.py       Resize & cache as 128x128 .npy arrays
│   ├── train_classical.py  SVD+SVM, HOG+SVM, HOG+RF
│   ├── train_cnn.py        3-block CNN with augmentation
│   ├── train_stacking.py   Stacking ensemble meta-learner
│   ├── train_distilbert.py DistilBERT fine-tuning on OCR text
│   ├── ocr_images.py       Batch OCR with Tesseract
│   ├── extract.py          Invoice field extraction + bounding boxes
│   ├── evaluate.py         Test set evaluation & reports
│   └── predict.py          Single-document CLI inference
├── templates/              HTML (index, classify, performance)
├── static/                 CSS
├── models/                 Trained model artifacts (git-ignored)
├── data/                   Raw + processed dataset (git-ignored)
└── reports/                Confusion matrices, comparison charts, results.json
```

## Dataset

[RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) (Ryerson Vision Lab) — 4 document classes. Images are resized to 128×128 grayscale and cached as NumPy arrays for fast loading. Downloads are checkpointed and resumable.
