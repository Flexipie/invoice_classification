# DocuVision — Intelligent Document Classification & Invoice Extraction

**Classify documents instantly. Extract invoice data automatically. No manual sorting, no guesswork.**

DocuVision is a pure computer vision pipeline that identifies document types — email, invoice, letter, or scientific report — using an ensemble of six machine learning models. When an invoice is detected, it goes further: extracting key fields (invoice number, dates, amounts, parties) and pinpointing their exact locations on the page.

No LLMs. No cloud APIs. Everything runs locally.

---

## Why DocuVision?

| Problem | Solution |
|---------|----------|
| Manual document sorting is slow and error-prone | Automated classification in under 1 second |
| Invoice data entry is tedious | Auto-extraction of 6 key fields with bounding boxes |
| Single models fail on edge cases | 6-model ensemble with 87.5% accuracy |
| Most tools require cloud connectivity | Fully offline — your data never leaves your machine |

---

## What It Does

### Document Classification
Upload any document image or PDF. DocuVision runs it through six independent classifiers and returns a consensus prediction with confidence scores.

**Supported document types:**
- **Email** — digital correspondence
- **Invoice** — billing and payment documents
- **Letter** — formal written correspondence
- **Scientific Report** — academic and research papers

### Invoice Field Extraction
When a document is classified as an invoice, DocuVision automatically extracts:

| Field | Example |
|-------|---------|
| Invoice Number | `INV-2024-001` |
| Invoice Date | `01/03/2024` |
| Due Date | `31/03/2024` |
| Issuer | `Acme Corp` |
| Recipient | `John Doe` |
| Total Amount | `$1,250.00` |

Each extracted field is mapped back to its location on the document with color-coded bounding boxes — making verification instant.

### PDF Support
PDFs are automatically rendered and processed. Text is extracted directly from PDF metadata when available, falling back to OCR for scanned documents.

---

## The Models

DocuVision combines classical machine learning, deep learning, and NLP into a single ensemble.

### Base Models

| Model | Approach | How It Works | Test Accuracy |
|-------|----------|-------------|---------------|
| **SVD + SVM** | Classical | Reduces raw pixel data to 200 principal components, then classifies with a radial basis function SVM | 79.3% |
| **HOG + SVM** | Classical | Extracts gradient orientation histograms (shape/edge features), classifies with linear SVM | 75.1% |
| **HOG + Random Forest** | Classical | Same gradient features, classified by an ensemble of 200 decision trees | 77.2% |
| **CNN** | Deep Learning | 3-block convolutional neural network (32→64→128 filters) with batch normalization and dropout | 86.5% |
| **DistilBERT-OCR** | Transformer/NLP | OCRs the document text, then classifies using a fine-tuned DistilBERT transformer | Optional |

### Stacking Ensemble (Final Classifier)

The secret weapon. A logistic regression meta-learner that takes the probability outputs of all base models and learns the optimal way to combine them.

```
SVD+SVM  ──→ [4 probabilities] ─┐
HOG+SVM  ──→ [4 probabilities] ─┤
HOG+RF   ──→ [4 probabilities] ─┼──→ [16-dim vector] ──→ Logistic Regression ──→ Final Prediction
CNN      ──→ [4 probabilities] ─┘
```

**Result: 87.5% accuracy** — consistently outperforming every individual model.

### Performance Breakdown

| Model | Email (F1) | Invoice (F1) | Letter (F1) | Scientific (F1) | Overall Accuracy |
|-------|-----------|-------------|------------|----------------|-----------------|
| SVD+SVM | 0.924 | 0.740 | 0.771 | 0.741 | 79.3% |
| HOG+SVM | 0.916 | 0.690 | 0.742 | 0.654 | 75.1% |
| HOG+RF | 0.907 | 0.729 | 0.765 | 0.696 | 77.2% |
| CNN | 0.950 | 0.848 | 0.848 | 0.817 | 86.5% |
| **Stacking** | **0.949** | **0.865** | **0.856** | **0.832** | **87.5%** |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML, CSS, JavaScript, Chart.js |
| **Backend** | Flask (Python) |
| **Classical ML** | scikit-learn (SVM, Random Forest, TruncatedSVD) |
| **Deep Learning** | PyTorch (CNN) |
| **NLP/Transformer** | HuggingFace Transformers (DistilBERT) |
| **Computer Vision** | OpenCV, scikit-image (HOG features) |
| **OCR** | Tesseract (via pytesseract) |
| **NER** | spaCy (entity extraction for names) |
| **PDF Processing** | pdfplumber |
| **Dataset** | RVL-CDIP via HuggingFace Datasets |

---

## Web Interface

### Classify Page
Upload a document and instantly see results from all six models side by side. The ensemble prediction is highlighted with a confidence badge. For invoices, extracted fields appear with bounding boxes overlaid on the document preview.

### Performance Dashboard
Interactive charts showing accuracy comparisons, per-class precision/recall/F1 scores, and confusion matrices for every model — built with Chart.js.

### Responsive Design
Fully responsive across desktop, tablet, and mobile. Clean, modern UI with a blue-accent design system.

---

## API

Three endpoints, all returning JSON.

### `POST /classify`
Classify a document image or PDF.

```bash
curl -F "file=@document.png" http://localhost:5000/classify
```

```json
{
  "label": "invoice",
  "confidence": 0.875,
  "models": {
    "Stacking": { "label": "invoice", "confidence": 0.875 },
    "CNN": { "label": "invoice", "confidence": 0.865 },
    "DistilBERT-OCR": { "label": "invoice", "confidence": 0.82 },
    "SVD+SVM": { "label": "invoice", "confidence": null },
    "HOG+SVM": { "label": "invoice", "confidence": null },
    "HOG+RF": { "label": "invoice", "confidence": 0.772 }
  }
}
```

### `POST /extract`
Classify + extract invoice fields (if invoice detected).

```bash
curl -F "file=@invoice.pdf" http://localhost:5000/extract
```

Returns classification results plus:
```json
{
  "fields": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "01/03/2024",
    "total_amount": "$1,250.00"
  },
  "bboxes": [
    { "field": "invoice_number", "x": 100, "y": 50, "w": 200, "h": 30 }
  ]
}
```

### `GET /health`
```json
{ "status": "ok", "stacking_ready": true }
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt-get install tesseract-ocr`

### Installation

```bash
cd document-classifier
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Training Pipeline

Run these scripts in order to train all models from scratch:

```bash
# 1. Download the RVL-CDIP dataset (3,000 images, ~10-30 min)
python scripts/download_data.py

# 2. Preprocess images to 128x128 grayscale .npy arrays
python scripts/preprocess.py

# 3. Train classical models (SVD+SVM, HOG+SVM, HOG+RF)
python scripts/train_classical.py

# 4. Train the CNN (20 epochs with cosine LR decay)
python scripts/train_cnn.py

# 5. Train the stacking ensemble (uses validation set)
python scripts/train_stacking.py

# 6. (Optional) OCR all images + fine-tune DistilBERT
python scripts/ocr_images.py
python scripts/train_distilbert.py

# 7. Evaluate all models on the test set
python scripts/evaluate.py
```

All scripts support environment variable overrides:
```bash
EPOCHS=30 BATCH_SIZE=64 LR=5e-4 python scripts/train_cnn.py
IMG_SIZE=64 python scripts/preprocess.py
```

### Running the App

```bash
python app.py                      # starts on port 5000
PORT=8080 python app.py            # custom port
```

Open `http://localhost:5000` in your browser.

### CLI Prediction

```bash
python scripts/predict.py path/to/document.png
python scripts/predict.py path/to/document.pdf --model hog_svm
python scripts/predict.py path/to/invoice.png --no-extract
```

---

## Project Structure

```
document-classifier/
├── app.py                        Flask API & web server
├── requirements.txt              Python dependencies
├── pipeline.ipynb                Full pipeline walkthrough notebook
├── setup_env.sh                  Environment setup script
│
├── scripts/
│   ├── download_data.py          Stream RVL-CDIP from HuggingFace
│   ├── preprocess.py             Resize & cache as .npy arrays
│   ├── train_classical.py        SVD+SVM, HOG+SVM, HOG+RF
│   ├── train_cnn.py              3-block CNN with augmentation
│   ├── train_stacking.py         Meta-learner ensemble
│   ├── train_distilbert.py       Fine-tune DistilBERT on OCR text
│   ├── ocr_images.py             Batch OCR with Tesseract
│   ├── evaluate.py               Test set evaluation & reports
│   ├── evaluate_distilbert.py    DistilBERT-specific evaluation
│   ├── predict.py                Single-document CLI inference
│   └── extract.py                Invoice field extraction + bboxes
│
├── templates/
│   ├── index.html                Home page
│   ├── classify.html             Upload & classification UI
│   └── performance.html          Model performance dashboard
│
├── static/
│   └── styles.css                Design system & responsive styles
│
├── data/
│   ├── raw/                      Downloaded images (git-ignored)
│   └── processed/                .npy arrays & metadata (git-ignored)
│
├── models/                       Trained model artifacts (git-ignored)
│   ├── svd_svm.pkl
│   ├── hog_svm.pkl
│   ├── hog_rf.pkl
│   ├── cnn_best.pth
│   ├── stacking.pkl
│   └── distilbert_best/
│
└── reports/                      Evaluation outputs
    ├── cm_*.png                  Confusion matrices
    ├── comparison.png            Model comparison chart
    └── results.json              Summary metrics
```

---

## Dataset

**Source:** [RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) (Ryerson Vision Lab), streamed via HuggingFace Datasets.

| Split | Images per Class | Total |
|-------|-----------------|-------|
| Train | 500 | 2,000 |
| Validation | 125 | 500 |
| Test | 125 | 500 |
| **Total** | | **3,000** |

All images are resized to 128x128 grayscale and cached as NumPy arrays for fast loading. Downloads are checkpointed and resumable.

---

## CNN Architecture

```
Input: 128×128×1 (grayscale)

Conv Block 1:  Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)     → 32×64×64
Conv Block 2:  Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)    → 64×32×32
Conv Block 3:  Conv2d(64→128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)   → 128×16×16

Flatten:       128×16×16 = 32,768 features
FC1:           32,768 → 512 → ReLU → Dropout(0.5)
FC2:           512 → 4 → Softmax
```

Trained with Adam optimizer, cosine annealing LR schedule, and light augmentation (random flip, rotation, color jitter).
