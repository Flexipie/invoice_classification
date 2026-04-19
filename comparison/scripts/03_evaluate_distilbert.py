"""
Evaluate the fine-tuned DistilBERT classifier on the test split.

Reads:
  data/processed/test_texts.json
  comparison/models/distilbert_best/   (produced by 02_train_distilbert.py)

Writes:
  comparison/reports/cm_distilbert.png
  comparison/reports/distilbert_test.json  (per-sample preds + probs)
  Appends "DistilBERT-OCR" entry to reports/results.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "comparison" / "models" / "distilbert_best"
COMP_REPORTS = ROOT / "comparison" / "reports"
REPO_REPORTS = ROOT / "reports"

LABELS = ["email", "invoice", "letter", "scientific_report"]

BATCH_SIZE = int(os.environ.get("BERT_EVAL_BATCH_SIZE", 16))
MAX_LEN    = int(os.environ.get("BERT_MAX_LEN", 256))


@torch.no_grad()
def predict_batch(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    enc = tokenizer(
        texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
    ).to(device)
    out = model(**enc)
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()
    return probs


def main() -> None:
    if not MODEL_DIR.exists():
        raise SystemExit(
            f"Model dir {MODEL_DIR} missing. Run 02_train_distilbert.py first."
        )

    test_path = PROCESSED_DIR / "test_texts.json"
    if not test_path.exists():
        raise SystemExit(
            f"{test_path} missing. Run 01_ocr_images.py first."
        )

    COMP_REPORTS.mkdir(parents=True, exist_ok=True)
    REPO_REPORTS.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 03_evaluate_distilbert (device={device}) ===")

    print(f"Loading model from {MODEL_DIR} …")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device).eval()

    rows = json.loads(test_path.read_text(encoding="utf-8"))
    # Keep alignment: empty-text rows get zero prob for all; easier to predict dummy
    texts  = [(r.get("text") or "") for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    print(f"Test rows: {len(rows)}")

    print("Running inference …")
    all_probs = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch = texts[start:end]
        # transformer needs non-empty strings; substitute a space for empties
        batch_safe = [t if t.strip() else " " for t in batch]
        all_probs[start:end] = predict_batch(model, tokenizer, batch_safe, device)
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  batch {start}/{len(texts)}")

    preds = all_probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    print(f"\nTest accuracy : {acc:.4f}")
    print(f"Test macro F1 : {f1m:.4f}")
    print(classification_report(labels, preds, target_names=LABELS, digits=4))

    # Confusion matrix plot
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"DistilBERT-OCR (test acc={acc:.3f})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    cm_path = COMP_REPORTS / "cm_distilbert.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix → {cm_path}")

    # Per-sample dump
    per_sample = [
        {
            "path": rows[i]["path"],
            "label": int(labels[i]),
            "pred":  int(preds[i]),
            "probs": all_probs[i].tolist(),
        }
        for i in range(len(rows))
    ]
    (COMP_REPORTS / "distilbert_test.json").write_text(
        json.dumps({
            "accuracy":        acc,
            "f1_macro":        f1m,
            "label_names":     LABELS,
            "classification_report": classification_report(
                labels, preds, target_names=LABELS, output_dict=True, digits=4
            ),
            "confusion_matrix": cm.tolist(),
            "samples":         per_sample,
        }, indent=2)
    )

    # Append to repo-level results.json (so 05_compare_all sees it naturally)
    results_path = REPO_REPORTS / "results.json"
    if results_path.exists():
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    existing["DistilBERT-OCR"] = float(acc)
    results_path.write_text(json.dumps(existing, indent=2))
    print(f"results.json updated with DistilBERT-OCR = {acc:.4f}")

    print("\nDone. Next: python comparison/scripts/04_run_existing_cv.py")


if __name__ == "__main__":
    main()
