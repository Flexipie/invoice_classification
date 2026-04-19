"""
Fine-tune a DistilBERT classifier on OCR'd RVL-CDIP text.

Input:
  data/processed/train_texts.json       (produced by ocr_images.py)
  data/processed/validation_texts.json

Output:
  models/distilbert_best/               (HF save_pretrained format)
  reports/distilbert_training.json      (train/val loss+acc per epoch)

Config (env vars):
  BERT_MODEL      default distilbert-base-uncased
  BERT_EPOCHS     default 2
  BERT_BATCH_SIZE default 8
  BERT_MAX_LEN    default 256
  BERT_LR         default 5e-5
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

LABELS = ["email", "invoice", "letter", "scientific_report"]
NUM_LABELS = len(LABELS)

MODEL_NAME = os.environ.get("BERT_MODEL", "distilbert-base-uncased")
EPOCHS     = int(os.environ.get("BERT_EPOCHS", 2))
BATCH_SIZE = int(os.environ.get("BERT_BATCH_SIZE", 8))
MAX_LEN    = int(os.environ.get("BERT_MAX_LEN", 256))
LR         = float(os.environ.get("BERT_LR", 5e-5))
SEED       = int(os.environ.get("BERT_SEED", 42))


def load_split(name: str) -> Dataset:
    path = PROCESSED_DIR / f"{name}_texts.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/ocr_images.py first."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in rows if (r.get("text") or "").strip()]
    return Dataset.from_list([
        {"text": r["text"], "label": int(r["label"])} for r in rows
    ])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=== Train DistilBERT ===")
    print(f"Model     : {MODEL_NAME}")
    print(f"Epochs    : {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max len   : {MAX_LEN}")
    print(f"LR        : {LR}")
    _dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device    : {_dev}")

    print("\nLoading splits …")
    train_ds = load_split("train")
    val_ds   = load_split("validation")
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    print("\nTokenising …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_LEN
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(tokenize,   batched=True, remove_columns=["text"])

    id2label = {i: name for i, name in enumerate(LABELS)}
    label2id = {name: i for i, name in enumerate(LABELS)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    out_dir = MODELS_DIR / "distilbert_best"
    tmp_dir = MODELS_DIR / "_hf_checkpoints"

    args = TrainingArguments(
        output_dir=str(tmp_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
        use_cpu=not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("\nTraining …")
    trainer.train()

    print("\nFinal validation eval:")
    final_metrics = trainer.evaluate()
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

    print(f"\nSaving best model → {out_dir}")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    history = [log for log in trainer.state.log_history if log]
    (REPORTS_DIR / "distilbert_training.json").write_text(
        json.dumps({"history": history, "final_val": final_metrics}, indent=2)
    )
    print(f"Training log → reports/distilbert_training.json")

    print("\nDone. Next: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
