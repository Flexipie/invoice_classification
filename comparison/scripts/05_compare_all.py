"""
Merge every model's test metrics into a single side-by-side comparison.

Pulls:
  reports/results.json              (written by evaluate.py + stacking + distilbert step)
  comparison/reports/distilbert_test.json

Re-computes per-model confusion matrices and classification reports against the
same held-out test set (data/processed/test_*.npy + test_texts.json) so that
accuracy / macro F1 / per-class F1 are directly comparable.

Outputs:
  comparison/reports/comparison.json
  comparison/reports/comparison.png   (grouped bar chart: accuracy + macro F1)
  comparison/reports/comparison.md    (human-readable report)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPO_REPORTS = ROOT / "reports"
COMP_REPORTS = ROOT / "comparison" / "reports"

LABELS = ["email", "invoice", "letter", "scientific_report"]
HOG_SIZE = 128
SVD_SIZE = 64
IMG_SIZE = 128

# Expose repo scripts on sys.path so we can import DocumentCNN
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Feature helpers (mirror train_classical.py exactly)
# ---------------------------------------------------------------------------

def _svd_feats(images):
    feats = []
    for img in images:
        pil = Image.fromarray(img).resize((SVD_SIZE, SVD_SIZE))
        feats.append(np.array(pil, dtype=np.float32).ravel())
    return np.stack(feats)


def _hog_feats(images):
    from skimage.feature import hog
    feats = []
    for img in images:
        pil = Image.fromarray(img).resize((HOG_SIZE, HOG_SIZE))
        arr = np.array(pil, dtype=np.float32) / 255.0
        feats.append(hog(arr, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), block_norm="L2-Hys",
                         feature_vector=True))
    return np.stack(feats)


def _predict_pipe_img(pipe_path: Path, X_test, feats_fn):
    data = joblib.load(pipe_path)
    pipe = data["pipe"] if isinstance(data, dict) else data
    return pipe.predict(feats_fn(X_test))


def _predict_cnn(X_test):
    import torch
    from torchvision import transforms
    from train_cnn import DocumentCNN

    ckpt = torch.load(MODELS_DIR / "cnn_best.pth", map_location="cpu")
    model = DocumentCNN(ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    preds = []
    with torch.no_grad():
        for img in X_test:
            out = model(tfm(img).unsqueeze(0))
            preds.append(int(out.argmax(1).item()))
    return np.array(preds)


def _predict_stacking(X_test):
    from train_stacking import build_meta_features  # imports feature helpers

    svd  = joblib.load(MODELS_DIR / "svd_svm.pkl")
    hsvm = joblib.load(MODELS_DIR / "hog_svm.pkl")
    hrf  = joblib.load(MODELS_DIR / "hog_rf.pkl")
    meta = joblib.load(MODELS_DIR / "stacking.pkl")
    svd  = svd if not isinstance(svd, dict) else svd["pipe"]
    hsvm = hsvm["pipe"] if isinstance(hsvm, dict) else hsvm
    hrf  = hrf["pipe"]  if isinstance(hrf, dict)  else hrf

    M = build_meta_features(X_test, svd, hsvm, hrf)
    return meta.predict(M)


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def summarize(name: str, y_true, y_pred) -> dict:
    rep = classification_report(
        y_true, y_pred, target_names=LABELS, output_dict=True, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    return {
        "model":           name,
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1":    {lbl: rep[lbl]["f1-score"] for lbl in LABELS},
        "per_class_precision": {lbl: rep[lbl]["precision"] for lbl in LABELS},
        "per_class_recall":    {lbl: rep[lbl]["recall"] for lbl in LABELS},
        "confusion_matrix":    cm.tolist(),
    }


def fmt_row(summary: dict) -> str:
    acc = summary["accuracy"]
    f1  = summary["f1_macro"]
    per = summary["per_class_f1"]
    cells = [summary["model"], f"{acc:.4f}", f"{f1:.4f}"] + \
            [f"{per[l]:.3f}" for l in LABELS]
    return "| " + " | ".join(cells) + " |"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    COMP_REPORTS.mkdir(parents=True, exist_ok=True)

    print("=== 05_compare_all ===")

    images_path = DATA_DIR / "test_images.npy"
    labels_path = DATA_DIR / "test_labels.npy"
    if not (images_path.exists() and labels_path.exists()):
        raise SystemExit("Missing data/processed/test_*.npy — run 00_prepare_data.py first")

    X_test = np.load(images_path)
    y_test = np.load(labels_path).astype(int)
    print(f"Test set: {len(X_test)} samples")

    all_summaries: list[dict] = []

    # --- Classical + CNN + Stacking ---
    specs = [
        ("SVD+SVM",  MODELS_DIR / "svd_svm.pkl",  _svd_feats),
        ("HOG+SVM",  MODELS_DIR / "hog_svm.pkl",  _hog_feats),
        ("HOG+RF",   MODELS_DIR / "hog_rf.pkl",   _hog_feats),
    ]
    for name, path, feats_fn in specs:
        if not path.exists():
            print(f"  skip {name}: {path.name} not found")
            continue
        print(f"\n>> {name}")
        preds = _predict_pipe_img(path, X_test, feats_fn)
        all_summaries.append(summarize(name, y_test, preds))

    if (MODELS_DIR / "cnn_best.pth").exists():
        print("\n>> CNN")
        preds = _predict_cnn(X_test)
        all_summaries.append(summarize("CNN", y_test, preds))

    if (MODELS_DIR / "stacking.pkl").exists():
        print("\n>> Stacking")
        try:
            preds = _predict_stacking(X_test)
            all_summaries.append(summarize("Stacking", y_test, preds))
        except Exception as e:
            print(f"  stacking eval failed: {e}")

    # --- DistilBERT-OCR (from its own dump) ---
    bert_dump = COMP_REPORTS / "distilbert_test.json"
    if bert_dump.exists():
        print("\n>> DistilBERT-OCR")
        blob = json.loads(bert_dump.read_text(encoding="utf-8"))
        samples = blob["samples"]
        # Align on test_texts.json order (same order as per-sample dump)
        y_bert_true = np.array([s["label"] for s in samples])
        y_bert_pred = np.array([s["pred"]  for s in samples])
        summary = summarize("DistilBERT-OCR", y_bert_true, y_bert_pred)
        all_summaries.append(summary)
    else:
        print("  skip DistilBERT-OCR: distilbert_test.json missing")

    if not all_summaries:
        raise SystemExit("No models found to compare.")

    # --- Sort by accuracy desc ---
    all_summaries.sort(key=lambda d: -d["accuracy"])

    # --- Save JSON ---
    merged = {
        "label_names": LABELS,
        "num_test":    int(len(y_test)),
        "models":      all_summaries,
    }
    (COMP_REPORTS / "comparison.json").write_text(json.dumps(merged, indent=2))
    print(f"\nSaved → comparison/reports/comparison.json")

    # --- Bar chart ---
    names = [s["model"] for s in all_summaries]
    accs  = [s["accuracy"] for s in all_summaries]
    f1s   = [s["f1_macro"] for s in all_summaries]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names)), 5))
    ax.bar(x - w/2, accs, w, label="Accuracy")
    ax.bar(x + w/2, f1s,  w, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title(f"Model comparison (test set, N={len(y_test)})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(COMP_REPORTS / "comparison.png", dpi=150)
    plt.close()
    print(f"Saved → comparison/reports/comparison.png")

    # --- Markdown report ---
    md = []
    md.append("# Model comparison\n")
    md.append(f"Held-out test set: **{len(y_test)} samples**, "
              f"4 balanced classes: {', '.join(LABELS)}.\n")
    md.append("## Summary\n")
    md.append("| Model | Accuracy | Macro F1 | " + " | ".join(f"F1 ({l})" for l in LABELS) + " |")
    md.append("|---|---|---|" + "|".join(["---"] * len(LABELS)) + "|")
    for s in all_summaries:
        md.append(fmt_row(s))
    md.append("")
    md.append("![Bar chart](comparison.png)\n")
    md.append("## Per-model confusion matrices\n")
    md.append("PNG files produced by the per-model evaluation steps:\n")
    md.append("- `../../reports/cm_svd+svm.png` · `../../reports/cm_hog+svm.png` · "
              "`../../reports/cm_hog+rf.png` · `../../reports/cm_cnn.png` · "
              "`../../reports/cm_stacking.png`")
    md.append("- `cm_distilbert.png` (in this folder)\n")
    md.append("## Discussion template\n")
    md.append(
        "- **Image-based models** (SVD+SVM, HOG+SVM, HOG+RF, CNN, Stacking) "
        "learn directly from pixel / gradient statistics of the 128×128 grayscale scans. "
        "They are robust to OCR failures on noisy or low-resolution scans but miss "
        "explicit textual cues.\n"
        "- **DistilBERT-OCR** depends entirely on Tesseract output. Where OCR fails "
        "(blurry scans, tables, non-English portions), the transformer has nothing to "
        "classify from; where OCR succeeds, it exploits the explicit lexical cues "
        "('invoice', 'amount due', 'Dear Dr.', etc.) that CV models cannot directly "
        "perceive.\n"
        "- The **stacking ensemble** combines complementary errors of the base CV models; "
        "adding DistilBERT to the stack would likely improve the ceiling further "
        "(left as future work).\n"
        "- **Practical trade-off for the live demo**: the transformer needs an OCR step "
        "at inference time (extra ~0.5–2s per document on CPU) whereas the CV stack "
        "runs in <100 ms on the same hardware. This is important to mention in the "
        "oral presentation.\n"
    )
    (COMP_REPORTS / "comparison.md").write_text("\n".join(md))
    print(f"Saved → comparison/reports/comparison.md")

    # --- Print pretty summary ---
    print("\n" + "=" * 60)
    print("  FINAL MODEL COMPARISON (test set)")
    print("=" * 60)
    print(f"  {'Model':<18s} {'Accuracy':>9s} {'Macro F1':>9s}")
    print(f"  {'-' * 18} {'-' * 9} {'-' * 9}")
    for s in all_summaries:
        print(f"  {s['model']:<18s} {s['accuracy']:>9.4f} {s['f1_macro']:>9.4f}")


if __name__ == "__main__":
    main()
