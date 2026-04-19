"""
Evaluate all trained models on the held-out test set.

Loads:
  models/svd_svm.pkl
  models/hog_svm.pkl
  models/hog_rf.pkl
  models/cnn_best.pth

Outputs:
  - Per-model accuracy + classification report printed to stdout
  - Confusion matrix plots saved to reports/cm_{model}.png
  - Summary table saved to reports/results.json
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

LABEL_NAMES = ["email", "invoice", "letter", "scientific_report"]
IMG_SIZE    = 128
HOG_SIZE    = 128
SVD_SIZE    = 64


def load_test():
    images = np.load(DATA_DIR / "test_images.npy")
    labels = np.load(DATA_DIR / "test_labels.npy")
    return images, labels


# ---------- feature helpers (mirrors train_classical.py) ----------

def extract_svd_features(images):
    feats = []
    for img in images:
        pil = Image.fromarray(img).resize((SVD_SIZE, SVD_SIZE))
        feats.append(np.array(pil, dtype=np.float32).ravel())
    return np.stack(feats)


def extract_hog_features(images):
    from skimage.feature import hog
    feats = []
    for img in images:
        pil = Image.fromarray(img).resize((HOG_SIZE, HOG_SIZE))
        arr = np.array(pil, dtype=np.float32) / 255.0
        descriptor = hog(
            arr,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        feats.append(descriptor)
    return np.stack(feats)


# ---------- CNN inference ----------

def load_cnn():
    import torch
    from train_cnn import DocumentCNN

    ckpt = torch.load(MODELS_DIR / "cnn_best.pth", map_location="cpu")
    model = DocumentCNN(ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_cnn(model, images):
    import torch
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    preds = []
    with torch.no_grad():
        for img in images:
            x = tfm(img).unsqueeze(0)      # (1, 1, H, W)
            out = model(x)
            preds.append(int(out.argmax(1).item()))
    return np.array(preds)


def load_stacking_bundle():
    import joblib
    from train_stacking import build_meta_features  # noqa: F401

    svd_svm_raw = joblib.load(MODELS_DIR / "svd_svm.pkl")
    hog_svm_raw = joblib.load(MODELS_DIR / "hog_svm.pkl")
    hog_rf_raw = joblib.load(MODELS_DIR / "hog_rf.pkl")
    meta_clf = joblib.load(MODELS_DIR / "stacking.pkl")

    return {
        "svd_svm": svd_svm_raw if not isinstance(svd_svm_raw, dict) else svd_svm_raw["pipe"],
        "hog_svm": hog_svm_raw["pipe"] if isinstance(hog_svm_raw, dict) else hog_svm_raw,
        "hog_rf": hog_rf_raw["pipe"] if isinstance(hog_rf_raw, dict) else hog_rf_raw,
        "meta_clf": meta_clf,
    }


def predict_stacking(images):
    from train_stacking import build_meta_features

    bundle = load_stacking_bundle()
    meta_features = build_meta_features(images, bundle["svd_svm"], bundle["hog_svm"], bundle["hog_rf"])
    return bundle["meta_clf"].predict(meta_features)


# ---------- plotting ----------

def save_confusion_matrix(y_true, y_pred, model_name: str, acc: float):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name}  (test acc={acc:.3f})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = REPORTS_DIR / f"cm_{model_name.lower().replace(' ', '_').replace('+', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {out}")


# ---------- main ----------

def evaluate_model(name, y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
    print(f"\n{'='*50}")
    print(f"  {name}   (test accuracy: {acc:.4f})")
    print('='*50)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    save_confusion_matrix(y_true, y_pred, name, acc)
    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix_image": f"/reports/cm_{name.lower().replace(' ', '_').replace('+', '_')}.png",
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test data …")
    X_test, y_test = load_test()
    print(f"  {X_test.shape[0]} test images")

    results = {}

    # --- SVD + SVM ---
    svd_path = MODELS_DIR / "svd_svm.pkl"
    if svd_path.exists():
        print("\nEvaluating SVD + SVM …")
        pipe = joblib.load(svd_path)
        X = extract_svd_features(X_test)
        preds = pipe.predict(X)
        results["SVD+SVM"] = evaluate_model("SVD+SVM", preds, y_test)
    else:
        print("  svd_svm.pkl not found, skipping.")

    # --- HOG + SVM ---
    hog_svm_path = MODELS_DIR / "hog_svm.pkl"
    if hog_svm_path.exists():
        print("\nEvaluating HOG + SVM …")
        data = joblib.load(hog_svm_path)
        pipe = data["pipe"] if isinstance(data, dict) else data
        X = extract_hog_features(X_test)
        preds = pipe.predict(X)
        results["HOG+SVM"] = evaluate_model("HOG+SVM", preds, y_test)
    else:
        print("  hog_svm.pkl not found, skipping.")

    # --- HOG + RF ---
    hog_rf_path = MODELS_DIR / "hog_rf.pkl"
    if hog_rf_path.exists():
        print("\nEvaluating HOG + RF …")
        data = joblib.load(hog_rf_path)
        pipe = data["pipe"] if isinstance(data, dict) else data
        X = extract_hog_features(X_test)
        preds = pipe.predict(X)
        results["HOG+RF"] = evaluate_model("HOG+RF", preds, y_test)
    else:
        print("  hog_rf.pkl not found, skipping.")

    # --- CNN ---
    cnn_path = MODELS_DIR / "cnn_best.pth"
    if cnn_path.exists():
        print("\nEvaluating CNN …")
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        model = load_cnn()
        preds = predict_cnn(model, X_test)
        results["CNN"] = evaluate_model("CNN", preds, y_test)
    else:
        print("  cnn_best.pth not found, skipping.")

    # --- Stacking ensemble ---
    stacking_path = MODELS_DIR / "stacking.pkl"
    if stacking_path.exists():
        print("\nEvaluating Stacking Ensemble …")
        preds = predict_stacking(X_test)
        results["Stacking Ensemble"] = evaluate_model("Stacking Ensemble", preds, y_test)
    else:
        print("  stacking.pkl not found, skipping.")

    # --- DistilBERT-OCR ---
    distilbert_dir = MODELS_DIR / "distilbert_best"
    test_texts_path = DATA_DIR / "test_texts.json"
    if distilbert_dir.exists() and test_texts_path.exists():
        print("\nEvaluating DistilBERT-OCR …")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(distilbert_dir))
            bert_model = AutoModelForSequenceClassification.from_pretrained(str(distilbert_dir))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            bert_model.to(device).eval()

            rows = json.loads(test_texts_path.read_text(encoding="utf-8"))
            texts = [(r.get("text") or "") for r in rows]
            bert_labels = np.array([int(r["label"]) for r in rows])

            all_probs = np.zeros((len(rows), len(LABEL_NAMES)), dtype=np.float32)
            batch_size = 16
            with torch.no_grad():
                for start in range(0, len(texts), batch_size):
                    batch = [t if t.strip() else " " for t in texts[start:start+batch_size]]
                    enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
                    out = bert_model(**enc)
                    all_probs[start:start+batch_size] = torch.softmax(out.logits, dim=1).cpu().numpy()

            bert_preds = all_probs.argmax(axis=1)
            results["DistilBERT-OCR"] = evaluate_model("DistilBERT-OCR", bert_preds, bert_labels)
        except Exception as e:
            print(f"  DistilBERT evaluation failed: {e}")
    else:
        if not distilbert_dir.exists():
            print("  models/distilbert_best/ not found, skipping DistilBERT.")
        if not test_texts_path.exists():
            print("  test_texts.json not found, skipping DistilBERT.")

    # --- Summary ---
    print("\n" + "="*50)
    print("  MODEL COMPARISON (test set)")
    print("="*50)
    for name, item in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"  {name:20s}: {item['accuracy']:.4f}")

    with open(REPORTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {REPORTS_DIR / 'results.json'}")

    # --- Comparison bar chart ---
    if len(results) > 1:
        print("\nGenerating comparison chart …")
        names = list(results.keys())
        accs = [results[n]["accuracy"] for n in names]
        colors = ["steelblue"] * len(names)
        best_idx = accs.index(max(accs))
        colors[best_idx] = "coral"

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_ylabel("test accuracy")
        ax.set_title("Model comparison — test set accuracy")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="random baseline")
        ax.legend()
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        chart_path = REPORTS_DIR / "comparison.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"  Comparison chart → {chart_path}")


if __name__ == "__main__":
    main()
