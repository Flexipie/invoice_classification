"""
Stacking meta-classifier.

Level-0 base models (already trained):
  SVD+SVM, HOG+SVM, HOG+RF  →  decision_function / predict_proba  (4 values each)
  CNN                         →  softmax probabilities              (4 values)

Level-1 meta-classifier:
  LogisticRegression trained on the 16-feature meta-feature vector produced
  by concatenating all base model outputs on the *validation* set.
  Using the validation set (not training set) avoids leaking base-model
  training data into the meta-learner.

Saved to models/stacking.pkl
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

LABEL_NAMES = ["email", "invoice", "letter", "scientific_report"]
IMG_SIZE    = 128
HOG_SIZE    = 128
SVD_SIZE    = 64

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Feature extraction (mirrors train_classical.py)
# ---------------------------------------------------------------------------

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
        feats.append(hog(arr, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), block_norm="L2-Hys",
                         feature_vector=True))
    return np.stack(feats)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Per-model meta-feature extraction
# ---------------------------------------------------------------------------

def meta_features_svd_svm(images, pipe) -> np.ndarray:
    """decision_function → softmax-normalised probabilities (N, 4)."""
    X = extract_svd_features(images)
    # SVD transform first (all steps except final SVM)
    X_transformed = pipe[:-1].transform(X)
    scores = pipe[-1].decision_function(X_transformed)   # (N, 4)
    return softmax(scores)


def meta_features_hog_svm(images, pipe) -> np.ndarray:
    X = extract_hog_features(images)
    X_transformed = pipe[:-1].transform(X)
    scores = pipe[-1].decision_function(X_transformed)
    return softmax(scores)


def meta_features_hog_rf(images, pipe) -> np.ndarray:
    X = extract_hog_features(images)
    return pipe.predict_proba(X)                         # (N, 4)


def meta_features_cnn(images) -> np.ndarray:
    import torch
    from torchvision import transforms
    from train_cnn import DocumentCNN

    ckpt  = torch.load(MODELS_DIR / "cnn_best.pth", map_location="cpu")
    model = DocumentCNN(ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    probs = []
    with torch.no_grad():
        for img in images:
            x   = tfm(img).unsqueeze(0)
            out = model(x)
            probs.append(torch.softmax(out, dim=1)[0].numpy())
    return np.stack(probs)                               # (N, 4)


def build_meta_features(images, svd_svm, hog_svm, hog_rf) -> np.ndarray:
    print("  SVD+SVM scores …")
    f1 = meta_features_svd_svm(images, svd_svm)
    print("  HOG+SVM scores …")
    f2 = meta_features_hog_svm(images, hog_svm)
    print("  HOG+RF probabilities …")
    f3 = meta_features_hog_rf(images, hog_rf)
    print("  CNN probabilities …")
    f4 = meta_features_cnn(images)
    return np.hstack([f1, f2, f3, f4])                  # (N, 16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load base models
    print("Loading base models …")
    svd_svm_data = joblib.load(MODELS_DIR / "svd_svm.pkl")
    svd_svm = svd_svm_data if not isinstance(svd_svm_data, dict) else svd_svm_data["pipe"]

    hog_svm_data = joblib.load(MODELS_DIR / "hog_svm.pkl")
    hog_svm = hog_svm_data["pipe"] if isinstance(hog_svm_data, dict) else hog_svm_data

    hog_rf_data = joblib.load(MODELS_DIR / "hog_rf.pkl")
    hog_rf = hog_rf_data["pipe"] if isinstance(hog_rf_data, dict) else hog_rf_data

    # ----- Build meta-features on validation set (meta-training data) -----
    print("\nBuilding meta-features on validation set …")
    X_val = np.load(DATA_DIR / "validation_images.npy")
    y_val = np.load(DATA_DIR / "validation_labels.npy")
    M_val = build_meta_features(X_val, svd_svm, hog_svm, hog_rf)
    print(f"  Meta-feature matrix: {M_val.shape}")

    # ----- Train meta-classifier -----
    print("\nTraining meta-classifier (LogisticRegression) …")
    meta_clf = Pipeline([
        ("scale", StandardScaler()),
        ("lr",    LogisticRegression(C=1.0, max_iter=1000,
                                     multi_class="multinomial",
                                     solver="lbfgs", random_state=42)),
    ])
    meta_clf.fit(M_val, y_val)

    val_preds = meta_clf.predict(M_val)
    val_acc   = accuracy_score(y_val, val_preds)
    print(f"  Meta-train (val set) accuracy: {val_acc:.4f}")

    # ----- Evaluate on test set -----
    print("\nBuilding meta-features on test set …")
    X_test = np.load(DATA_DIR / "test_images.npy")
    y_test = np.load(DATA_DIR / "test_labels.npy")
    M_test = build_meta_features(X_test, svd_svm, hog_svm, hog_rf)

    test_preds = meta_clf.predict(M_test)
    test_acc   = accuracy_score(y_test, test_preds)

    print(f"\n{'='*50}")
    print(f"  Stacking ensemble  (test accuracy: {test_acc:.4f})")
    print(f"{'='*50}")
    print(classification_report(y_test, test_preds, target_names=LABEL_NAMES))

    # Confusion matrix
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    cm   = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Stacking ensemble  (test acc={test_acc:.3f})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "cm_stacking.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix → reports/cm_stacking.png")

    # Save model
    joblib.dump(meta_clf, MODELS_DIR / "stacking.pkl")
    print(f"  Model saved → models/stacking.pkl")

    # Update results.json
    results_path = REPORTS_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}
    results["Stacking"] = test_acc
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Final model comparison (test set) ===")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← best" if acc == max(results.values()) else ""
        print(f"  {name:20s}: {acc:.4f}{marker}")


if __name__ == "__main__":
    main()
