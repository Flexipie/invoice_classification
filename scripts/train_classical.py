"""
Train classical CV models for document classification.

Model A — SVD + SVM
  Flatten grayscale images → TruncatedSVD(200 components) → SVC(rbf)
  Captures global structure through principal directions of pixel variance.

Model B — HOG + SVM  /  HOG + Random Forest
  skimage HOG features (orientations=9, pixels_per_cell=8, cells_per_block=2)
  encodes local gradient distributions — robust to illumination changes.
  Two classifiers compared: SVM (linear kernel) and Random Forest.

Trained models are saved as joblib files under models/.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

HOG_SIZE = 128   # images must be this size for HOG feature count to be consistent
SVD_SIZE = 64    # smaller resize for SVD to keep memory manageable


def load_npy(split: str):
    images = np.load(DATA_DIR / f"{split}_images.npy")   # (N, H, W) uint8
    labels = np.load(DATA_DIR / f"{split}_labels.npy")   # (N,)
    return images, labels


def load_label_map() -> dict:
    with open(DATA_DIR / "label_map.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def extract_svd_features(images: np.ndarray, size: int = SVD_SIZE) -> np.ndarray:
    """Resize to size×size and flatten to 1-D vectors."""
    from PIL import Image
    feats = []
    for img in images:
        pil = Image.fromarray(img).resize((size, size))
        feats.append(np.array(pil, dtype=np.float32).ravel())
    return np.stack(feats)


def extract_hog_features(images: np.ndarray, size: int = HOG_SIZE) -> np.ndarray:
    """Extract HOG descriptor from each image."""
    from PIL import Image
    feats = []
    for img in images:
        pil = Image.fromarray(img).resize((size, size))
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


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_svd_svm(X_train, y_train, X_val, y_val, label_map):
    print("\n--- Model A: SVD + SVM ---")
    print("  Extracting pixel features …")
    X_tr = extract_svd_features(X_train)
    X_v  = extract_svd_features(X_val)

    pipe = Pipeline([
        ("svd",   TruncatedSVD(n_components=200, random_state=42)),
        ("scale", StandardScaler()),
        ("svm",   SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                      random_state=42)),
    ])

    print("  Fitting SVD + SVM (this may take a few minutes) …")
    pipe.fit(X_tr, y_train)

    val_preds = pipe.predict(X_v)
    acc = accuracy_score(y_val, val_preds)
    print(f"  Validation accuracy: {acc:.4f}")

    names = [label_map[i] for i in sorted(label_map)]
    print(classification_report(y_val, val_preds, target_names=names))

    out = MODELS_DIR / "svd_svm.pkl"
    joblib.dump(pipe, out)
    print(f"  Saved → {out}")
    return pipe, acc


def train_hog_svm(X_train, y_train, X_val, y_val, label_map):
    print("\n--- Model B1: HOG + SVM ---")
    print("  Extracting HOG features …")
    X_tr = extract_hog_features(X_train)
    X_v  = extract_hog_features(X_val)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("svm",   SVC(kernel="linear", C=1.0, class_weight="balanced",
                      random_state=42)),
    ])

    print("  Fitting HOG + SVM …")
    pipe.fit(X_tr, y_train)

    val_preds = pipe.predict(X_v)
    acc = accuracy_score(y_val, val_preds)
    print(f"  Validation accuracy: {acc:.4f}")

    names = [label_map[i] for i in sorted(label_map)]
    print(classification_report(y_val, val_preds, target_names=names))

    out = MODELS_DIR / "hog_svm.pkl"
    joblib.dump({"pipe": pipe, "feature": "hog"}, out)
    print(f"  Saved → {out}")
    return pipe, acc


def train_hog_rf(X_train, y_train, X_val, y_val, label_map):
    print("\n--- Model B2: HOG + Random Forest ---")
    print("  Extracting HOG features …")
    X_tr = extract_hog_features(X_train)
    X_v  = extract_hog_features(X_val)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("rf",    RandomForestClassifier(n_estimators=200, max_depth=None,
                                         n_jobs=-1, random_state=42,
                                         class_weight="balanced")),
    ])

    print("  Fitting HOG + Random Forest …")
    pipe.fit(X_tr, y_train)

    val_preds = pipe.predict(X_v)
    acc = accuracy_score(y_val, val_preds)
    print(f"  Validation accuracy: {acc:.4f}")

    names = [label_map[i] for i in sorted(label_map)]
    print(classification_report(y_val, val_preds, target_names=names))

    out = MODELS_DIR / "hog_rf.pkl"
    joblib.dump({"pipe": pipe, "feature": "hog"}, out)
    print(f"  Saved → {out}")
    return pipe, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    label_map = load_label_map()

    print("Loading preprocessed data …")
    X_train, y_train = load_npy("train")
    X_val,   y_val   = load_npy("validation")
    print(f"  train: {X_train.shape}  val: {X_val.shape}")

    results = {}

    _, acc = train_svd_svm(X_train, y_train, X_val, y_val, label_map)
    results["SVD+SVM"] = acc

    _, acc = train_hog_svm(X_train, y_train, X_val, y_val, label_map)
    results["HOG+SVM"] = acc

    _, acc = train_hog_rf(X_train, y_train, X_val, y_val, label_map)
    results["HOG+RF"] = acc

    print("\n=== Summary ===")
    for name, acc in results.items():
        print(f"  {name:15s}: {acc:.4f}")

    print("\nNext step: python scripts/train_cnn.py")


if __name__ == "__main__":
    main()
