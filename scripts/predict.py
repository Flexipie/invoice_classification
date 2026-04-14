"""
Single-image inference CLI.

Usage:
  python scripts/predict.py path/to/document.png [--model stacking|cnn|svd_svm|hog_svm|hog_rf]

Prints the predicted class and confidence.
If the document is classified as 'invoice', runs extraction automatically.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "data" / "processed"
sys.path.insert(0, str(Path(__file__).parent))

LABEL_NAMES = ["email", "invoice", "letter", "scientific_report"]
IMG_SIZE    = 128
HOG_SIZE    = 128
SVD_SIZE    = 64


def load_image_as_array(path: str) -> np.ndarray:
    """Load image (or first page of PDF) as grayscale uint8 numpy array."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            page = pdf.pages[0]
            pil = page.to_image(resolution=150).original.convert("L")
    else:
        pil = Image.open(p).convert("L")
    return np.array(pil, dtype=np.uint8)


def predict_cnn(img: np.ndarray) -> tuple[str, float]:
    import torch
    from torchvision import transforms
    from train_cnn import DocumentCNN

    ckpt = torch.load(ROOT / "models" / "cnn_best.pth", map_location="cpu")
    model = DocumentCNN(ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    with torch.no_grad():
        x = tfm(img).unsqueeze(0)
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        idx   = int(probs.argmax())
        conf  = float(probs[idx])

    return LABEL_NAMES[idx], conf


def predict_classical(img: np.ndarray, model_name: str) -> tuple[str, None]:
    import joblib
    from PIL import Image as PILImage

    model_path = ROOT / "models" / f"{model_name}.pkl"
    data = joblib.load(model_path)
    pipe = data["pipe"] if isinstance(data, dict) else data
    feature_type = data.get("feature", "svd") if isinstance(data, dict) else "svd"

    if feature_type == "hog":
        from skimage.feature import hog
        pil = PILImage.fromarray(img).resize((HOG_SIZE, HOG_SIZE))
        arr = np.array(pil, dtype=np.float32) / 255.0
        feat = hog(arr, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys",
                   feature_vector=True)
    else:
        pil = PILImage.fromarray(img).resize((SVD_SIZE, SVD_SIZE))
        feat = np.array(pil, dtype=np.float32).ravel()

    pred = pipe.predict(feat.reshape(1, -1))[0]
    return LABEL_NAMES[pred], None


def predict_stacking(img: np.ndarray) -> tuple[str, float]:
    import joblib
    from train_stacking import build_meta_features

    svd_svm_data = joblib.load(ROOT / "models" / "svd_svm.pkl")
    svd_svm = svd_svm_data if not isinstance(svd_svm_data, dict) else svd_svm_data["pipe"]
    hog_svm_data = joblib.load(ROOT / "models" / "hog_svm.pkl")
    hog_svm = hog_svm_data["pipe"] if isinstance(hog_svm_data, dict) else hog_svm_data
    hog_rf_data = joblib.load(ROOT / "models" / "hog_rf.pkl")
    hog_rf = hog_rf_data["pipe"] if isinstance(hog_rf_data, dict) else hog_rf_data

    meta_clf = joblib.load(ROOT / "models" / "stacking.pkl")
    M = build_meta_features(np.array([img]), svd_svm, hog_svm, hog_rf)
    probs = meta_clf.predict_proba(M)[0]
    idx  = int(probs.argmax())
    return LABEL_NAMES[idx], float(probs[idx])


def main():
    parser = argparse.ArgumentParser(description="Document classifier")
    parser.add_argument("image", help="Path to image or PDF")
    parser.add_argument(
        "--model",
        default="stacking",
        choices=["stacking", "cnn", "svd_svm", "hog_svm", "hog_rf"],
        help="Model to use (default: stacking)",
    )
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip invoice extraction even if classified as invoice")
    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    img = load_image_as_array(args.image)
    print(f"  Shape: {img.shape}")

    if args.model == "stacking":
        label, conf = predict_stacking(img)
        conf_str = f" (confidence: {conf:.2%})"
    elif args.model == "cnn":
        label, conf = predict_cnn(img)
        conf_str = f" (confidence: {conf:.2%})"
    else:
        label, _ = predict_classical(img, args.model)
        conf_str = ""

    print(f"\nPredicted class: {label}{conf_str}")

    if label == "invoice" and not args.no_extract:
        print("\nRunning invoice extraction …")
        from extract import extract_invoice_info
        info = extract_invoice_info(img)
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
