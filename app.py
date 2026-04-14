"""
Flask API for document classification and invoice extraction.

Endpoints:
  POST /classify   — classify an uploaded image or PDF
  POST /extract    — classify + extract invoice fields if invoice

Both endpoints accept multipart/form-data with a 'file' field.

Run:
  python app.py                    # default port 5000
  PORT=8080 python app.py          # custom port
"""

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared state: load best model once at startup
# ---------------------------------------------------------------------------

LABEL_NAMES = ["email", "invoice", "letter", "scientific_report"]
IMG_SIZE    = 128

_cnn_model    = None
_cnn_loaded   = False


def get_cnn_model():
    global _cnn_model, _cnn_loaded
    if not _cnn_loaded:
        try:
            import torch
            from train_cnn import DocumentCNN
            ckpt_path = Path(__file__).parent / "models" / "cnn_best.pth"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model = DocumentCNN(ckpt["num_classes"])
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            _cnn_model = model
            print("[startup] CNN model loaded.")
        except Exception as e:
            print(f"[startup] CNN not available: {e}")
        _cnn_loaded = True
    return _cnn_model


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_from_request() -> np.ndarray:
    if "file" not in request.files:
        raise ValueError("No 'file' field in request.")
    f = request.files["file"]
    data = f.read()

    if f.filename and f.filename.lower().endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pil = pdf.pages[0].to_image(resolution=150).original.convert("L")
    else:
        pil = Image.open(io.BytesIO(data)).convert("L")

    return np.array(pil, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def classify_image(img: np.ndarray) -> tuple[str, float | None]:
    model = get_cnn_model()
    if model is not None:
        import torch
        from torchvision import transforms

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

    # Fallback: HOG + SVM if CNN not available
    import joblib
    from skimage.feature import hog

    hog_path = Path(__file__).parent / "models" / "hog_svm.pkl"
    if not hog_path.exists():
        raise RuntimeError("No trained model found. Run train_cnn.py or train_classical.py first.")

    data = joblib.load(hog_path)
    pipe = data["pipe"] if isinstance(data, dict) else data

    pil = Image.fromarray(img).resize((128, 128))
    arr = np.array(pil, dtype=np.float32) / 255.0
    feat = hog(arr, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    pred = int(pipe.predict(feat.reshape(1, -1))[0])
    return LABEL_NAMES[pred], None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/classify", methods=["POST"])
def classify():
    try:
        img = load_image_from_request()
        label, confidence = classify_image(img)
        resp = {"label": label}
        if confidence is not None:
            resp["confidence"] = round(confidence, 4)
        return jsonify(resp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract", methods=["POST"])
def extract():
    try:
        img = load_image_from_request()
        label, confidence = classify_image(img)
        resp = {"label": label}
        if confidence is not None:
            resp["confidence"] = round(confidence, 4)

        if label == "invoice":
            from extract import extract_invoice_info
            resp["fields"] = extract_invoice_info(img)
        return jsonify(resp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    cnn_ready = (get_cnn_model() is not None)
    return jsonify({"status": "ok", "cnn_ready": cnn_ready})


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pre-load model
    get_cnn_model()
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port} …")
    app.run(host="0.0.0.0", port=port, debug=False)
