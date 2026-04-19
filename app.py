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
import os
import sys
import socket
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

app = Flask(__name__)

ROOT_DIR = Path(__file__).parent
REPORTS_DIR = ROOT_DIR / "reports"

# ---------------------------------------------------------------------------
# Shared state: load best model once at startup
# ---------------------------------------------------------------------------

LABEL_NAMES = ["email", "invoice", "letter", "scientific_report"]
IMG_SIZE    = 128

_stack        = None
_stack_loaded = False


def get_stack():
    global _stack, _stack_loaded
    if not _stack_loaded:
        try:
            import joblib
            from train_stacking import build_meta_features  # noqa: F401 — imported for use below

            models_dir = Path(__file__).parent / "models"
            svd_raw  = joblib.load(models_dir / "svd_svm.pkl")
            hog_svm_raw = joblib.load(models_dir / "hog_svm.pkl")
            hog_rf_raw  = joblib.load(models_dir / "hog_rf.pkl")
            meta_clf = joblib.load(models_dir / "stacking.pkl")

            _stack = {
                "svd_svm":  svd_raw  if not isinstance(svd_raw,  dict) else svd_raw["pipe"],
                "hog_svm":  hog_svm_raw["pipe"] if isinstance(hog_svm_raw, dict) else hog_svm_raw,
                "hog_rf":   hog_rf_raw["pipe"]  if isinstance(hog_rf_raw,  dict) else hog_rf_raw,
                "meta_clf": meta_clf,
            }
            print("[startup] Stacking ensemble loaded.")
        except Exception as e:
            print(f"[startup] Stacking not available: {e}")
        _stack_loaded = True
    return _stack


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
    # Primary: stacking ensemble
    stack = get_stack()
    if stack is not None:
        from train_stacking import build_meta_features
        M = build_meta_features(np.array([img]),
                                stack["svd_svm"], stack["hog_svm"], stack["hog_rf"])
        probs = stack["meta_clf"].predict_proba(M)[0]
        idx   = int(probs.argmax())
        return LABEL_NAMES[idx], float(probs[idx])

    # Fallback: CNN alone
    try:
        import torch
        from torchvision import transforms
        from train_cnn import DocumentCNN

        ckpt_path = Path(__file__).parent / "models" / "cnn_best.pth"
        ckpt  = torch.load(ckpt_path, map_location="cpu")
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
            x   = tfm(img).unsqueeze(0)
            out = model(x)
            probs = torch.softmax(out, dim=1)[0]
            idx   = int(probs.argmax())
        return LABEL_NAMES[idx], float(probs[idx])
    except Exception:
        pass

    raise RuntimeError("No trained model found. Run train_stacking.py first.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/upload", methods=["GET"])
def upload():
    return redirect(url_for("classify_page"))


@app.route("/classify", methods=["GET"])
def classify_page():
    return render_template("classify.html")


@app.route("/performance", methods=["GET"])
def performance_page():
    return render_template("performance.html")


@app.route("/demo", methods=["GET"])
def demo():
    return redirect(url_for("classify_page"))

@app.route("/classify", methods=["POST"])
def classify():
    try:
        img = load_image_from_request()
        label, confidence = classify_image(img)
        resp: dict[str, object] = {"label": label}
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
        resp: dict[str, object] = {"label": label}
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
    stack_ready = (get_stack() is not None)
    return jsonify({"status": "ok", "stacking_ready": stack_ready})


@app.route("/reports/<path:filename>")
def reports_file(filename):
    return send_from_directory(REPORTS_DIR, filename)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    get_stack()
    requested_port = int(os.environ.get("PORT", 5000))

    def find_free_port(preferred_port: int) -> int:
        for port in (preferred_port, 8080, 5001, 8000):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if sock.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        raise RuntimeError("No free local port found.")

    port = find_free_port(requested_port)
    print(f"Starting server on http://127.0.0.1:{port} …")
    app.run(host="0.0.0.0", port=port, debug=False)
