#!/usr/bin/env bash
# Create .venv, install Python deps, download spaCy model, verify Tesseract.
# Usage: from this directory, run:  bash setup_env.sh
#        or:  chmod +x setup_env.sh && ./setup_env.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

echo "==> Project: $ROOT"

# --- Python version (3.9+; spaCy pin in requirements.txt is for 3.9) ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.9+ and try again."
  exit 1
fi

ver="$(
  python3 - <<'PY'
import sys
print("%d.%d" % sys.version_info[:2])
PY
)"
major="${ver%%.*}"
minor="${ver#*.}"
if [[ "$major" -lt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "${minor:-0}" -lt 9 ]]; }; then
  echo "ERROR: Need Python 3.9 or newer (found $(python3 --version 2>&1))."
  exit 1
fi

echo "==> Using $(python3 --version)"

# --- venv ---
if [[ ! -x "$PY" ]]; then
  echo "==> Creating virtual environment: $VENV"
  python3 -m venv "$VENV"
else
  echo "==> Virtual environment already exists: $VENV"
fi

echo "==> Upgrading pip"
"$PIP" install --upgrade pip

echo "==> Installing requirements"
"$PIP" install -r "$ROOT/requirements.txt"

echo "==> Downloading spaCy model en_core_web_sm"
"$PY" -m spacy download en_core_web_sm

# --- Optional: verify Hugging Face access (streaming; does not mirror the full dataset) ---
echo "==> Checking Hugging Face dataset (metadata only; full corpus downloads on first ocr.py run)"
if "$PY" - <<'PY'
from datasets import load_dataset
ds = load_dataset("chainyo/rvl-cdip", split="train", streaming=True)
next(iter(ds))
print("  OK: chainyo/rvl-cdip is reachable.")
PY
then
  :
else
  echo "  WARNING: Could not reach chainyo/rvl-cdip (offline or no network). OCR will try again when you run it."
fi

# --- Tesseract (system binary) ---
echo "==> Checking Tesseract OCR"
if ! command -v tesseract >/dev/null 2>&1; then
  echo ""
  echo "WARNING: 'tesseract' not found on PATH. pytesseract needs the system binary."
  echo "  macOS:  brew install tesseract"
  echo "  Ubuntu: sudo apt-get install tesseract-ocr"
  echo ""
else
  echo "  Found: $(command -v tesseract) ($("$PY" -c "import pytesseract as pt; print(pt.get_tesseract_version())" 2>/dev/null || echo 'version check failed'))"
fi

echo ""
echo "Done. Activate the environment and run the pipeline:"
echo "  source \"$VENV/bin/activate\""
echo "  python scripts/ocr.py"
echo ""
