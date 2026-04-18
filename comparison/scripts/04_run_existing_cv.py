"""
Run the repo's existing CV pipeline end-to-end, without modifying any file.

Each existing script writes to reports/results.json and saves models under
models/. We just orchestrate them in order, forwarding stdout live.

Steps:
  1. scripts/train_classical.py   (SVD+SVM, HOG+SVM, HOG+RF)
  2. scripts/train_cnn.py         (CNN)
  3. scripts/train_stacking.py    (Stacking meta-classifier + also writes test accs)
  4. scripts/evaluate.py          (rebuilds results.json for the 4 base models)

Order matters: evaluate.py overwrites results.json, so we run it BEFORE
train_stacking.py — which both evaluates stacking on the test set and appends
the stacking entry to results.json.

Any step that has already produced its model file is skipped if SKIP_EXISTING=1.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
MODELS_DIR = ROOT / "models"

SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "1") != "0"

# (script, output files that, if all exist, allow skipping)
PIPELINE = [
    ("train_classical.py", [MODELS_DIR / "svd_svm.pkl",
                            MODELS_DIR / "hog_svm.pkl",
                            MODELS_DIR / "hog_rf.pkl"]),
    ("train_cnn.py",       [MODELS_DIR / "cnn_best.pth"]),
    ("evaluate.py",        []),               # always run — writes test metrics
    ("train_stacking.py",  [MODELS_DIR / "stacking.pkl"]),
]


def run(script: str) -> None:
    print(f"\n=== Running scripts/{script} ===", flush=True)
    # Force non-interactive matplotlib backend (evaluate.py otherwise crashes
    # on Windows with a Tcl thread error when Python shuts down). Also ensure
    # UTF-8 stdout so the repo's unicode arrows don't crash cp1252.
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    rc = subprocess.call([sys.executable, str(SCRIPTS_DIR / script)], cwd=ROOT, env=env)
    if rc != 0:
        raise SystemExit(f"{script} failed with exit code {rc}")


def main() -> None:
    print("=== 04_run_existing_cv ===")
    print(f"SKIP_EXISTING: {SKIP_EXISTING}")

    for script, output_files in PIPELINE:
        if SKIP_EXISTING and output_files and all(p.exists() for p in output_files):
            print(f"\n[skip] {script}: outputs already present ({[str(p.name) for p in output_files]})")
            continue
        run(script)

    print("\nDone. Next: python comparison/scripts/05_compare_all.py")


if __name__ == "__main__":
    main()
