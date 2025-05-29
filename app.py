"""
app.py  –  REST wrapper for Untitled.py
---------------------------------------
• POST /train   – re‑runs Untitled.py, regenerates fraud_pipe.joblib
• POST /predict – expects {"features": [...]}, returns score + boolean
"""
import os
import subprocess
import joblib
import numpy as np
from flask import Flask, request, jsonify, abort

MODEL_PATH = "fraud_pipe.joblib"
TRAIN_SCRIPT = "final_(1)-Copy.py"

app = Flask(__name__)
pipe = None                       # will hold {"scaler": …, "model": …, "thr": …}

# ─────────────────────────────────────────────────────────────
def load_pipe():
    global pipe
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found – run /train first to generate it.")
    pipe = joblib.load(MODEL_PATH)

def run_training():
    """Execute Untitled.py and return (exit_code, stdout, stderr)."""
    proc = subprocess.run(
        ["python", TRAIN_SCRIPT],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr
# ─────────────────────────────────────────────────────────────

@app.route("/ping", methods=["GET"])
def ping():
    return {"status": "ok", "model_loaded": pipe is not None}

@app.route("/train", methods=["POST"])
def train():
    """
    Trigger a full re‑train.  (Keep this endpoint protected
    or internal only in production!)
    """
    exit_code, out, err = run_training()
    if exit_code != 0:
        return (
            jsonify(
                {"status": "error", "exit_code": exit_code, "stderr": err[:500]}
            ),
            500,
        )

    # reload the new artefact
    try:
        load_pipe()
    except Exception as e:
        return (
            jsonify({"status": "error", "detail": f"Model load failed: {e}"}), 500
        )

    return {"status": "success", "stdout_tail": out[-500:]}

@app.route("/predict", methods=["POST"])
def predict():
    if pipe is None:
        abort(400, "Model not loaded – call /train first or upload fraud_pipe.joblib")

    data = request.get_json(force=True)
    feats = np.asarray(data.get("features"), dtype=float)

    if feats.shape != (30,):
        abort(400, "Expecting 30 numerical values in 'features'.")

    scaler, model, thr = pipe["scaler"], pipe["model"], pipe["thr"]
    score = model.predict_proba(scaler.transform(feats.reshape(1, -1)))[0, 1]
    label = bool(score >= thr)

    return jsonify({"score": float(score), "fraud": label})

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load model on cold‑start if it exists
    if os.path.exists(MODEL_PATH):
        load_pipe()
    app.run(host="0.0.0.0", port=5000)
