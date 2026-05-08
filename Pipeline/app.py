"""
=============================================================
  MALARIA CELL CLASSIFIER — LOCAL WEB APP (Flask)
=============================================================
  Run  : python app.py
  Open : http://localhost:5000
=============================================================
"""
import os
import io
import re
import uuid
import shutil
import sqlite3
import base64
import warnings
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_lib

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from PIL import Image
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_NAME  = "best_model.keras"
MODEL_PATH  = os.path.join(ROOT_DIR, "Dataset", "model", MODEL_NAME)
REPORT_PATH = os.path.join(ROOT_DIR, "Dataset", "reports", "classification_report.txt")
PLOTS_DIR   = os.path.join(ROOT_DIR, "Dataset", "plots")
IMG_SIZE   = (128, 128)
CLASSES    = ["Parasitized", "Uninfected"]

# ─── FEEDBACK / DATABASE ─────────────────────────────────────────────────────
APP_DIR          = os.path.dirname(os.path.abspath(__file__))
DB_PATH          = os.path.join(APP_DIR, "feedback.db")
FEEDBACK_IMG_DIR = os.path.join(APP_DIR, "static", "feedback_images")
os.makedirs(FEEDBACK_IMG_DIR, exist_ok=True)

app = Flask(__name__)


def init_db():
    """Create feedback table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path       TEXT    NOT NULL,
            model_prediction TEXT    NOT NULL,
            confidence       REAL    NOT NULL,
            true_label       TEXT,
            is_correct       INTEGER,
            timestamp        TEXT    NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def parse_report(path):
    metrics = {
        "loss": None,
        "accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "epochs": None,
        "training_time": None,
    }
    class_metrics = []

    if not os.path.exists(path):
        return metrics, class_metrics

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if "loss:" in stripped and metrics["loss"] is None:
                match = re.search(r"loss:\s*([0-9.]+)", stripped)
                if match:
                    metrics["loss"] = match.group(1)
            if stripped.startswith("accuracy") and "macro avg" not in stripped and "weighted avg" not in stripped:
                match = re.search(r"accuracy\s+([0-9.]+)", stripped)
                if match:
                    metrics["accuracy"] = match.group(1)
            if stripped.startswith("macro avg"):
                parts = stripped.split()
                if len(parts) >= 4:
                    metrics["macro_precision"] = parts[1]
                    metrics["macro_recall"]    = parts[2]
                    metrics["macro_f1"]        = parts[3]
            if stripped.startswith("Training time:"):
                match = re.search(r"Training time:\s*([0-9.]+)", stripped)
                if match:
                    metrics["training_time"] = match.group(1)
            if stripped.startswith("Epochs run:"):
                match = re.search(r"Epochs run:\s*(\d+)", stripped)
                if match:
                    metrics["epochs"] = match.group(1)
            if stripped.startswith("Parasitized") or stripped.startswith("Uninfected"):
                parts = stripped.split()
                if len(parts) >= 5:
                    class_metrics.append({
                        "label":     parts[0],
                        "precision": parts[1],
                        "recall":    parts[2],
                        "f1_score":  parts[3],
                        "support":   parts[4],
                    })
    return metrics, class_metrics


if not os.path.exists(MODEL_PATH):
    print(f"[FATAL] Model not found at: {MODEL_PATH}", flush=True)
    print(f"[FATAL] __file__   = {__file__}", flush=True)
    print(f"[FATAL] ROOT_DIR   = {ROOT_DIR}", flush=True)
    import sys; sys.exit(1)

print(f"[INFO] Loading model from {MODEL_PATH}", flush=True)
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("[INFO] Model loaded successfully.", flush=True)
except Exception as e:
    import traceback
    print(f"[FATAL] Failed to load model: {e}", flush=True)
    traceback.print_exc()
    import sys; sys.exit(1)

model_metrics, class_metrics = parse_report(REPORT_PATH)


def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, np.expand_dims(arr, 0)


def make_gradcam(mdl, img_batch):
    last_conv_name = None
    last_conv_idx  = None
    for i, layer in enumerate(mdl.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            last_conv_idx  = i

    if last_conv_name is None:
        return None

    try:
        conv_model = tf.keras.Model(
            inputs=mdl.inputs,
            outputs=mdl.get_layer(last_conv_name).output,
        )
        remaining = mdl.layers[last_conv_idx + 1:]

        with tf.GradientTape() as tape:
            conv_out = conv_model(img_batch, training=False)
            tape.watch(conv_out)
            x = conv_out
            for layer in remaining:
                x = layer(x, training=False)
            loss = x[:, 0]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None


def overlay_cam(img_arr, heatmap, alpha=0.45):
    if heatmap is None:
        return img_arr
    h_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(
            (img_arr.shape[1], img_arr.shape[0]), Image.BILINEAR
        )
    ) / 255.0
    colored = cm_lib.get_cmap("jet")(h_resized)[:, :, :3]
    return np.clip((1 - alpha) * img_arr + alpha * colored, 0, 1)


def arr_to_b64(arr):
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(arr, 0, 1), format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/plots/<filename>")
def serve_plot(filename):
    """Serve plot images from the Dataset/plots directory."""
    path = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="image/png")


@app.route("/bg.png")
def serve_bg():
    """Serve the background image from the app root directory."""
    root = os.path.dirname(os.path.abspath(__file__))
    for name in ("bg.png", "bg.png.png"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            return send_file(path, mimetype="image/png")
    return "Not found", 404


@app.route("/logo.png")
def serve_logo():
    """Serve logo.png from the app root directory."""
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "logo.png")
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "Not found", 404


@app.route("/")
def index():
    confusion_exists = os.path.exists(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    roc_exists       = os.path.exists(os.path.join(PLOTS_DIR, "roc_curve.png"))
    return render_template(
        "index.html",
        active_tab="home",
        metrics=model_metrics,
        class_metrics=class_metrics,
        model_name=MODEL_NAME,
        confusion_exists=confusion_exists,
        roc_exists=roc_exists,
    )


@app.route("/upload")
def upload():
    return render_template("upload.html", active_tab="upload", model_name=MODEL_NAME)


@app.route("/result", methods=["POST"])
def result():
    if "image" not in request.files:
        return redirect(url_for("upload"))

    file = request.files["image"]

    # --- Save image permanently for feedback / retraining ---
    img_id       = str(uuid.uuid4())
    img_filename = f"{img_id}.png"
    img_save_path = os.path.join(FEEDBACK_IMG_DIR, img_filename)
    file.seek(0)
    try:
        pil_img = Image.open(file.stream)
        pil_img.save(img_save_path)
    except Exception:
        return render_template(
            "upload.html",
            active_tab="upload",
            model_name=MODEL_NAME,
            error_message="Unable to read the uploaded image. Please choose a valid image file.",
        )

    img_arr, img_batch = preprocess(pil_img)
    prob = float(model.predict(img_batch, verbose=0)[0][0])
    pred_idx = int(prob >= 0.5)
    prediction = CLASSES[pred_idx]
    confidence_value        = prob if pred_idx == 1 else 1.0 - prob
    probability_parasitized = 100.0 * (1.0 - prob)
    probability_uninfected  = 100.0 * prob

    # --- Log prediction to feedback DB ---
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO feedback (image_path, model_prediction, confidence, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (img_save_path, prediction, round(confidence_value, 6), datetime.now().isoformat())
    )
    feedback_id = c.lastrowid
    conn.commit()
    conn.close()

    heatmap = make_gradcam(model, img_batch)
    overlay = overlay_cam(img_arr, heatmap)

    return render_template(
        "result.html",
        active_tab="upload",
        prediction=prediction,
        result_message=(
            "This cell shows signs of malaria infection (Plasmodium parasite detected)."
            if prediction == "Parasitized"
            else "This cell appears healthy — no malaria parasite was detected."
        ),
        confidence_pct=f"{confidence_value * 100:.1f}",
        probabilities=[
            {"label": "Parasitized", "percent": f"{probability_parasitized:.1f}", "color": "#ef4444"},
            {"label": "Uninfected",  "percent": f"{probability_uninfected:.1f}",  "color": "#22c55e"},
        ],
        raw_sigmoid=f"{prob:.6f}",
        original_b64=arr_to_b64(img_arr),
        gradcam_b64=arr_to_b64(overlay),
        model_name=MODEL_NAME,
        feedback_id=feedback_id,
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    """Receive user feedback (correct / wrong) for a prediction."""
    feedback_id = request.form.get("feedback_id")
    is_correct  = int(request.form.get("is_correct", 1))
    true_label  = request.form.get("true_label") or None

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE feedback SET is_correct = ?, true_label = ? WHERE id = ?",
        (is_correct, true_label, feedback_id)
    )
    conn.commit()
    conn.close()

    return render_template("feedback_thankyou.html", is_correct=is_correct)


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/feedback")
def admin_feedback():
    """Admin dashboard — view all collected feedback."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM feedback ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()

    total     = len(rows)
    correct   = sum(1 for r in rows if r[5] == 1)
    wrong     = sum(1 for r in rows if r[5] == 0)
    pending   = sum(1 for r in rows if r[5] is None)
    corrected = sum(1 for r in rows if r[4] is not None)

    return render_template(
        "admin_feedback.html",
        rows=rows,
        total=total,
        correct=correct,
        wrong=wrong,
        pending=pending,
        corrected=corrected,
    )


@app.route("/admin/retrain", methods=["POST"])
def admin_retrain():
    """Trigger model retraining via retrain.py."""
    import subprocess
    retrain_script = os.path.join(APP_DIR, "retrain.py")
    result = subprocess.run(
        ["python", retrain_script],
        capture_output=True, text=True, cwd=APP_DIR
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    return render_template("admin_feedback.html",
        retrain_output=output,
        rows=[], total=0, correct=0, wrong=0, pending=0, corrected=0
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        pil_img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    img_arr, img_batch = preprocess(pil_img)
    prob = float(model.predict(img_batch, verbose=0)[0][0])
    pred_idx   = int(prob >= 0.5)
    prediction = CLASSES[pred_idx]
    confidence = prob if pred_idx == 1 else 1.0 - prob

    return jsonify({
        "prediction":       prediction,
        "confidence":       round(confidence, 6),
        "raw_sigmoid":      round(prob, 6),
        "prob_parasitized": round(1.0 - prob, 6),
        "prob_uninfected":  round(prob, 6),
    })


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("    Malaria Cell Classifier  —  Web Interface")
    print("  Open your browser at: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
