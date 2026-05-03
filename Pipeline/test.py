"""
=============================================================
  MALARIA CELL IMAGE CLASSIFIER — TEST & INFERENCE SCRIPT
=============================================================
  Model   : Custom CNN  (best_model.keras)
  Modes
    1. Single image   → python test.py --image path/to/cell.png
    2. Folder batch   → python test.py --folder path/to/folder
    3. Dataset eval   → python test.py --dataset  (uses val split of training data)
  Output  : console report + PNG visualisations saved next to this script
=============================================================
"""

import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm_lib

warnings.filterwarnings("ignore")

# ── dependency check ─────────────────────────────────────────────────────────
REQUIRED = ["tensorflow", "PIL", "seaborn", "sklearn"]
missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg if pkg != "PIL" else "Pillow")

if missing:
    print(f"\n[ERROR] Missing packages: {missing}")
    print(f"  Run: pip install {' '.join(missing)}\n")
    sys.exit(1)

import tensorflow as tf
from PIL import Image
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\harsh\OneDrive\Desktop\AI\ai_trained_dataset\model\best_model.keras"
DATA_DIR    = r"C:\Users\harsh\Downloads\archive (4)\cell_images\cell_images"
OUTPUT_DIR  = r"C:\Users\harsh\OneDrive\Desktop\AI\ai_trained_dataset"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
CLASSES     = ["Parasitized", "Uninfected"]   # index 0 / 1

BANNER = "=" * 60


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load the saved Keras model."""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at:\n  {MODEL_PATH}")
        sys.exit(1)
    print(f"[INFO] Loading model from:\n  {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Model loaded successfully. Input shape: {model.input_shape}")
    return model


def preprocess_image(path):
    """Load, resize, and normalise a single image → (1, H, W, 3)."""
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, np.expand_dims(arr, 0)


def confidence_bar_color(prob, threshold=0.5):
    """Return a color string based on confidence level."""
    if prob < 0.3 or prob > 0.7:
        return "#2ecc71"   # strong prediction → green
    return "#e67e22"       # uncertain → orange


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    Generate a Grad-CAM heatmap for the given image.
    Automatically finds the last Conv2D layer if not specified.
    """
    # Find last conv layer if not given
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        return None

    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output,
                     model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]   # binary sigmoid output

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")
        return None


def overlay_gradcam(original_img_arr, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original image."""
    if heatmap is None:
        return original_img_arr

    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(
            (original_img_arr.shape[1], original_img_arr.shape[0]),
            Image.BILINEAR
        )
    ) / 255.0

    colormap = cm_lib.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]

    superimposed = (1 - alpha) * original_img_arr + alpha * heatmap_colored
    return np.clip(superimposed, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — SINGLE IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(model, image_path):
    """Run inference on a single image and produce a detailed visual report."""
    print(f"\n{BANNER}")
    print("  MODE : SINGLE IMAGE PREDICTION")
    print(BANNER)
    print(f"  Image : {image_path}\n")

    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    img_arr, img_batch = preprocess_image(image_path)

    # Prediction
    prob = float(model.predict(img_batch, verbose=0)[0][0])
    pred_class = CLASSES[int(prob >= 0.5)]
    confidence = prob if prob >= 0.5 else 1.0 - prob

    # Grad-CAM
    heatmap = make_gradcam_heatmap(model, img_batch)
    overlay = overlay_gradcam(img_arr, heatmap)

    # ── Print result ─────────────────────────────────────────
    print(f"  ┌──────────────────────────────────────┐")
    print(f"  │  Prediction   : {pred_class:<22}│")
    print(f"  │  Confidence   : {confidence*100:>5.1f}%                │")
    print(f"  │  Raw sigmoid  : {prob:.6f}               │")
    print(f"  │  (0=Parasitized, 1=Uninfected)       │")
    print(f"  └──────────────────────────────────────┘")

    # ── Plot ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax_orig = fig.add_subplot(gs[0])
    ax_cam  = fig.add_subplot(gs[1])
    ax_conf = fig.add_subplot(gs[2])

    # Original image
    ax_orig.imshow(img_arr)
    ax_orig.set_title("Input Image", color="white", fontsize=12, pad=8)
    ax_orig.axis("off")
    ax_orig.set_facecolor("#1a1a2e")

    # Grad-CAM overlay
    ax_cam.imshow(overlay)
    ax_cam.set_title("Grad-CAM Heatmap", color="white", fontsize=12, pad=8)
    ax_cam.axis("off")
    ax_cam.set_facecolor("#1a1a2e")

    # Confidence bar chart
    class_probs = [1.0 - prob, prob]   # [Parasitized prob, Uninfected prob]
    bar_colors = ["#e74c3c", "#2ecc71"]
    bars = ax_conf.barh(CLASSES, class_probs, color=bar_colors, height=0.5)
    ax_conf.set_xlim(0, 1)
    ax_conf.set_xlabel("Probability", color="white", fontsize=10)
    ax_conf.set_title("Class Probabilities", color="white", fontsize=12, pad=8)
    ax_conf.tick_params(colors="white")
    ax_conf.set_facecolor("#16213e")
    for spine in ax_conf.spines.values():
        spine.set_edgecolor("#444")
    for bar, val in zip(bars, class_probs):
        ax_conf.text(min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                     f"{val*100:.1f}%", va="center", color="white", fontsize=10,
                     fontweight="bold")

    # Suptitle
    verdict_color = "#e74c3c" if pred_class == "Parasitized" else "#2ecc71"
    fig.suptitle(
        f"Prediction: {pred_class}  ({confidence*100:.1f}% confidence)",
        fontsize=14, fontweight="bold", color=verdict_color, y=1.02
    )

    out_path = os.path.join(RESULTS_DIR, "single_prediction.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  [INFO] Visual saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — FOLDER BATCH
# ─────────────────────────────────────────────────────────────────────────────

def predict_folder(model, folder_path, n_display=16):
    """
    Run inference on all .png/.jpg images in a folder.
    Prints a summary and saves a grid of predictions.
    """
    print(f"\n{BANNER}")
    print("  MODE : FOLDER BATCH PREDICTION")
    print(BANNER)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if os.path.splitext(f)[1].lower() in exts
    ]

    if not image_files:
        print(f"[ERROR] No supported images found in: {folder_path}")
        return

    print(f"  Found {len(image_files)} images in:\n  {folder_path}\n")

    results = []
    for idx, path in enumerate(image_files, 1):
        try:
            img_arr, img_batch = preprocess_image(path)
            prob = float(model.predict(img_batch, verbose=0)[0][0])
            pred_class = CLASSES[int(prob >= 0.5)]
            confidence = prob if prob >= 0.5 else 1.0 - prob
            results.append({
                "file": os.path.basename(path),
                "img":  img_arr,
                "prob": prob,
                "pred": pred_class,
                "conf": confidence,
            })
        except Exception as e:
            print(f"  [WARN] Skipped {os.path.basename(path)}: {e}")

        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(image_files)} …")

    # Summary
    parasitized = sum(1 for r in results if r["pred"] == "Parasitized")
    uninfected  = len(results) - parasitized

    print(f"\n  ┌──────────────────────────────────┐")
    print(f"  │  Total images   : {len(results):<15}│")
    print(f"  │  Parasitized    : {parasitized:<15}│")
    print(f"  │  Uninfected     : {uninfected:<15}│")
    print(f"  │  Avg confidence : {np.mean([r['conf'] for r in results])*100:>5.1f}%          │")
    print(f"  └──────────────────────────────────┘")

    # Per-file table (first 30)
    print(f"\n  {'File':<35} {'Prediction':<15} {'Prob':>8}  {'Confidence':>10}")
    print(f"  {'-'*75}")
    for r in results[:30]:
        print(f"  {r['file']:<35} {r['pred']:<15} {r['prob']:>8.4f}  {r['conf']*100:>8.1f}%")
    if len(results) > 30:
        print(f"  … and {len(results)-30} more (see the saved PNG for all)")

    # Grid visualisation (up to n_display)
    _save_prediction_grid(results[:n_display], title="Folder Batch Predictions",
                          out_name="folder_predictions.png")


def _save_prediction_grid(results, title, out_name, cols=4):
    """Save a grid of images with prediction labels."""
    n     = len(results)
    rows  = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3.5, rows * 4),
                              facecolor="#1a1a2e")
    fig.suptitle(title, fontsize=14, fontweight="bold", color="white", y=1.01)

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#1a1a2e")
        if i < n:
            r = results[i]
            ax.imshow(r["img"])
            color  = "#e74c3c" if r["pred"] == "Parasitized" else "#2ecc71"
            symbol = "🔴" if r["pred"] == "Parasitized" else "🟢"
            ax.set_title(
                f"{symbol} {r['pred']}\n{r['conf']*100:.1f}% conf",
                fontsize=8, color=color, pad=4
            )
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, out_name)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  [INFO] Grid visual saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — FULL DATASET EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(model):
    """
    Evaluate the model on the full labelled dataset (val split),
    and reproduce confusion matrix + ROC curve.
    """
    print(f"\n{BANNER}")
    print("  MODE : FULL DATASET EVALUATION")
    print(BANNER)

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] Data directory not found:\n  {DATA_DIR}")
        print("  Please check the DATA_DIR path at the top of test.py")
        return

    # Build a generator (no augmentation, just rescale)
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.15)

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = "binary",
        subset      = "validation",
        seed        = 42,
        shuffle     = False,
        classes     = CLASSES,
    )

    print(f"\n  Validation samples : {val_gen.samples:,}")
    print(f"  Class map          : {val_gen.class_indices}")
    print("\n  Running evaluation …")

    # Keras evaluate
    results = model.evaluate(val_gen, verbose=1)
    names   = model.metrics_names

    print(f"\n  {BANNER}")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*32}")
    for n, v in zip(names, results):
        print(f"  {n:<20} {v:>10.4f}")
    print(f"  {BANNER}")

    # Predictions
    val_gen.reset()
    y_prob = model.predict(val_gen, verbose=1).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = val_gen.classes

    print(f"\n  Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # ── Confusion Matrix ─────────────────────────────────────
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#1a1a2e")
    fig.suptitle("Confusion Matrix — Test Evaluation",
                 fontsize=15, fontweight="bold", color="white")

    for ax, data, fmt, ttl in zip(
        axes,
        [cm, cm_pct], ["d", ".1f"],
        ["Counts", "Percentage (%)"]
    ):
        ax.set_facecolor("#16213e")
        sns.heatmap(data, annot=True, fmt=fmt,
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    cmap="coolwarm", ax=ax, linewidths=0.5,
                    annot_kws={"size": 13},
                    cbar_kws={"shrink": 0.8})
        ax.set_title(ttl, fontsize=12, color="white")
        ax.set_ylabel("True Label", fontsize=11, color="white")
        ax.set_xlabel("Predicted Label", fontsize=11, color="white")
        ax.tick_params(colors="white")

    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  [INFO] Confusion matrix saved → {cm_path}")

    # ── ROC Curve ────────────────────────────────────────────
    auc_score = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
            label=f"ROC Curve  (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#e74c3c")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12, color="white")
    ax.set_ylabel("True Positive Rate",  fontsize=12, color="white")
    ax.set_title("ROC Curve — Malaria Cell Classifier",
                 fontsize=14, fontweight="bold", color="white")
    ax.legend(fontsize=11, facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(alpha=0.2, color="white")

    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, "test_roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [INFO] ROC curve saved → {roc_path}")

    # ── Sample Predictions Grid ───────────────────────────────
    val_gen.reset()
    imgs_batch, _ = next(val_gen)
    batch_prob     = model.predict(imgs_batch, verbose=0).flatten()
    batch_pred     = (batch_prob >= 0.5).astype(int)
    batch_true     = val_gen.classes[:len(imgs_batch)]

    sample_results = [
        {
            "img":  imgs_batch[i],
            "prob": batch_prob[i],
            "pred": CLASSES[batch_pred[i]],
            "conf": batch_prob[i] if batch_pred[i] == 1 else 1 - batch_prob[i],
            "correct": int(batch_true[i]) == batch_pred[i],
            "true": CLASSES[int(batch_true[i])],
        }
        for i in range(min(16, len(imgs_batch)))
    ]

    n, cols = len(sample_results), 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3.5, rows * 4),
                              facecolor="#1a1a2e")
    fig.suptitle("Sample Predictions  (✓ correct  |  ✗ wrong)",
                 fontsize=13, fontweight="bold", color="white", y=1.01)

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#1a1a2e")
        if i < n:
            r = sample_results[i]
            ax.imshow(r["img"])
            color  = "#2ecc71" if r["correct"] else "#e74c3c"
            symbol = "✓" if r["correct"] else "✗"
            ax.set_title(
                f"{symbol}  P: {r['pred']}\nT: {r['true']}\n{r['conf']*100:.1f}%",
                fontsize=7.5, color=color, pad=4
            )
        ax.axis("off")

    plt.tight_layout()
    sp_path = os.path.join(RESULTS_DIR, "test_sample_predictions.png")
    plt.savefig(sp_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [INFO] Sample predictions saved → {sp_path}")

    # Final summary line
    print(f"\n  {'='*50}")
    print(f"  ✅  Evaluation complete")
    print(f"  Accuracy : {results[names.index('accuracy')]*100:.2f}%")
    print(f"  AUC      : {auc_score:.4f}")
    print(f"  Results  → {RESULTS_DIR}")
    print(f"  {'='*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Malaria CNN — Test & Inference",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test.py --dataset\n"
            "  python test.py --image  path/to/cell.png\n"
            "  python test.py --folder path/to/images/\n"
        )
    )
    parser.add_argument("--image",   type=str, help="Path to a single cell image")
    parser.add_argument("--folder",  type=str, help="Path to a folder of images")
    parser.add_argument("--dataset", action="store_true",
                        help="Evaluate on the full labelled dataset (val split)")

    args = parser.parse_args()

    if not (args.image or args.folder or args.dataset):
        # Default: run dataset evaluation
        print("[INFO] No mode specified. Defaulting to --dataset evaluation.")
        args.dataset = True

    print(f"\n{BANNER}")
    print(f"  MALARIA CELL CLASSIFIER — TEST / INFERENCE")
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPUs found : {len(tf.config.list_physical_devices('GPU'))}")
    print(f"{BANNER}\n")

    model = load_model()
    model.summary(print_fn=lambda x: None)   # suppress noisy summary

    if args.image:
        predict_single(model, args.image)
    elif args.folder:
        predict_folder(model, args.folder)
    elif args.dataset:
        evaluate_dataset(model)


if __name__ == "__main__":
    main()
