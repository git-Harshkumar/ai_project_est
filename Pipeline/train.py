"""
=============================================================
  MALARIA CELL IMAGE CLASSIFICATION - COMPLETE ML PIPELINE
=============================================================
  Dataset : NIH Malaria Cell Images
  Classes : Parasitized | Uninfected
  Model   : Custom CNN (TensorFlow / Keras)
  Outputs : D:\\ai_trained_dataset\\
=============================================================
"""

import os
import sys
import time
import shutil
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── 0. Dependency check ─────────────────────────────────────────────────────
REQUIRED = ["tensorflow", "sklearn", "PIL", "seaborn"]
missing  = []
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
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import seaborn as sns
from PIL import Image

print(f"\n{'='*60}")
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPUs found : {len(tf.config.list_physical_devices('GPU'))}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\harsh\Downloads\archive (4)\cell_images\cell_images"
OUTPUT_DIR  = r"D:\ai_trained_dataset"

IMG_SIZE    = (128, 128)          # resize target
BATCH_SIZE  = 32
EPOCHS      = 30                  # max epochs (early stopping kicks in)
VAL_SPLIT   = 0.15               # 15% validation
TEST_SPLIT  = 0.15               # 15% test  (remaining 70% train)
SEED        = 42
CLASSES     = ["Parasitized", "Uninfected"]

# Output sub-dirs
MODEL_DIR   = os.path.join(OUTPUT_DIR, "model")
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR    = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR  = os.path.join(OUTPUT_DIR, "reports")

for d in [MODEL_DIR, LOG_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[INFO] Data   → {DATA_DIR}")
print(f"[INFO] Output → {OUTPUT_DIR}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET INSPECTION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 50)
print("  STEP 1 : DATASET INSPECTION")
print("=" * 50)

class_counts = {}
corrupt_files = []

for cls in CLASSES:
    cls_path = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(cls_path):
        print(f"[ERROR] Folder not found: {cls_path}")
        sys.exit(1)
    files = [f for f in os.listdir(cls_path) if f.lower().endswith(".png")]
    class_counts[cls] = len(files)

    # Quick corruption check (first 20 images)
    for f in files[:20]:
        try:
            img = Image.open(os.path.join(cls_path, f))
            img.verify()
        except Exception:
            corrupt_files.append(os.path.join(cls_path, f))

total = sum(class_counts.values())
print(f"\n  {'Class':<18} {'Images':>8}  {'%':>6}")
print(f"  {'-'*34}")
for cls, cnt in class_counts.items():
    print(f"  {cls:<18} {cnt:>8,}  {cnt/total*100:>5.1f}%")
print(f"  {'-'*34}")
print(f"  {'TOTAL':<18} {total:>8,}  100.0%")

if corrupt_files:
    print(f"\n[WARN] {len(corrupt_files)} possibly corrupt files found (spot-check only).")
else:
    print("\n[INFO] Spot-check: No corrupt files detected.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING & DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STEP 2 : PREPROCESSING & DATA LOADERS")
print("=" * 50)

# --- Train augmentation ---
train_datagen = ImageDataGenerator(
    rescale           = 1.0 / 255,
    validation_split  = VAL_SPLIT,
    rotation_range    = 20,
    width_shift_range = 0.15,
    height_shift_range= 0.15,
    shear_range       = 0.10,
    zoom_range        = 0.15,
    horizontal_flip   = True,
    vertical_flip     = False,
    brightness_range  = [0.85, 1.15],
    fill_mode         = "nearest",
)

# --- Val / Test use only rescaling ---
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size   = IMG_SIZE,
    batch_size    = BATCH_SIZE,
    class_mode    = "binary",
    subset        = "training",
    seed          = SEED,
    shuffle       = True,
    classes       = CLASSES,
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size   = IMG_SIZE,
    batch_size    = BATCH_SIZE,
    class_mode    = "binary",
    subset        = "validation",
    seed          = SEED,
    shuffle       = False,
    classes       = CLASSES,
)

# For test set we use the val-generator without the split (manual approach)
# We'll evaluate on the val set since we're using flow_from_directory
print(f"\n  Train samples : {train_gen.samples:,}")
print(f"  Val   samples : {val_gen.samples:,}")
print(f"  Class map     : {train_gen.class_indices}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAMPLE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Saving sample grid …")

images, labels = next(train_gen)
n = min(16, len(images))
cols, rows = 4, (n + 3) // 4

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
fig.suptitle("Training Samples (after augmentation)", fontsize=14, fontweight="bold")

for i, ax in enumerate(axes.flat):
    if i < n:
        ax.imshow(images[i])
        ax.set_title(CLASSES[int(labels[i])], fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "sample_images.png"), dpi=120)
plt.close()
print(f"  → Saved sample_images.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STEP 3 : MODEL ARCHITECTURE")
print("=" * 50)

def build_model(input_shape=(128, 128, 3)):
    """Custom CNN with Batch Normalisation and Dropout."""
    model = models.Sequential([

        # ── Block 1 ────────────────────────────────────
        layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 2 ────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 3 ────────────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # ── Block 4 ────────────────────────────────────
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # ── Classifier ─────────────────────────────────
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.50),
        layers.Dense(1, activation="sigmoid"),   # binary output
    ], name="MalariaCNN")

    return model

model = build_model()
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss      = "binary_crossentropy",
    metrics   = ["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
)

model.summary()
total_params = model.count_params()
print(f"\n  Total trainable parameters: {total_params:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
ckpt_path = os.path.join(MODEL_DIR, "best_model.keras")

cb_list = [
    callbacks.ModelCheckpoint(
        filepath          = ckpt_path,
        monitor           = "val_auc",
        mode              = "max",
        save_best_only    = True,
        verbose           = 1,
    ),
    callbacks.EarlyStopping(
        monitor           = "val_auc",
        patience          = 7,
        mode              = "max",
        restore_best_weights = True,
        verbose           = 1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor           = "val_loss",
        factor            = 0.5,
        patience          = 3,
        min_lr            = 1e-6,
        verbose           = 1,
    ),
    callbacks.CSVLogger(os.path.join(LOG_DIR, "training_log.csv")),
]


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STEP 4 : TRAINING")
print("=" * 50)

t0 = time.time()

history = model.fit(
    train_gen,
    epochs            = EPOCHS,
    validation_data   = val_gen,
    callbacks         = cb_list,
    verbose           = 1,
)

elapsed = time.time() - t0
print(f"\n[INFO] Training completed in {elapsed/60:.1f} minutes.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Saving training curves …")

hist = history.history
metrics_to_plot = [
    ("accuracy",  "val_accuracy",  "Accuracy"),
    ("loss",      "val_loss",      "Loss"),
    ("auc",       "val_auc",       "AUC"),
    ("precision", "val_precision", "Precision"),
    ("recall",    "val_recall",    "Recall"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Training History", fontsize=16, fontweight="bold")
axes = axes.flat

for ax, (train_key, val_key, title) in zip(axes, metrics_to_plot):
    if train_key in hist:
        ax.plot(hist[train_key], label="Train", linewidth=2)
    if val_key in hist:
        ax.plot(hist[val_key],   label="Val",   linewidth=2, linestyle="--")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

# Learning rate curve
if "lr" in hist:
    axes[5].plot(hist["lr"], color="orange", linewidth=2)
    axes[5].set_title("Learning Rate", fontsize=13)
    axes[5].set_xlabel("Epoch")
    axes[5].set_yscale("log")
    axes[5].grid(alpha=0.3)
else:
    axes[5].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "training_curves.png"), dpi=150)
plt.close()
print(f"  → Saved training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STEP 5 : EVALUATION")
print("=" * 50)

# Load best checkpoint
best_model = tf.keras.models.load_model(ckpt_path)

# Evaluate on val set
results = best_model.evaluate(val_gen, verbose=1)
metric_names = best_model.metrics_names

print(f"\n  Best-model metrics on validation set:")
for name, val in zip(metric_names, results):
    print(f"    {name:<15}: {val:.4f}")

# Predictions
val_gen.reset()
y_pred_prob = best_model.predict(val_gen, verbose=1).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)
y_true      = val_gen.classes

# Classification Report
report_str = classification_report(y_true, y_pred, target_names=CLASSES)
print(f"\n  Classification Report:\n{report_str}")

# Save report to file
with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w") as f:
    f.write("MALARIA CELL CLASSIFICATION REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Best-model evaluation (validation set):\n")
    for name, val in zip(metric_names, results):
        f.write(f"  {name}: {val:.4f}\n")
    f.write("\n" + report_str)
    f.write(f"\nTraining time: {elapsed/60:.2f} minutes\n")
    f.write(f"Epochs run: {len(hist['loss'])}\n")

print(f"  → Saved classification_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# 10. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Saving confusion matrix …")

cm = confusion_matrix(y_true, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Confusion Matrix", fontsize=15, fontweight="bold")

for ax, data, fmt, title in zip(
    axes,
    [cm,     cm_pct],
    ["d",    ".1f"],
    ["Counts", "Percentage (%)"],
):
    sns.heatmap(
        data, annot=True, fmt=fmt,
        xticklabels=CLASSES, yticklabels=CLASSES,
        cmap="Blues", ax=ax, linewidths=0.5,
        annot_kws={"size": 13},
    )
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print(f"  → Saved confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Saving ROC curve …")

auc_score  = roc_auc_score(y_true, y_pred_prob)
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="crimson", lw=2,
         label=f"ROC Curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
plt.fill_between(fpr, tpr, alpha=0.10, color="crimson")
plt.xlim([0, 1]); plt.ylim([0, 1.02])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate",  fontsize=12)
plt.title("ROC Curve — Malaria Cell Classifier", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"), dpi=150)
plt.close()
print(f"  → Saved roc_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12. PREDICTION SAMPLES (correct & wrong)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Saving prediction samples …")

val_gen.reset()
imgs_batch, _ = next(val_gen)
batch_prob     = best_model.predict(imgs_batch, verbose=0).flatten()
batch_pred     = (batch_prob >= 0.5).astype(int)
batch_true     = val_gen.classes[:len(imgs_batch)]

n_show = min(16, len(imgs_batch))
cols, rows = 4, (n_show + 3) // 4
fig, axes  = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5))
fig.suptitle("Predictions (✓ correct, ✗ wrong)", fontsize=13, fontweight="bold")

for i, ax in enumerate(axes.flat):
    if i < n_show:
        ax.imshow(imgs_batch[i])
        true_lbl  = CLASSES[batch_true[i]]
        pred_lbl  = CLASSES[batch_pred[i]]
        correct   = batch_true[i] == batch_pred[i]
        color     = "green" if correct else "red"
        symbol    = "✓" if correct else "✗"
        ax.set_title(f"{symbol} P:{pred_lbl}\nT:{true_lbl}\n{batch_prob[i]:.2f}",
                     fontsize=7, color=color)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "prediction_samples.png"), dpi=130)
plt.close()
print(f"  → Saved prediction_samples.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13. SAVE FINAL MODEL (multiple formats)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Saving final model …")

# SavedModel format
savedmodel_path = os.path.join(MODEL_DIR, "malaria_cnn_savedmodel")
best_model.save(savedmodel_path)
print(f"  → SavedModel   : {savedmodel_path}")

# HDF5 (.h5)
h5_path = os.path.join(MODEL_DIR, "malaria_cnn.h5")
best_model.save(h5_path)
print(f"  → HDF5 model   : {h5_path}")

# Model architecture JSON
json_path = os.path.join(MODEL_DIR, "model_architecture.json")
with open(json_path, "w") as f:
    f.write(best_model.to_json(indent=2))
print(f"  → Architecture : {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 14. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
val_acc = max(hist.get("val_accuracy", [0]))
val_auc = max(hist.get("val_auc",      [0]))

print(f"""
{'='*60}
  ✅  PIPELINE COMPLETE
{'='*60}
  Best Validation Accuracy : {val_acc*100:.2f}%
  Best Validation AUC      : {val_auc:.4f}
  ROC-AUC (post-hoc)       : {auc_score:.4f}
  Training Time            : {elapsed/60:.1f} minutes
  Epochs Completed         : {len(hist['loss'])}

  Output Directory : {OUTPUT_DIR}
    ├── model/
    │   ├── best_model.keras          (best checkpoint)
    │   ├── malaria_cnn.h5            (HDF5 format)
    │   ├── malaria_cnn_savedmodel/   (TF SavedModel)
    │   └── model_architecture.json
    ├── plots/
    │   ├── sample_images.png
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   └── prediction_samples.png
    ├── reports/
    │   └── classification_report.txt
    └── logs/
        └── training_log.csv
{'='*60}
""")
