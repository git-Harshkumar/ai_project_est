"""
=============================================================
  MALARIA CLASSIFIER — ACTIVE LEARNING RETRAINING SCRIPT
=============================================================
  Usage: python retrain.py
  - Reads corrected feedback rows from feedback.db
  - Copies images into retrain_data/{Parasitized,Uninfected}/
  - Fine-tunes the existing model for 5 epochs (lr=1e-5)
  - Backs up old model as best_model_backup.keras
  - Saves retrained model to the original MODEL_PATH
=============================================================
"""

import os
import sys
import shutil
import sqlite3
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (mirrors app.py configuration)
# ─────────────────────────────────────────────────────────────────────────────
APP_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.abspath(os.path.join(APP_DIR, ".."))

DB_PATH   = os.path.join(APP_DIR, "feedback.db")
MODEL_DIR = os.path.join(ROOT_DIR, "Dataset", "model")
MODEL_NAME = "best_model.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
BACKUP_PATH = os.path.join(MODEL_DIR, "best_model_backup.keras")

RETRAIN_DIR = os.path.join(APP_DIR, "retrain_data")
CLASSES = ["Parasitized", "Uninfected"]
IMG_SIZE = (128, 128)
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LR = 1e-5
BATCH_SIZE = 16

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — collect corrected feedback rows
# ─────────────────────────────────────────────────────────────────────────────
print("[1/5] Querying feedback database for corrected rows …")

if not os.path.exists(DB_PATH):
    print(f"[ERROR] feedback.db not found at {DB_PATH}. Run the app first.")
    sys.exit(1)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute(
    "SELECT image_path, true_label FROM feedback "
    "WHERE true_label IS NOT NULL AND true_label != ''"
)
rows = c.fetchall()
conn.close()

if not rows:
    print("[WARN] No corrected feedback found. Nothing to retrain on.")
    sys.exit(0)

print(f"       Found {len(rows)} corrected sample(s).")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — build mini-dataset directory
# ─────────────────────────────────────────────────────────────────────────────
print("[2/5] Preparing retrain_data directory …")

# Clean previous run
if os.path.exists(RETRAIN_DIR):
    shutil.rmtree(RETRAIN_DIR)

for cls in CLASSES:
    os.makedirs(os.path.join(RETRAIN_DIR, cls), exist_ok=True)

copied = 0
skipped = 0
for img_path, true_label in rows:
    if true_label not in CLASSES:
        print(f"       [SKIP] Unknown label '{true_label}' for {img_path}")
        skipped += 1
        continue
    if not os.path.exists(img_path):
        print(f"       [SKIP] Image not found: {img_path}")
        skipped += 1
        continue
    dest_dir  = os.path.join(RETRAIN_DIR, true_label)
    dest_file = os.path.join(dest_dir, os.path.basename(img_path))
    shutil.copy2(img_path, dest_file)
    copied += 1

print(f"       Copied {copied} image(s), skipped {skipped}.")

if copied == 0:
    print("[ERROR] No valid images to retrain on. Aborting.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — load existing model
# ─────────────────────────────────────────────────────────────────────────────
print("[3/5] Loading existing model …")

import tensorflow as tf

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}.")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"       Model loaded: {MODEL_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — build tf.data pipeline from mini-dataset
# ─────────────────────────────────────────────────────────────────────────────
print("[4/5] Building data pipeline …")

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Collect file paths and labels
file_paths = []
labels     = []
class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

for cls in CLASSES:
    cls_dir = os.path.join(RETRAIN_DIR, cls)
    for fname in os.listdir(cls_dir):
        fpath = os.path.join(cls_dir, fname)
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            file_paths.append(fpath)
            labels.append(class_to_idx[cls])

print(f"       Dataset: {len(file_paths)} image(s) across {len(CLASSES)} classes.")

if len(file_paths) == 0:
    print("[ERROR] Dataset is empty after filtering. Aborting.")
    sys.exit(1)

paths_ds  = tf.data.Dataset.from_tensor_slices(file_paths)
labels_ds = tf.data.Dataset.from_tensor_slices(labels)
dataset   = tf.data.Dataset.zip((paths_ds, labels_ds))
dataset   = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset   = dataset.cache().shuffle(buffer_size=max(len(file_paths), 32))
dataset   = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — fine-tune and save
# ─────────────────────────────────────────────────────────────────────────────
print(f"[5/5] Fine-tuning for {FINE_TUNE_EPOCHS} epoch(s) at lr={FINE_TUNE_LR} …")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(dataset, epochs=FINE_TUNE_EPOCHS, verbose=1)

final_acc  = history.history["accuracy"][-1]
final_loss = history.history["loss"][-1]
print(f"\n       Fine-tuning complete. Final acc={final_acc:.4f}  loss={final_loss:.4f}")

# Backup old model
print(f"\n       Backing up old model → {BACKUP_PATH}")
shutil.copy2(MODEL_PATH, BACKUP_PATH)

# Save retrained model
model.save(MODEL_PATH)
print(f"       Retrained model saved → {MODEL_PATH}")

print("\n✅  Retraining finished successfully!")
print("   Restart the Flask app to load the updated model.\n")
