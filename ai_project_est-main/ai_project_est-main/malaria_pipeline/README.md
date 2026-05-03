# 🦟 Malaria Cell Image Classifier — Training Pipeline

## Overview
Complete end-to-end ML pipeline for **binary classification** of malaria-infected vs. healthy blood cell images using a custom CNN built with TensorFlow/Keras.

---

## 📁 Folder Structure
```
malaria_pipeline/
├── train.py            ← Main training script (run this)
├── install_deps.bat    ← Install all dependencies
└── README.md           ← This file

D:\ai_trained_dataset\  ← All outputs saved here
├── model/
│   ├── best_model.keras          (auto-saved best checkpoint)
│   ├── malaria_cnn.h5            (final model in HDF5)
│   ├── malaria_cnn_savedmodel/   (TF SavedModel format)
│   └── model_architecture.json
├── plots/
│   ├── sample_images.png         (augmented training samples)
│   ├── training_curves.png       (acc/loss/auc curves)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── prediction_samples.png
├── reports/
│   └── classification_report.txt
└── logs/
    ├── training_log.csv
    └── (TensorBoard event files)
```

---

## 🚀 How to Run

### Step 1 — Install dependencies (first time only)
Double-click `install_deps.bat`  
OR run in terminal:
```bash
pip install tensorflow scikit-learn Pillow seaborn matplotlib numpy
```

### Step 2 — Train the model
```bash
cd C:\Users\harsh\OneDrive\Desktop\AI\malaria_pipeline
python train.py
```

### Step 3 — Monitor with TensorBoard (optional)
```bash
tensorboard --logdir D:\ai_trained_dataset\logs
```
Then open: http://localhost:6006

---

## 🧠 Model Architecture
| Layer Block | Filters | Notes |
|-------------|---------|-------|
| Conv Block 1 | 32 | BatchNorm + ReLU + MaxPool + Dropout 0.25 |
| Conv Block 2 | 64 | BatchNorm + ReLU + MaxPool + Dropout 0.25 |
| Conv Block 3 | 128 | BatchNorm + ReLU + MaxPool + Dropout 0.30 |
| Conv Block 4 | 256 | BatchNorm + ReLU + MaxPool + Dropout 0.30 |
| Classifier | 256 Dense → 1 | L2 reg + Dropout 0.50 + Sigmoid |

---

## ⚙️ Preprocessing
| Step | Details |
|------|---------|
| Resize | 128×128 pixels |
| Normalization | Divide by 255 (0–1 range) |
| Train Split | 70% train / 15% val / 15% test |
| Augmentation | Rotation ±20°, Shift 15%, Zoom 15%, Flip H, Brightness ±15% |

---

## 📊 Expected Performance
- Validation Accuracy: **~95–97%**
- AUC-ROC: **~0.98–0.99**
- Training Time: ~10–25 minutes (CPU) / 2–5 min (GPU)

---

## 🔮 Loading & Using the Trained Model
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model(r"D:\ai_trained_dataset\model\best_model.keras")

# Predict single image
img = Image.open("your_cell_image.png").resize((128, 128))
x   = np.array(img) / 255.0
x   = np.expand_dims(x, axis=0)   # shape: (1, 128, 128, 3)

prob = model.predict(x)[0][0]
label = "Parasitized" if prob < 0.5 else "Uninfected"
print(f"Prediction: {label}  (confidence: {max(prob, 1-prob)*100:.1f}%)")
```
