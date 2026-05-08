# 🦠 Malaria Cell Image Classifier — Complete Project Documentation

> **Live Demo**: [https://ai-project-est.onrender.com](https://ai-project-est.onrender.com)
> **GitHub**: [https://github.com/git-Harshkumar/ai_project_est](https://github.com/git-Harshkumar/ai_project_est)

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Data Preprocessing & Augmentation](#3-data-preprocessing--augmentation)
4. [Model Architecture (CNN)](#4-model-architecture-cnn)
5. [Training Strategy](#5-training-strategy)
6. [Model Evaluation & Results](#6-model-evaluation--results)
7. [GradCAM Explainability](#7-gradcam-explainability)
8. [Web Application (Flask)](#8-web-application-flask)
9. [Deployment on Render](#9-deployment-on-render)
10. [Tech Stack](#10-tech-stack)
11. [Project File Structure](#11-project-file-structure)

---

## 1. Project Overview

This project is an **end-to-end deep learning pipeline** that automatically detects malaria in blood cell microscopy images.

- **Input**: A single blood cell microscopy image (PNG/JPG)
- **Output**: Classification as either **Parasitized** (infected) or **Uninfected** (healthy), along with a confidence score and a GradCAM heatmap showing *which part of the image* the model looked at.
- **Problem Type**: Binary Image Classification
- **Approach**: Custom Convolutional Neural Network (CNN) trained from scratch

**Why CNNs for this task?**
CNNs are ideal for image classification because they automatically learn spatial features (edges, textures, shapes) at multiple scales through their convolutional layers — unlike traditional ML which requires handcrafted features.

---

## 2. Dataset

| Property | Value |
|---|---|
| **Name** | NIH Malaria Cell Images |
| **Source** | National Institutes of Health (NIH) |
| **Total Images** | ~27,558 images |
| **Parasitized** | ~13,779 images |
| **Uninfected** | ~13,779 images |
| **Format** | PNG (microscopy images, varying sizes) |
| **Balance** | Perfectly balanced (50/50) |

### Dataset Splits
| Split | Percentage | Purpose |
|---|---|---|
| Train | 70% | Used to update model weights |
| Validation | 15% | Used to tune hyperparameters, monitor overfitting |
| (Val used as Test) | 15% | Final performance evaluation |

> The dataset is **perfectly balanced** — equal number of Parasitized and Uninfected images — so no class weighting was needed.

---

## 3. Data Preprocessing & Augmentation

### Preprocessing (Applied to ALL splits)
Every image goes through these steps before being fed to the model:

1. **Resize** → All images resized to **128×128 pixels** (standardized input)
2. **Normalize** → Pixel values scaled from `[0, 255]` to `[0.0, 1.0]` by dividing by 255
3. **Convert to RGB** → Ensures all images have 3 channels (R, G, B)

### Data Augmentation (Training set ONLY)
Augmentation artificially increases training diversity to prevent overfitting. Random transformations are applied to each image during training:

| Augmentation | Value | What it does |
|---|---|---|
| `rotation_range` | 20° | Randomly rotates image up to 20 degrees |
| `width_shift_range` | 15% | Randomly shifts image horizontally |
| `height_shift_range` | 15% | Randomly shifts image vertically |
| `shear_range` | 10% | Slants the image (like skewing) |
| `zoom_range` | 15% | Randomly zooms in/out |
| `horizontal_flip` | True | Randomly flips image left-right |
| `brightness_range` | [0.85, 1.15] | Varies brightness by ±15% |
| `fill_mode` | "nearest" | Fills empty pixels with nearest pixel value |

> **Why augmentation?** The training data is finite. Augmentation makes the model see different "versions" of each image, forcing it to learn the actual features (parasite shape/texture) rather than memorizing exact pixel patterns.

---

## 4. Model Architecture (CNN)

The model is a **custom CNN named `MalariaCNN`** built using TensorFlow/Keras Sequential API. It follows a standard pattern of increasing filter depth with spatial downsampling.

### Input
- Shape: `(128, 128, 3)` — 128×128 pixel RGB image

### Architecture Overview

```
INPUT (128×128×3)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  BLOCK 1: Conv2D(32) → BN → ReLU               │
│           Conv2D(32) → BN → ReLU               │
│           MaxPool(2×2) → Dropout(0.25)          │
│  Output: 64×64×32                               │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  BLOCK 2: Conv2D(64) → BN → ReLU               │
│           Conv2D(64) → BN → ReLU               │
│           MaxPool(2×2) → Dropout(0.25)          │
│  Output: 32×32×64                               │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  BLOCK 3: Conv2D(128) → BN → ReLU              │
│           Conv2D(128) → BN → ReLU              │
│           MaxPool(2×2) → Dropout(0.30)          │
│  Output: 16×16×128                              │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  BLOCK 4: Conv2D(256) → BN → ReLU              │
│           MaxPool(2×2) → Dropout(0.30)          │
│  Output: 8×8×256                                │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  CLASSIFIER HEAD                                │
│  GlobalAveragePooling2D → (256,)                │
│  Dense(256) + L2 Reg → BN → ReLU                │
│  Dropout(0.50)                                  │
│  Dense(1, sigmoid) → probability                │
└─────────────────────────────────────────────────┘
    │
    ▼
OUTPUT: Single number between 0 and 1
(≥ 0.5 = Uninfected, < 0.5 = Parasitized)
```

### Layer-by-Layer Explanation

| Layer | What it does |
|---|---|
| **Conv2D(32/64/128/256, kernel=3×3)** | Slides a 3×3 filter across the image to detect features (edges, textures, shapes). Increasing filters = detecting more complex features |
| **BatchNormalization (BN)** | Normalizes the activations within a batch. Stabilizes and speeds up training, reduces sensitivity to initialization |
| **Activation("relu")** | ReLU sets all negative values to 0. Adds non-linearity so the network can learn complex patterns |
| **MaxPooling2D(2×2)** | Takes the max value in each 2×2 region. Halves the spatial size, making the model more robust to small shifts |
| **Dropout(0.25 / 0.30 / 0.50)** | Randomly "turns off" a fraction of neurons during training. Prevents overfitting |
| **GlobalAveragePooling2D** | Converts feature maps to a single vector by averaging. Better generalization than Flatten |
| **Dense(256) + L2** | Fully connected layer. L2 regularization penalizes large weights, reducing overfitting |
| **Dense(1, sigmoid)** | Output layer. Sigmoid squashes to [0, 1] — interpreted as probability of being "Uninfected" |

---

## 5. Training Strategy

### Optimizer & Loss
| Setting | Value | Why |
|---|---|---|
| **Optimizer** | Adam (lr=0.001) | Adaptive learning rate, works well out of the box |
| **Loss Function** | Binary Crossentropy | Standard for binary classification |
| **Max Epochs** | 30 | Upper limit; early stopping used |
| **Batch Size** | 32 | Good balance of stability and speed |

### Callbacks (Automatic Training Control)

**1. ModelCheckpoint**
- Saves the model weights whenever validation AUC improves
- Ensures you always keep the **best version**, even if training later degrades

**2. EarlyStopping** (patience=7)
- Monitors `val_auc`. If it doesn't improve for 7 consecutive epochs, training stops
- Prevents overfitting and wasted compute
- Your model stopped at **epoch 12** (out of max 30)

**3. ReduceLROnPlateau** (patience=3, factor=0.5)
- If `val_loss` doesn't improve for 3 epochs, the learning rate is halved (e.g., 0.001 → 0.0005)
- Helps the model find finer optima when it's close to converging
- Minimum LR floor: 1e-6

**4. CSVLogger**
- Saves loss, accuracy, AUC, precision, recall for every epoch to a CSV file
- Useful for plotting training curves

### Training Time
- **Total time**: ~665 minutes (~11 hours) — trained on CPU (no GPU detected)
- **Epochs completed**: 12 (early stopping triggered)

---

## 6. Model Evaluation & Results

### Performance on Validation Set (2,066 images per class)

| Metric | Value |
|---|---|
| **Overall Accuracy** | **92%** |
| **Loss** | 0.2584 |
| **Macro Precision** | 0.93 |
| **Macro Recall** | 0.92 |
| **Macro F1-Score** | 0.92 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Parasitized** | 0.98 | 0.85 | 0.91 | 2,066 |
| **Uninfected** | 0.87 | 0.98 | 0.92 | 2,066 |

### Understanding the Metrics

| Metric | Formula | What it means here |
|---|---|---|
| **Precision (Parasitized=0.98)** | TP / (TP+FP) | When model says "Parasitized", it's right 98% of the time |
| **Recall (Parasitized=0.85)** | TP / (TP+FN) | Model catches 85% of all actual Parasitized cells |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean of Precision and Recall |

> **Medical context**: High precision for Parasitized (0.98) means very few healthy cells are wrongly flagged as infected (few false alarms). However, recall of 0.85 means 15% of actual infected cells are missed — in real clinical use, you'd want higher recall, possibly by lowering the classification threshold.

### Generated Visualizations
- `confusion_matrix.png` — Shows counts and percentages of correct/wrong predictions
- `roc_curve.png` — ROC curve showing AUC score
- `training_curves.png` — Loss, accuracy, AUC, precision, recall over epochs
- `prediction_samples.png` — Sample predictions with correct (✓) and wrong (✗) labels

---

## 7. GradCAM Explainability

**GradCAM (Gradient-weighted Class Activation Mapping)** is a technique to visualize *which regions of the input image* influenced the model's decision.

### How it works (Step by step)
1. **Identify the last Conv2D layer** (Block 4's Conv2D with 256 filters) — this layer has the highest-level spatial features
2. **Run a forward pass** and record the output of that conv layer
3. **Run a backward pass** (GradientTape) to compute gradients of the predicted class score with respect to that conv layer's output
4. **Pool the gradients** by averaging across the spatial dimensions → get importance weights per filter
5. **Weight the feature maps** by these importance scores → heatmap
6. **Apply ReLU** to keep only positive influences
7. **Resize and overlay** the heatmap on the original image using a "jet" colormap

### Result
- **Red/warm areas** = Regions the model focused on most strongly
- **Blue/cool areas** = Less relevant regions

This helps verify the model is looking at the actual parasite (purple stained region) rather than background artifacts.

---

## 8. Web Application (Flask)

### Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home page — shows model metrics, confusion matrix, ROC curve |
| `/upload` | GET | Upload page — drag & drop or file picker |
| `/result` | POST | Classifies the uploaded image, shows prediction + GradCAM |
| `/predict` | POST | JSON API endpoint — returns raw prediction data |
| `/plots/<filename>` | GET | Serves plot images (confusion matrix, etc.) |

### Prediction Flow (what happens when you upload)
```
User uploads image
    ↓
Open with Pillow → Convert to RGB → Resize to 128×128
    ↓
Normalize pixels (÷ 255) → Add batch dimension → shape (1, 128, 128, 3)
    ↓
model.predict() → single sigmoid value (0 to 1)
    ↓
≥ 0.5 → "Uninfected" | < 0.5 → "Parasitized"
    ↓
GradCAM heatmap generated from last Conv2D layer
    ↓
Overlay heatmap on original image (alpha blending)
    ↓
Both images encoded to Base64 and sent to result.html
```

---

## 9. Deployment on Render

### Configuration Files

| File | Purpose |
|---|---|
| `Pipeline/requirements.txt` | All Python dependencies |
| `Pipeline/Procfile` | Tells Render to run gunicorn (production WSGI server) |
| `Pipeline/runtime.txt` | Pins Python version |
| `Pipeline/.python-version` | Pins Python version (backup) |
| `render.yaml` | Render blueprint — auto-configures the service |

### Why gunicorn instead of `python app.py`?
Flask's built-in dev server is single-threaded and not production-safe. Gunicorn is a production WSGI server that handles concurrent requests properly.

### Render Free Tier Limitations
| Limitation | Impact |
|---|---|
| 0.1 vCPU (shared) | Inference takes 30–90 seconds |
| 512 MB RAM | TF + model uses ~400 MB; tight but works |
| Spins down after 15 min | First request after sleep takes 10–15 sec |
| No persistent disk | Plots/reports loaded from git at startup |

---

## 10. Tech Stack

| Category | Technology | Version | Purpose |
|---|---|---|---|
| **Deep Learning** | TensorFlow / Keras | 2.16.2 | Model building, training, inference |
| **Data Processing** | NumPy | 1.26.4 | Array operations, image math |
| **Image Processing** | Pillow (PIL) | 10.4.0 | Opening, resizing, converting images |
| **Visualization** | Matplotlib | 3.9.4 | Training curves, heatmap overlay |
| **Evaluation** | scikit-learn | latest | Classification report, confusion matrix, ROC |
| **Visualization** | Seaborn | latest | Heatmap plots for confusion matrix |
| **Web Framework** | Flask | 3.1.0 | Web server, routing, templating |
| **WSGI Server** | Gunicorn | 21.2.0 | Production web server |
| **Deployment** | Render.com | — | Cloud hosting |
| **Version Control** | Git / GitHub | — | Code management |
| **Language** | Python | 3.11.0 | Everything |

---

## 11. Project File Structure

```
ai_project_est/                    ← Git repo root
│
├── Dataset/                       ← Training outputs (model, plots, reports)
│   ├── model/
│   │   └── best_model.keras       ← Saved model (best checkpoint, ~8 MB)
│   ├── plots/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── training_curves.png
│   │   ├── sample_images.png
│   │   └── prediction_samples.png
│   ├── reports/
│   │   └── classification_report.txt  ← Accuracy, F1, precision, recall
│   └── logs/
│       └── training_log.csv       ← Per-epoch metrics
│
├── Pipeline/                      ← Application code (Render root dir)
│   ├── app.py                     ← Flask web application
│   ├── train.py                   ← Full ML training pipeline
│   ├── test.py                    ← Model testing / inference script
│   ├── requirements.txt           ← Python dependencies
│   ├── Procfile                   ← Gunicorn start command for Render
│   ├── runtime.txt                ← Python version pin
│   ├── .python-version            ← Python version pin (backup)
│   ├── logo.png                   ← App logo
│   ├── static/
│   │   └── css/
│   │       └── style.css          ← App styling
│   └── templates/
│       ├── base.html              ← Base HTML layout
│       ├── index.html             ← Home page
│       ├── upload.html            ← Upload page
│       └── result.html            ← Results page
│
├── render.yaml                    ← Render deployment blueprint
├── .python-version                ← Python version (repo root)
└── setup_render.bat               ← Helper script for local setup
```

---

## Quick Reference: Key Numbers

| What | Value |
|---|---|
| Input image size | 128 × 128 pixels |
| Number of classes | 2 (Parasitized, Uninfected) |
| Total training images | ~19,290 (70% of 27,558) |
| Total validation images | ~4,132 (15% of 27,558) |
| Model parameters | ~3.2 million |
| Training epochs | 12 (early stopped from max 30) |
| Training time | ~11 hours (CPU only) |
| Best validation accuracy | **92%** |
| Macro F1-Score | **0.92** |
| Model file size | 7.95 MB (.keras format) |
| Live URL | https://ai-project-est.onrender.com |
