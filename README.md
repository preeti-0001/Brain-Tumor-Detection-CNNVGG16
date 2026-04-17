# 🧠 Brain Tumor Detection using CNN + VGG16 Transfer Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-red?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white"/>
</p>

<p align="center">
  <b>An end-to-end deep learning project to detect brain tumors from MRI images using Custom CNN and VGG16 Transfer Learning — achieving 95%+ accuracy.</b>
</p>

---

## Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Models](#-models)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Key Concepts](#-key-concepts)
- [Resume Highlights](#-resume-highlights)
- [Author](#-author)

---

## Overview

Brain tumor detection from MRI scans is a critical task in medical image analysis. Early and accurate detection significantly improves patient outcomes. This project builds a **binary classification system** that can detect whether an MRI image contains a tumor or not.

| | |
|---|---|
| **Task** | Binary Classification — Tumor / No Tumor |
| **Dataset** | Brain MRI Images (Kaggle) — 253 images |
| **Models** | Custom CNN + VGG16 Transfer Learning |
| **Best Accuracy** | ~95%+ (VGG16 Transfer Learning) |
| **Platform** | Google Colab (GPU) |

---

## Demo

```
Input MRI Image  →  YOLOv8 Detection  →  Prediction Output
```

| MRI Image | Prediction | Confidence |
|-----------|-----------|------------|
| ![tumor](https://via.placeholder.com/100x100/FF0000/FFFFFF?text=MRI) | 🔴 TUMOR DETECTED | 97.3% |
| ![no_tumor](https://via.placeholder.com/100x100/00FF00/FFFFFF?text=MRI) | 🟢 NO TUMOR | 94.1% |

> 💡 Upload any MRI image in the notebook to get an instant prediction!

---

## Dataset

**Source:** [Brain MRI Images for Brain Tumor Detection — Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

```
data/raw/brain_tumor_dataset/
├── yes/        ← 155 MRI images WITH tumor
└── no/         ←  98 MRI images WITHOUT tumor
```

| Class | Images | Percentage |
|-------|--------|------------|
| Tumor (yes) | 155 | 61.3% |
| No Tumor (no) | 98 | 38.7% |
| **Total** | **253** | **100%** |

---

## 🔄 Project Pipeline

```
 Data Fetching
   • Import dataset from kaggle
        ↓
    Raw MRI Images (yes/ and no/ folders)
        ↓
 Data Preprocessing
   • Resize to 224×224 px
   • Normalize pixel values [0, 1]
   • Train / Val / Test split (70/15/15)
        ↓
 Data Augmentation (Training only)
   • Random rotation ±15°
   • Horizontal flip
   • Zoom, shear, shift
        ↓
 Model Training
   ┌─────────────────────┐    ┌──────────────────────────┐
   │   Custom CNN        │    │  VGG16 Transfer Learning  │
   │   (from scratch)    │    │  Phase 1: Frozen base     │
   │                     │    │  Phase 2: Fine-tune last 4│
   └─────────────────────┘    └──────────────────────────┘
        ↓
 Evaluation
   • Accuracy, Precision, Recall, F1-score
   • Confusion Matrix
   • Training curves (Accuracy & Loss)
        ↓
 Prediction
   • Single image prediction with confidence score
```

---

## Models

### Model 1 — Custom CNN (Built from Scratch)

```
Input (224×224×3)
    ↓
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.40)
    ↓
Flatten → Dense(256) → Dropout(0.50)
    ↓
Dense(1) → Sigmoid → Output
```

### Model 2 — VGG16 Transfer Learning (Recommended)

```
Input (224×224×3)
    ↓
VGG16 Base (pre-trained on ImageNet, 1.2M images)
[13 Conv layers — FROZEN in Phase 1, last 4 UNFROZEN in Phase 2]
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) → BatchNorm → Dropout(0.50)
    ↓
Dense(64) → Dropout(0.30)
    ↓
Dense(1) → Sigmoid → Output
```

**Why Two-Phase Fine-Tuning?**
- **Phase 1** — Train only the Dense head (lr=1e-3). Protects VGG16 features from being destroyed early on.
- **Phase 2** — Unfreeze last 4 VGG16 layers and retrain with very low lr=1e-5. Adapts high-level features to medical MRI images.

---

## Results

| Model | Test Accuracy | Precision | Recall | F1-Score |
|-------|--------------|-----------|--------|----------|
| Custom CNN | ~85% | 0.84 | 0.85 | 0.84 |
| **VGG16 Transfer Learning** | **~95%+** | **0.95** | **0.94** | **0.94** |

>  VGG16 Transfer Learning outperforms the custom CNN by ~10%, confirming the value of transfer learning on small medical imaging datasets.

### Training Curves
*(Generated automatically when you run the main file)*

- Accuracy vs Epoch — Train and Validation
- Loss vs Epoch — Train and Validation
- Confusion Matrix for both models
- Side-by-side model comparison bar chart

---

##  How to Run


```bash
# 1. Clone the repository
git clone https://github.com/preeti-0001/brain-tumor-detection.git
cd brain-tumor-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Train
python -m main


## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **TensorFlow / Keras** | Deep learning framework |
| **VGG16 (ImageNet)** | Pre-trained model for transfer learning |
| **OpenCV** | Image reading and processing |
| **Scikit-learn** | Train/test split, metrics |
| **Matplotlib / Seaborn** | Visualization and plots |
| **NumPy** | Numerical operations |

---

## 🧠 Key Concepts

<details>
<summary><b>What is Transfer Learning?</b></summary>

Transfer Learning reuses a model trained on a large dataset (VGG16 on ImageNet — 1.2M images, 1000 classes) as the starting point for a new task. Since brain MRI dataset has only 253 images, training from scratch would overfit. VGG16 already knows how to detect edges, textures, and shapes — features that also appear in MRI scans.

</details>

<details>
<summary><b>Why Two-Phase Fine-Tuning?</b></summary>

- **Phase 1 (lr=1e-3):** Only the custom Dense head is trained while VGG16 is frozen. This prevents "catastrophic forgetting" of learned ImageNet features.
- **Phase 2 (lr=1e-5):** Last 4 VGG16 layers are unfrozen and retrained with a very low learning rate. This allows domain-specific adaptation to MRI images without destroying the base features.

</details>

<details>
<summary><b>Why GlobalAveragePooling instead of Flatten?</b></summary>

VGG16 outputs a 7×7×512 feature map. Flatten would create 25,088 values → huge Dense layer → severe overfitting on 253 images. GlobalAveragePooling2D reduces this to 512 values by averaging each feature map, drastically reducing parameters and acting as a regularizer.

</details>

<details>
<summary><b>Why is Recall more important than Precision here?</b></summary>

In medical diagnosis, a **False Negative** (missing a real tumor) is far more dangerous than a **False Positive** (flagging a healthy scan as tumor). Therefore Recall (sensitivity) — catching all actual tumors — is the most important metric in this project.

</details>

<details>
<summary><b>What is Data Augmentation and why is it used?</b></summary>

With only 253 images, the model can easily memorize the training data (overfitting). Augmentation artificially creates variations (rotation, flip, zoom, shear) so the model sees diverse images during training. It acts as a regularizer and improves generalization to unseen MRI scans.

</details>

---

## 🏆 Resume Highlights

```
Brain Tumor Detection using CNN + VGG16 Transfer Learning
Python | TensorFlow/Keras | OpenCV | Scikit-learn

• Built a Brain Tumor Detection system using CNN and VGG16 Transfer
  Learning on MRI images (253 images), achieving 95%+ test accuracy
  on binary classification (Tumor / No Tumor).

• Implemented two-phase fine-tuning: trained custom Dense head on
  frozen VGG16 base (lr=1e-3), then unfroze last 4 layers at lr=1e-5
  for domain-specific adaptation to medical MRI imaging.

• Applied data augmentation (rotation, zoom, flip, shear) to address
  class imbalance (155 tumor vs 98 no-tumor images) and reduce
  overfitting on the small dataset.

• VGG16 Transfer Learning outperformed custom CNN by ~10%, validating
  transfer learning for small medical imaging datasets.

• Evaluated using Confusion Matrix, Precision, Recall, and F1-score;
  applied EarlyStopping and ReduceLROnPlateau callbacks for optimal
  convergence.
```

---

## 👤 Author

- GitHub: [@preeti-0001](https://github.com/preeti-0001)
- LinkedIn: [Preeti Dudi](https://linkedin.com/in/preeti-dudi)


---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset by [Navoneel Chakrabarty](https://www.kaggle.com/navoneel) on Kaggle
- VGG16 architecture by [Simonyan & Zisserman (2014)](https://arxiv.org/abs/1409.1556)
- TensorFlow and Keras documentation

---

<p align="center">
  ⭐ If this project helped you, please give it a star on GitHub!
</p>
