# 🎗️ Breast Cancer Classification using ANN

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **production-quality**, end-to-end breast cancer classification project using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras. The model classifies tumors as **Malignant** or **Benign** based on 30 numeric features derived from fine needle aspirate (FNA) images of breast masses.

---

## 📊 Results

| Metric      | Score   |
|-------------|---------|
| Accuracy    | 95.61%  |
| Precision   | 97.18%  |
| Recall      | 95.83%  |
| F1 Score    | 96.50%  |
| ROC-AUC     | 99.34%  |

> Model trained on the Wisconsin Breast Cancer Dataset (569 samples, 30 features).

---

## 🗂️ Project Structure

```
breast-cancer-classification/
│
├── data/
│   └── breast_cancer.csv           ← Wisconsin Breast Cancer dataset
│
├── models/
│   └── saved_model.keras           ← Trained ANN (auto-saved)
│
├── outputs/
│   ├── plots/                      ← 9 EDA & evaluation figures
│   │   ├── 01_class_distribution.png
│   │   ├── 02_feature_histograms.png
│   │   ├── 03_box_plots.png
│   │   ├── 04_correlation_heatmap.png
│   │   ├── 05_violin_plots.png
│   │   ├── 06_scatter_plots.png
│   │   ├── 07_training_history.png
│   │   ├── 08_confusion_matrix_roc.png
│   │   └── 09_metrics_summary.png
│   ├── metrics/
│   │   ├── metrics.json            ← Evaluation metrics (JSON)
│   │   └── classification_report.txt
│   └── predictions/                ← Saved inference results
│
├── Breast_Cancer_Classification.ipynb   ← Main notebook (41 cells)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧬 Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)** from the UCI ML Repository.

- **Samples:** 569 (357 Benign, 212 Malignant)
- **Features:** 30 numeric features (mean, standard error, worst values)
- **Target:** Binary — `0 = Malignant`, `1 = Benign`

Features are computed from digitized FNA images and describe characteristics of cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension).

---

## 🧠 Model Architecture

```
Input (30 features)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.30)
    ↓
Dense(64)  → BatchNorm → ReLU → Dropout(0.25)
    ↓
Dense(32)  → BatchNorm → ReLU → Dropout(0.20)
    ↓
Dense(1, sigmoid)  ← Binary output probability
```

**Optimizer:** Adam (lr=0.001) | **Loss:** Binary Cross-Entropy  
**Regularization:** L2 weight decay + Dropout + BatchNormalization  
**Callbacks:** EarlyStopping (patience=20) + ReduceLROnPlateau + ModelCheckpoint

---

## 🚀 Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Notebook

```bash
jupyter notebook Breast_Cancer_Classification.ipynb
```

Or use **VS Code**, **JupyterLab**, or **Google Colab**.

---

## 🔮 Quick Inference

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load saved model
model = tf.keras.models.load_model('models/saved_model.keras')

# Provide 30 feature values for a new patient
new_patient = np.array([[17.99, 10.38, 122.80, 1001.0, 0.1184, 0.2776,
                          0.3001, 0.1471, 0.2419, 0.0787, 1.095, 0.9053,
                          8.589, 153.4, 0.0064, 0.049, 0.0537, 0.0159,
                          0.0300, 0.0062, 25.38, 17.33, 184.6, 2019.0,
                          0.162, 0.666, 0.712, 0.265, 0.460, 0.1189]])

# Apply the same scaler (fit on training data in notebook)
# scaler.transform(new_patient)  → model.predict(...)
```

> See Section 11 of the notebook for the complete, ready-to-use `predict_single()` function.

---

## 📈 Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Import Libraries | All dependencies + seed setup |
| 2 | Load Dataset | CSV loading + preview |
| 3 | EDA | Shape, stats, missing values, class distribution |
| 4 | Visualizations | 6 plot types auto-saved to `outputs/plots/` |
| 5 | Preprocessing | Feature scaling, train/test split |
| 6 | Build ANN | Keras Sequential model |
| 7 | Train Model | With EarlyStopping + ReduceLR callbacks |
| 8 | Training History | Loss, accuracy, precision curves |
| 9 | Evaluation | Accuracy, F1, ROC-AUC, Confusion Matrix |
| 10 | Save Outputs | Model + metrics + classification report |
| 11 | Predictive System | Single-record inference with JSON output |
| 12 | Summary | Results dashboard |

---

## ⚠️ Disclaimer

This project is a **research and educational prototype only**. The model has **not been clinically validated** and must not be used for real medical diagnosis or treatment decisions.

---

## 📄 License

This project is licensed under the MIT License.
