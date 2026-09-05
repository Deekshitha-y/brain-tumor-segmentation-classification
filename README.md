# 🧠 Multi-Task Brain Tumor Classification and Segmentation with Explainable AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified deep learning framework for simultaneous **Brain Tumor Classification** and **Tumor Segmentation** using a multi-task learning architecture based on **EfficientNetV2-S** and a **U-Net++ style decoder**, enhanced with **Grad-CAM Explainable AI (XAI)**.

---

## 📌 Project Overview

Brain tumor diagnosis from MRI scans typically requires two separate workflows:
1. **Classification**: Identifying the specific type of tumor (Meningioma, Glioma, or Pituitary).
2. **Segmentation**: Delineating the precise anatomical boundaries of the lesion for surgical planning and treatment.

This project unifies both tasks into a single **Multi-Task Learning (MTL)** network. A shared encoder extracts general visual features, while specialized downstream heads jointly optimize both classification and segmentation, yielding state-of-the-art accuracy while remaining computationally efficient.

---

## 📊 Dataset (Figshare)

This project utilizes the benchmark **Brain Tumor Dataset** published by Jun Cheng on Figshare:
* **Source**: [Figshare Brain Tumor Dataset (Cheng et al.)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
* **Modality**: T1-weighted contrast-enhanced MRI (CE-MRI)
* **Total Scans**: 3,064 slices (`.mat` files) across 233 patients
* **Classes**:
  * **Meningioma**: 708 slices (Label 1)
  * **Glioma**: 1,426 slices (Label 2)
  * **Pituitary Tumor**: 930 slices (Label 3)
* **Ground Truth**: Each `.mat` record includes the 2D image matrix, tumor border/mask (`tumorMask`), and diagnostic class label (`label`).

---

## 🏗️ Model Architecture

The custom `MultiTaskUNetPlusPlus` model features:

```text
                     ┌───────────────────────────────┐
                     │     Input MRI (256x256x3)     │
                     └───────────────┬───────────────┘
                                     │
                 ┌───────────────────▼───────────────────┐
                 │  Shared Backbone (EfficientNetV2-S)   │
                 │     Feature Levels: e0, e1, e2, e3, e4│
                 └───────────────┬───────────────┬───────┘
                                 │               │
        ┌────────────────────────┘               └────────────────────────┐
        ▼                                                                 ▼
┌───────────────────────────────┐                       ┌─────────────────────────────────┐
│     Classification Head       │                       │     Segmentation Decoder        │
│ • Global Average Pooling      │                       │ • U-Net++ Dense Lateral Convs   │
│ • Linear(512) + BN + ReLU     │                       │ • Multi-scale Bilinear Upsample │
│ • Dropout(0.4)                │                       │ • Skip Connection Concat        │
│ • Output: 3 Tumor Classes     │                       │ • Output: 1-channel Mask Logits │
└───────────────────────────────┘                       └─────────────────────────────────┘
```

* **Backbone**: ImageNet-pretrained `tf_efficientnetv2_s` via `timm`.
* **Joint Loss**:
  $$\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_{\text{seg}} + 1.0 \times \mathcal{L}_{\text{cls}}$$
  * $\mathcal{L}_{\text{seg}} = 0.5 \times \text{DiceLoss} + 0.5 \times \text{BCEWithLogitsLoss}$
  * $\mathcal{L}_{\text{cls}} = \text{CrossEntropyLoss}(\text{label\_smoothing}=0.05)$

---

## 🏆 Benchmark Results

Evaluated on an independent **Test Set (460 images / 15% stratified split)**:

### 1. Classification Performance

| Tumor Type | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Meningioma** | **97.27%** | **100.00%** | **0.9862** | 107 |
| **Glioma** | **100.00%** | **99.07%** | **0.9953** | 214 |
| **Pituitary** | **100.00%** | **99.28%** | **0.9964** | 139 |
| **Overall Accuracy** | — | — | **99.35%** | **460** |
| **Macro Average** | **99.09%** | **99.45%** | **0.9926** | **460** |

### 2. Segmentation Metrics

| Metric | Mean Score | Best (Max) | Description |
| :--- | :---: | :---: | :--- |
| **Dice Coefficient** | **0.8089** | **0.9864** | Overlap between predicted & true tumor |
| **IoU (Jaccard Index)** | **0.7247** | **0.9732** | Area of intersection over union |
| **Pixel Accuracy** | **99.47%** | **99.98%** | Ratio of correctly identified pixels |
| **Hausdorff Distance** | **12.24** | **1.00** | Boundary shape fidelity (lower is better) |

---

## 🔍 Explainable AI (Grad-CAM)

To ensure clinical trustworthiness, the network incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)**:
* Visualizes which anatomical features in the MRI the model attends to when making a tumor classification.
* Validates whether the classification focus aligns precisely with the delineated segmentation contours, preventing shortcut learning.

---

## 📂 Repository Structure

```text
.
├── .gitignore                      # Excludes weights, .mat files, and cache
├── README.md                       # Comprehensive documentation
├── requirements.txt                # Python package dependencies
└── brain-tumor-c-s-final.ipynb     # Complete training and evaluation notebook
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Deekshitha-y/brain-tumor-segmentation-classification.git
cd brain-tumor-segmentation-classification
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Create and activate environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the Dataset
1. Download the Figshare dataset from [here](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427).
2. Extract the `.mat` files into a directory (e.g., `brain_tumor_data/` or configure the input path inside the notebook).

### 4. Run the Notebook
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook brain-tumor-c-s-final.ipynb
```

---

## 📚 Citation

If you use the dataset, please cite the original author:
```bibtex
@article{cheng2015enhanced,
  title={Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition},
  author={Cheng, Jun and Huang, Wei and Tao, Shenghai and Lu, Chengwei and Zhu, Zeyu and Yan, Jing and Wang, Yanan and Zhao, Zhen and others},
  journal={PloS one},
  volume={10},
  number={10},
  pages={e0140381},
  year={2015},
  publisher={Public Library of Science}
}
```

---

## 👩‍💻 Author

* **Deekshitha Yasarapu** - [@Deekshitha-y](https://github.com/Deekshitha-y)

