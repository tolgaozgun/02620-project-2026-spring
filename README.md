# TCGA Cancer Subtype Discovery and Classification

This project implements an end-to-end machine learning pipeline for cancer subtype discovery and classification using TCGA (The Cancer Genome Atlas) Pan-Cancer RNA-seq data.

## Project Overview

The pipeline analyzes gene expression profiles across 6 major cancer types:
- **BRCA** (Breast Invasive Carcinoma)
- **LUAD** (Lung Adenocarcinoma)
- **KIRC** (Kidney Renal Clear Cell Carcinoma)
- **PRAD** (Prostate Adenocarcinoma)
- **COAD** (Colon Adenocarcinoma)
- **GBM** (Glioblastoma Multiforme)

### Methods Implemented
1. **Unsupervised Discovery**: PCA for dimensionality reduction and K-Means for molecular subtype clustering (NumPy from scratch).
2. **Supervised Classification**: 
   - Multi-class Logistic Regression with One-vs-Rest and L2 regularization (NumPy from scratch).
   - Multi-Layer Perceptron (MLP) classification (PyTorch).

## Setup and Installation

### Prerequisites
- Python 3.9+
- Conda (recommended)

### Environment Setup
```bash
conda create -n cancer-research python=3.9
conda activate cancer-research
pip install -r requirements.txt
```
*(Note: If requirements.txt is missing, manual installs for `numpy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`, `seaborn`, `openpyxl`, and `tqdm` are required.)*

## How to Run

To execute the entire pipeline (data loading, preprocessing, model training, and evaluation):

```bash
python src/main.py
```

### Data Acquisition
On the first run, `src/data_loader.py` will automatically download the **Pan-Cancer Atlas mRNA Expression Matrix** (~1.88GB) and the **Clinical Data Resource** from the GDC API. This process requires a stable internet connection and may take several minutes.

## Project Structure
- `src/`: Core implementation logic and model scripts.
- `docs/`: Project proposal and documentation.
- `data/`: Local storage for raw and processed datasets (excluded from Git).
- `results/`: Visualization outputs and performance metrics.

## License
MIT
