# Project Findings: Cancer Subtype Discovery and Classification

## 1. Executive Summary
This project successfully analyzed TCGA Pan-Cancer RNA-seq data to discover molecular subtypes and classify cancer types. The high performance of both unsupervised and supervised models confirms that gene expression profiles are highly characteristic of primary cancer types, even after batch correction in the PANCAN dataset.

## 2. Dataset Characteristics
- **Total Samples**: 3,607
- **Gene Features**: 20,297 (Initial), ~18,000+ (After constant gene removal)
- **Cancer Types Analyzed**:
  - BRCA (Breast): ~1,000+ samples
  - LUAD (Lung): ~500+ samples
  - KIRC (Kidney): ~500+ samples
  - PRAD (Prostate): ~400+ samples
  - COAD (Colon): ~200+ samples
  - GBM (Brain): ~100+ samples

## 3. Method 1: Unsupervised Discovery (PCA + K-Means)
The goal was to see if cancers naturally cluster by type without labels.
- **Top 50 PCs Explained Variance**: ~47.2%
- **Adjusted Rand Index (ARI)**: **0.8003**
- **Normalized Mutual Information (NMI)**: **0.8547**
- **Silhouette Score**: **0.2304**

**Observation**: The high ARI indicates that the molecular profiles of different cancers are distinct enough that K-Means can recover the cancer type labels with ~80% accuracy based solely on expression data.

## 4. Supervised Classification Performance
We compared a custom NumPy Logistic Regression (linear) with a PyTorch MLP (non-linear).

| Metric (Mean 5-fold CV) | Logistic Regression (NumPy) | MLP (PyTorch) |
| :--- | :--- | :--- |
| **Macro F1 Score** | **0.9975** | **0.9969** |
| **Precision** | **0.9970** | **0.9968** |
| **Recall** | **0.9980** | **0.9971** |

**Observation**: Both models achieved near-perfect performance. The linear separator (Logistic Regression) was sufficient for classification on the PCA-reduced feature space, suggesting that the cancer types are linearly separable in the top 50 principal components.

## 5. Conclusions and Biological Relevance
- **Molecular Distinctness**: The high clustering and classification performance confirm that BRCA, LUAD, KIRC, etc., have unique transcriptomic signatures that are stable across thousands of patients.
- **Model Efficiency**: Scratch-implemented NumPy models performed on par with deep learning approaches for this specific multi-class problem, illustrating the power of well-regularized linear models in high-dimensional biological data.
- **Pipeline Robustness**: The use of batch-corrected Pan-Cancer Atlas data ensured that findings were not driven by technical artifacts between different centers.

