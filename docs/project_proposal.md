# Cancer Subtype Discovery and Classification from TCGA Gene Expression Data

**Authors:** Tolga Ozgun

## Introduction

### Problem Statement

Cancer is not a single disease, even within a single organ, tumors can differ dramatically in molecular profile, prognosis, and treatment response. The challenge we address is: given high-dimensional bulk RNA-seq gene expression profiles from The Cancer Genome Atlas (TCGA), can we (1) discover meaningful molecular subtypes in an unsupervised fashion, (2) accurately classify cancer types using interpretable linear models, and (3) improve classification further with a deep neural network?

### Motivation

Identifying tumor subtypes from gene expression data has direct clinical implications: patients with distinct molecular profiles often respond differently to therapies. TCGA provides one of the largest and most well-curated multi-cancer genomic datasets available, making it an ideal testbed for benchmarking ML methods on a high-stakes biological problem. The dataset is high-dimensional (\~20,000 genes) with moderate sample counts (\~10,000 samples spanning 33 cancer types), presenting a genuine and realistic ML challenge.

### Dataset

We will use the TCGA Pan-Cancer (PANCAN) gene expression dataset, available through the GDC Data Portal ([https://portal.gdc.cancer.gov/repository](https://portal.gdc.cancer.gov/repository)). Specifically, we will download the HTSeq-FPKM-UQ RNA-seq expression matrix. We will focus on a subset of 5–6 cancer types (e.g., BRCA, LUAD, KIRC, PRAD, COAD, GBM) with well-represented sample counts (\~500–1000 samples each), yielding a dataset of approximately 3,000–5,000 samples × 20,000 gene features. Cancer type labels (available in the clinical metadata files) serve as ground truth for supervised tasks.

### 

### Overview of Three Methods

1. **PCA \+ K-Means Clustering (implemented from scratch in NumPy):** We reduce the gene expression matrix to a low-dimensional representation using PCA, then run K-Means to discover clusters and compare them to known cancer type labels.  
2. **Logistic Regression with L2 Regularization (implemented from scratch in NumPy):** We train a multi-class logistic regression classifier (one-vs-rest) on the PCA-reduced features to predict cancer type, using gradient descent and cross-validation for hyperparameter tuning.  
3. **Multi-Layer Perceptron (MLP, implemented using PyTorch):** We train a feedforward neural network directly on PCA-reduced or raw (top-variance gene) features to classify cancer types, comparing performance against the logistic regression baseline.

### Evaluation

For clustering (Method 1), we will use Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) against true cancer type labels, as well as silhouette score for internal validation. For classification (Methods 2 and 3), we will use macro-averaged F1 score, precision, recall, and a confusion matrix across cancer types. All classifiers will be evaluated via 5-fold stratified cross-validation. We will compare Methods 2 and 3 directly to assess the gain from non-linearity.

### Related Works

- Hutter et al. "The Cancer Genome Atlas: creating lasting value beyond its data." *Cell*, 2018\.  
- Weinstein et al. "The Cancer Genome Atlas Pan-Cancer analysis project." *Nature Genetics*, 2013\.

