# Data Directory

The data files (mRNA-seq matrices and clinical metadata) are not included in this repository.

## Why?
The primary RNA-seq expression matrix (`pancan_expr.tsv`) is approximately **1.88GB** uncompressed. This exceeds standard version control limits and GitHub's recommended repository size.

## How to Get the Data
The script `src/data_loader.py` is configured to download the **official Pan-Cancer Atlas mRNA Expression Matrix** and the **Clinical Data Resource (CDR)** directly from the **NCI Genomic Data Commons (GDC) API**.

Run the following command to automatically download and preprocess the data:
```bash
python src/data_loader.py
```
This will result in a sub-filtered dataset of ~3,607 samples specialized for the analysis.
