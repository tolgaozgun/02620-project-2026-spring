import os
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with a progress bar."""
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        return
    
    print(f"Downloading {url} to {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)

def load_data():
    # 1. Paths
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # GDC UUIDs for Pan-Cancer Atlas (Official Publication Data)
    expr_uuid = "3586c0da-64d0-4b74-a449-5ff4d9136611" # EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv
    clinical_uuid = "1b5f413e-a8d1-4d10-92eb-7c4ae739ed81" # TCGA-CDR-SupplementalTableS1.xlsx
    
    expr_url = f"https://api.gdc.cancer.gov/data/{expr_uuid}"
    clinical_url = f"https://api.gdc.cancer.gov/data/{clinical_uuid}"
    
    expr_file = os.path.join(data_dir, "pancan_expr.tsv")
    clinical_file = os.path.join(data_dir, "clinical_pancan.xlsx")
    
    # 2. Download
    download_file(expr_url, expr_file)
    download_file(clinical_url, clinical_file)
    
    # 3. Load Clinical Data
    print("Loading clinical data...")
    clinical_df = pd.read_excel(clinical_file, sheet_name="TCGA-CDR")
    target_types = ['BRCA', 'LUAD', 'KIRC', 'PRAD', 'COAD', 'GBM']
    clinical_df = clinical_df[clinical_df['type'].isin(target_types)]
    sample_to_type = dict(zip(clinical_df['bcr_patient_barcode'], clinical_df['type']))
    
    # 4. Optimized Expression Data Loading
    print("Filtering expression matrix columns...")
    # Read only the header first to find valid columns
    header = pd.read_csv(expr_file, sep='\t', nrows=0)
    
    valid_columns = ['gene_id'] # Keep the gene_id column
    column_mapping = {}
    
    for col in header.columns:
        patient_id = col[:12]
        if patient_id in sample_to_type:
            valid_columns.append(col)
            column_mapping[col] = sample_to_type[patient_id]
            
    print(f"Number of valid samples found: {len(valid_columns) - 1}")
    
    # Read only the relevant columns to save memory and time
    print("Loading subset of expression matrix...")
    expr_df = pd.read_csv(expr_file, sep='\t', usecols=valid_columns, index_col=0)
    
    # Transpose to Samples x Genes
    print("Transposing and labeling...")
    expr_df = expr_df.T
    expr_df['cancer_type'] = [column_mapping[idx] for idx in expr_df.index]
    
    # Log transformation: log2(x + 1)
    print("Applying log2 transformation...")
    numeric_cols = expr_df.columns[expr_df.columns != 'cancer_type']
    # Filter for protein coding or high variance if needed
    # First, handle NaNs (fill with 0 as missing detected means unexpressed)
    print("Handling NaNs and constant columns...")
    expr_df[numeric_cols] = np.log2(expr_df[numeric_cols].astype(float) + 1)
    expr_df[numeric_cols] = expr_df[numeric_cols].fillna(0)
    
    # Drop columns with 0 variance
    variances = expr_df[numeric_cols].var()
    non_constant_genes = variances[variances > 1e-6].index
    expr_df = expr_df[list(non_constant_genes) + ['cancer_type']]
    
    print(f"Final features count: {len(non_constant_genes)}")
    
    # 5. Save processed data
    output_file = os.path.join(data_dir, "processed_pancan.csv")
    print(f"Saving processed data to {output_file}...")
    expr_df.to_csv(output_file)
    print("Data loading and preprocessing complete.")
    return expr_df

if __name__ == "__main__":
    load_data()
