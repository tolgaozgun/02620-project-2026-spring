import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score

import data_loader
from pca import PCA
from kmeans import KMeans
from logistic_regression import LogisticRegressionOvR
from mlp import train_mlp
import evaluate

def main():
    # 1. Load Data
    data_path = "data/processed_pancan.csv"
    if not os.path.exists(data_path):
        print("Processed data not found. Running data loader...")
        df = data_loader.load_data()
    else:
        df = pd.read_csv(data_path, index_col=0)
    
    print(f"Dataset shape: {df.shape}")
    
    # 2. Prepare Features and Labels
    X = df.drop(columns=['cancer_type']).values
    y_raw = df['cancer_type'].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    n_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Method 1: PCA + K-Means (Unsupervised)
    print("\n--- Method 1: PCA + K-Means ---")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio (top 50): {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_classes, random_state=42)
    kmeans.fit(X_pca)
    
    # Evaluate Clustering
    evaluate.evaluate_clustering(X_pca, y, kmeans.labels_)
    evaluate.plot_pca(X_pca, y_raw, title="PCA Projection with True Labels")
    
    # 4. Method 2: Logistic Regression (Supervised - NumPy)
    print("\n--- Method 2: Logistic Regression (OvR) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pca, y)):
        X_train, X_val = X_pca[train_idx], X_pca[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Hyperparameter tuning for l2_lambda as specified in proposal
        best_fold_f1 = -1
        best_fold_model = None
        
        for l2 in [0.001, 0.01, 0.1]:
            temp_model = LogisticRegressionOvR(lr=0.1, l2_lambda=l2, n_iters=1000)
            temp_model.fit(X_train, y_train)
            y_pred_temp = temp_model.predict(X_val)
            f1_temp = f1_score(y_val, y_pred_temp, average='macro')
            
            if f1_temp > best_fold_f1:
                best_fold_f1 = f1_temp
                best_fold_model = temp_model
        
        y_pred = best_fold_model.predict(X_val)
        res = evaluate.evaluate_classification(y_val, y_pred, class_names, title=f"LR Fold {fold+1}")
        lr_scores.append(res['F1'])
        
    print(f"Logistic Regression Mean F1: {np.mean(lr_scores):.4f}")
    
    # 5. Method 3: Multi-Layer Perceptron (Supervised - PyTorch)
    print("\n--- Method 3: PyTorch MLP ---")
    mlp_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pca, y)):
        X_train, X_val = X_pca[train_idx], X_pca[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model, history = train_mlp(X_train, y_train, X_val, y_val, n_classes, epochs=50)
        
        # Final Eval
        import torch
        model.eval()
        with torch.no_grad():
            outputs = model(torch.FloatTensor(X_val))
            _, predicted_tensor = torch.max(outputs, 1)
            y_pred = predicted_tensor.cpu().numpy() if hasattr(predicted_tensor, 'numpy') else predicted_tensor.cpu().tolist()
            
        res = evaluate.evaluate_classification(y_val, y_pred, class_names, title=f"MLP Fold {fold+1}")
        mlp_scores.append(res['F1'])
        
    print(f"MLP Mean F1: {np.mean(mlp_scores):.4f}")
    
    print("\nProject Complete. Results saved in results/ directory.")

if __name__ == "__main__":
    main()
