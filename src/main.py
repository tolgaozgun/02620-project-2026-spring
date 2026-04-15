import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pca import PCA
from kmeans import KMeans
from logistic_regression import LogisticRegressionOvR
from mlp import train_mlp
import evaluate

def setup_run_directory():
    """Create a timestamped run directory for logging results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/runs/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "analysis"), exist_ok=True)

    return run_dir

class Logger:
    """Custom logger to write to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding='utf-8')
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging(run_dir):
    """Setup console and file logging."""
    log_file = os.path.join(run_dir, "logs", "training_log.txt")
    sys.stdout = Logger(log_file)
    return log_file

def train_val_test_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train/validation/test sets.
    Default: 70% train, 15% val, 15% test
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate train and validation from remaining data
    # Adjust val_size to account for the test set already removed
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size,
        random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_model(model, model_path, model_info):
    """Save model and its metadata."""
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    # Save model weights
    if hasattr(model, 'state_dict'):  # PyTorch model
        torch.save(model.state_dict(), model_path)
    else:  # Scikit-learn or custom model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    # Save model metadata
    info_path = model_path.replace('.pkl', '_info.json').replace('.pt', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Model info saved to: {info_path}")

def plot_learning_curves(history, save_path):
    """Plot training and validation learning curves."""
    if not history or 'train_loss' not in history:
        print("No history data available for learning curves")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training loss
    axes[0].plot(history['train_loss'], label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].legend()
    axes[0].grid(True)

    # Plot validation accuracy
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Validation Accuracy', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy Over Time')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to: {save_path}")

def detect_overfitting(y_train_pred, y_train_true, y_val_pred, y_val_true, model_name):
    """Detect and report potential overfitting."""
    train_acc = accuracy_score(y_train_true, y_train_pred)
    val_acc = accuracy_score(y_val_true, y_val_pred)
    overfitting_ratio = train_acc - val_acc

    print(f"\n--- Overfitting Analysis: {model_name} ---")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Difference (Train - Val): {overfitting_ratio:.4f}")

    if overfitting_ratio > 0.10:
        print("⚠️  MODERATE OVERFITTING: Training >> Validation")
        print("   Recommendation: Increase regularization or reduce model complexity")
    elif overfitting_ratio > 0.05:
        print("⚠️  MILD OVERFITTING: Consider increasing regularization")
    else:
        print("✅ NO SIGNIFICANT OVERFITTING: Good generalization")

    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'overfitting_ratio': overfitting_ratio
    }

def main():
    """Improved main function with proper train/val/test split and model saving."""
    # Setup
    print("=" * 80)
    print("CANCER SUBTYPE DISCOVERY - IMPROVED PIPELINE")
    print("=" * 80)
    print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    run_dir = setup_run_directory()
    log_file = setup_logging(run_dir)

    print(f"\n📁 Run directory: {run_dir}")
    print(f"📝 Log file: {log_file}")
    print("=" * 80)

    # 1. Load Data
    data_path = "data/processed_pancan.csv"
    if not os.path.exists(data_path):
        print("Processed data not found. Run data loader first.")
        return

    df = pd.read_csv(data_path, index_col=0)
    print(f"📊 Dataset loaded: {df.shape}")
    print(f"   Samples: {len(df)}, Features: {len(df.columns) - 1}")

    # 2. Prepare Features and Labels
    X = df.drop(columns=['cancer_type']).values
    y_raw = df['cancer_type'].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    n_classes = len(class_names)

    print(f"🎯 Classes: {list(class_names)}")
    print(f"📈 Class distribution: {pd.Series(y_raw).value_counts().to_dict()}")

    # 3. PROPER TRAIN/VAL/TEST SPLIT
    print("\n" + "=" * 80)
    print("DATA SPLITTING")
    print("=" * 80)
    print("Using 70% train / 15% validation / 15% test split")

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, test_size=0.15, val_size=0.15, random_state=42
    )

    print(f"✅ Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"✅ Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"✅ Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Save split information
    split_info = {
        'total_samples': int(len(X)),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'n_features': int(X.shape[1]),
        'n_classes': n_classes,
        'classes': list(class_names)
    }

    with open(os.path.join(run_dir, 'analysis', 'data_split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # 4. PREPROCESSING (Fit on training data only!)
    print("\n" + "=" * 80)
    print("PREPROCESSING")
    print("=" * 80)
    print("⚠️  IMPORTANT: Scalers and PCA fit on TRAINING data only")

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Fit PCA on training data
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"✅ Data scaled and PCA-transformed")
    print(f"✅ PCA explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    # Save preprocessing objects
    with open(os.path.join(run_dir, 'models', 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(run_dir, 'models', 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    # 5. METHOD 1: UNSUPERVISED DISCOVERY (Training data only)
    print("\n" + "=" * 80)
    print("METHOD 1: PCA + K-Means (Unsupervised Discovery)")
    print("=" * 80)
    print("Running on training set only to avoid data leakage")

    kmeans = KMeans(n_clusters=n_classes, random_state=42)
    kmeans.fit(X_train_pca)

    clustering_results = evaluate.evaluate_clustering(X_train_pca, y_train, kmeans.labels_)
    evaluate.plot_pca(X_train_pca, le.inverse_transform(y_train), title="PCA Projection (Training Set)",
                     output_path=os.path.join(run_dir, "graphs", "pca_projection.png"))

    # 6. METHOD 2: LOGISTIC REGRESSION
    print("\n" + "=" * 80)
    print("METHOD 2: Logistic Regression (One-vs-Rest)")
    print("=" * 80)
    print("Training on training set, tuning on validation set")

    # Hyperparameter tuning
    best_lr_f1 = -1
    best_lr_model = None
    best_lr_lambda = None

    for l2_lambda in [0.001, 0.01, 0.1]:
        print(f"Training LR with L2={l2_lambda}...")

        lr_model = LogisticRegressionOvR(lr=0.1, l2_lambda=l2_lambda, n_iters=1000)
        lr_model.fit(X_train_pca, y_train)

        # Evaluate on validation set
        y_val_pred = lr_model.predict(X_val_pca)
        val_f1 = f1_score(y_val, y_val_pred, average='macro')

        print(f"   Validation F1: {val_f1:.4f}")

        if val_f1 > best_lr_f1:
            best_lr_f1 = val_f1
            best_lr_model = lr_model
            best_lr_lambda = l2_lambda

    print(f"\n✅ Best LR model: L2={best_lr_lambda}, Val F1={best_lr_f1:.4f}")

    # Overfitting analysis
    y_train_pred = best_lr_model.predict(X_train_pca)
    lr_overfitting = detect_overfitting(y_train_pred, y_train, y_val_pred, y_val, "Logistic Regression")

    # Final test evaluation
    y_test_pred = best_lr_model.predict(X_test_pca)
    lr_test_results = evaluate.evaluate_classification(y_test, y_test_pred, class_names,
                                                     title="Logistic Regression (Test Set)",
                                                     output_path=os.path.join(run_dir, "graphs", "lr_test_confusion.png"))

    # Save best model
    save_model(best_lr_model, os.path.join(run_dir, 'models', 'best_lr_model.pkl'),
              {'model_type': 'LogisticRegressionOvR', 'l2_lambda': best_lr_lambda,
               'val_f1': best_lr_f1, 'test_f1': lr_test_results['F1']})

    # 7. METHOD 3: MLP
    print("\n" + "=" * 80)
    print("METHOD 3: Multi-Layer Perceptron")
    print("=" * 80)
    print("Training on training set, evaluating on validation set")

    import torch

    mlp_model, history = train_mlp(X_train_pca, y_train, X_val_pca, y_val, n_classes, epochs=50)

    # Plot learning curves
    plot_learning_curves(history, os.path.join(run_dir, "graphs", "mlp_learning_curves.png"))

    # Overfitting analysis
    mlp_model.eval()
    with torch.no_grad():
        y_train_pred_mlp = mlp_model(torch.FloatTensor(X_train_pca)).argmax(dim=1).numpy()
        y_val_pred_mlp = mlp_model(torch.FloatTensor(X_val_pca)).argmax(dim=1).numpy()

    mlp_overfitting = detect_overfitting(y_train_pred_mlp, y_train, y_val_pred_mlp, y_val, "MLP")

    # Final test evaluation
    y_test_pred_mlp = mlp_model(torch.FloatTensor(X_test_pca)).argmax(dim=1).numpy()
    mlp_test_results = evaluate.evaluate_classification(y_test, y_test_pred_mlp, class_names,
                                                       title="MLP (Test Set)",
                                                       output_path=os.path.join(run_dir, "graphs", "mlp_test_confusion.png"))

    # Save best model
    torch.save(mlp_model.state_dict(), os.path.join(run_dir, 'models', 'best_mlp_model.pt'))
    with open(os.path.join(run_dir, 'models', 'best_mlp_model_info.json'), 'w') as f:
        json.dump({'model_type': 'MLP', 'val_acc': history['val_acc'][-1],
                  'test_f1': mlp_test_results['F1']}, f, indent=2)

    # 8. FINAL COMPARISON AND SUMMARY
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    results_summary = {
        'clustering_ari': clustering_results['ARI'],
        'lr_test_f1': lr_test_results['F1'],
        'mlp_test_f1': mlp_test_results['F1'],
        'lr_overfitting': lr_overfitting['overfitting_ratio'],
        'mlp_overfitting': mlp_overfitting['overfitting_ratio'],
        'best_lr_lambda': best_lr_lambda
    }

    with open(os.path.join(run_dir, 'analysis', 'final_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n📊 Test Set Performance:")
    print(f"   Logistic Regression: {lr_test_results['F1']:.4f}")
    print(f"   MLP: {mlp_test_results['F1']:.4f}")
    print(f"   Clustering ARI: {clustering_results['ARI']:.4f}")

    print(f"\n🔍 Overfitting Analysis:")
    print(f"   LR overfitting ratio: {lr_overfitting['overfitting_ratio']:.4f}")
    print(f"   MLP overfitting ratio: {mlp_overfitting['overfitting_ratio']:.4f}")

    print(f"\n💾 All results saved to: {run_dir}")
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()