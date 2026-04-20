"""
Cancer Subtype Discovery Pipeline
TCGA Pan-Cancer RNA-seq | BRCA / LUAD / KIRC / PRAD / COAD / GBM

Generates all figures and analysis into a timestamped run directory.
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pca import PCA
from kmeans import KMeans
from logistic_regression import LogisticRegressionOvR
from mlp import MLP, train_mlp
import evaluate

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10,
})

CANCER_COLORS = {
    "BRCA": "#E64B35", "COAD": "#4DBBD5", "GBM": "#00A087",
    "KIRC": "#3C5488", "LUAD": "#F39B7F", "PRAD": "#8491B4",
}

# ── Infrastructure ────────────────────────────────────────────────────────────

def setup_run_directory():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", "runs", f"run_{ts}")
    for sub in ("logs", "graphs", "graphs/new_figures", "models", "analysis"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def G(run_dir, name):
    """Shorthand: path to graphs/ subfolder."""
    return os.path.join(run_dir, "graphs", name)


def NF(run_dir, name):
    """Shorthand: path to graphs/new_figures/ subfolder."""
    return os.path.join(run_dir, "graphs", "new_figures", name)


def A(run_dir, name):
    """Shorthand: path to analysis/ subfolder."""
    return os.path.join(run_dir, "analysis", name)


# ── Data helpers ──────────────────────────────────────────────────────────────

def holdout_split(X, y, test_size=0.15, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def detect_overfitting(train_acc, val_acc, model_name):
    diff = train_acc - val_acc
    print(f"\n--- Overfitting Analysis: {model_name} ---")
    print(f"  Train Accuracy: {train_acc:.4f}  |  Val Accuracy: {val_acc:.4f}"
          f"  |  Diff: {diff:.4f}")
    if diff > 0.10:
        print("  WARNING: MODERATE OVERFITTING")
    elif diff > 0.05:
        print("  WARNING: MILD OVERFITTING")
    else:
        print("  OK: No significant overfitting")
    return {"train_acc": train_acc, "val_acc": val_acc, "diff": diff}


# ── Inline figure helpers ─────────────────────────────────────────────────────

def fig_class_distribution(class_names, y_raw, path):
    counts = pd.Series(y_raw).value_counts().reindex(class_names)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(class_names, counts.values,
                  color=[CANCER_COLORS[c] for c in class_names],
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(val), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Cancer Type"); ax.set_ylabel("Number of Samples")
    ax.set_title("Sample Count per Cancer Type (TCGA Pan-Cancer)")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def fig_kmeans_side_by_side(X_pca, y_names, class_names,
                             cluster_labels, centers, ari, nmi, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for cancer in class_names:
        mask = y_names == cancer
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        label=cancer, color=CANCER_COLORS[cancer],
                        alpha=0.5, s=15, linewidths=0)
    axes[0].set_title("True Cancer Type Labels")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(title="Cancer Type", fontsize=9, markerscale=1.5)

    palette = sns.color_palette("Set2", len(centers))
    n_clusters = len(centers)
    for k in range(n_clusters):
        mask = cluster_labels == k
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        label=f"Cluster {k}", color=palette[k],
                        alpha=0.5, s=15, linewidths=0)
    for k, center in enumerate(centers):
        axes[1].scatter(center[0], center[1], marker="X", s=120,
                        color=palette[k], edgecolors="black",
                        linewidths=0.8, zorder=5)
    axes[1].set_title(f"K-Means Clusters (k={n_clusters})  |  ARI={ari:.3f}, NMI={nmi:.3f}")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].legend(title="Cluster", fontsize=9, markerscale=1.5)

    plt.suptitle("Unsupervised Discovery: PCA + K-Means vs Ground Truth", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def fig_cluster_purity(cluster_labels, y_names, class_names, path):
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for k in range(n):
        mask = cluster_labels == k
        for j, cancer in enumerate(class_names):
            matrix[k, j] = np.sum((y_names == cancer) & mask)
    norm = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(norm, annot=matrix, fmt="d",
                xticklabels=class_names,
                yticklabels=[f"Cluster {k}" for k in range(n)],
                cmap="YlOrRd", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Proportion of cluster"})
    ax.set_xlabel("True Cancer Type"); ax.set_ylabel("K-Means Cluster")
    ax.set_title("Cluster Composition — K-Means vs True Labels\n(cell values = sample counts)")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def fig_cv_folds(lr_folds, mlp_folds, path):
    x = np.arange(5); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, lr_folds,  w, label="Logistic Regression", color="#3C5488", alpha=0.85)
    ax.bar(x + w/2, mlp_folds, w, label="MLP",                  color="#E64B35", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
    ax.set_ylim(max(0, min(lr_folds + mlp_folds) - 0.01), 1.003)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("5-Fold CV Macro F1 — Logistic Regression vs MLP")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.axhline(np.mean(lr_folds),  color="#3C5488", linestyle="--", lw=1, alpha=0.7)
    ax.axhline(np.mean(mlp_folds), color="#E64B35", linestyle="--", lw=1, alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def fig_perclass_f1(y_test, y_pred_lr, y_pred_mlp, class_names, n_classes, path):
    def per_class(y_true, y_pred):
        return [f1_score(y_true == k, y_pred == k) for k in range(n_classes)]

    lr_pc = per_class(y_test, y_pred_lr)
    mlp_pc = per_class(y_test, y_pred_mlp)
    x = np.arange(n_classes); w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, lr_pc,  w, label="Logistic Regression", color="#3C5488", alpha=0.85)
    ax.bar(x + w/2, mlp_pc, w, label="MLP",                  color="#E64B35", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylim(0.88, 1.02); ax.set_ylabel("F1 Score (binary OvR)")
    ax.set_title("Per-Class F1 Score on Test Set — LR vs MLP")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def fig_summary_comparison(lr_results, mlp_results, path):
    metrics = ["Macro F1", "Macro Precision", "Macro Recall"]
    lr_vals  = [lr_results["F1"],  lr_results["Precision"],  lr_results["Recall"]]
    mlp_vals = [mlp_results["F1"], mlp_results["Precision"], mlp_results["Recall"]]
    x = np.arange(3); w = 0.3

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, lr_vals,  w, label="Logistic Regression", color="#3C5488", alpha=0.85)
    b2 = ax.bar(x + w/2, mlp_vals, w, label="MLP",                  color="#E64B35", alpha=0.85)
    for b, v in list(zip(b1, lr_vals)) + list(zip(b2, mlp_vals)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0.975, 1.005); ax.set_ylabel("Score")
    ax.set_title("Test Set Performance — Logistic Regression vs MLP")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def fig_mlp_learning_curves(history, path):
    if not history or "train_loss" not in history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], color="#3C5488", linewidth=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("MLP Training Loss"); axes[0].grid(True)
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], color="#E64B35", linestyle="--", linewidth=1.5)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Accuracy")
        axes[1].set_title("MLP Validation Accuracy"); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


# ── New extended figures ───────────────────────────────────────────────────────

def fig_decision_boundary(lr_model, mlp_model, X_test_pca, y_test,
                           class_names, n_classes, path):
    """
    LR vs MLP decision regions on the PC1-PC2 plane.
    All other PCs fixed at 0 (PCA-space mean).
    """
    pad = 2.0
    x1_min = X_test_pca[:, 0].min() - pad
    x1_max = X_test_pca[:, 0].max() + pad
    x2_min = X_test_pca[:, 1].min() - pad
    x2_max = X_test_pca[:, 1].max() + pad

    res = 200
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, res),
                         np.linspace(x2_min, x2_max, res))
    n_grid = xx.ravel().shape[0]
    grid_50d = np.zeros((n_grid, X_test_pca.shape[1]))
    grid_50d[:, 0] = xx.ravel()
    grid_50d[:, 1] = yy.ravel()

    # LR predictions
    z_lr = lr_model.predict(grid_50d).reshape(xx.shape)

    # MLP predictions
    mlp_model.eval()
    with torch.no_grad():
        z_mlp = mlp_model(torch.FloatTensor(grid_50d)).argmax(dim=1).numpy().reshape(xx.shape)

    disagree = (z_lr != z_mlp).astype(float)

    colors_list = [CANCER_COLORS[c] for c in class_names]
    cmap_cls = matplotlib.colors.ListedColormap(colors_list)
    levels = np.arange(-0.5, n_classes + 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for ax, z, title in [
        (axes[0], z_lr,  "Logistic Regression"),
        (axes[1], z_mlp, "MLP [512, 256]"),
    ]:
        ax.contourf(xx, yy, z, alpha=0.22, cmap=cmap_cls, levels=levels)
        for k, cancer in enumerate(class_names):
            mask = y_test == k
            ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                       color=CANCER_COLORS[cancer], s=14, alpha=0.75,
                       label=cancer, linewidths=0, zorder=2)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.legend(title="True Label", fontsize=8, markerscale=1.5,
                  bbox_to_anchor=(1.01, 1), loc="upper left")

    # Disagreement panel
    axes[2].pcolormesh(xx, yy, disagree, cmap="RdYlGn_r", alpha=0.45,
                       vmin=0, vmax=1, shading="auto")
    for k, cancer in enumerate(class_names):
        mask = y_test == k
        axes[2].scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                        color=CANCER_COLORS[cancer], s=14, alpha=0.75,
                        label=cancer, linewidths=0, zorder=2)
    axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")
    axes[2].set_title("Regions Where LR \u2260 MLP (red)")
    axes[2].legend(title="True Label", fontsize=8, markerscale=1.5,
                   bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.suptitle(
        "Decision Boundaries in PC1\u2013PC2 Space  (all other PCs fixed at 0)",
        fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def fig_perclass_perm_heatmap(perclass_drops, top_pc_indices, class_names, path):
    """
    Heatmap: rows = top PCs (sorted by total importance),
             cols = cancer types,
             values = macro-F1 drop for that class when that PC is permuted.
    """
    pc_labels = [f"PC{i+1}" for i in top_pc_indices]

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(perclass_drops, annot=True, fmt=".3f",
                xticklabels=class_names,
                yticklabels=pc_labels,
                cmap="Reds", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Per-class F1 drop when permuted"})
    ax.set_xlabel("Cancer Type")
    ax.set_ylabel("Principal Component (sorted by total importance)")
    ax.set_title(
        "Per-Class Permutation Importance\n"
        "Which PCs Matter for Which Cancer Type?")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def fig_summary_3models(lr_results, svm_results, mlp_results, path):
    """Test-set F1/Precision/Recall bar chart for all three models."""
    metrics = ["Macro F1", "Macro Precision", "Macro Recall"]
    lr_vals  = [lr_results["F1"],  lr_results["Precision"],  lr_results["Recall"]]
    svm_vals = [svm_results["F1"], svm_results["Precision"], svm_results["Recall"]]
    mlp_vals = [mlp_results["F1"], mlp_results["Precision"], mlp_results["Recall"]]

    x = np.arange(3); w = 0.22
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w,   lr_vals,  w, label="Logistic Regression", color="#3C5488", alpha=0.85)
    b2 = ax.bar(x,       svm_vals, w, label="RBF-SVM",             color="#F39B7F", alpha=0.85)
    b3 = ax.bar(x + w,   mlp_vals, w, label="MLP [512, 256]",      color="#E64B35", alpha=0.85)

    for bars, vals in [(b1, lr_vals), (b2, svm_vals), (b3, mlp_vals)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0004,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    all_vals = lr_vals + svm_vals + mlp_vals
    ax.set_ylim(max(0.965, min(all_vals) - 0.005), 1.005)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Test Set Performance — LR vs RBF-SVM vs MLP")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def fig_cv_folds_3models(lr_folds, svm_folds, mlp_folds, path):
    """5-fold CV F1 bar chart for all three models."""
    x = np.arange(5); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w,   lr_folds,  w, label="Logistic Regression", color="#3C5488", alpha=0.85)
    ax.bar(x,       svm_folds, w, label="RBF-SVM",             color="#F39B7F", alpha=0.85)
    ax.bar(x + w,   mlp_folds, w, label="MLP [512, 256]",      color="#E64B35", alpha=0.85)

    all_folds = lr_folds + svm_folds + mlp_folds
    ax.set_ylim(max(0, min(all_folds) - 0.01), 1.003)
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("5-Fold CV Macro F1 — LR vs RBF-SVM vs MLP")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for folds, color in [(lr_folds, "#3C5488"), (svm_folds, "#F39B7F"), (mlp_folds, "#E64B35")]:
        ax.axhline(np.mean(folds), color=color, linestyle="--", lw=1, alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_lr_cv(X_dev, y_dev, lambdas, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {l: [] for l in lambdas}

    for fold_idx, (tr, va) in enumerate(skf.split(X_dev, y_dev)):
        X_tr, X_va = X_dev[tr], X_dev[va]
        y_tr, y_va = y_dev[tr], y_dev[va]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        pca = PCA(n_components=50)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_va_p = pca.transform(X_va_s)

        for l in lambdas:
            m = LogisticRegressionOvR(lr=0.1, l2_lambda=l, n_iters=1000)
            m.fit(X_tr_p, y_tr)
            results[l].append(f1_score(y_va, m.predict(X_va_p), average="macro"))

        print(f"  Fold {fold_idx + 1}/{n_splits} done")

    return {l: {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": v}
            for l, v in results.items()}


def run_mlp_cv(X_dev, y_dev, n_classes, epochs=50, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_f1s = []

    for fold_idx, (tr, va) in enumerate(skf.split(X_dev, y_dev)):
        X_tr, X_va = X_dev[tr], X_dev[va]
        y_tr, y_va = y_dev[tr], y_dev[va]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        pca = PCA(n_components=50)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_va_p = pca.transform(X_va_s)

        model, _ = train_mlp(X_tr_p, y_tr, X_va_p, y_va, n_classes, epochs=epochs)
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_va_p)).argmax(dim=1).numpy()
        fold_f1s.append(f1_score(y_va, y_pred, average="macro"))
        print(f"  Fold {fold_idx + 1}/{n_splits}: F1 = {fold_f1s[-1]:.4f}")

    return {"mean": float(np.mean(fold_f1s)), "std": float(np.std(fold_f1s)),
            "folds": fold_f1s}


def run_svm_cv(X_dev, y_dev, n_splits=5, random_state=42):
    """5-fold stratified CV for RBF-SVM (preprocessing inside each fold)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_f1s = []

    for fold_idx, (tr, va) in enumerate(skf.split(X_dev, y_dev)):
        X_tr, X_va = X_dev[tr], X_dev[va]
        y_tr, y_va = y_dev[tr], y_dev[va]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        pca_fold = PCA(n_components=50)
        X_tr_p = pca_fold.fit_transform(X_tr_s)
        X_va_p = pca_fold.transform(X_va_s)

        svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        svm.fit(X_tr_p, y_tr)
        fold_f1s.append(f1_score(y_va, svm.predict(X_va_p), average="macro"))
        print(f"  Fold {fold_idx + 1}/{n_splits}: F1 = {fold_f1s[-1]:.4f}")

    return {"mean": float(np.mean(fold_f1s)), "std": float(np.std(fold_f1s)),
            "folds": fold_f1s}


# ── Feature importance ────────────────────────────────────────────────────────

def compute_lr_gene_weights(lr_model, pca, class_names):
    """Project OvR weights from PCA space back to gene space."""
    weights_per_class = {}
    for cancer, (w, _) in zip(class_names, lr_model.models):
        gene_weights = pca.components.T @ w
        weights_per_class[cancer] = gene_weights
    return weights_per_class


def compute_permutation_importance(model, X_test_pca, y_test, baseline_f1,
                                   n_repeats=3, random_state=42):
    """
    Drop in macro-F1 when each PC is permuted (averaged over n_repeats).
    Returns array of shape (n_components,).
    """
    rng = np.random.default_rng(random_state)
    n_components = X_test_pca.shape[1]
    importances = np.zeros(n_components)

    model.eval()
    for j in range(n_components):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test_pca.copy()
            rng.shuffle(X_perm[:, j])
            with torch.no_grad():
                y_pred = model(torch.FloatTensor(X_perm)).argmax(dim=1).numpy()
            drops.append(baseline_f1 - f1_score(y_test, y_pred, average="macro"))
        importances[j] = np.mean(drops)

    return importances


def compute_perclass_permutation(model, X_test_pca, y_test, n_classes,
                                  top_pc_indices, n_repeats=3, random_state=42):
    """
    Per-class F1 drop when each of the given PC columns is permuted.
    Returns array of shape (len(top_pc_indices), n_classes).
    """
    rng = np.random.default_rng(random_state + 1)  # different seed to avoid correlation

    model.eval()
    with torch.no_grad():
        y_pred_base = model(torch.FloatTensor(X_test_pca)).argmax(dim=1).numpy()
    baseline = np.array([f1_score(y_test == k, y_pred_base == k)
                         for k in range(n_classes)])

    result = np.zeros((len(top_pc_indices), n_classes))
    for i, pc_idx in enumerate(top_pc_indices):
        drops = np.zeros((n_repeats, n_classes))
        for r in range(n_repeats):
            X_perm = X_test_pca.copy()
            rng.shuffle(X_perm[:, pc_idx])
            with torch.no_grad():
                y_pred = model(torch.FloatTensor(X_perm)).argmax(dim=1).numpy()
            drops[r] = baseline - np.array([f1_score(y_test == k, y_pred == k)
                                            for k in range(n_classes)])
        result[i] = drops.mean(axis=0)

    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("CANCER SUBTYPE DISCOVERY PIPELINE")
    print("TCGA Pan-Cancer RNA-seq | BRCA / LUAD / KIRC / PRAD / COAD / GBM")
    print("=" * 80)
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    run_dir = setup_run_directory()
    sys.stdout = Logger(os.path.join(run_dir, "logs", "training_log.txt"))
    print(f"Run directory: {run_dir}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1 — DATA LOADING")
    print("=" * 80)

    data_path = "data/processed_pancan.csv"
    if not os.path.exists(data_path):
        print("ERROR: Run src/data_loader.py first.")
        return

    df = pd.read_csv(data_path, index_col=0)
    gene_names = [c for c in df.columns if c != "cancer_type"]
    print(f"Dataset: {df.shape[0]} samples × {len(gene_names)} genes")

    X = df[gene_names].values
    y_raw = df["cancer_type"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    n_classes = len(class_names)

    print(f"Classes ({n_classes}): {list(class_names)}")
    for cls, cnt in pd.Series(y_raw).value_counts().items():
        print(f"  {cls}: {cnt}")

    fig_class_distribution(class_names, y_raw, G(run_dir, "fig01_class_distribution.png"))

    # ── 2. Hold-out test split ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2 — TRAIN / TEST SPLIT (85% dev / 15% test, stratified)")
    print("=" * 80)

    X_dev, X_test, y_dev, y_test = holdout_split(X, y, test_size=0.15)
    print(f"Development: {len(X_dev)} samples  |  Test (held-out): {len(X_test)} samples")

    with open(A(run_dir, "data_split.json"), "w") as f:
        json.dump({"total": int(len(X)), "dev": int(len(X_dev)),
                   "test": int(len(X_test)), "n_genes": len(gene_names),
                   "n_classes": n_classes, "classes": list(class_names)}, f, indent=2)

    # ── 3. Shared preprocessing (dev set) ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 3 — PREPROCESSING (fit on development set)")
    print("=" * 80)

    scaler = StandardScaler()
    X_dev_s  = scaler.fit_transform(X_dev)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=50)
    X_dev_pca  = pca.fit_transform(X_dev_s)
    X_test_pca = pca.transform(X_test_s)

    print(f"StandardScaler + PCA(50) fit on {len(X_dev)} development samples")
    print(f"Top 50 PCs explain {pca.total_explained_variance_ratio_*100:.2f}% of total variance")

    with open(os.path.join(run_dir, "models", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(run_dir, "models", "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    evaluate.plot_explained_variance(
        pca.explained_variance_ratio_, pca.total_explained_variance_ratio_,
        G(run_dir, "fig02_pca_variance.png"))

    evaluate.plot_pc_loadings(
        pca.components, gene_names, n_top=15,
        output_path=G(run_dir, "fig12_pc_loadings.png"))

    # ── 4. METHOD 1 — PCA + K-Means ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("METHOD 1 — PCA + K-Means (Unsupervised Discovery)")
    print("=" * 80)

    kmeans = KMeans(n_clusters=n_classes, random_state=42)
    kmeans.fit(X_dev_pca)
    print(f"KMeans | Inertia: {kmeans.inertia_:.2f}")

    clustering_results = evaluate.evaluate_clustering(X_dev_pca, y_dev, kmeans.labels_)

    y_dev_names = le.inverse_transform(y_dev)

    evaluate.plot_pca(
        X_dev_pca, y_dev_names,
        title="PCA Projection — True Cancer Type Labels",
        legend_title="Cancer Type",
        output_path=G(run_dir, "fig03_pca_true_labels.png"))

    fig_kmeans_side_by_side(
        X_dev_pca, y_dev_names, class_names,
        kmeans.labels_, kmeans.cluster_centers_,
        clustering_results["ARI"], clustering_results["NMI"],
        G(run_dir, "fig04_kmeans_vs_truth.png"))

    fig_cluster_purity(kmeans.labels_, y_dev_names, class_names,
                       G(run_dir, "fig05_cluster_purity.png"))

    with open(A(run_dir, "clustering_results.json"), "w") as f:
        json.dump({**clustering_results, "inertia": kmeans.inertia_}, f, indent=2)

    # ── 5. METHOD 2 — Logistic Regression (5-fold CV) ─────────────────────────
    print("\n" + "=" * 80)
    print("METHOD 2 — Logistic Regression OvR (5-fold Stratified CV)")
    print("=" * 80)

    lambdas = [0.001, 0.01, 0.1]
    lr_cv = run_lr_cv(X_dev, y_dev, lambdas)

    print("\nCV macro-F1:")
    best_lambda, best_f1 = max(
        ((l, lr_cv[l]["mean"]) for l in lambdas), key=lambda x: x[1])
    for l in lambdas:
        marker = " <-- best" if l == best_lambda else ""
        print(f"  L2={l}: {lr_cv[l]['mean']:.4f} ± {lr_cv[l]['std']:.4f}{marker}")

    # Retrain on full dev set with shared preprocessing
    np.random.seed(42)
    print(f"\nRetraining LR (L2={best_lambda}) on full development set...")
    lr_model = LogisticRegressionOvR(lr=0.1, l2_lambda=best_lambda, n_iters=1000)
    lr_model.fit(X_dev_pca, y_dev)

    y_dev_pred_lr  = lr_model.predict(X_dev_pca)
    y_test_pred_lr = lr_model.predict(X_test_pca)

    detect_overfitting(accuracy_score(y_dev, y_dev_pred_lr), lr_cv[best_lambda]["mean"],
                       "Logistic Regression")

    lr_test_results = evaluate.evaluate_classification(
        y_test, y_test_pred_lr, class_names,
        title="Logistic Regression (Test Set)",
        output_path=G(run_dir, "fig07_lr_confusion.png"))

    with open(os.path.join(run_dir, "models", "lr_model.pkl"), "wb") as f:
        pickle.dump(lr_model, f)

    # ── 6. LR feature importance (gene-space projection) ──────────────────────
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE — LR: projecting weights to gene space")
    print("=" * 80)

    lr_gene_weights = compute_lr_gene_weights(lr_model, pca, class_names)

    gene_summary = evaluate.plot_lr_gene_importance(
        lr_gene_weights, class_names, gene_names, n_top=15,
        output_path=G(run_dir, "fig11_lr_gene_importance.png"))

    print("\nTop 5 genes per cancer type:")
    top_genes_report = {}
    for cancer in class_names:
        top5 = gene_summary[cancer][:5]
        top5_names = [g for g, _ in top5]
        top_genes_report[cancer] = top5_names
        print(f"  {cancer}: {', '.join(top5_names)}")

    with open(A(run_dir, "lr_top_genes.json"), "w") as f:
        json.dump({c: [(g, round(w, 6)) for g, w in gene_summary[c]]
                   for c in class_names}, f, indent=2)

    # ── 7. METHOD 3 — MLP (5-fold CV) ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("METHOD 3 — MLP (5-fold Stratified CV)")
    print("=" * 80)

    torch.manual_seed(42)
    mlp_cv = run_mlp_cv(X_dev, y_dev, n_classes, epochs=50)
    print(f"\nMLP CV macro-F1: {mlp_cv['mean']:.4f} ± {mlp_cv['std']:.4f}")

    # Retrain on full dev set — hold a small internal val slice for curve monitoring
    print("\nRetraining MLP on full development set...")
    X_dev_tr, X_dev_monitor, y_dev_tr, y_dev_monitor = train_test_split(
        X_dev_pca, y_dev, test_size=0.10, random_state=42, stratify=y_dev)
    torch.manual_seed(42)
    mlp_model, history = train_mlp(X_dev_tr, y_dev_tr, X_dev_monitor, y_dev_monitor,
                                   n_classes, epochs=50)

    fig_mlp_learning_curves(history, G(run_dir, "fig08_mlp_learning_curves.png"))

    mlp_model.eval()
    with torch.no_grad():
        y_dev_pred_mlp  = mlp_model(torch.FloatTensor(X_dev_pca)).argmax(dim=1).numpy()
        y_test_pred_mlp = mlp_model(torch.FloatTensor(X_test_pca)).argmax(dim=1).numpy()

    detect_overfitting(accuracy_score(y_dev, y_dev_pred_mlp), mlp_cv["mean"], "MLP")

    mlp_test_results = evaluate.evaluate_classification(
        y_test, y_test_pred_mlp, class_names,
        title="MLP (Test Set)",
        output_path=G(run_dir, "fig09_mlp_confusion.png"))

    torch.save(mlp_model.state_dict(),
               os.path.join(run_dir, "models", "mlp_model.pt"))

    # ── 8. MLP permutation importance ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE — MLP: permutation importance over PCA dimensions")
    print("=" * 80)

    baseline_f1 = mlp_test_results["F1"]
    perm_scores = compute_permutation_importance(
        mlp_model, X_test_pca, y_test, baseline_f1, n_repeats=3)

    evaluate.plot_permutation_importance(
        perm_scores, output_path=G(run_dir, "fig13_mlp_permutation_importance.png"))

    top10_pcs = np.argsort(perm_scores)[::-1][:10]
    print("Top 10 most important PCs (by F1 drop when permuted):")
    for rank, pc in enumerate(top10_pcs, 1):
        print(f"  {rank}. PC{pc+1}: F1 drop = {perm_scores[pc]:.4f}")

    with open(A(run_dir, "mlp_permutation_importance.json"), "w") as f:
        json.dump({"per_pc": {f"PC{i+1}": float(perm_scores[i])
                              for i in range(len(perm_scores))},
                   "top10_pcs": [int(i+1) for i in top10_pcs]}, f, indent=2)

    # ── 9. METHOD 4 — RBF-SVM (5-fold CV) ────────────────────────────────────
    print("\n" + "=" * 80)
    print("METHOD 4 — RBF-SVM (5-fold Stratified CV)")
    print("=" * 80)

    svm_cv = run_svm_cv(X_dev, y_dev)
    print(f"\nSVM CV macro-F1: {svm_cv['mean']:.4f} ± {svm_cv['std']:.4f}")

    print("\nRetraining RBF-SVM on full development set...")
    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm_model.fit(X_dev_pca, y_dev)

    y_test_pred_svm = svm_model.predict(X_test_pca)
    svm_test_results = evaluate.evaluate_classification(
        y_test, y_test_pred_svm, class_names,
        title="RBF-SVM (Test Set)",
        output_path=NF(run_dir, "figN1_svm_confusion.png"))

    print(f"\nSVM vs LR test F1 gap: "
          f"{(svm_test_results['F1'] - lr_test_results['F1'])*100:+.2f} pp")
    print(f"MLP vs SVM test F1 gap: "
          f"{(mlp_test_results['F1'] - svm_test_results['F1'])*100:+.2f} pp")

    with open(os.path.join(run_dir, "models", "svm_model.pkl"), "wb") as f:
        pickle.dump(svm_model, f)

    # ── 10. NEW FIGURES (new_figures/ subfolder) ───────────────────────────────
    print("\n" + "=" * 80)
    print("GENERATING NEW ANALYSIS FIGURES")
    print("=" * 80)

    # 4A — Decision boundary on PC1-PC2
    print("\nGenerating decision boundary figure...")
    y_test_names = le.inverse_transform(y_test)
    fig_decision_boundary(
        lr_model, mlp_model, X_test_pca, y_test,
        class_names, n_classes,
        NF(run_dir, "figN2_decision_boundary.png"))

    # 4C — Per-class permutation importance heatmap (top 8 PCs)
    print("\nComputing per-class permutation importance (top 8 PCs)...")
    top8_pcs = np.argsort(perm_scores)[::-1][:8]
    perclass_drops = compute_perclass_permutation(
        mlp_model, X_test_pca, y_test, n_classes, top8_pcs, n_repeats=3)
    fig_perclass_perm_heatmap(
        perclass_drops, top8_pcs, class_names,
        NF(run_dir, "figN3_perclass_perm_heatmap.png"))

    with open(A(run_dir, "mlp_perclass_permutation.json"), "w") as f:
        json.dump({f"PC{top8_pcs[i]+1}": {class_names[k]: float(perclass_drops[i, k])
                                           for k in range(n_classes)}
                   for i in range(len(top8_pcs))}, f, indent=2)

    # 3-model summary comparison
    fig_summary_3models(lr_test_results, svm_test_results, mlp_test_results,
                        NF(run_dir, "figN4_summary_3models.png"))

    # 3-model CV folds
    fig_cv_folds_3models(
        lr_cv[best_lambda]["folds"], svm_cv["folds"], mlp_cv["folds"],
        NF(run_dir, "figN5_cv_folds_3models.png"))

    # ── 11. Combined figures ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GENERATING COMBINED FIGURES")
    print("=" * 80)

    fig_cv_folds(lr_cv[best_lambda]["folds"], mlp_cv["folds"],
                 G(run_dir, "fig06_cv_folds.png"))

    fig_perclass_f1(y_test, y_test_pred_lr, y_test_pred_mlp,
                    class_names, n_classes,
                    G(run_dir, "fig10_perclass_f1.png"))

    fig_summary_comparison(lr_test_results, mlp_test_results,
                           G(run_dir, "fig_summary_comparison.png"))

    # ── 12. Save consolidated results ─────────────────────────────────────────
    final = {
        "clustering": {**clustering_results, "inertia": kmeans.inertia_},
        "lr": {
            "best_lambda": best_lambda,
            "cv_f1_mean": lr_cv[best_lambda]["mean"],
            "cv_f1_std": lr_cv[best_lambda]["std"],
            "cv_fold_f1s": lr_cv[best_lambda]["folds"],
            "test_f1": lr_test_results["F1"],
            "test_precision": lr_test_results["Precision"],
            "test_recall": lr_test_results["Recall"],
            "cv_all_lambdas": {str(l): lr_cv[l] for l in lambdas},
        },
        "svm": {
            "cv_f1_mean": svm_cv["mean"],
            "cv_f1_std": svm_cv["std"],
            "cv_fold_f1s": svm_cv["folds"],
            "test_f1": svm_test_results["F1"],
            "test_precision": svm_test_results["Precision"],
            "test_recall": svm_test_results["Recall"],
        },
        "mlp": {
            "cv_f1_mean": mlp_cv["mean"],
            "cv_f1_std": mlp_cv["std"],
            "cv_fold_f1s": mlp_cv["folds"],
            "test_f1": mlp_test_results["F1"],
            "test_precision": mlp_test_results["Precision"],
            "test_recall": mlp_test_results["Recall"],
        },
        "feature_importance": {
            "lr_top_genes": {c: [g for g, _ in gene_summary[c][:5]] for c in class_names},
            "mlp_top10_pcs": [int(i+1) for i in top10_pcs],
        },
        "run_dir": run_dir,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(A(run_dir, "final_results.json"), "w") as f:
        json.dump(final, f, indent=2)

    # ── 13. Print final summary ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nMethod 1 — PCA + K-Means (Unsupervised)")
    print(f"  ARI={clustering_results['ARI']:.4f}  NMI={clustering_results['NMI']:.4f}"
          f"  Silhouette={clustering_results['Silhouette']:.4f}")

    print(f"\nMethod 2 — Logistic Regression (L2={best_lambda})")
    print(f"  5-fold CV F1: {lr_cv[best_lambda]['mean']:.4f} ± {lr_cv[best_lambda]['std']:.4f}")
    print(f"  Test F1: {lr_test_results['F1']:.4f}  "
          f"Precision: {lr_test_results['Precision']:.4f}  "
          f"Recall: {lr_test_results['Recall']:.4f}")

    print(f"\nMethod 3 — RBF-SVM")
    print(f"  5-fold CV F1: {svm_cv['mean']:.4f} ± {svm_cv['std']:.4f}")
    print(f"  Test F1: {svm_test_results['F1']:.4f}  "
          f"Precision: {svm_test_results['Precision']:.4f}  "
          f"Recall: {svm_test_results['Recall']:.4f}")

    print(f"\nMethod 4 — MLP [512, 256]")
    print(f"  5-fold CV F1: {mlp_cv['mean']:.4f} ± {mlp_cv['std']:.4f}")
    print(f"  Test F1: {mlp_test_results['F1']:.4f}  "
          f"Precision: {mlp_test_results['Precision']:.4f}  "
          f"Recall: {mlp_test_results['Recall']:.4f}")

    print(f"\nMLP vs LR test F1 gap:  {(mlp_test_results['F1'] - lr_test_results['F1'])*100:+.2f} pp")
    print(f"MLP vs SVM test F1 gap: {(mlp_test_results['F1'] - svm_test_results['F1'])*100:+.2f} pp")
    print(f"SVM vs LR test F1 gap:  {(svm_test_results['F1'] - lr_test_results['F1'])*100:+.2f} pp")

    print(f"\nAll results saved to: {run_dir}")
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
