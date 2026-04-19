"""
Generate all figures needed for the final report.
Produces a richer set of plots than the main pipeline.
Run from the project root: conda run -n cancer-research python src/generate_report_figures.py
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, classification_report)
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca import PCA
from kmeans import KMeans
from logistic_regression import LogisticRegressionOvR
from mlp import MLP, train_mlp

# ── output directory ──────────────────────────────────────────────────────────
FIG_DIR = "results/report_figures"
os.makedirs(FIG_DIR, exist_ok=True)

CANCER_COLORS = {
    "BRCA": "#E64B35", "COAD": "#4DBBD5", "GBM": "#00A087",
    "KIRC": "#3C5488", "LUAD": "#F39B7F", "PRAD": "#8491B4",
}
PALETTE = list(CANCER_COLORS.values())

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 150,
})

# ── load & split (same seed as pipeline) ──────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/processed_pancan.csv", index_col=0)
X = df.drop(columns=["cancer_type"]).values
y_raw = df["cancer_type"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
n_classes = len(class_names)

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

scaler = StandardScaler()
X_dev_s = scaler.fit_transform(X_dev)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=50)
X_dev_pca = pca.fit_transform(X_dev_s)
X_test_pca = pca.transform(X_test_s)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Class distribution bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 1: class distribution")
counts = pd.Series(y_raw).value_counts().reindex(class_names)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(class_names, counts.values,
              color=[CANCER_COLORS[c] for c in class_names], edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            str(val), ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Cancer Type")
ax.set_ylabel("Number of Samples")
ax.set_title("Sample Count per Cancer Type (TCGA Pan-Cancer)")
ax.set_ylim(0, counts.max() * 1.15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_class_distribution.pdf"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — PCA explained variance (scree + cumulative)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 2: explained variance")
evr = pca.explained_variance_ratio_
cumev = np.cumsum(evr)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].bar(range(1, 51), evr * 100, color="steelblue", alpha=0.85)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance (%)")
axes[0].set_title("Per-Component Explained Variance")
axes[0].grid(axis="y", alpha=0.3)

axes[1].plot(range(1, 51), cumev * 100, marker="o", markersize=3,
             color="darkorange", linewidth=1.8)
axes[1].axhline(pca.total_explained_variance_ratio_ * 100, color="red",
                linestyle="--", alpha=0.6,
                label=f"Top 50 total: {pca.total_explained_variance_ratio_*100:.1f}%")
axes[1].set_xlabel("Number of Principal Components")
axes[1].set_ylabel("Cumulative Explained Variance (%)")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("PCA on TCGA Pan-Cancer Gene Expression (development set, n=3,065)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_pca_variance.pdf"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — PCA scatter: true labels (PC1 vs PC2)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3: PCA true labels")
y_dev_names = le.inverse_transform(y_dev)
fig, ax = plt.subplots(figsize=(8, 6))
for cancer in class_names:
    mask = y_dev_names == cancer
    ax.scatter(X_dev_pca[mask, 0], X_dev_pca[mask, 1],
               label=cancer, color=CANCER_COLORS[cancer],
               alpha=0.55, s=18, linewidths=0)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA Projection — True Cancer Type Labels")
ax.legend(title="Cancer Type", bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_pca_true_labels.pdf"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — K-Means cluster assignments vs true labels (side by side)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4: KMeans clusters vs true labels")
kmeans = KMeans(n_clusters=n_classes, random_state=42)
kmeans.fit(X_dev_pca)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Left: true labels
for cancer in class_names:
    mask = y_dev_names == cancer
    axes[0].scatter(X_dev_pca[mask, 0], X_dev_pca[mask, 1],
                    label=cancer, color=CANCER_COLORS[cancer],
                    alpha=0.5, s=15, linewidths=0)
axes[0].set_title("True Cancer Type Labels")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].legend(title="Cancer Type", fontsize=9, markerscale=1.5)

# Right: K-Means clusters
cluster_palette = sns.color_palette("Set2", n_classes)
for k in range(n_classes):
    mask = kmeans.labels_ == k
    axes[1].scatter(X_dev_pca[mask, 0], X_dev_pca[mask, 1],
                    label=f"Cluster {k}", color=cluster_palette[k],
                    alpha=0.5, s=15, linewidths=0)
# Plot centroids
for k, center in enumerate(kmeans.cluster_centers_):
    axes[1].scatter(center[0], center[1], marker="X", s=120,
                    color=cluster_palette[k], edgecolors="black", linewidths=0.8, zorder=5)
axes[1].set_title(f"K-Means Clusters (k={n_classes})  |  ARI=0.794, NMI=0.849")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].legend(title="Cluster", fontsize=9, markerscale=1.5)

plt.suptitle("Unsupervised Discovery: PCA + K-Means vs Ground Truth", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_kmeans_vs_truth.pdf"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Cluster purity heatmap (cluster × cancer type)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 5: cluster purity heatmap")
purity_matrix = np.zeros((n_classes, n_classes), dtype=int)
for k in range(n_classes):
    mask = kmeans.labels_ == k
    for j, cancer in enumerate(class_names):
        purity_matrix[k, j] = np.sum((y_dev_names == cancer) & mask)

# Normalise rows to get proportions
purity_norm = purity_matrix / purity_matrix.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 5.5))
sns.heatmap(purity_norm, annot=purity_matrix, fmt="d",
            xticklabels=class_names,
            yticklabels=[f"Cluster {k}" for k in range(n_classes)],
            cmap="YlOrRd", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Proportion of cluster"})
ax.set_xlabel("True Cancer Type")
ax.set_ylabel("K-Means Cluster")
ax.set_title("Cluster Composition — K-Means vs True Labels\n(cell values = sample counts)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_cluster_purity.pdf"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — CV F1 per fold: LR vs MLP
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 6: CV fold comparison")
lr_folds  = [0.9988, 0.9922, 0.9969, 0.9962, 1.0000]
mlp_folds = [0.9954, 0.9975, 0.9963, 0.9918, 1.0000]
folds = [f"Fold {i+1}" for i in range(5)]

x = np.arange(5)
w = 0.35
fig, ax = plt.subplots(figsize=(8, 4.5))
bars1 = ax.bar(x - w/2, lr_folds,  w, label="Logistic Regression", color="#3C5488", alpha=0.85)
bars2 = ax.bar(x + w/2, mlp_folds, w, label="MLP",                  color="#E64B35", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(folds)
ax.set_ylim(0.985, 1.003)
ax.set_ylabel("Macro F1 Score")
ax.set_title("5-Fold CV Macro F1 — Logistic Regression vs MLP")
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.axhline(np.mean(lr_folds),  color="#3C5488", linestyle="--", linewidth=1, alpha=0.7)
ax.axhline(np.mean(mlp_folds), color="#E64B35", linestyle="--", linewidth=1, alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_cv_folds.pdf"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — LR confusion matrix (test set)  — recompute cleanly
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 7 & 8: confusion matrices (retrain LR & MLP)")
# Retrain LR
lr_model = LogisticRegressionOvR(lr=0.1, l2_lambda=0.001, n_iters=1000)
lr_model.fit(X_dev_pca, y_dev)
y_pred_lr = lr_model.predict(X_test_pca)

cm_lr = confusion_matrix(y_test, y_pred_lr)
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(cm_lr, annot=True, fmt="d",
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", linewidths=0.5, ax=ax,
            annot_kws={"size": 11})
ax.set_title("Logistic Regression — Test Set Confusion Matrix")
ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig7_lr_confusion.pdf"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — MLP confusion matrix + learning curve
# ─────────────────────────────────────────────────────────────────────────────
mlp_model, history = train_mlp(X_dev_pca, y_dev, X_test_pca, y_test,
                                n_classes, epochs=50)
mlp_model.eval()
with torch.no_grad():
    y_pred_mlp = mlp_model(torch.FloatTensor(X_test_pca)).argmax(dim=1).numpy()

cm_mlp = confusion_matrix(y_test, y_pred_mlp)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sns.heatmap(cm_mlp, annot=True, fmt="d",
            xticklabels=class_names, yticklabels=class_names,
            cmap="Oranges", linewidths=0.5, ax=axes[0],
            annot_kws={"size": 11})
axes[0].set_title("MLP — Test Set Confusion Matrix")
axes[0].set_ylabel("True Label"); axes[0].set_xlabel("Predicted Label")

axes[1].plot(history["train_loss"], label="Training Loss", color="#3C5488", linewidth=1.5)
ax2 = axes[1].twinx()
ax2.plot(history["val_acc"], label="Val Accuracy", color="#E64B35",
         linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss", color="#3C5488")
ax2.set_ylabel("Validation Accuracy", color="#E64B35")
axes[1].set_title("MLP Training Dynamics")
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc="center right")
axes[1].grid(alpha=0.3)

plt.suptitle("MLP — Test Evaluation and Training Curves", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig8_mlp_eval.pdf"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Per-class F1: LR vs MLP side by side
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 9: per-class F1 comparison")
from sklearn.metrics import f1_score as f1_per

def per_class_f1(y_true, y_pred, n):
    return [f1_score(y_true == k, y_pred == k) for k in range(n)]

lr_pc  = per_class_f1(y_test, y_pred_lr,  n_classes)
mlp_pc = per_class_f1(y_test, y_pred_mlp, n_classes)

x = np.arange(n_classes)
w = 0.35
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x - w/2, lr_pc,  w, label="Logistic Regression", color="#3C5488", alpha=0.85)
ax.bar(x + w/2, mlp_pc, w, label="MLP",                  color="#E64B35", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(class_names)
ax.set_ylim(0.88, 1.02)
ax.set_ylabel("F1 Score (binary OvR)")
ax.set_title("Per-Class F1 Score on Test Set — LR vs MLP")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig9_perclass_f1.pdf"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Summary comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 10: summary bar chart")
metrics = ["Macro F1", "Macro Precision", "Macro Recall"]
lr_vals  = [0.9927, 0.9877, 0.9982]
mlp_vals = [0.9922, 0.9877, 0.9973]

x = np.arange(len(metrics))
w = 0.3
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w/2, lr_vals,  w, label="Logistic Regression", color="#3C5488", alpha=0.85)
b2 = ax.bar(x + w/2, mlp_vals, w, label="MLP",                  color="#E64B35", alpha=0.85)
for b, v in zip(list(b1) + list(b2), lr_vals + mlp_vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0005,
            f"{v:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0.975, 1.005)
ax.set_ylabel("Score")
ax.set_title("Test Set Performance — Logistic Regression vs MLP")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig10_summary_comparison.pdf"), bbox_inches="tight")
plt.close()

print(f"\nAll figures saved to {FIG_DIR}/")
print("Files:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
