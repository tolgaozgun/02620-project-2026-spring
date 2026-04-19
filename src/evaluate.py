import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)

# ── helpers ───────────────────────────────────────────────────────────────────

def _savefig(path, fallback):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.savefig(fallback, dpi=300, bbox_inches="tight")
    plt.close()


def _clean(gene_name):
    """Strip entrez ID suffix: 'TP53|7157' → 'TP53'."""
    return gene_name.split("|")[0] if "|" in gene_name else gene_name


# ── clustering ────────────────────────────────────────────────────────────────

def evaluate_clustering(X, true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    sil = silhouette_score(X, pred_labels)

    print("Clustering Results:")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"  Silhouette Score: {sil:.4f}")

    return {"ARI": ari, "NMI": nmi, "Silhouette": sil}


# ── classification ────────────────────────────────────────────────────────────

def evaluate_classification(y_true, y_pred, classes, title="Model", output_path=None):
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{title} Classification Results:")
    print(f"  Macro F1:        {f1:.4f}")
    print(f"  Macro Precision: {precision:.4f}")
    print(f"  Macro Recall:    {recall:.4f}")
    print(f"\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes,
                cmap="Blues", linewidths=0.5, annot_kws={"size": 11})
    plt.title(f"Confusion Matrix — {title}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    _savefig(output_path, f"results/confusion_matrix_{title.lower().replace(' ', '_')}.png")

    return {"F1": f1, "Precision": precision, "Recall": recall}


# ── PCA visualisation ─────────────────────────────────────────────────────────

CANCER_COLORS = {
    "BRCA": "#E64B35", "COAD": "#4DBBD5", "GBM": "#00A087",
    "KIRC": "#3C5488", "LUAD": "#F39B7F", "PRAD": "#8491B4",
}

def plot_pca(X_pca, labels, title="PCA Projection", legend_title="Class", output_path=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    unique = list(dict.fromkeys(labels))
    palette = [CANCER_COLORS.get(l, "#999999") for l in unique]
    for label, color in zip(unique, palette):
        mask = np.array(labels) == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=label, color=color, alpha=0.55, s=18, linewidths=0)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9)
    plt.tight_layout()
    _savefig(output_path, "results/pca_projection.png")


def plot_explained_variance(explained_variance_ratio, total_ratio, output_path=None):
    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].bar(range(1, n + 1), explained_variance_ratio * 100,
                color="steelblue", alpha=0.85)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Per-Component Explained Variance")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(range(1, n + 1), cumulative * 100,
                 marker="o", markersize=3, color="darkorange", linewidth=1.8)
    axes[1].axhline(total_ratio * 100, color="red", linestyle="--", alpha=0.6,
                    label=f"Top {n}: {total_ratio*100:.1f}%")
    axes[1].set_xlabel("Number of Principal Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle(f"PCA — top {n} components explain {total_ratio*100:.1f}% of total variance",
                 fontsize=12)
    plt.tight_layout()
    _savefig(output_path, "results/explained_variance.png")


# ── feature importance ────────────────────────────────────────────────────────

def plot_lr_gene_importance(weights_per_class, class_names, gene_names, n_top=15,
                            output_path=None):
    """
    2×3 grid of horizontal bar charts — top genes per cancer type from LR
    weights projected back to gene space.

    weights_per_class : dict {class_name: ndarray of shape (n_genes,)}
    gene_names        : list of raw gene names ('SYMBOL|EntrezID')
    """
    display_names = [_clean(g) for g in gene_names]
    n_classes = len(class_names)
    n_rows, n_cols = 2, 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    axes = axes.flatten()

    for idx, cancer in enumerate(class_names):
        w = weights_per_class[cancer]
        top_idx = np.argsort(np.abs(w))[::-1][:n_top]
        top_names = [display_names[i] for i in top_idx]
        top_vals  = w[top_idx]

        colors = ["#E64B35" if v > 0 else "#3C5488" for v in top_vals]
        y_pos = np.arange(n_top)

        ax = axes[idx]
        ax.barh(y_pos, top_vals[::-1], color=colors[::-1], alpha=0.85, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(cancer, fontweight="bold",
                     color=CANCER_COLORS.get(cancer, "#333333"), fontsize=12)
        ax.set_xlabel("Gene-space weight", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    for idx in range(n_classes, len(axes)):
        axes[idx].set_visible(False)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#E64B35", label="Positively associated"),
                    Patch(facecolor="#3C5488", label="Negatively associated")]
    fig.legend(handles=legend_elems, loc="lower right", fontsize=10)
    fig.suptitle(f"Logistic Regression — Top {n_top} Discriminative Genes per Cancer Type\n"
                 "(weights projected from PCA space back to gene space)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _savefig(output_path, "results/lr_gene_importance.png")

    # Return top gene names per class for reporting
    summary = {}
    for cancer in class_names:
        w = weights_per_class[cancer]
        top_idx = np.argsort(np.abs(w))[::-1][:n_top]
        summary[cancer] = [(display_names[i], float(w[i])) for i in top_idx]
    return summary


def plot_pc_loadings(components, gene_names, n_top=15, output_path=None):
    """
    Horizontal bar charts of top gene loadings for PC1 and PC2.

    components : ndarray (n_components, n_genes)
    gene_names : list of raw gene names
    """
    display_names = [_clean(g) for g in gene_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for pc_idx, ax in enumerate(axes):
        loadings = components[pc_idx]
        top_idx = np.argsort(np.abs(loadings))[::-1][:n_top]
        top_names = [display_names[i] for i in top_idx]
        top_vals  = loadings[top_idx]

        colors = ["#E64B35" if v > 0 else "#3C5488" for v in top_vals]
        y_pos = np.arange(n_top)

        ax.barh(y_pos, top_vals[::-1], color=colors[::-1], alpha=0.85, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"PC{pc_idx + 1} — top {n_top} gene loadings", fontsize=12)
        ax.set_xlabel("Loading coefficient", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("PCA Loadings: genes most strongly captured by PC1 and PC2", fontsize=13)
    plt.tight_layout()
    _savefig(output_path, "results/pc_loadings.png")


def plot_permutation_importance(importance_scores, output_path=None):
    """
    Bar chart of MLP macro-F1 drop when each PC dimension is permuted.

    importance_scores : ndarray (n_components,) — drop in F1 per PC
    """
    n = len(importance_scores)
    order = np.argsort(importance_scores)[::-1]
    sorted_scores = importance_scores[order]
    sorted_labels = [f"PC{i+1}" for i in order]

    # Colour: top 10 highlighted
    colors = ["#E64B35" if i < 10 else "#AAAAAA" for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(n), sorted_scores, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
    ax.set_xlabel("Principal Component (sorted by importance)")
    ax.set_ylabel("Macro-F1 drop when permuted")
    ax.set_title("MLP Permutation Importance — Per-PC Contribution to Classification")
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#E64B35", label="Top 10 most important"),
                        Patch(facecolor="#AAAAAA", label="Remaining PCs")],
               loc="upper right", fontsize=9)
    plt.tight_layout()
    _savefig(output_path, "results/mlp_permutation_importance.png")
