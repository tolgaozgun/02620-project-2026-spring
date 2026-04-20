"""
Multi-Seed Robustness Experiment
=================================
Runs the full supervised pipeline (LR, RBF-SVM, MLP) with 10 different
random seeds to estimate variance due to train/test split and model
initialization, independent of cross-validation fold assignment.

Hyperparameters are fixed at the values selected during the main CV run:
  LR  : L2 lambda = 0.001
  SVM : C=1.0, kernel=rbf, gamma=scale
  MLP : [512, 256], Adam lr=0.001, 50 epochs

Results are saved to results/multi_seed/ as JSON + figures.
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca import PCA
from logistic_regression import LogisticRegressionOvR
from mlp import MLP, train_mlp

# ── Config ────────────────────────────────────────────────────────────────────

SEEDS = [0, 7, 13, 21, 37, 42, 55, 77, 99, 123]
OUT_DIR = os.path.join("results", "multi_seed")
os.makedirs(OUT_DIR, exist_ok=True)

CANCER_COLORS = {
    "BRCA": "#E64B35", "COAD": "#4DBBD5", "GBM": "#00A087",
    "KIRC": "#3C5488", "LUAD": "#F39B7F", "PRAD": "#8491B4",
}

# ── Load data once ────────────────────────────────────────────────────────────

def load_data():
    data_path = "data/processed_pancan.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Run src/data_loader.py first.")
    df = pd.read_csv(data_path, index_col=0)
    gene_names = [c for c in df.columns if c != "cancer_type"]
    X = df[gene_names].values
    y_raw = df["cancer_type"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le.classes_


# ── Single seed run ───────────────────────────────────────────────────────────

def run_seed(X, y, class_names, seed):
    n_classes = len(class_names)
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")

    # 1. Train/test split
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y)

    # 2. Preprocessing — fit on dev only
    scaler = StandardScaler()
    X_dev_s  = scaler.fit_transform(X_dev)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=50)
    X_dev_pca  = pca.fit_transform(X_dev_s)
    X_test_pca = pca.transform(X_test_s)

    result = {"seed": seed, "n_dev": len(X_dev), "n_test": len(X_test)}

    # 3. Logistic Regression (L2=0.001, fixed)
    np.random.seed(seed)
    lr = LogisticRegressionOvR(lr=0.1, l2_lambda=0.001, n_iters=1000)
    lr.fit(X_dev_pca, y_dev)
    y_pred_lr = lr.predict(X_test_pca)
    result["lr"] = {
        "f1":        float(f1_score(y_test, y_pred_lr, average="macro")),
        "precision": float(precision_score(y_test, y_pred_lr, average="macro")),
        "recall":    float(recall_score(y_test, y_pred_lr, average="macro")),
        "per_class_f1": [float(f1_score(y_test == k, y_pred_lr == k))
                         for k in range(n_classes)],
    }
    print(f"  LR  F1={result['lr']['f1']:.4f}")

    # 4. RBF-SVM (C=1.0, gamma=scale, fixed)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    svm.fit(X_dev_pca, y_dev)
    y_pred_svm = svm.predict(X_test_pca)
    result["svm"] = {
        "f1":        float(f1_score(y_test, y_pred_svm, average="macro")),
        "precision": float(precision_score(y_test, y_pred_svm, average="macro")),
        "recall":    float(recall_score(y_test, y_pred_svm, average="macro")),
        "per_class_f1": [float(f1_score(y_test == k, y_pred_svm == k))
                         for k in range(n_classes)],
    }
    print(f"  SVM F1={result['svm']['f1']:.4f}")

    # 5. MLP (same architecture, seed controls init + data order)
    X_dev_tr, X_dev_mon, y_dev_tr, y_dev_mon = train_test_split(
        X_dev_pca, y_dev, test_size=0.10, random_state=seed, stratify=y_dev)
    torch.manual_seed(seed)
    mlp_model, _ = train_mlp(X_dev_tr, y_dev_tr, X_dev_mon, y_dev_mon,
                              n_classes, epochs=50)
    mlp_model.eval()
    with torch.no_grad():
        y_pred_mlp = mlp_model(torch.FloatTensor(X_test_pca)).argmax(dim=1).numpy()
    result["mlp"] = {
        "f1":        float(f1_score(y_test, y_pred_mlp, average="macro")),
        "precision": float(precision_score(y_test, y_pred_mlp, average="macro")),
        "recall":    float(recall_score(y_test, y_pred_mlp, average="macro")),
        "per_class_f1": [float(f1_score(y_test == k, y_pred_mlp == k))
                         for k in range(n_classes)],
    }
    print(f"  MLP F1={result['mlp']['f1']:.4f}")

    return result


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_f1_distribution(all_results, class_names, out_dir):
    """Strip + box plots of F1 across seeds, one panel per model."""
    models  = ["lr", "svm", "mlp"]
    labels  = ["Logistic Regression", "RBF-SVM", "MLP [512,256]"]
    colors  = ["#3C5488", "#F39B7F", "#E64B35"]

    # ── 1. Overall macro-F1 distribution ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [[r[m]["f1"] for r in all_results] for m in models]
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for i, (d, color) in enumerate(zip(data, colors), 1):
        ax.scatter([i] * len(d), d, color=color, zorder=3, s=40, edgecolors="black", lw=0.6)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title(f"Macro-F1 Distribution Across {len(SEEDS)} Random Seeds")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "ms_fig1_f1_distribution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

    # ── 2. F1 per seed (line plot) ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    seed_labels = [str(s) for s in SEEDS]
    for m, label, color in zip(models, labels, colors):
        f1s = [r[m]["f1"] for r in all_results]
        ax.plot(seed_labels, f1s, marker="o", label=label, color=color,
                linewidth=1.5, markersize=6)
    ax.set_xlabel("Random Seed"); ax.set_ylabel("Test Macro-F1")
    ax.set_title("Test Macro-F1 per Seed — LR vs RBF-SVM vs MLP")
    ax.legend(); ax.grid(alpha=0.3)
    ymin = min(r[m]["f1"] for r in all_results for m in models)
    ax.set_ylim(max(0.97, ymin - 0.005), 1.003)
    plt.tight_layout()
    path = os.path.join(out_dir, "ms_fig2_f1_per_seed.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

    # ── 3. Per-class F1 heatmap (MLP, all seeds) ──────────────────────────────
    mlp_pc = np.array([r["mlp"]["per_class_f1"] for r in all_results])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(mlp_pc, annot=True, fmt=".3f",
                xticklabels=class_names,
                yticklabels=[f"seed={s}" for s in SEEDS],
                cmap="YlOrRd_r", vmin=0.88, vmax=1.0,
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "Per-class F1"})
    ax.set_xlabel("Cancer Type"); ax.set_ylabel("Random Seed")
    ax.set_title("MLP Per-Class F1 Across 10 Seeds")
    plt.tight_layout()
    path = os.path.join(out_dir, "ms_fig3_mlp_perclass_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

    # ── 4. Summary bar chart: mean ± std ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3); w = 0.5
    means = [np.mean([r[m]["f1"] for r in all_results]) for m in models]
    stds  = [np.std( [r[m]["f1"] for r in all_results]) for m in models]
    bars = ax.bar(x, means, w, yerr=stds, capsize=6,
                  color=colors, alpha=0.82, edgecolor="white",
                  error_kw=dict(elinewidth=1.5, ecolor="black"))
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + s + 0.0008,
                f"{m:.4f}\n±{s:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title(f"Mean ± Std Test Macro-F1 Across {len(SEEDS)} Seeds")
    ax.set_ylim(max(0.97, min(means) - max(stds) - 0.01), 1.008)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "ms_fig4_mean_std_summary.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results, class_names):
    models = ["lr", "svm", "mlp"]
    labels = ["LR", "SVM", "MLP"]

    print("\n" + "="*60)
    print("MULTI-SEED SUMMARY (10 seeds)")
    print("="*60)
    print(f"{'Model':<6} {'Mean F1':>9} {'Std F1':>9} {'Min F1':>9} {'Max F1':>9}")
    print("-"*45)
    for m, l in zip(models, labels):
        f1s = [r[m]["f1"] for r in all_results]
        print(f"{l:<6} {np.mean(f1s):>9.4f} {np.std(f1s):>9.4f} "
              f"{np.min(f1s):>9.4f} {np.max(f1s):>9.4f}")

    print("\nPer-seed F1:")
    print(f"{'Seed':<6}", end="")
    for l in labels:
        print(f"  {l:>8}", end="")
    print()
    for r in all_results:
        print(f"{r['seed']:<6}", end="")
        for m in models:
            print(f"  {r[m]['f1']:>8.4f}", end="")
        print()

    print("\nPer-class F1 stats (MLP):")
    mlp_pc = np.array([r["mlp"]["per_class_f1"] for r in all_results])
    print(f"{'Class':<8} {'Mean':>8} {'Std':>8} {'Min':>8}")
    print("-"*35)
    for k, cls in enumerate(class_names):
        print(f"{cls:<8} {mlp_pc[:,k].mean():>8.4f} "
              f"{mlp_pc[:,k].std():>8.4f} {mlp_pc[:,k].min():>8.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Multi-Seed Robustness Experiment")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {OUT_DIR}")

    X, y, class_names = load_data()
    print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features | "
          f"{len(class_names)} classes")

    all_results = []
    for seed in SEEDS:
        result = run_seed(X, y, class_names, seed)
        all_results.append(result)

    # Save raw results
    json_path = os.path.join(OUT_DIR, "multi_seed_results.json")
    with open(json_path, "w") as f:
        json.dump({"seeds": SEEDS, "class_names": list(class_names),
                   "results": all_results}, f, indent=2)
    print(f"\nSaved: {json_path}")

    print_summary(all_results, class_names)
    fig_f1_distribution(all_results, class_names, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
