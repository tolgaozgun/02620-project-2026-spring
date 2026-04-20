"""
Multi-Seed Robustness Experiment
=================================
Calls run_pipeline() from main.py with 10 different seeds to estimate
variance due to train/test split, CV fold assignment, and model initialisation.

Everything is identical to the main pipeline — same preprocessing, same 5-fold
CV hyperparameter search, same model architectures. Only the seed changes.

Results are saved to results/multi_seed/ as JSON + figures.
Each individual run also saves its full output to results/runs/run_*_seedX/.
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import run_pipeline

SEEDS = [0, 7, 13, 21, 37, 42, 55, 77, 99, 123]
OUT_DIR = os.path.join("results", "multi_seed")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Figures ───────────────────────────────────────────────────────────────────

def make_figures(all_results, class_names):
    models = ["lr", "svm", "mlp"]
    labels = ["Logistic Regression", "RBF-SVM", "MLP [512,256]"]
    colors = ["#3C5488", "#F39B7F", "#E64B35"]

    # 1. Box + strip — overall macro-F1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [[r[m]["test_f1"] for r in all_results] for m in models]
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for i, (d, color) in enumerate(zip(data, colors), 1):
        ax.scatter([i] * len(d), d, color=color, zorder=3,
                   s=45, edgecolors="black", linewidths=0.6)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title(f"Macro-F1 Distribution Across {len(SEEDS)} Seeds\n(full CV pipeline per seed)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(os.path.join(OUT_DIR, "ms_fig1_f1_distribution.png"))

    # 2. Line plot — F1 per seed
    fig, ax = plt.subplots(figsize=(10, 4.5))
    seed_labels = [str(s) for s in SEEDS]
    for m, label, color in zip(models, labels, colors):
        f1s = [r[m]["test_f1"] for r in all_results]
        ax.plot(seed_labels, f1s, marker="o", label=label,
                color=color, linewidth=1.5, markersize=6)
    ax.set_xlabel("Random Seed"); ax.set_ylabel("Test Macro-F1")
    ax.set_title("Test Macro-F1 per Seed — LR vs RBF-SVM vs MLP")
    ax.legend(); ax.grid(alpha=0.3)
    ymin = min(r[m]["test_f1"] for r in all_results for m in models)
    ax.set_ylim(max(0.97, ymin - 0.005), 1.003)
    plt.tight_layout()
    _save(os.path.join(OUT_DIR, "ms_fig2_f1_per_seed.png"))

    # 3. Per-class F1 heatmap — MLP
    mlp_pc = np.array([r["mlp"]["per_class_f1"] for r in all_results])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(mlp_pc, annot=True, fmt=".3f",
                xticklabels=class_names,
                yticklabels=[f"seed={s}" for s in SEEDS],
                cmap="YlOrRd_r", vmin=0.88, vmax=1.0,
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "Per-class F1"})
    ax.set_xlabel("Cancer Type"); ax.set_ylabel("Random Seed")
    ax.set_title("MLP Per-Class F1 Across 10 Seeds (full CV pipeline)")
    plt.tight_layout()
    _save(os.path.join(OUT_DIR, "ms_fig3_mlp_perclass_heatmap.png"))

    # 4. Mean ± std bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3); w = 0.5
    means = [np.mean([r[m]["test_f1"] for r in all_results]) for m in models]
    stds  = [np.std( [r[m]["test_f1"] for r in all_results]) for m in models]
    bars = ax.bar(x, means, w, yerr=stds, capsize=6,
                  color=colors, alpha=0.82, edgecolor="white",
                  error_kw=dict(elinewidth=1.5, ecolor="black"))
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + s + 0.0008,
                f"{m:.4f}\n±{s:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title(f"Mean ± Std Test Macro-F1 Across {len(SEEDS)} Seeds")
    ax.set_ylim(max(0.97, min(means) - max(stds) - 0.01), 1.010)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(os.path.join(OUT_DIR, "ms_fig4_mean_std_summary.png"))

    # 5. CV F1 vs test F1 scatter (all seeds, all models)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, m, label, color in zip(axes, models, labels, colors):
        cv_f1s   = [r[m]["cv_f1_mean"] for r in all_results]
        test_f1s = [r[m]["test_f1"]    for r in all_results]
        ax.scatter(cv_f1s, test_f1s, color=color, s=60,
                   edgecolors="black", linewidths=0.6, zorder=3)
        for s, cx, ty in zip(SEEDS, cv_f1s, test_f1s):
            ax.annotate(str(s), (cx, ty), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        lo = min(cv_f1s + test_f1s) - 0.003
        hi = max(cv_f1s + test_f1s) + 0.003
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("CV Macro-F1 (mean)")
        ax.set_ylabel("Test Macro-F1")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    plt.suptitle("CV F1 vs Test F1 per Seed — diagonal = perfect agreement",
                 fontsize=12)
    plt.tight_layout()
    _save(os.path.join(OUT_DIR, "ms_fig5_cv_vs_test.png"))


def _save(path):
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_and_save_summary(all_results, class_names):
    models = ["lr", "svm", "mlp"]
    labels = ["LR", "SVM", "MLP"]

    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY (10 seeds, full CV pipeline)")
    print("=" * 70)
    print(f"{'Model':<6} {'Mean F1':>9} {'Std F1':>9} {'Min F1':>9} {'Max F1':>9}")
    print("-" * 46)
    summary = {}
    for m, l in zip(models, labels):
        f1s = [r[m]["test_f1"] for r in all_results]
        cv_means = [r[m]["cv_f1_mean"] for r in all_results]
        print(f"{l:<6} {np.mean(f1s):>9.4f} {np.std(f1s):>9.4f} "
              f"{np.min(f1s):>9.4f} {np.max(f1s):>9.4f}")
        summary[m] = {
            "test_f1_mean": float(np.mean(f1s)),
            "test_f1_std":  float(np.std(f1s)),
            "test_f1_min":  float(np.min(f1s)),
            "test_f1_max":  float(np.max(f1s)),
            "cv_f1_mean":   float(np.mean(cv_means)),
            "cv_f1_std":    float(np.std(cv_means)),
        }

    print("\nPer-seed results:")
    print(f"{'Seed':<6}", end="")
    for l in labels:
        print(f"  {'CV-'+l:>10}  {'Test-'+l:>10}", end="")
    print()
    for r in all_results:
        print(f"{r['seed']:<6}", end="")
        for m in models:
            print(f"  {r[m]['cv_f1_mean']:>10.4f}  {r[m]['test_f1']:>10.4f}", end="")
        print()

    print("\nBest lambda per seed (LR):")
    for r in all_results:
        print(f"  seed={r['seed']}: lambda={r['lr']['best_lambda']}")

    # Save JSON
    out = {
        "seeds": SEEDS,
        "class_names": list(class_names),
        "summary": summary,
        "per_seed": all_results,
    }
    path = os.path.join(OUT_DIR, "multi_seed_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MULTI-SEED ROBUSTNESS EXPERIMENT")
    print(f"Seeds: {SEEDS}")
    print("Running full pipeline (5-fold CV) per seed via run_pipeline()")
    print("=" * 70)

    class_names = None
    all_results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n[{i+1}/{len(SEEDS)}] ── Seed {seed} ──────────────────────────")
        final = run_pipeline(seed=seed, log_to_file=True)

        # Extract what we need for aggregate analysis
        r = {
            "seed": seed,
            "run_dir": final["run_dir"],
            "lr": {
                "cv_f1_mean":  final["lr"]["cv_f1_mean"],
                "cv_f1_std":   final["lr"]["cv_f1_std"],
                "best_lambda": final["lr"]["best_lambda"],
                "test_f1":     final["lr"]["test_f1"],
                "test_precision": final["lr"]["test_precision"],
                "test_recall":    final["lr"]["test_recall"],
                "per_class_f1":   final["lr"].get("per_class_f1", []),
            },
            "svm": {
                "cv_f1_mean":  final["svm"]["cv_f1_mean"],
                "cv_f1_std":   final["svm"]["cv_f1_std"],
                "test_f1":     final["svm"]["test_f1"],
                "test_precision": final["svm"]["test_precision"],
                "test_recall":    final["svm"]["test_recall"],
                "per_class_f1":   final["svm"].get("per_class_f1", []),
            },
            "mlp": {
                "cv_f1_mean":  final["mlp"]["cv_f1_mean"],
                "cv_f1_std":   final["mlp"]["cv_f1_std"],
                "test_f1":     final["mlp"]["test_f1"],
                "test_precision": final["mlp"]["test_precision"],
                "test_recall":    final["mlp"]["test_recall"],
                "per_class_f1":   final["mlp"].get("per_class_f1", []),
            },
        }
        all_results.append(r)

        if class_names is None:
            class_names = final.get("class_names",
                                    ["BRCA", "COAD", "GBM", "KIRC", "LUAD", "PRAD"])

        # Save checkpoint after each seed in case of interruption
        ckpt = os.path.join(OUT_DIR, "multi_seed_checkpoint.json")
        with open(ckpt, "w") as f:
            json.dump({"seeds_done": [r["seed"] for r in all_results],
                       "results": all_results}, f, indent=2)

        print(f"  → LR={r['lr']['test_f1']:.4f}  "
              f"SVM={r['svm']['test_f1']:.4f}  "
              f"MLP={r['mlp']['test_f1']:.4f}")

    print_and_save_summary(all_results, class_names)
    make_figures(all_results, class_names)
    print("\nAll done.")


if __name__ == "__main__":
    main()
