"""
compare_models.py
-----------------
Side-by-side visual comparison of the DEFAULT vs TUNED XGBoost models.

Generates:
  1. Metric comparison bar chart (accuracy, F1, precision, recall)
  2. Confusion matrices side-by-side
  3. Overlaid ROC curves
  4. Overlaid Precision-Recall curves
  5. Probability distribution comparison
  6. Feature importance comparison (top-20)

All saved to reports/figures/compare_*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

# Plot settings
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualization"))
import plot_settings  # noqa

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DEFAULT = PROJECT_ROOT / "models" / "xgb_bbbp_default.json"
MODEL_TUNED = PROJECT_ROOT / "models" / "xgb_bbbp_tuned.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CLASS_NAMES = ["Non-penetrating (0)", "Penetrating (1)"]
CLR_DEFAULT = "#3498db"   # blue
CLR_TUNED   = "#e74c3c"   # red


def save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] ✓ {path.name}")


def load_models():
    default = XGBClassifier(); default.load_model(str(MODEL_DEFAULT))
    tuned   = XGBClassifier(); tuned.load_model(str(MODEL_TUNED))
    return {"Default": default, "Tuned": tuned}


# ── 1. Metric comparison bar chart ────────────────────────────────────
def plot_metric_comparison(models, X_train, y_train, X_test, y_test):
    metric_fns = {
        "Accuracy": accuracy_score,
        "F1 Score": f1_score,
        "Precision": precision_score,
        "Recall": recall_score,
    }
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for ax, (split_name, X, y) in zip(axes, [("Train", X_train, y_train), ("Test", X_test, y_test)]):
        metric_names = list(metric_fns.keys())
        default_vals = [metric_fns[m](y, models["Default"].predict(X)) for m in metric_names]
        tuned_vals   = [metric_fns[m](y, models["Tuned"].predict(X))   for m in metric_names]

        x = np.arange(len(metric_names))
        w = 0.32
        b1 = ax.bar(x - w/2, default_vals, w, label="Default", color=CLR_DEFAULT, edgecolor="white", zorder=3)
        b2 = ax.bar(x + w/2, tuned_vals,   w, label="Tuned",   color=CLR_TUNED,   edgecolor="white", zorder=3)

        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=12)
        ax.set_title(f"{split_name} Set", fontsize=14)
        ax.set_ylim([0.55, 1.10])
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Default vs Tuned — Metric Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "compare_metrics.png")


# ── 2. Confusion matrices ─────────────────────────────────────────────
def plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test), normalize="true")
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap="Blues" if name == "Default" else "Reds",
                  values_format=".2%", colorbar=False)
        ax.set_title(f"Confusion Matrix — {name}", fontsize=13)

    fig.suptitle("Test Set Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "compare_confusion_matrices.png")


# ── 3. Overlaid ROC curves ────────────────────────────────────────────
def plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 7))

    for (name, model), color, ls in zip(models.items(),
                                         [CLR_DEFAULT, CLR_TUNED],
                                         ["-", "--"]):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5, ls=ls,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — Default vs Tuned", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    save(fig, "compare_roc.png")


# ── 4. Overlaid Precision-Recall curves ───────────────────────────────
def plot_pr_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 7))

    for (name, model), color, ls in zip(models.items(),
                                         [CLR_DEFAULT, CLR_TUNED],
                                         ["-", "--"]):
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, color=color, lw=2.5, ls=ls,
                label=f"{name} (AP = {ap:.3f})")

    prevalence = y_test.mean()
    ax.axhline(prevalence, color="gray", ls=":", lw=1,
               label=f"Baseline (prevalence = {prevalence:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curve — Default vs Tuned", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=12)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    save(fig, "compare_precision_recall.png")


# ── 5. Probability distribution comparison ────────────────────────────
def plot_probability_comparison(models, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, (name, model), color in zip(axes, models.items(), [CLR_DEFAULT, CLR_TUNED]):
        y_prob = model.predict_proba(X_test)[:, 1]
        ax.hist(y_prob[y_test == 0], bins=30, alpha=0.55, color="#e74c3c",
                label="Non-penetrating (0)", edgecolor="white", zorder=3)
        ax.hist(y_prob[y_test == 1], bins=30, alpha=0.55, color="#2ecc71",
                label="Penetrating (1)", edgecolor="white", zorder=3)
        ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Threshold = 0.5")
        ax.set_xlabel("Predicted Probability (class 1)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{name} Model", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Prediction Probability Distribution — Default vs Tuned",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "compare_probability.png")


# ── 6. Feature importance comparison (top-20) ─────────────────────────
def plot_feature_importance_comparison(models, feature_names):
    TOP = 20
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=False)

    for ax, (name, model), color in zip(axes, models.items(), [CLR_DEFAULT, CLR_TUNED]):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:TOP]
        top_f = [feature_names[i] for i in idx]
        top_v = imp[idx]
        ax.barh(range(len(top_f)), top_v[::-1], color=color, edgecolor="white")
        ax.set_yticks(range(len(top_f)))
        ax.set_yticklabels(top_f[::-1], fontsize=10)
        ax.set_xlabel("Importance (gain)", fontsize=11)
        ax.set_title(f"Top {TOP} Features — {name}", fontsize=13)

    fig.suptitle("Feature Importance — Default vs Tuned",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "compare_feature_importance.png")


# ── main ───────────────────────────────────────────────────────────────
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models = load_models()

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    feature_names = X_train.columns.tolist()

    print("[compare] Generating default-vs-tuned comparison plots …\n")

    plot_metric_comparison(models, X_train, y_train, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_pr_curves(models, X_test, y_test)
    plot_probability_comparison(models, X_test, y_test)
    plot_feature_importance_comparison(models, feature_names)

    print(f"\n[compare] All 6 comparison figures saved to {FIGURES_DIR}/")
    print("[compare] Done ✓")


if __name__ == "__main__":
    main()
