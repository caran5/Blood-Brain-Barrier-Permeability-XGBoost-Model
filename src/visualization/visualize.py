"""
visualize.py
------------
Comprehensive model evaluation visualizations:
  1. Feature Importance (top-25)
  2. Confusion Matrices (train & test)
  3. ROC Curve
  4. Precision-Recall Curve
  5. Learning Curve (accuracy & F1 vs training size)
  6. Metric Comparison Bar Chart (accuracy, F1, precision, recall)
  7. Class Distribution (train vs test)

All figures are saved to reports/figures/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import learning_curve

# Import project plot settings
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_settings  # noqa: E402, F401 — applies rcParams on import

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_bbbp.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TOP_N = 25  # how many features to show in importance plot
RANDOM_STATE = 42
CLASS_NAMES = ["Non-penetrating (0)", "Penetrating (1)"]


# ── helper ─────────────────────────────────────────────────────────────
def save(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] ✓ {path.name}")


# ── individual plot functions ──────────────────────────────────────────

def plot_feature_importance(model, feature_names):
    """Bar chart of top-N feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:TOP_N]
    top_features = [feature_names[i] for i in idx]
    top_importances = importances[idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top_features)), top_importances[::-1],
            color="#3498db", edgecolor="white")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top {TOP_N} Feature Importances — XGBoost BBBP Classifier")
    plt.tight_layout()
    save(fig, "feature_importance.png")


def plot_confusion_matrices(model, X_train, y_train, X_test, y_test):
    """Side-by-side normalised confusion matrices for train & test."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, X, y, title in [
        (axes[0], X_train, y_train, "Train"),
        (axes[1], X_test, y_test, "Test"),
    ]:
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap="Blues", values_format=".2%", colorbar=False)
        ax.set_title(f"Confusion Matrix — {title}")

    plt.tight_layout()
    save(fig, "confusion_matrices.png")


def plot_roc_curve(model, X_test, y_test):
    """ROC curve with AUC score."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#e74c3c")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — XGBoost BBBP Classifier")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    save(fig, "roc_curve.png")


def plot_precision_recall_curve(model, X_test, y_test):
    """Precision-Recall curve with average precision."""
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, color="#2ecc71", lw=2.5,
            label=f"PR Curve (AP = {ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.12, color="#2ecc71")
    # Baseline = prevalence of positive class
    prevalence = y_test.mean()
    ax.axhline(prevalence, color="gray", ls="--", lw=1,
               label=f"Baseline (prevalence = {prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve — XGBoost BBBP Classifier")
    ax.legend(loc="lower left", fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    save(fig, "precision_recall_curve.png")


def plot_learning_curve(model, X_train, y_train):
    """Learning curves: accuracy and F1 vs. training set size."""
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="accuracy",
        shuffle=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    _, train_f1, val_f1 = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="f1",
        shuffle=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, tr, va, metric in [
        (axes[0], train_scores, val_scores, "Accuracy"),
        (axes[1], train_f1, val_f1, "F1 Score"),
    ]:
        tr_mean, tr_std = tr.mean(axis=1), tr.std(axis=1)
        va_mean, va_std = va.mean(axis=1), va.std(axis=1)

        ax.fill_between(train_sizes_abs, tr_mean - tr_std, tr_mean + tr_std,
                         alpha=0.15, color="#3498db")
        ax.fill_between(train_sizes_abs, va_mean - va_std, va_mean + va_std,
                         alpha=0.15, color="#e74c3c")
        ax.plot(train_sizes_abs, tr_mean, "o-", color="#3498db", lw=2, label="Train")
        ax.plot(train_sizes_abs, va_mean, "o-", color="#e74c3c", lw=2, label="Validation (CV)")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(metric)
        ax.set_title(f"Learning Curve — {metric}")
        ax.legend(loc="lower right", fontsize=11)
        ax.set_ylim([0.5, 1.02])

    plt.tight_layout()
    save(fig, "learning_curves.png")


def plot_metric_comparison(model, X_train, y_train, X_test, y_test):
    """Grouped bar chart comparing accuracy, F1, precision, recall on train & test."""
    metrics = {}
    for name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        y_pred = model.predict(X)
        metrics[name] = {
            "Accuracy": accuracy_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
        }

    metric_names = list(metrics["Train"].keys())
    train_vals = [metrics["Train"][m] for m in metric_names]
    test_vals = [metrics["Test"][m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, train_vals, width, label="Train",
                   color="#3498db", edgecolor="white", zorder=3)
    bars2 = ax.bar(x + width / 2, test_vals, width, label="Test",
                   color="#e74c3c", edgecolor="white", zorder=3)

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance — Train vs Test")
    ax.set_ylim([0, 1.12])
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "metric_comparison.png")


def plot_class_distribution(y_train, y_test):
    """Class distribution side-by-side for train and test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y, title in [(axes[0], y_train, "Train"), (axes[1], y_test, "Test")]:
        counts = y.value_counts().sort_index()
        colors = ["#e74c3c", "#2ecc71"]
        bars = ax.bar(CLASS_NAMES, counts.values, color=colors, edgecolor="white", zorder=3)
        for bar, val in zip(bars, counts.values):
            ax.annotate(f"{val}\n({val / len(y):.1%})",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_title(f"Class Distribution — {title} (n={len(y)})")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save(fig, "class_distribution.png")


def plot_probability_histogram(model, X_test, y_test):
    """Histogram of predicted probabilities split by true class."""
    y_prob = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_prob[y_test == 0], bins=30, alpha=0.6, color="#e74c3c",
            label="Non-penetrating (0)", edgecolor="white", zorder=3)
    ax.hist(y_prob[y_test == 1], bins=30, alpha=0.6, color="#2ecc71",
            label="Penetrating (1)", edgecolor="white", zorder=3)
    ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Decision threshold (0.5)")
    ax.set_xlabel("Predicted Probability (class 1)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Probability Distribution by True Class")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "probability_histogram.png")


# ── main ───────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    # Load data
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    feature_names = X_train.columns.tolist()

    print("[visualize] Generating plots …\n")

    # 1. Feature importance
    plot_feature_importance(model, feature_names)

    # 2. Confusion matrices
    plot_confusion_matrices(model, X_train, y_train, X_test, y_test)

    # 3. ROC curve
    plot_roc_curve(model, X_test, y_test)

    # 4. Precision-recall curve
    plot_precision_recall_curve(model, X_test, y_test)

    # 5. Learning curves (takes a moment — retrains at multiple sizes)
    print("[visualize]   ⏳ Computing learning curves (5-fold CV × 10 sizes) …")
    plot_learning_curve(model, X_train, y_train)

    # 6. Metric comparison bar chart
    plot_metric_comparison(model, X_train, y_train, X_test, y_test)

    # 7. Class distribution
    plot_class_distribution(y_train, y_test)

    # 8. Probability histogram
    plot_probability_histogram(model, X_test, y_test)

    print(f"\n[visualize] All 8 figures saved to {FIGURES_DIR}/")
    print("[visualize] Done ✓")


if __name__ == "__main__":
    main()
