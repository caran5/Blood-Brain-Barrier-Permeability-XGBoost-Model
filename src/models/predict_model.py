"""
predict_model.py
----------------
Load the trained XGBoost model and evaluate it on both the
train and test sets.  Reports Accuracy and F1 Score.

Metric definitions
------------------
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
    The fraction of all predictions that are correct.
    Easy to interpret but can be misleading with imbalanced classes.

- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
    The harmonic mean of precision and recall.
    More informative than accuracy when class distributions are skewed
    because it penalises both false positives and false negatives.
"""

import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_bbbp.json"


def evaluate(model, X, y, split_name: str) -> dict:
    """Predict and print accuracy / F1 for a given split."""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"\n{'=' * 50}")
    print(f"  {split_name.upper()} SET RESULTS")
    print(f"{'=' * 50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n{classification_report(y, y_pred, target_names=['non-penetrating (0)', 'penetrating (1)'])}")
    return {"accuracy": acc, "f1": f1}


def main():
    # Load model
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    print(f"[predict_model] Loaded model from {MODEL_PATH}")

    # Load data
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    # Evaluate on both splits
    train_metrics = evaluate(model, X_train, y_train, "train")
    test_metrics = evaluate(model, X_test, y_test, "test")

    # Overfitting check
    acc_gap = train_metrics["accuracy"] - test_metrics["accuracy"]
    f1_gap = train_metrics["f1"] - test_metrics["f1"]
    print(f"\n{'=' * 50}")
    print(f"  OVERFITTING CHECK")
    print(f"{'=' * 50}")
    print(f"  Accuracy gap (train − test): {acc_gap:+.4f}")
    print(f"  F1 gap       (train − test): {f1_gap:+.4f}")
    if acc_gap > 0.10:
        print("  ⚠  Significant overfitting detected — consider regularisation or fewer features.")
    else:
        print("  ✓  Gap is within a reasonable range.")

    # Metric explanations
    print(f"\n{'=' * 50}")
    print("  METRIC DEFINITIONS")
    print(f"{'=' * 50}")
    print("  Accuracy = (TP + TN) / (TP + TN + FP + FN)")
    print("    → Fraction of all predictions that are correct.")
    print("    → Simple but can be misleading with imbalanced classes.\n")
    print("  F1 Score = 2 × (Precision × Recall) / (Precision + Recall)")
    print("    → Harmonic mean of precision and recall.")
    print("    → Better than accuracy for skewed class distributions")
    print("      because it penalises both false positives and false negatives.")

    print("\n[predict_model] Done ✓")


if __name__ == "__main__":
    main()
