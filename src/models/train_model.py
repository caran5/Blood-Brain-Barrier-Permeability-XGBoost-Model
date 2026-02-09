"""
train_model.py
--------------
Train an XGBoost classifier on the featurized training data
using DEFAULT hyperparameters (no tuning).
Saves the trained model to models/.
"""

import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgb_bbbp.json"

RANDOM_STATE = 42


def main():
    # Load featurized training data
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()

    print(f"[train_model] Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # ── Instantiate XGBoost with default hyper-parameters ──────────────
    # tree_method="hist" → CPU histogram-based algorithm (fast, no GPU needed)
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        tree_method="hist",      # CPU is sufficient for ~2 K rows
        random_state=RANDOM_STATE,
    )

    print("[train_model] Fitting XGBClassifier (default hyperparameters, CPU) …")
    model.fit(X_train, y_train)

    # ── Save ────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"[train_model] Model saved → {MODEL_PATH}")
    print("[train_model] Done ✓")


if __name__ == "__main__":
    main()
