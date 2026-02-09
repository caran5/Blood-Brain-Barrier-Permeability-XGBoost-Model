"""
tune_model.py
-------------
Hyperparameter tuning for XGBoost BBBP classifier using
RandomizedSearchCV (5-fold stratified CV, scoring=F1).

Saves:
  - models/xgb_bbbp_tuned.json          (best model)
  - reports/tuning_results.csv           (full CV results)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
N_ITER = 100  # number of random samples


def main():
    # ── Load data ──────────────────────────────────────────────────────
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()

    print(f"[tune_model] Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # ── Compute scale_pos_weight for imbalance ─────────────────────────
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count
    print(f"[tune_model] Class balance: {neg_count} neg / {pos_count} pos → scale_pos_weight={spw:.2f}")

    # ── Hyperparameter search space ────────────────────────────────────
    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 700, 1000],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.3, 0.5, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [1, 1.5, 2, 5],
        "scale_pos_weight": [1, spw],
    }

    # ── Base estimator ─────────────────────────────────────────────────
    base_model = XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    # ── Randomized search ──────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="f1",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )

    print(f"[tune_model] Running RandomizedSearchCV ({N_ITER} iterations, 5-fold CV) …")
    search.fit(X_train, y_train)

    # ── Results ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  BEST HYPERPARAMETERS")
    print(f"{'=' * 60}")
    for k, v in sorted(search.best_params_.items()):
        print(f"  {k:25s} = {v}")
    print(f"\n  Best CV F1 Score: {search.best_score_:.4f}")
    print(f"{'=' * 60}")

    # Save full CV results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / "tuning_results.csv"
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    results_df.to_csv(results_path, index=False)
    print(f"\n[tune_model] Full CV results → {results_path}")

    # Print top-5 configs
    print(f"\n  Top-5 configurations:")
    top5 = results_df.head(5)[["rank_test_score", "mean_test_score", "std_test_score", "mean_train_score", "params"]]
    for _, row in top5.iterrows():
        print(f"    Rank {int(row['rank_test_score'])}: "
              f"CV F1={row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}  "
              f"(train F1={row['mean_train_score']:.4f})")

    # ── Save best model ───────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tuned_path = MODEL_DIR / "xgb_bbbp_tuned.json"
    search.best_estimator_.save_model(str(tuned_path))
    print(f"\n[tune_model] Tuned model saved → {tuned_path}")

    # Also overwrite the default model path so predict_model.py and
    # visualize.py pick up the tuned version automatically
    default_path = MODEL_DIR / "xgb_bbbp.json"
    search.best_estimator_.save_model(str(default_path))
    print(f"[tune_model] Also saved as → {default_path}  (overwrites default)")

    print("\n[tune_model] Done ✓")


if __name__ == "__main__":
    main()
