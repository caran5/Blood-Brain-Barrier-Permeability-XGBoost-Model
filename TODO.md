# TODO — BBBP Blood-Brain Barrier Penetration Model

> **Dataset:** `BBBP.csv` — 2,051 compounds with SMILES strings and a binary label (`p_np`: 1 = penetrates, 0 = does not penetrate).

---

## 1. Define the Problem

- [ ] **Task type:** Binary classification — predict whether a molecule penetrates the blood-brain barrier (`p_np` = 1) or not (`p_np` = 0).
- [ ] **Input data:** Molecular SMILES strings (+ derived molecular descriptors/fingerprints).
- [ ] **Desired output:** A predicted class label (0 or 1) for each molecule.
- [ ] **Success criteria:** Accuracy and F1-score on the held-out test set; compare against published SOTA benchmarks for BBBP (e.g., MoleculeNet leaderboard).

---

## 2. Collect and Prepare Data

### 2a. Data Collection
- [ ] Place the raw `BBBP.csv` into `data/raw/` to follow the project template conventions.
  ```
  cp BBBP.csv data/raw/BBBP.csv
  ```

### 2b. Data Cleaning (`src/data/make_dataset.py`)
- [ ] Load `data/raw/BBBP.csv`.
- [ ] Check for and handle **missing values** (especially missing SMILES strings — drop those rows).
- [ ] Validate SMILES strings with RDKit; drop or flag any that fail sanitization.
- [ ] Check class balance of `p_np` — note any imbalance for later (stratified splitting).
- [ ] Remove duplicate compounds if any.

### 2c. Train / Test Split (80-20)
- [ ] Split the cleaned data **80 % train / 20 % test** using `sklearn.model_selection.train_test_split` with `stratify=y` to preserve class ratios.
- [ ] Save splits to:
  - `data/processed/train.csv`
  - `data/processed/test.csv`
- [ ] **Do NOT touch the test set until final evaluation.**

### 2d. Feature Engineering (`src/features/build_features.py`)
- [ ] Convert SMILES → **RDKit molecular descriptors** (e.g., MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, etc.) and/or **Morgan fingerprints** (circular fingerprints, radius 2, 2048 bits).
- [ ] Save the resulting feature matrices:
  - `data/processed/X_train.csv`, `data/processed/y_train.csv`
  - `data/processed/X_test.csv`, `data/processed/y_test.csv`
- [ ] (Optional) Save an `interim/` version before final processing.

---

## 3. Environment Setup (Conda)

- [ ] Create a Conda environment:
  ```
  conda create -n bbbp python=3.11 -y
  conda activate bbbp
  ```
- [ ] Install dependencies:
  ```
  conda install -c conda-forge xgboost scikit-learn pandas numpy rdkit matplotlib seaborn -y
  ```
- [ ] **Export** the environment so others can reproduce it:
  ```
  conda env export > environment.yml
  ```
- [ ] Also populate `requirements.txt`:
  ```
  pip freeze > requirements.txt
  ```

---

## 4. Select an Appropriate Architecture

- [ ] **Model choice: XGBoost Classifier** (`xgboost.XGBClassifier`).
  - Gradient-boosted decision trees — strong baseline for tabular/molecular-descriptor data.
  - Good balance of performance and interpretability (built-in feature importance).
- [ ] **Do NOT optimize hyperparameters this week** — use default XGBoost settings.
- [ ] **CPU vs GPU:**
  - XGBoost supports both. For a dataset of ~2 K rows, **CPU is sufficient** and simpler to set up (`tree_method="hist"`).
  - GPU (`tree_method="gpu_hist"`, `device="cuda"`) is beneficial for much larger datasets or extensive hyperparameter sweeps; not needed here.
  - Document which device was used in the training script.

---

## 5. Train the Model (`src/models/train_model.py`)

- [ ] Load processed features (`X_train`, `y_train`).
- [ ] Instantiate `XGBClassifier` with **default** hyperparameters (no tuning this week).
  ```python
  from xgboost import XGBClassifier
  model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
  model.fit(X_train, y_train)
  ```
- [ ] Save the trained model to `models/xgb_bbbp.json` (or `.ubj`):
  ```python
  model.save_model("models/xgb_bbbp.json")
  ```

---

## 6. Evaluate the Model (`src/models/predict_model.py`)

### 6a. Metrics on Train and Test Sets
- [ ] Generate predictions on **both** train and test sets.
- [ ] Calculate and report:
  | Metric | Train | Test |
  |--------|-------|------|
  | **Accuracy** | — | — |
  | **F1 Score** | — | — |

- [ ] **What these metrics mean:**
  - **Accuracy** = (TP + TN) / (TP + TN + FP + FN) — the fraction of all predictions that are correct. Easy to interpret but can be misleading with imbalanced classes.
  - **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall) — the harmonic mean of precision and recall. More informative than accuracy when class distributions are skewed because it penalizes both false positives and false negatives.

- [ ] If train accuracy >> test accuracy, note potential **overfitting**.

### 6b. Feature Importance (`src/visualization/visualize.py`)
- [ ] Extract feature importances from the trained XGBoost model:
  ```python
  model.feature_importances_
  # or use xgboost.plot_importance(model)
  ```
- [ ] Plot a bar chart of the top-N most important features.
- [ ] Save the figure to `reports/figures/feature_importance.png`.

### 6c. Compare to SOTA (future)
- [ ] Look up MoleculeNet / OGB BBBP benchmark results.
- [ ] Note where this baseline XGBoost model stands relative to published numbers.

---

## 7. Hyperparameter Tuning

> **Goal:** Systematically search for a better set of XGBoost hyperparameters to
> close the train–test gap (overfitting) and improve test-set F1 / accuracy.

### 7a. Strategy Selection
- [ ] Choose a search method:
  - **RandomizedSearchCV** — good first pass, budget-friendly (recommended).
  - **GridSearchCV** — exhaustive but expensive; use only if the param grid is small.
  - **Optuna / Bayesian optimisation** — most sample-efficient; best for large grids.
- [ ] Use **5-fold stratified cross-validation** on the training set only.
- [ ] Primary scoring metric: **F1 (binary)** (accounts for class imbalance).

### 7b. Hyperparameter Search Space
- [ ] Define the search grid / distributions for the key XGBoost knobs:

  | Parameter | Range / Options | Why |
  |---|---|---|
  | `n_estimators` | 100, 200, 500, 1000 | Number of boosting rounds |
  | `max_depth` | 3, 4, 5, 6, 7, 8 | Tree depth — lower = less overfitting |
  | `learning_rate` (eta) | 0.01, 0.05, 0.1, 0.2, 0.3 | Step size shrinkage |
  | `subsample` | 0.6, 0.7, 0.8, 0.9, 1.0 | Row sampling per tree |
  | `colsample_bytree` | 0.5, 0.6, 0.7, 0.8, 1.0 | Feature sampling per tree |
  | `min_child_weight` | 1, 3, 5, 7 | Minimum sum of instance weight in a leaf |
  | `gamma` | 0, 0.1, 0.3, 0.5, 1.0 | Minimum loss reduction to make a split |
  | `reg_alpha` (L1) | 0, 0.01, 0.1, 1.0 | L1 regularisation |
  | `reg_lambda` (L2) | 1, 1.5, 2, 5 | L2 regularisation |
  | `scale_pos_weight` | 1, (neg_count/pos_count) | Helps with class imbalance |

### 7c. Run the Search
- [ ] Implement in `src/models/tune_model.py`:
  ```python
  from sklearn.model_selection import RandomizedSearchCV
  search = RandomizedSearchCV(
      XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42),
      param_distributions=param_grid,
      n_iter=100,          # number of random samples
      scoring="f1",
      cv=5,
      verbose=1,
      n_jobs=-1,
      random_state=42,
  )
  search.fit(X_train, y_train)
  ```
- [ ] Log all CV results to a CSV for analysis:
  ```python
  pd.DataFrame(search.cv_results_).to_csv("reports/tuning_results.csv")
  ```

### 7d. Evaluate Best Model
- [ ] Print the best hyperparameters found:
  ```python
  print(search.best_params_)
  print(f"Best CV F1: {search.best_score_:.4f}")
  ```
- [ ] Re-evaluate the best estimator on the held-out test set.
- [ ] Compare **before vs after** tuning:

  | Metric | Default (test) | Tuned (test) | Δ |
  |---|---|---|---|
  | Accuracy | 0.8922 | — | — |
  | F1 Score | 0.9317 | — | — |

- [ ] Save the tuned model to `models/xgb_bbbp_tuned.json`.
- [ ] Re-generate all visualisations with the tuned model (confusion matrix, ROC, PR curve, learning curve, etc.).

### 7e. Ablation / Sensitivity (Optional)
- [ ] Plot validation F1 vs each key hyperparameter to understand sensitivity.
- [ ] Check if reducing `max_depth` or increasing `reg_lambda` closes the overfitting gap.

---

## 8. Deploy the Model (Future Scope)

- [ ] Wrap the trained model in an inference script or API for novel predictions.
- [ ] Accept a SMILES string → featurize → predict → return class + probability.
- [ ] Validate predictions experimentally (wet-lab confirmation for promising candidates).

---

## 9. Reproducibility Checklist

- [ ] `data/raw/` contains the original untouched CSV.
- [ ] `data/processed/` contains the train/test splits and feature matrices.
- [ ] `models/` contains the saved XGBoost model artifact.
- [ ] `reports/figures/` contains the feature importance plot.
- [ ] `environment.yml` and `requirements.txt` fully specify the environment.
- [ ] All scripts in `src/` are runnable end-to-end from a fresh clone + environment setup.
- [ ] `notebooks/` contains any exploratory analysis notebooks (optional).
- [ ] Random seeds are set everywhere (`random_state=42`) for reproducibility.

---

## Quick-Start Order of Operations

```text
1.  conda create -n bbbp python=3.11 -y && conda activate bbbp
2.  conda install -c conda-forge xgboost scikit-learn pandas numpy rdkit matplotlib seaborn -y
3.  cp BBBP.csv data/raw/BBBP.csv
4.  python src/data/make_dataset.py          # clean + split
5.  python src/features/build_features.py    # featurize
6.  python src/models/train_model.py         # train XGBoost (default HPs)
7.  python src/models/predict_model.py       # evaluate baseline
8.  python src/visualization/visualize.py    # plot all figures
9.  python src/models/tune_model.py          # hyperparameter search
10. python src/models/predict_model.py       # re-evaluate tuned model
11. python src/visualization/visualize.py    # re-generate plots
12. conda env export > environment.yml
13. pip freeze > requirements.txt
```
