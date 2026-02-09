# Explanation of the BBBP Classification Project

## What We Did and the Results

We built a binary classification pipeline to predict whether small-molecule compounds can penetrate the blood-brain barrier (BBB) using the BBBP dataset (2,050 compounds). Starting from raw SMILES strings, we cleaned the data (removing 11 invalid molecules), validated every structure with RDKit, and performed a stratified 80/20 train-test split (1,631 train / 408 test), preserving the 76.5 % positive-class ratio. Each SMILES was featurised into 2,058 numerical columns—10 physicochemical descriptors (molecular weight, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, ring count, heavy-atom count, fraction Csp³, aromatic ring count) plus a 2,048-bit Morgan circular fingerprint (radius 2). We trained an XGBoost gradient-boosted tree classifier, first with default hyperparameters and then with a tuned configuration found via RandomizedSearchCV (100 iterations × 5-fold stratified CV, optimising F1). The default model achieved 89.22 % accuracy and 0.9317 F1 on the test set, while the tuned model (which introduced stronger regularisation via `reg_lambda = 5`, `colsample_bytree = 0.6`, `subsample = 0.7`, and `min_child_weight = 3`) improved to 89.71 % accuracy and 0.9348 F1 on the test set. Critically, the overfitting gap (train − test accuracy) shrank from 10.0 percentage points to 8.6 percentage points, confirming that regularisation helped the model generalise better. All training was performed on CPU (`tree_method = "hist"`), which was more than sufficient for a dataset of this size.

## Visualizations and Metrics Explained

### 1. Feature Importance (`feature_importance.png`)
This horizontal bar chart shows the top-25 features ranked by XGBoost's gain-based importance—i.e., how much each feature reduces the loss function across all the splits that use it. Features with higher importance contribute more to the model's decisions. In our case, specific Morgan fingerprint bits (which encode substructural motifs) and physicochemical descriptors like MolLogP and TPSA dominate, which aligns with domain knowledge that lipophilicity and polar surface area are key drivers of BBB penetration.

### 2. Confusion Matrices (`confusion_matrices.png`, `compare_confusion_matrices.png`)
A confusion matrix is a 2×2 grid showing how many samples were correctly or incorrectly classified for each class. The rows represent the true labels and the columns represent the predicted labels, yielding four counts: true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). Our matrices are normalised to percentages so you can read them as "of all true class-0 molecules, what fraction did the model get right?" On the test set, the tuned model correctly classifies 68 % of non-penetrating compounds and 96 % of penetrating compounds—the imbalance reflects both the skewed class distribution and the inherent difficulty of predicting the minority class. The comparison figure puts the default and tuned models side-by-side so you can see that tuning slightly improved the minority-class recall.

### 3. ROC Curve (`roc_curve.png`, `compare_roc.png`)
The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (recall) on the y-axis against the False Positive Rate (1 − specificity) on the x-axis as the classification threshold sweeps from 0 to 1. The Area Under the Curve (AUC) summarises overall discrimination: 1.0 is perfect, 0.5 is random guessing. The default model achieved an AUC of 0.9375 and the tuned model 0.9389 on the test set—both strong, and the overlaid comparison shows the two curves nearly overlap, indicating similar ranking quality.

### 4. Precision-Recall Curve (`precision_recall_curve.png`, `compare_precision_recall.png`)
The Precision-Recall (PR) curve plots precision (of the predicted positives, how many are truly positive) against recall (of the true positives, how many did we find) at every threshold. The Average Precision (AP) summarises the area under this curve. PR curves are more informative than ROC when classes are imbalanced because they focus on the positive class. Our models achieved an AP of 0.978 (default) and 0.980 (tuned), meaning that at nearly every recall level the model maintains very high precision. The dashed baseline represents the class prevalence (76.5 %)—the precision you would get by predicting every sample as positive.

### 5. Learning Curves (`learning_curves.png`)
Learning curves plot training-set and cross-validation performance as a function of the number of training samples used. They diagnose two key issues: **high bias** (both curves plateau at a low score, meaning the model is too simple) and **high variance / overfitting** (a large gap between training and validation scores). Our plot shows that training accuracy is nearly perfect regardless of dataset size, while validation accuracy steadily climbs and starts to plateau around 1,200 samples. The persistent gap between the two curves signals some overfitting, though it narrows with more data—suggesting that the model could benefit from additional training data or further regularisation.

### 6. Metric Comparison Bar Chart (`metric_comparison.png`, `compare_metrics.png`)
This grouped bar chart puts four key metrics side-by-side for quick comparison. **Accuracy** is the overall fraction of correct predictions—simple but potentially misleading when one class dominates. **F1 Score** is the harmonic mean of precision and recall, giving equal weight to both false positives and false negatives—it is a more balanced metric for imbalanced datasets. **Precision** answers "when the model predicts positive, how often is it right?" (0.907 on test for the tuned model). **Recall** (also called sensitivity or true positive rate) answers "of all actual positives, how many did the model catch?" (0.965 on test for the tuned model). The comparison version shows default (blue) vs tuned (red) on both train and test splits, making it easy to see that tuning slightly lifted test-set metrics while meaningfully lowering the train-set metrics—evidence of reduced overfitting.

### 7. Class Distribution (`class_distribution.png`)
This bar chart simply shows how many samples belong to each class in the train and test sets. It confirms that the stratified split preserved the original 76.5 % positive / 23.5 % negative ratio in both partitions. Understanding class imbalance is essential because it affects which metrics are meaningful: with 76.5 % positives, a naïve "predict all positive" classifier would already achieve 76.5 % accuracy, which is why F1, precision, and recall are more informative here.

### 8. Probability Histogram (`probability_histogram.png`, `compare_probability.png`)
This histogram shows the distribution of predicted probabilities for each true class. Ideally, true negatives (class 0) should cluster near 0 and true positives (class 1) near 1, with clear separation at the 0.5 decision threshold. Our models show strong separation for the majority class (penetrating molecules pile up near 1.0), but the minority class (non-penetrating) has a wider spread with some mass above 0.5—these are the false positives. The comparison version lets you see whether tuning improved this separation (it did slightly, producing a tighter peak for non-penetrating compounds below the threshold).

---

### Summary Table

| Metric | Default (Train) | Default (Test) | Tuned (Train) | Tuned (Test) |
|---|---|---|---|---|
| **Accuracy** | 0.9926 | 0.8922 | 0.9828 | **0.8971** |
| **F1 Score** | 0.9952 | 0.9317 | 0.9888 | **0.9348** |
| **Precision** | 0.9920 | 0.9036 | 0.9849 | **0.9066** |
| **Recall** | 0.9984 | 0.9615 | 0.9928 | **0.9647** |
| **ROC AUC** | 0.9998 | 0.9375 | 0.9988 | **0.9389** |
| **Avg Precision** | 1.0000 | 0.9781 | 0.9996 | **0.9796** |
| **Overfit gap (Acc)** | — | 0.1005 ⚠️ | — | **0.0858** ✓ |
