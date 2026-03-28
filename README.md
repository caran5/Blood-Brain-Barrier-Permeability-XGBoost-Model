# Blood-Brain Barrier Penetration (BBBP) Prediction with XGBoost

A machine learning project for predicting whether drug molecules can penetrate the blood-brain barrier (BBB) using XGBoost and other classification algorithms. This repository implements models trained on the BBBP dataset to classify compounds as BBB penetrating (BBB+) or non-penetrating (BBB-).

## 📋 Overview

The blood-brain barrier is a highly selective membrane that controls the passage of chemical compounds between the bloodstream and the brain. Predicting BBB permeability is crucial in drug discovery, especially for developing treatments targeting the central nervous system (CNS). Approximately 98% of small molecules cannot cross the BBB naturally, making efficient computational prediction tools invaluable for accelerating drug development.

This project builds machine learning classification models to predict BBB penetration potential for drug compounds, enabling researchers to prioritize candidates early in the discovery process.

## 🎯 Key Features

- **XGBoost-based classification** for robust predictions
- **Multiple model implementations** for comparison (baseline classifiers included)
- **Structured project layout** following data science best practices
- **Comprehensive evaluation metrics** (ROC-AUC, precision, recall, F1-score)
- **Reproducible pipeline** with configurable parameters
- **Documentation and references** for methodology and related research

## 📊 Dataset

The project uses the **BBBP dataset** from MoleculeNet, containing:
- **2,053 compounds** with binary labels (BBB+ or BBB-)
- Compounds include drugs, hormones, and neurotransmitters
- Classification threshold: logBB ≥ -1 (Kp ≥ 0.1) for BBB penetrating

The dataset includes:
- Compound names
- SMILES strings (molecular representations)
- Binary penetration labels (0 = non-penetrating, 1 = penetrating)
- Calculated molecular descriptors

## 🏗️ Project Structure

```
bbbp_xgboost/
├── data/                    # Raw and processed datasets
│   └── BBBP.csv            # Main BBBP dataset
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── src/                    # Source code modules
│   ├── preprocessing.py    # Data loading and feature engineering
│   ├── models.py           # Model definitions and training
│   └── evaluation.py       # Metrics and evaluation functions
├── models/                 # Saved model artifacts
├── reports/                # Generated reports and visualizations
├── references/             # Research papers and documentation
├── docs/                   # Additional documentation
├── environment.yml         # Conda environment specification
├── requirements.txt        # Python dependencies
├── BBBP.csv               # Dataset file
├── README.md              # This file
├── TODO.md                # Development roadmap
└── explain.md             # Model explanation and methodology
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/caran5/bbbp_xgboost.git
   cd bbbp_xgboost
   ```

2. **Create a virtual environment (recommended)**
   
   Using conda:
   ```bash
   conda env create -f environment.yml
   conda activate bbbp_env
   ```
   
   Or using pip:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
   ```

## 💻 Usage

### Running the Pipeline

1. **Start a Jupyter notebook**
   ```bash
   jupyter notebook notebooks/
   ```

2. **Execute the main workflow**
   - Navigate to the relevant notebook
   - Follow cells sequentially to load data, train models, and evaluate performance

### Training a Model

```python
from src.models import XGBoostBBBPredictor
from src.preprocessing import load_and_preprocess_data

# Load and prepare data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/BBBP.csv')

# Train model
model = XGBoostBBBPredictor()
model.train(X_train, y_train)

# Evaluate
scores = model.evaluate(X_test, y_test)
print(f"ROC-AUC Score: {scores['roc_auc']:.4f}")
```

### Making Predictions

```python
# Predict for new compounds
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

## 📈 Model Performance

The project evaluates models using standard classification metrics:

- **ROC-AUC**: Receiver Operating Characteristic Area Under Curve
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient

Baseline performance on the BBBP dataset:
- KernelSVM: ROC-AUC ~0.73
- Random Forest: ROC-AUC ~0.75+
- XGBoost: ROC-AUC ~0.77+ (competitive)

*Note: Performance varies based on data preprocessing, feature engineering, and hyperparameter tuning.*

## 🔍 Methodology

### Feature Engineering
- Molecular descriptors (physicochemical properties)
- Fingerprints (structural representations)
- Diversity of features to capture BBB penetration mechanisms

### Model Selection
- **XGBoost**: Gradient boosting for robust classification
- **Baseline models**: Random Forest, Logistic Regression for comparison
- **Cross-validation**: Scaffold split validation recommended for generalization

### Key Considerations
- **Class imbalance**: BBB+ and BBB- compounds have different distributions
- **Molecular diversity**: Compounds span diverse chemical space
- **Interpretability**: Feature importance analysis to understand predictions

## 📚 References

The project is based on established research in computational drug discovery:

1. Sakiyama, H., Fukuda, M., & Okuno, T. (2021). Prediction of Blood-Brain Barrier Penetration (BBBP) Based on Molecular Descriptors. *Molecules*, 26(24), 7428.

2. Martins, I. F., et al. (2012). A Bayesian modular neural network architecture for QSAR modelling. *Journal of Chemical Information and Modeling*, 52(6), 1428-1446.

3. Waring, M. J., et al. (2015). An analysis of the attrition of drug candidates from four major pharmaceutical companies. *Nature Reviews Drug Discovery*, 14(7), 475-486.

See `references/` directory for additional papers and documentation.

## 🛠️ Development

### Contributing
Contributions are welcome! Areas for improvement:
- Advanced feature engineering techniques
- Deep learning approaches (neural networks)
- Uncertainty quantification in predictions
- Enhanced model interpretability
- Additional validation datasets

See `TODO.md` for planned enhancements.

### Running Tests
```bash
python -m pytest tests/  # If test suite is available
```

## 📋 Requirements

Key dependencies:
- **xgboost**: Gradient boosting library
- **scikit-learn**: Machine learning toolkit
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib, seaborn**: Visualization
- **jupyter**: Interactive notebooks

Full list in `requirements.txt` and `environment.yml`

## 📝 License

This project is provided as-is for educational and research purposes. Check the repository for specific licensing information.

## 🤝 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation in `docs/` and `explain.md`
- Review notebooks for implementation details

## 🔗 Related Resources

- **MoleculeNet**: http://moleculenet.ai/datasets-1
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **RDKit (Cheminformatics)**: https://www.rdkit.org/
- **Drug Discovery Resources**: https://www.ncbi.nlm.nih.gov/research/bionlp/

---

**Last Updated**: 2026  
**Status**: Active Development
