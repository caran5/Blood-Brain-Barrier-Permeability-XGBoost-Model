"""
build_features.py
-----------------
Read the processed train/test CSVs, convert SMILES into numerical features
using RDKit molecular descriptors + Morgan fingerprints, and save the
feature matrices for modelling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem


# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── descriptor list ────────────────────────────────────────────────────
DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3",
    "NumAromaticRings",
]

DESCRIPTOR_FUNCS = {name: func for name, func in Descriptors.descList if name in DESCRIPTOR_NAMES}

# Morgan fingerprint settings
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048


def smiles_to_descriptors(smiles: str) -> dict:
    """Compute a dict of selected RDKit molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: np.nan for name in DESCRIPTOR_NAMES}
    return {name: DESCRIPTOR_FUNCS[name](mol) for name in DESCRIPTOR_NAMES}


def smiles_to_morgan(smiles: str) -> np.ndarray:
    """Compute a Morgan fingerprint bit-vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(MORGAN_NBITS, np.nan)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    return np.array(fp, dtype=np.int8)


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Build a full feature DataFrame from a DataFrame that has a 'smiles' column."""
    # Molecular descriptors
    desc_df = pd.DataFrame(df["smiles"].apply(smiles_to_descriptors).tolist())
    desc_df.index = df.index

    # Morgan fingerprints
    fp_matrix = np.vstack(df["smiles"].apply(smiles_to_morgan).values)
    fp_cols = [f"morgan_{i}" for i in range(MORGAN_NBITS)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=df.index)

    features = pd.concat([desc_df, fp_df], axis=1)
    return features


def process_split(split_name: str) -> None:
    """Load a split CSV, featurize, and save X and y."""
    csv_path = PROCESSED_DIR / f"{split_name}.csv"
    df = pd.read_csv(csv_path)
    print(f"[build_features] Featurizing {split_name} ({len(df)} rows) …")

    X = featurize(df)
    y = df["p_np"]

    X_path = PROCESSED_DIR / f"X_{split_name}.csv"
    y_path = PROCESSED_DIR / f"y_{split_name}.csv"
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)
    print(f"[build_features]   → {X_path.name}  ({X.shape[1]} features)")
    print(f"[build_features]   → {y_path.name}")


def main():
    process_split("train")
    process_split("test")
    print(f"[build_features] Done ✓")


if __name__ == "__main__":
    main()
