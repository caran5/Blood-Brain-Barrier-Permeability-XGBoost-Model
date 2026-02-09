"""
make_dataset.py
---------------
Load raw BBBP.csv, clean the data, validate SMILES with RDKit,
and split into 80 % train / 20 % test (stratified).
Outputs are saved to data/processed/.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from rdkit import Chem

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "BBBP.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw BBBP CSV."""
    df = pd.read_csv(path)
    print(f"[make_dataset] Loaded {len(df)} rows from {path.name}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, missing values, and invalid SMILES."""
    initial_len = len(df)

    # Drop rows with missing SMILES or target
    df = df.dropna(subset=["smiles", "p_np"]).copy()
    print(f"[make_dataset] After dropping NaN: {len(df)} rows (removed {initial_len - len(df)})")

    # Remove duplicates based on SMILES
    before_dup = len(df)
    df = df.drop_duplicates(subset=["smiles"]).copy()
    print(f"[make_dataset] After dropping duplicates: {len(df)} rows (removed {before_dup - len(df)})")

    # Validate SMILES with RDKit
    valid_mask = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    n_invalid = (~valid_mask).sum()
    df = df[valid_mask].copy()
    print(f"[make_dataset] After SMILES validation: {len(df)} rows (removed {n_invalid} invalid)")

    # Report class balance
    counts = df["p_np"].value_counts()
    print(f"[make_dataset] Class distribution:\n{counts.to_string()}")
    print(f"[make_dataset]   Positive rate: {counts.get(1, 0) / len(df):.2%}")

    return df


def split_and_save(df: pd.DataFrame, output_dir: Path) -> None:
    """80-20 stratified split → train.csv & test.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["p_np"],
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[make_dataset] Train set: {len(train_df)} rows → {train_path}")
    print(f"[make_dataset] Test  set: {len(test_df)} rows → {test_path}")


def main():
    df = load_raw_data(RAW_DATA)
    df = clean_data(df)
    split_and_save(df, PROCESSED_DIR)
    print("[make_dataset] Done ✓")


if __name__ == "__main__":
    main()
