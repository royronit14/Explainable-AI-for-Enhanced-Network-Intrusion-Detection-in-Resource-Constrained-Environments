"""
Data loading and split utilities.

Extracted from the notebook: loads UNSW-NB15 training CSV, drops
'id' and 'attack_cat' columns, separates features/target, and performs
the same stratified train/test split as the research implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Constants preserved from the notebook (source of truth).
DEFAULT_DATASET_PATH = Path("UNSW-NB15/UNSW_NB15_training.csv")
COLUMNS_TO_DROP = ["id", "attack_cat"]
TARGET_COLUMN = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the UNSW-NB15 CSV and drop the non-feature columns.

    Mirrors the notebook's behavior: drops any subset of
    ['id', 'attack_cat'] that actually exists in the loaded frame.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    existing = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    return df


def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into X_train, X_test, y_train, y_test.

    Uses stratified split with the same random_state and test_size as
    the original notebook.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
