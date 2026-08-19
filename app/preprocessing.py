"""
Preprocessing pipeline.

Reproduces the notebook's ColumnTransformer:
  - StandardScaler on numerical columns
  - OneHotEncoder(handle_unknown='ignore') on categorical columns
  - remainder='passthrough'

The fitted preprocessor is serialized with joblib so the exact same
transform is applied at inference time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_column_types(
    df: pd.DataFrame,
) -> Tuple[pd.Index, pd.Index]:
    """Return (numerical_columns, categorical_columns) for a raw X frame."""
    numerical = df.select_dtypes(include=np.number).columns
    categorical = df.select_dtypes(include=["object"]).columns
    return numerical, categorical


def build_preprocessor(
    numerical_features: pd.Index,
    categorical_features: pd.Index,
) -> ColumnTransformer:
    """Construct the ColumnTransformer used by the notebook."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(numerical_features)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_features)),
        ],
        remainder="passthrough",
    )


def fit_preprocessor(
    X_train_raw: pd.DataFrame,
) -> Tuple[ColumnTransformer, np.ndarray, pd.Index, pd.Index]:
    """Fit a preprocessor on the training frame and return the transformed matrix.

    Returns (preprocessor, X_train_transformed, numerical_features, categorical_features).
    """
    numerical_features, categorical_features = split_column_types(X_train_raw)
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X_train = preprocessor.fit_transform(X_train_raw)
    return preprocessor, X_train, numerical_features, categorical_features


def transform(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Apply a fitted preprocessor to new data."""
    return preprocessor.transform(X)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the post-transform feature names."""
    return list(preprocessor.get_feature_names_out())


def save_preprocessor(
    preprocessor: ColumnTransformer,
    path: str | Path = "models/preprocessor.joblib",
) -> Path:
    """Persist a fitted preprocessor."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out)
    return out


def load_preprocessor(
    path: str | Path = "models/preprocessor.joblib",
) -> ColumnTransformer:
    """Load a persisted preprocessor."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {p}")
    return joblib.load(p)
