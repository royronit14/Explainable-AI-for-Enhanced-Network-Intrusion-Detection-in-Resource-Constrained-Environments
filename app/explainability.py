"""
SHAP-based explainability utilities.

Mirrors the notebook:
  - shap.TreeExplainer on the trained XGBoost
  - SHAP values sampled over a 1000-row slice of X_test
  - Top-K features ranked by mean |SHAP| across that sample
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import shap

DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_TOP_K = 10


def create_explainer(model) -> shap.TreeExplainer:
    """Build a TreeExplainer for a tree-based model (e.g. XGBoost).

    Workaround for xgboost>=2.1 / shap compatibility: xgboost stores
    `base_score` in `learner_model_param` as a JSON-array string like
    '[5.5E-1]' which SHAP's loader fails to parse. We monkey-patch
    the global builtin `float` during loader construction to coerce
    such strings.
    """
    import builtins as _bi

    _orig_builtin_float = _bi.float

    def _safe_float(value):
        try:
            return _orig_builtin_float(value)
        except (TypeError, ValueError):
            if isinstance(value, str):
                stripped = value.strip().strip("[]")
                first = stripped.split(",")[0].strip()
                return _orig_builtin_float(first)
            raise

    from shap.explainers import _tree as _shap_tree

    orig_loader_init = _shap_tree.XGBTreeModelLoader.__init__

    def _patched_loader_init(self, original_model):
        _bi.float = _safe_float
        try:
            orig_loader_init(self, original_model)
        finally:
            _bi.float = _orig_builtin_float

    _shap_tree.XGBTreeModelLoader.__init__ = _patched_loader_init
    try:
        return shap.TreeExplainer(model)
    finally:
        _shap_tree.XGBTreeModelLoader.__init__ = orig_loader_init


def get_shap_values(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    max_sample: int = DEFAULT_SAMPLE_SIZE,
    random_state: int = 42,
) -> np.ndarray:
    """Compute SHAP values on a sample (default 1000 rows).

    Matches notebook behavior of `shap.sample(X_test, 1000)`.
    """
    if hasattr(X, "shape") and X.shape[0] > max_sample:
        sample = shap.sample(X, max_sample)
    else:
        sample = X

    sv = explainer.shap_values(sample)
    return sv, sample


def get_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Return a DataFrame of (feature_name, mean|SHAP|) sorted descending."""
    vals = np.abs(shap_values).mean(0)
    if np.ndim(vals) > 1:
        # Multi-class: average across classes
        vals = np.mean(np.abs(shap_values), axis=(0, 2))
    return (
        pd.DataFrame(
            list(zip(feature_names, vals)),
            columns=["feature_name", "feature_importance_val"],
        )
        .sort_values(by="feature_importance_val", ascending=False)
        .reset_index(drop=True)
    )


def select_top_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = DEFAULT_TOP_K,
) -> List[str]:
    """Return the names of the top-K features by mean |SHAP|."""
    fi = get_feature_importance(shap_values, feature_names)
    return fi.head(top_k)["feature_name"].tolist()


def save_top_features(
    features: List[str],
    path: str | Path = "models/top10_features.json",
) -> Path:
    """Persist the top-K feature list as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"top_features": features, "k": len(features)}, f, indent=2)
    return out


def load_top_features(
    path: str | Path = "models/top10_features.json",
) -> List[str]:
    """Load the top-K feature list from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Top-K features file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["top_features"])
