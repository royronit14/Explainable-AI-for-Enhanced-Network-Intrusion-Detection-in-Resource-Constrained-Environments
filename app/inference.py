"""
Reusable inference engine.

Loads persisted artifacts (preprocessor + models + top-K features) and
provides single + batch prediction with local SHAP explanations.

This module never trains models; it only consumes artifacts produced by
``app.train``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from app.explainability import create_explainer

MODELS_DIR = Path("models")
ARTIFACT_NAMES = {
    "preprocessor": "preprocessor.joblib",
    "full": "xgb_full.joblib",
    "light": "xgb_light.joblib",
    "top_features": "top10_features.json",
}

# UNSW-NB15 binary label semantics (from notebook / dataset).
LABEL_MAPPING: Dict[int, str] = {0: "Normal", 1: "Attack"}
VALID_MODELS: Tuple[str, ...] = ("full", "light")


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_top_features(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Top-K features file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    feats = data.get("top_features")
    if not feats:
        raise ValueError(f"No 'top_features' key with a list in {path}")
    return list(feats)


def load_artifacts(
    models_dir: str | Path = MODELS_DIR,
) -> Dict[str, Any]:
    """Load all inference artifacts from ``models_dir``.

    Returns a dict with keys:
        preprocessor, xgb_full, xgb_light, top_features, full_explainer,
        light_explainer, raw_feature_names.
    """
    base = Path(models_dir)
    preprocessor = joblib.load(base / ARTIFACT_NAMES["preprocessor"])
    xgb_full = joblib.load(base / ARTIFACT_NAMES["full"])
    xgb_light = joblib.load(base / ARTIFACT_NAMES["light"])
    top_features = _load_top_features(base / ARTIFACT_NAMES["top_features"])

    # The lightweight model was trained on transformed (post-OneHot/Scaler)
    # feature names. Capture the raw input schema so we can validate user
    # input in the original 39-feature space.
    transformed_names = list(preprocessor.get_feature_names_out())
    feature_index = {f: i for i, f in enumerate(transformed_names)}
    missing = [f for f in top_features if f not in feature_index]
    if missing:
        raise ValueError(
            f"Persisted top_features not found in preprocessor output: {missing}"
        )

    return {
        "preprocessor": preprocessor,
        "xgb_full": xgb_full,
        "xgb_light": xgb_light,
        "top_features": top_features,
        "full_explainer": create_explainer(xgb_full),
        "light_explainer": create_explainer(xgb_light),
        "raw_feature_names": None,  # populated lazily by _raw_feature_names
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _raw_feature_names(artifacts: Mapping[str, Any]) -> List[str]:
    """Recover the original (pre-transform) feature names from the preprocessor.

    sklearn stores raw column names under ``_feature_names_in`` for
    transformers fit on a DataFrame. Falls back to numeric indices if absent.
    """
    if artifacts.get("raw_feature_names"):
        return artifacts["raw_feature_names"]
    pre = artifacts["preprocessor"]
    names = getattr(pre, "feature_names_in_", None)
    if names is None:
        raise RuntimeError(
            "Preprocessor does not expose raw feature names. "
            "Re-fit with a DataFrame so feature_names_in_ is populated."
        )
    names_list = [str(n) for n in names]
    artifacts["raw_feature_names"] = names_list
    return names_list


def validate_input(
    data: Mapping[str, Any] | pd.DataFrame,
    artifacts: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate a single-row input.

    Returns a single-row DataFrame with columns in the preprocessor's
    expected order. Raises ``ValueError`` with a precise message on failure.
    """
    expected_cols = _raw_feature_names(artifacts)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if len(df) != 1:
            raise ValueError(
                f"validate_input expects exactly 1 row; got {len(df)}. "
                "Use predict_batch for multi-row input."
            )
    else:
        df = pd.DataFrame([dict(data)])

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

    extra = [c for c in df.columns if c not in expected_cols]
    if extra:
        # Drop silently? No — be explicit.
        raise ValueError(f"Unexpected feature(s): {', '.join(extra)}")

    # Numeric coercion + NaN detection.
    for c in expected_cols:
        series = df[c]
        if series.isna().any():
            raise ValueError(f"Feature '{c}' contains missing/NaN values")
        try:
            df[c] = pd.to_numeric(series, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature '{c}' must be numeric; got {series.iloc[0]!r}"
            ) from exc

    return df[expected_cols]


def validate_batch(
    df: pd.DataFrame,
    artifacts: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate a multi-row DataFrame."""
    expected_cols = _raw_feature_names(artifacts)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("predict_batch expects a pandas DataFrame")
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

    extra = [c for c in df.columns if c not in expected_cols]
    if extra:
        raise ValueError(f"Unexpected feature(s): {', '.join(extra)}")

    coerced = df[expected_cols].apply(pd.to_numeric, errors="raise")
    if coerced.isna().any().any():
        bad = [c for c in expected_cols if coerced[c].isna().any()]
        raise ValueError(f"Missing/NaN values in feature(s): {', '.join(bad)}")

    return coerced


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_model(artifacts: Mapping[str, Any], model_name: str):
    if model_name == "full":
        return artifacts["xgb_full"], artifacts["full_explainer"], artifacts["preprocessor"]
    if model_name == "light":
        return artifacts["xgb_light"], artifacts["light_explainer"], None
    raise ValueError(
        f"Unknown model_name '{model_name}'. Valid options: {VALID_MODELS}"
    )


def _transform(
    df: pd.DataFrame,
    artifacts: Mapping[str, Any],
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Return (model_input, light_indices_or_None).

    For 'full' we use the entire transformed matrix. For 'light' we select
    the columns indexed by the persisted top-K feature names.
    """
    pre = artifacts["preprocessor"]
    X_full = pre.transform(df)
    if model_name == "full":
        return X_full, None

    transformed_names = list(pre.get_feature_names_out())
    index = {n: i for i, n in enumerate(transformed_names)}
    top = artifacts["top_features"]
    idx = np.array([index[f] for f in top], dtype=int)
    return X_full[:, idx], idx


def _label_for(pred_value: int | np.integer) -> str:
    return LABEL_MAPPING.get(int(pred_value), str(pred_value))


def _shap_local(
    explainer,
    model_input: np.ndarray,
    feature_names: Sequence[str],
) -> List[Dict[str, float]]:
    """Return a list of {feature, value} dicts for a single row."""
    shap_vals = explainer.shap_values(model_input)
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:
        # Multi-class: pick the predicted class column.
        arr = arr[0]
    row = arr[0] if arr.ndim == 2 else arr
    feats: List[Dict[str, float]] = []
    for name, val in zip(feature_names, row):
        feats.append({"feature": str(name), "shap_value": float(val)})
    feats.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
    return feats


# ---------------------------------------------------------------------------
# Public prediction API
# ---------------------------------------------------------------------------

def predict_single(
    input_data: Mapping[str, Any] | pd.DataFrame,
    artifacts: Mapping[str, Any] | None = None,
    model_name: str = "full",
) -> Dict[str, Any]:
    """Predict one sample and return a structured result with local SHAP."""
    if artifacts is None:
        artifacts = load_artifacts()
    if model_name not in VALID_MODELS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Valid options: {VALID_MODELS}"
        )

    df = validate_input(input_data, artifacts)
    model, explainer, _ = _select_model(artifacts, model_name)

    model_input, _ = _transform(df, artifacts, model_name)
    pred = int(model.predict(model_input)[0])
    proba = model.predict_proba(model_input)[0]
    prob_attack = float(proba[1]) if len(proba) > 1 else float(proba[0])

    feature_names: Sequence[str]
    if model_name == "full":
        feature_names = list(artifacts["preprocessor"].get_feature_names_out())
    else:
        feature_names = list(artifacts["top_features"])

    explanation = _shap_local(explainer, model_input, feature_names)

    return {
        "prediction": pred,
        "label": _label_for(pred),
        "probability": prob_attack,
        "probabilities": {LABEL_MAPPING.get(i, str(i)): float(p) for i, p in enumerate(proba)},
        "model": model_name,
        "explanation": explanation,
    }


def predict_batch(
    df: pd.DataFrame,
    artifacts: Mapping[str, Any] | None = None,
    model_name: str = "full",
) -> pd.DataFrame:
    """Predict for a multi-row DataFrame; returns an annotated DataFrame.

    SHAP is intentionally NOT computed per row here; use ``predict_single``
    for explanations.
    """
    if artifacts is None:
        artifacts = load_artifacts()
    if model_name not in VALID_MODELS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Valid options: {VALID_MODELS}"
        )

    validated = validate_batch(df, artifacts)
    model, _, _ = _select_model(artifacts, model_name)
    model_input, _ = _transform(validated, artifacts, model_name)

    preds = model.predict(model_input)
    probas = model.predict_proba(model_input)

    out = validated.copy().reset_index(drop=True)
    out["prediction"] = [int(p) for p in preds]
    out["label"] = [_label_for(p) for p in preds]
    if probas.shape[1] > 1:
        out["probability"] = probas[:, 1].astype(float)
    else:
        out["probability"] = probas[:, 0].astype(float)
    return out


__all__ = [
    "load_artifacts",
    "validate_input",
    "validate_batch",
    "predict_single",
    "predict_batch",
    "LABEL_MAPPING",
    "VALID_MODELS",
]
