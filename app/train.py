"""
Reusable training pipeline.

Reproduces the published research pipeline:
  1. Load UNSW-NB15 training CSV
  2. Stratified 80/20 split (random_state=42)
  3. ColumnTransformer (StandardScaler + OneHotEncoder)
  4. Compare Decision Tree, MLP, XGBoost (preserved from notebook)
  5. SHAP TreeExplainer on XGBoost, sample 1000 rows
  6. Top-10 SHAP features -> lightweight XGBoost

Artifacts written under models/:
  - preprocessor.joblib
  - xgb_full.joblib
  - xgb_light.joblib
  - top10_features.json
  - model_metadata.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from app.data import load_dataset, split_data
from app.explainability import (
    create_explainer,
    get_shap_values,
    save_top_features,
    select_top_features,
)
from app.preprocessing import (
    fit_preprocessor,
    get_feature_names,
    save_preprocessor,
    transform,
)

MODELS_DIR = Path("models")
RANDOM_STATE = 42
TOP_K = 10


def _build_models() -> Dict[str, object]:
    """Construct the three models from the notebook with the same config."""
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "MLP Classifier": MLPClassifier(random_state=RANDOM_STATE, max_iter=500),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def _evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100),
        "precision": float(precision_score(y_true, y_pred, average="weighted") * 100),
        "recall": float(recall_score(y_true, y_pred, average="weighted") * 100),
        "f1": float(f1_score(y_true, y_pred, average="weighted") * 100),
    }


def _train_comparison(X_train, y_train, X_test, y_test) -> Dict[str, Dict]:
    """Train Decision Tree, MLP, XGBoost and report metrics.

    Mirrors notebook Part 3.
    """
    results: Dict[str, Dict] = {}
    for name, model in _build_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        y_pred = model.predict(X_test)
        metrics = _evaluate(y_test, y_pred)
        metrics["training_time_s"] = float(train_time)
        results[name] = {"model": model, "metrics": metrics}
        print(
            f"[comparison] {name:14s} acc={metrics['accuracy']:.2f}% "
            f"f1={metrics['f1']:.2f}% time={train_time:.2f}s"
        )
    return results


def train_full_and_lightweight(
    include_comparison: bool = True,
) -> Dict:
    """Run the full training pipeline and write artifacts to models/.

    Returns a summary dict with metrics and artifact paths.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading dataset ===")
    df = load_dataset()
    print(f"Loaded shape: {df.shape}")

    print("=== Splitting ===")
    X_train_raw, X_test_raw, y_train, y_test = split_data(df)

    print("=== Fitting preprocessor ===")
    preprocessor, X_train, numerical_features, categorical_features = fit_preprocessor(
        X_train_raw
    )
    X_test = transform(preprocessor, X_test_raw)
    feature_names = get_feature_names(preprocessor)
    print(f"Transformed feature count: {X_train.shape[1]}")

    # --- Optional comparison (Decision Tree + MLP + XGBoost) ---
    comparison_results = None
    if include_comparison:
        print("=== Training comparison models (DT, MLP, XGBoost) ===")
        comparison_results = _train_comparison(X_train, y_train, X_test, y_test)

    # --- Final full XGBoost model (same config as the comparison entry) ---
    print("=== Training full XGBoost model ===")
    full_model = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
    t0 = time.time()
    full_model.fit(X_train, y_train)
    full_train_time = time.time() - t0
    y_pred_full = full_model.predict(X_test)
    full_metrics = _evaluate(y_test, y_pred_full)
    full_metrics["training_time_s"] = float(full_train_time)
    print(
        f"[full-xgb] acc={full_metrics['accuracy']:.2f}% "
        f"f1={full_metrics['f1']:.2f}% time={full_train_time:.2f}s"
    )

    # --- SHAP Top-10 selection ---
    print("=== Computing SHAP feature importance ===")
    explainer = create_explainer(full_model)
    shap_values, X_sample = get_shap_values(explainer, X_test)

    # For binary classification shap_values has shape (n, n_features);
    # for multi-class it has shape (n, n_features, n_classes).
    # The notebook uses mean over the first axis for binary.
    if np.ndim(shap_values) == 3:
        # Reduce over class axis as well so importance is per-feature.
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        shap_for_topk = np.abs(shap_values).mean(axis=2)
    else:
        shap_for_topk = shap_values

    top_features = select_top_features(shap_for_topk, feature_names, top_k=TOP_K)
    print(f"Top-{TOP_K} features: {top_features}")

    # --- Lightweight model on Top-10 ---
    print("=== Training lightweight XGBoost on top-K features ===")
    feature_index = {f: i for i, f in enumerate(feature_names)}
    top_idx = [feature_index[f] for f in top_features]
    X_train_light = X_train[:, top_idx]
    X_test_light = X_test[:, top_idx]

    light_model = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
    t0 = time.time()
    light_model.fit(X_train_light, y_train)
    light_train_time = time.time() - t0
    y_pred_light = light_model.predict(X_test_light)
    light_metrics = _evaluate(y_test, y_pred_light)
    light_metrics["training_time_s"] = float(light_train_time)
    print(
        f"[light-xgb] acc={light_metrics['accuracy']:.2f}% "
        f"f1={light_metrics['f1']:.2f}% time={light_train_time:.2f}s"
    )

    # --- Save artifacts ---
    print("=== Saving artifacts ===")
    preprocessor_path = save_preprocessor(preprocessor, MODELS_DIR / "preprocessor.joblib")
    full_path = MODELS_DIR / "xgb_full.joblib"
    light_path = MODELS_DIR / "xgb_light.joblib"
    top_features_path = save_top_features(top_features, MODELS_DIR / "top10_features.json")

    joblib.dump(full_model, full_path)
    joblib.dump(light_model, light_path)

    metadata = {
        "model_type": "XGBoost",
        "random_state": RANDOM_STATE,
        "top_k": TOP_K,
        "n_features_full": int(X_train.shape[1]),
        "n_features_light": int(X_train_light.shape[1]),
        "full_metrics": full_metrics,
        "light_metrics": light_metrics,
    }
    if comparison_results is not None:
        metadata["comparison_metrics"] = {
            name: res["metrics"] for name, res in comparison_results.items()
        }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Artifacts written:")
    for p in [preprocessor_path, full_path, light_path, top_features_path, metadata_path]:
        print(f"  - {p}")

    return {
        "preprocessor_path": str(preprocessor_path),
        "xgb_full_path": str(full_path),
        "xgb_light_path": str(light_path),
        "top_features_path": str(top_features_path),
        "metadata_path": str(metadata_path),
        "top_features": top_features,
        "full_metrics": full_metrics,
        "light_metrics": light_metrics,
        "comparison_metrics": (
            {name: res["metrics"] for name, res in comparison_results.items()}
            if comparison_results
            else None
        ),
    }


def main() -> None:
    summary = train_full_and_lightweight(include_comparison=True)
    print("\n=== Training complete ===")
    print(f"Full model accuracy:    {summary['full_metrics']['accuracy']:.2f}%")
    print(f"Lightweight accuracy:   {summary['light_metrics']['accuracy']:.2f}%")
    print(f"Top-{summary['metadata_path'] and TOP_K} features: {summary['top_features']}")


if __name__ == "__main__":
    main()
