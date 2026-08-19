"""
High-value tests for the XAI NIDS Phase 1+2 pipeline.

Tests use a tiny slice of the real UNSW-NB15 training CSV so the suite
remains fast and reflects the actual data schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data import COLUMNS_TO_DROP, load_dataset  # noqa: E402
from app.inference import (  # noqa: E402
    LABEL_MAPPING,
    load_artifacts,
    predict_batch,
    predict_single,
)

ARTIFACT_DIR = ROOT / "models"
DATASET_PATH = ROOT / "UNSW-NB15" / "UNSW_NB15_training.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts(ARTIFACT_DIR)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """A tiny DataFrame using only the raw feature columns (no label)."""
    df = load_dataset(DATASET_PATH)
    df = df.drop(columns=["label"])
    return df.iloc[:3].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Test 1 — All artifacts load
# ---------------------------------------------------------------------------

def test_artifacts_load(artifacts):
    assert artifacts["preprocessor"] is not None
    assert artifacts["xgb_full"] is not None
    assert artifacts["xgb_light"] is not None
    assert isinstance(artifacts["top_features"], list)
    assert len(artifacts["top_features"]) == 10
    # Sanity: all persisted top features actually appear in the preprocessor
    transformed = set(artifacts["preprocessor"].get_feature_names_out())
    assert all(f in transformed for f in artifacts["top_features"])


# ---------------------------------------------------------------------------
# Test 2 — Full single prediction
# ---------------------------------------------------------------------------

def test_predict_single_full(sample_df, artifacts):
    res = predict_single(sample_df.iloc[[0]], artifacts, model_name="full")
    assert "prediction" in res
    assert res["label"] in LABEL_MAPPING.values()
    assert 0.0 <= res["probability"] <= 1.0
    assert res["model"] == "full"
    assert isinstance(res["explanation"], list) and len(res["explanation"]) > 0


# ---------------------------------------------------------------------------
# Test 3 — Lightweight single prediction
# ---------------------------------------------------------------------------

def test_predict_single_lightweight(sample_df, artifacts):
    res = predict_single(sample_df.iloc[[0]], artifacts, model_name="light")
    assert "prediction" in res
    assert res["label"] in LABEL_MAPPING.values()
    assert 0.0 <= res["probability"] <= 1.0
    assert res["model"] == "light"
    # The explanation must use the persisted Top-10 transformed feature names.
    feats = {e["feature"] for e in res["explanation"]}
    assert feats == set(artifacts["top_features"])


# ---------------------------------------------------------------------------
# Test 4 — Invalid input produces a clear error
# ---------------------------------------------------------------------------

def test_invalid_input_missing_column(sample_df, artifacts):
    bad = sample_df.iloc[[0]].drop(columns=["sttl"])
    with pytest.raises(ValueError, match="Missing required feature"):
        predict_single(bad, artifacts)


def test_invalid_input_nan_value(sample_df, artifacts):
    bad = sample_df.iloc[[0]].copy()
    bad["sttl"] = np.nan
    with pytest.raises(ValueError, match="missing/NaN"):
        predict_single(bad, artifacts)


def test_invalid_input_non_numeric(sample_df, artifacts):
    bad = sample_df.iloc[[0]].copy()
    bad["sttl"] = "not-a-number"
    with pytest.raises(ValueError, match="must be numeric"):
        predict_single(bad, artifacts)


# ---------------------------------------------------------------------------
# Test 5 — SHAP returns local feature contributions
# ---------------------------------------------------------------------------

def test_shap_local_values(sample_df, artifacts):
    res = predict_single(sample_df.iloc[[0]], artifacts, model_name="full")
    expl = res["explanation"]
    assert isinstance(expl, list)
    assert all({"feature", "shap_value"} <= set(e.keys()) for e in expl)
    # Sorted by |value| descending.
    abs_vals = [abs(e["shap_value"]) for e in expl]
    assert abs_vals == sorted(abs_vals, reverse=True)


# ---------------------------------------------------------------------------
# Test 6 — Batch prediction
# ---------------------------------------------------------------------------

def test_predict_batch(sample_df, artifacts):
    out = predict_batch(sample_df, artifacts, model_name="full")
    assert len(out) == len(sample_df) == 3
    for col in ("prediction", "label", "probability"):
        assert col in out.columns
    assert set(out["label"]).issubset(set(LABEL_MAPPING.values()))
    assert (out["probability"].between(0.0, 1.0)).all()


# ---------------------------------------------------------------------------
# Test 7 — Research consistency (sanity vs. notebook reference)
# ---------------------------------------------------------------------------

def test_research_consistency_metadata():
    """The persisted metadata should reflect the published accuracy bands."""
    meta_path = ARTIFACT_DIR / "model_metadata.json"
    if not meta_path.exists():
        pytest.skip("model_metadata.json not present")
    import json
    with meta_path.open() as f:
        meta = json.load(f)
    full_acc = meta["full_metrics"]["accuracy"]
    light_acc = meta["light_metrics"]["accuracy"]
    # Reference values from the notebook: 97.64% / 97.30%.
    assert 96.5 <= full_acc <= 98.5, f"Full acc out of band: {full_acc}"
    assert 96.0 <= light_acc <= 98.5, f"Light acc out of band: {light_acc}"
