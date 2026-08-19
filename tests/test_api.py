"""FastAPI endpoint tests for the XAI NIDS backend."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data import load_dataset  # noqa: E402
from app.main import app  # noqa: E402

DATASET_PATH = ROOT / "UNSW-NB15" / "UNSW_NB15_training.csv"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_features() -> dict:
    df = load_dataset(DATASET_PATH).drop(columns=["label"]).iloc[:1]
    return df.iloc[0].to_dict()


@pytest.fixture(scope="module")
def sample_csv_bytes() -> bytes:
    df = load_dataset(DATASET_PATH).drop(columns=["label"]).iloc[:5]
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# 1. Root
# ---------------------------------------------------------------------------

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Explainable AI Network Intrusion Detection"
    assert body["status"] == "running"


# ---------------------------------------------------------------------------
# 2. Health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("healthy", "degraded")
    assert isinstance(body["artifacts_loaded"], bool)


# ---------------------------------------------------------------------------
# 3. Model info
# ---------------------------------------------------------------------------

def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "XGBoost"
    assert "full_model" in body and "lightweight_model" in body
    assert body["full_model"]["features"] >= body["lightweight_model"]["features"]
    assert 0.0 <= body["full_model"]["accuracy"] <= 1.0
    assert 0.0 <= body["lightweight_model"]["accuracy"] <= 1.0
    assert isinstance(body["top_features"], list) and len(body["top_features"]) >= 1


# ---------------------------------------------------------------------------
# 4. Single prediction
# ---------------------------------------------------------------------------

def test_predict_single_full(client, sample_features):
    r = client.post("/predict", json={"model": "full", "features": sample_features})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "prediction" in body and "label" in body and "probability" in body
    assert body["model"] == "full"
    assert isinstance(body["explanation"], list) and len(body["explanation"]) >= 1


def test_predict_single_light(client, sample_features):
    r = client.post("/predict", json={"model": "light", "features": sample_features})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "light"
    # Explanation must use the persisted Top-10 transformed feature names.
    feats = {e["feature"] for e in body["explanation"]}
    assert len(feats) == 10


# ---------------------------------------------------------------------------
# 5. Invalid model
# ---------------------------------------------------------------------------

def test_predict_invalid_model(client, sample_features):
    r = client.post("/predict", json={"model": "bogus", "features": sample_features})
    assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 6. Invalid input (missing feature)
# ---------------------------------------------------------------------------

def test_predict_invalid_input_missing_feature(client, sample_features):
    bad = {k: v for k, v in sample_features.items() if k != "sttl"}
    r = client.post("/predict", json={"model": "full", "features": bad})
    assert r.status_code == 400, r.text
    assert "Missing required feature" in r.text


# ---------------------------------------------------------------------------
# 6b. Lightweight-only-input regression (must reject clean 4xx)
# ---------------------------------------------------------------------------
# The lightweight model is a slice of the full transformed matrix (see
# app/train.py). Therefore it is mathematically invalid to submit only the
# 10 top raw columns. The API must reject that with a clean 400 instead of
# silently inventing the missing 29 features.
LIGHT_RAW_10 = {
    "sttl",
    "ct_dst_src_ltm",
    "ct_dst_sport_ltm",
    "sbytes",
    "smean",
    "ct_srv_dst",
    "ct_srv_src",
    "ct_dst_ltm",
    "dbytes",
    "dmean",
}


def test_predict_light_requires_full_schema(client, sample_features):
    """10-field lightweight request must be rejected, not silently completed."""
    light_only = {k: v for k, v in sample_features.items() if k in LIGHT_RAW_10}
    assert len(light_only) == 10
    r = client.post("/predict", json={"model": "light", "features": light_only})
    assert r.status_code == 400, r.text
    assert "Missing required feature" in r.text


def test_predict_light_accepts_full_schema(client, sample_features):
    """Lightweight with full 39 features works and returns 10 SHAP entries."""
    r = client.post("/predict", json={"model": "light", "features": sample_features})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "light"
    assert isinstance(body["explanation"], list)
    assert len({e["feature"] for e in body["explanation"]}) == 10


# ---------------------------------------------------------------------------
# 7. Batch prediction
# ---------------------------------------------------------------------------

def test_predict_batch(client, sample_csv_bytes):
    files = {"file": ("sample.csv", sample_csv_bytes, "text/csv")}
    r = client.post("/predict/batch?model=full", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "full"
    assert body["total_rows"] == 5
    assert body["normal"] + body["attack"] == 5
    assert isinstance(body["results"], list) and len(body["results"]) == 5
    for row in body["results"]:
        assert {"prediction", "label", "probability"} <= set(row.keys())
