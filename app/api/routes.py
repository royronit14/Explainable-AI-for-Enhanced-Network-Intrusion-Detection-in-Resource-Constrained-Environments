"""HTTP routes for the XAI NIDS FastAPI backend.

Thin layer over ``app.inference`` — no ML logic here.
"""
from __future__ import annotations

import io
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.schemas import (
    BatchPredictResponse,
    BatchRow,
    HealthResponse,
    ModelInfoResponse,
    ModelName,
    RootResponse,
    SinglePredictRequest,
    SinglePredictResponse,
)
from app.inference import (
    LABEL_MAPPING,
    VALID_MODELS,
    load_artifacts,
    predict_batch,
    predict_single,
)

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

router = APIRouter()


# ---------------------------------------------------------------------------
# Lifecycle: load artifacts once at startup
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_artifacts() -> Dict[str, Any]:
    """Load inference artifacts once and cache them for the process lifetime."""
    try:
        return load_artifacts(MODELS_DIR)
    except Exception as exc:  # pragma: no cover - logged for diagnostics
        logger.exception("Failed to load inference artifacts: %s", exc)
        return {}


def _model_metadata() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    try:
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not read model_metadata.json: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/", response_model=RootResponse, summary="Service banner")
def root() -> RootResponse:
    return RootResponse(name="Explainable AI Network Intrusion Detection", status="running")


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    arts = get_artifacts()
    loaded = bool(arts) and all(
        arts.get(k) is not None
        for k in ("preprocessor", "xgb_full", "xgb_light", "top_features")
    )
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        artifacts_loaded=loaded,
    )


@router.get("/model-info", response_model=ModelInfoResponse, summary="Model metadata")
def model_info() -> ModelInfoResponse:
    arts = get_artifacts()
    if not arts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded.",
        )
    meta = _model_metadata()
    full = meta.get("full_metrics", {})
    light = meta.get("light_metrics", {})
    return ModelInfoResponse(
        model="XGBoost",
        full_model={
            "features": int(meta.get("n_features_full", 0)),
            "accuracy": float(full.get("accuracy", 0.0)) / 100.0,
        },
        lightweight_model={
            "features": int(meta.get("n_features_light", 0)),
            "accuracy": float(light.get("accuracy", 0.0)) / 100.0,
        },
        top_features=list(arts.get("top_features", [])),
    )


@router.post(
    "/predict",
    response_model=SinglePredictResponse,
    summary="Single-sample prediction with local SHAP",
)
def predict(req: SinglePredictRequest) -> SinglePredictResponse:
    if req.model not in VALID_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{req.model}'. Valid options: {list(VALID_MODELS)}",
        )
    arts = get_artifacts()
    if not arts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded.",
        )
    try:
        result = predict_single(req.features, arts, model_name=req.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from exc

    return SinglePredictResponse(**result)


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch prediction from uploaded CSV",
)
async def predict_batch_endpoint(
    model: ModelName = "full",
    file: UploadFile = File(..., description="CSV with the raw feature columns"),
) -> BatchPredictResponse:
    if model not in VALID_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{model}'. Valid options: {list(VALID_MODELS)}",
        )
    if file.content_type and "csv" not in file.content_type.lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected a CSV upload; got content-type '{file.content_type}'.",
        )

    arts = get_artifacts()
    if not arts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse CSV: {exc}",
        ) from exc

    try:
        out = predict_batch(df, arts, model_name=model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed.",
        ) from exc

    preds = out["prediction"].astype(int).tolist()
    labels = out["label"].tolist()
    probs = out["probability"].astype(float).tolist()
    normal = sum(1 for lab in labels if lab == LABEL_MAPPING[0])
    attack = sum(1 for lab in labels if lab == LABEL_MAPPING[1])

    results = [
        BatchRow(prediction=p, label=lab, probability=pr)
        for p, lab, pr in zip(preds, labels, probs)
    ]
    return BatchPredictResponse(
        model=model,
        total_rows=len(results),
        normal=normal,
        attack=attack,
        results=results,
    )
