"""Pydantic schemas for the HTTP API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ModelName = Literal["full", "light"]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    artifacts_loaded: bool


class RootResponse(BaseModel):
    name: str
    status: str


class ModelInfoResponse(BaseModel):
    model: str
    full_model: Dict[str, Any]
    lightweight_model: Dict[str, Any]
    top_features: List[str]


class SinglePredictRequest(BaseModel):
    model: ModelName = Field(default="full", description="'full' or 'light'")
    features: Dict[str, float] = Field(
        ..., description="Raw feature dict matching the preprocessor schema"
    )


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


class SinglePredictResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    probabilities: Dict[str, float]
    model: ModelName
    explanation: List[ShapFeature]


class BatchRow(BaseModel):
    prediction: int
    label: str
    probability: float


class BatchPredictResponse(BaseModel):
    model: ModelName
    total_rows: int
    normal: int
    attack: int
    results: List[BatchRow]
