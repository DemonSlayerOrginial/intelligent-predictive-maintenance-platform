from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/failure_model.joblib")
FEATURES_PATH = Path("models/feature_columns.json")

app = FastAPI(title="Predictive Maintenance ML API", version="0.1.0")
model = None
feature_columns: list[str] = []


class PredictionRequest(BaseModel):
    machine_id: str = Field(examples=["M184"])
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    machine_id: str
    failure_probability_24h: float
    risk_level: str


def risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "critical"
    if probability >= 0.45:
        return "warning"
    return "healthy"


@app.on_event("startup")
def load_model() -> None:
    global model, feature_columns
    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        model = joblib.load(MODEL_PATH)
        feature_columns = json.loads(FEATURES_PATH.read_text())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not trained/loaded yet")
    missing = [c for c in feature_columns if c not in request.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing {len(missing)} required features. First few: {missing[:8]}")
    row = pd.DataFrame([{c: request.features[c] for c in feature_columns}])
    probability = float(model.predict_proba(row)[0, 1])
    return PredictionResponse(machine_id=request.machine_id,failure_probability_24h=round(probability, 4),risk_level=risk_level(probability))
