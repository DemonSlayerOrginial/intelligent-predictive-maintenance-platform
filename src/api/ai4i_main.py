from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.ai4i_features import build_ai4i_features

MODEL_PATH = Path("models/ai4i_best_model.joblib")
METADATA_PATH = Path("models/ai4i_metadata.json")

app = FastAPI(title="AI4I Failure Risk API", version="0.2.0")
model = None
metadata: dict = {}


class MachineSnapshot(BaseModel):
    machine_id: str = Field(examples=["CNC-184"])
    type: str = Field(pattern="^[LMH]$", examples=["L"])
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: int
    torque_nm: float
    tool_wear_min: int


class FailureRiskResponse(BaseModel):
    machine_id: str
    failure_probability: float
    decision_threshold: float
    predicted_failure: bool
    risk_level: str


def to_risk_level(probability: float, threshold: float) -> str:
    if probability >= max(0.75, threshold):
        return "critical"
    if probability >= threshold:
        return "warning"
    return "healthy"


@app.on_event("startup")
def load_model() -> None:
    global model, metadata
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text())


@app.get("/health")
def health() -> dict:
    return {"status": "ok","model_loaded": model is not None,"model": metadata.get("best_model")}


@app.post("/predict", response_model=FailureRiskResponse)
def predict(snapshot: MachineSnapshot) -> FailureRiskResponse:
    if model is None:
        raise HTTPException(status_code=503,detail="AI4I model not available. Run python -m src.models.train_ai4i first.")
    raw = pd.DataFrame([{
        "type": snapshot.type,
        "air_temperature_k": snapshot.air_temperature_k,
        "process_temperature_k": snapshot.process_temperature_k,
        "rotational_speed_rpm": snapshot.rotational_speed_rpm,
        "torque_nm": snapshot.torque_nm,
        "tool_wear_min": snapshot.tool_wear_min,
    }])
    features = build_ai4i_features(raw)
    feature_columns = metadata["feature_columns"]
    probability = float(model.predict_proba(features[feature_columns])[0, 1])
    threshold = float(metadata["decision_threshold"])
    return FailureRiskResponse(
        machine_id=snapshot.machine_id,
        failure_probability=round(probability, 4),
        decision_threshold=round(threshold, 4),
        predicted_failure=probability >= threshold,
        risk_level=to_risk_level(probability, threshold),
    )
