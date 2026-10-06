from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app

from src.models.streaming_inference import StreamingModelBundle

app = FastAPI(title="Predictive Maintenance Inference", version="1.0.0")
app.mount("/metrics", make_asgi_app())

REQUESTS = Counter("inference_requests_total", "Total inference requests", ["status"])
LATENCY = Histogram("inference_latency_seconds", "Inference request latency")
RISK = Counter("inference_risk_predictions_total", "Predictions by risk level", ["risk_level"])

bundle: StreamingModelBundle | None = None


class FeaturePayload(BaseModel):
    timestamp: str
    machine_id: str
    temperature: float
    vibration: float
    pressure: float
    rpm: float
    voltage: float
    load: float
    error_count: int
    hours_since_maintenance: int
    temperature_mean_6h: float
    temperature_std_24h: float
    temperature_delta_6h: float
    vibration_mean_6h: float
    vibration_std_24h: float
    vibration_delta_6h: float
    pressure_mean_6h: float
    pressure_std_24h: float
    pressure_delta_6h: float
    rpm_mean_6h: float
    rpm_std_24h: float
    rpm_delta_6h: float
    voltage_mean_6h: float
    voltage_std_24h: float
    voltage_delta_6h: float
    load_mean_6h: float
    load_std_24h: float
    load_delta_6h: float
    errors_24h: int
    errors_6h: int


@app.on_event("startup")
def startup() -> None:
    global bundle
    try:
        bundle = StreamingModelBundle(Path("models"))
    except FileNotFoundError:
        bundle = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if bundle else "degraded", "models_loaded": bundle is not None}


@app.post("/predict")
def predict(payload: FeaturePayload) -> dict:
    if bundle is None:
        REQUESTS.labels(status="unavailable").inc()
        raise HTTPException(
            status_code=503,
            detail="Streaming models not found. Run python -m src.models.train_streaming_models.",
        )
    started = time.perf_counter()
    try:
        result = bundle.predict(payload.model_dump())
        REQUESTS.labels(status="ok").inc()
        RISK.labels(risk_level=result["risk_level"]).inc()
        return result
    except Exception:
        REQUESTS.labels(status="error").inc()
        raise
    finally:
        LATENCY.observe(time.perf_counter() - started)
