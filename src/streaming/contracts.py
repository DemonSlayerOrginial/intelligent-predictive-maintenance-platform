from __future__ import annotations

from datetime import datetime, timezone


def build_prediction_event(features: dict, prediction: dict) -> dict:
    return {
        "timestamp": features["timestamp"],
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "machine_id": features["machine_id"],
        "failure_probability": float(prediction["failure_probability"]),
        "predicted_failure": bool(prediction["predicted_failure"]),
        "anomaly_score": float(prediction["anomaly_score"]),
        "anomaly_detected": bool(prediction.get("anomaly_detected", False)),
        "remaining_useful_life_hours": float(prediction["remaining_useful_life_hours"]),
        "risk_level": str(prediction["risk_level"]),
        "features": {k: v for k, v in features.items() if k not in {"timestamp", "machine_id"}},
        "model_versions": prediction.get("model_versions", {}),
    }
