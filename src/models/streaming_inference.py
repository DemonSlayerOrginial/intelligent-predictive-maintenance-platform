from __future__ import annotations

import json
from math import exp
from pathlib import Path

import joblib
import pandas as pd


class StreamingModelBundle:
    def __init__(self, model_dir: Path = Path("models")) -> None:
        self.model_dir = model_dir
        self.failure_model = joblib.load(model_dir / "failure_model.joblib")
        self.anomaly_model = joblib.load(model_dir / "anomaly_model.joblib")
        self.rul_model = joblib.load(model_dir / "rul_model.joblib")
        self.metadata = json.loads((model_dir / "streaming_metadata.json").read_text())
        self.features = list(self.metadata["feature_columns"])

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + exp(-value))
        exp_v = exp(value)
        return exp_v / (1.0 + exp_v)

    def predict(self, feature_payload: dict) -> dict:
        frame = pd.DataFrame([{name: feature_payload[name] for name in self.features}])
        failure_probability = float(self.failure_model.predict_proba(frame)[0, 1])
        threshold = float(self.metadata["failure"].get("threshold", 0.35))

        raw_anomaly = float(-self.anomaly_model.decision_function(frame)[0])
        median = float(self.metadata["anomaly"]["score_median"])
        iqr = float(self.metadata["anomaly"]["score_iqr"])
        anomaly_score = self._sigmoid((raw_anomaly - median) / max(iqr, 1e-6))
        anomaly_detected = bool(self.anomaly_model.predict(frame)[0] == -1)

        rul = float(max(0.0, self.rul_model.predict(frame)[0]))
        predicted_failure = failure_probability >= threshold

        if failure_probability >= max(0.75, threshold) or (anomaly_detected and anomaly_score >= 0.98) or rul <= 24:
            risk = "critical"
        elif predicted_failure or anomaly_detected or rul <= 72:
            risk = "warning"
        else:
            risk = "healthy"

        return {
            "failure_probability": round(failure_probability, 5),
            "decision_threshold": threshold,
            "predicted_failure": predicted_failure,
            "anomaly_score": round(anomaly_score, 5),
            "anomaly_detected": anomaly_detected,
            "remaining_useful_life_hours": round(rul, 2),
            "risk_level": risk,
            "model_versions": {"bundle": self.metadata.get("registry_version", "local")},
        }
