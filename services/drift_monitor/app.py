from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from prometheus_client import Gauge, start_http_server

from src.mlops.drift import compute_feature_drift

DRIFT_FRACTION = Gauge("model_feature_drift_fraction", "Fraction of monitored features in PSI drift")
DRIFTED_FEATURES = Gauge("model_drifted_features", "Number of features with PSI >= 0.20")


def load_recent_features(limit: int) -> pd.DataFrame:
    import psycopg
    url = os.getenv("DATABASE_URL", "postgresql://pmp:pmp@localhost:5432/pmp")
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT features FROM predictions ORDER BY created_at DESC LIMIT %s",(limit,))
            rows = cur.fetchall()
    return pd.DataFrame([row[0] for row in rows])


def main() -> None:
    start_http_server(int(os.getenv("METRICS_PORT", "9105")))
    interval = int(os.getenv("DRIFT_CHECK_SECONDS", "60"))
    sample_size = int(os.getenv("DRIFT_SAMPLE_SIZE", "1000"))
    reports = Path("reports/drift")
    reports.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            recent = load_recent_features(sample_size)
            if len(recent) >= 100:
                report = compute_feature_drift(recent)
                report["checked_at"] = datetime.now(timezone.utc).isoformat()
                DRIFT_FRACTION.set(report["drift_fraction"])
                DRIFTED_FEATURES.set(report["features_drifted"])
                (reports / "latest.json").write_text(json.dumps(report, indent=2))
                if report["requires_retraining_review"]:
                    Path("reports/retrain_request.json").write_text(json.dumps({
                        "requested_at": report["checked_at"],
                        "reason": "feature_drift",
                        "drift_fraction": report["drift_fraction"],
                        "status": "waiting_for_delayed_labels",
                    },indent=2))
                print(f"Drift: {report['features_drifted']}/{report['features_evaluated']} features; fraction={report['drift_fraction']:.1%}")
        except Exception as exc:
            print(f"Drift monitor error: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
