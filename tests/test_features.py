import pandas as pd
from src.features.build_features import build_features


def test_features_use_past_and_current_values():
    rows = []
    for hour in range(30):
        rows.append({
            "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour),
            "machine_id": "M001",
            "temperature": 60 + hour,
            "vibration": 2 + hour * 0.1,
            "pressure": 50 - hour * 0.1,
            "rpm": 1500 + hour,
            "voltage": 230,
            "load": 0.5,
            "error_count": 0,
            "hours_since_maintenance": hour,
            "failure": 0,
            "failure_within_24h": 0,
        })
    out = build_features(pd.DataFrame(rows))
    assert not out.empty
    assert "temperature_mean_6h" in out.columns
    assert "errors_24h" in out.columns
