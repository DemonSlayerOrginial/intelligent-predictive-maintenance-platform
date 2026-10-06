from __future__ import annotations
import pandas as pd

BASE_SIGNALS = ["temperature", "vibration", "pressure", "rpm", "voltage", "load"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe rolling features using only current/past telemetry."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    grouped = data.groupby("machine_id", group_keys=False)
    for col in BASE_SIGNALS:
        data[f"{col}_mean_6h"] = grouped[col].transform(lambda s: s.rolling(6, min_periods=3).mean())
        data[f"{col}_std_24h"] = grouped[col].transform(lambda s: s.rolling(24, min_periods=6).std())
        data[f"{col}_delta_6h"] = grouped[col].transform(lambda s: s - s.shift(6))
    data["errors_24h"] = grouped["error_count"].transform(lambda s: s.rolling(24, min_periods=1).sum())
    data["errors_6h"] = grouped["error_count"].transform(lambda s: s.rolling(6, min_periods=1).sum())
    feature_cols = [c for c in data.columns if c.endswith(("_mean_6h", "_std_24h", "_delta_6h"))]
    return data.dropna(subset=feature_cols).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "machine_id", "failure", "failure_within_24h"}
    return [c for c in df.columns if c not in excluded]
