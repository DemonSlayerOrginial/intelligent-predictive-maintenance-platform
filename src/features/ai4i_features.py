from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.ai4i import BASE_FEATURES, ID_COLUMNS, LEAKAGE_COLUMNS, TARGET

ENGINEERED_FEATURES = [
    "temperature_delta_k",
    "mechanical_power_w",
    "tool_wear_torque",
    "torque_per_1000_rpm",
]


def build_ai4i_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temperature_delta_k"] = out["process_temperature_k"] - out["air_temperature_k"]
    out["mechanical_power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * (2.0 * np.pi / 60.0)
    out["tool_wear_torque"] = out["tool_wear_min"] * out["torque_nm"]
    out["torque_per_1000_rpm"] = out["torque_nm"] / (out["rotational_speed_rpm"] / 1000.0)
    return out


def get_ai4i_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = BASE_FEATURES + ENGINEERED_FEATURES
    return [column for column in candidates if column in df.columns]


def assert_no_target_leakage(feature_columns: list[str]) -> None:
    forbidden = set(LEAKAGE_COLUMNS + ID_COLUMNS + [TARGET])
    leaked = forbidden.intersection(feature_columns)
    if leaked:
        raise ValueError(f"Target leakage detected in features: {sorted(leaked)}")
