import pandas as pd

from src.data.ai4i import LEAKAGE_COLUMNS, TARGET
from src.features.ai4i_features import assert_no_target_leakage, build_ai4i_features, get_ai4i_feature_columns
from src.models.ai4i_metrics import select_fbeta_threshold


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "udi": [1, 2, 3, 4],
        "product_id": ["L1", "M1", "H1", "L2"],
        "type": ["L", "M", "H", "L"],
        "air_temperature_k": [298.0, 299.0, 300.0, 301.0],
        "process_temperature_k": [308.0, 309.5, 310.0, 311.5],
        "rotational_speed_rpm": [1500, 1400, 1800, 1200],
        "torque_nm": [40.0, 45.0, 30.0, 60.0],
        "tool_wear_min": [10, 50, 100, 200],
        TARGET: [0, 0, 0, 1],
        "twf": [0, 0, 0, 1],"hdf": [0, 0, 0, 0],"pwf": [0, 0, 0, 0],"osf": [0, 0, 0, 1],"rnf": [0, 0, 0, 0],
    })


def test_engineered_features_and_no_leakage():
    df = build_ai4i_features(sample_df())
    assert "temperature_delta_k" in df.columns
    assert "mechanical_power_w" in df.columns
    assert "tool_wear_torque" in df.columns
    features = get_ai4i_feature_columns(df)
    assert_no_target_leakage(features)
    assert TARGET not in features
    assert not set(LEAKAGE_COLUMNS).intersection(features)


def test_fbeta_threshold_is_probability():
    threshold = select_fbeta_threshold(
        y_true=pd.Series([0, 0, 1, 1]).to_numpy(),
        probabilities=pd.Series([0.05, 0.25, 0.55, 0.90]).to_numpy(),
        beta=2.0,
    )
    assert 0.0 <= threshold <= 1.0
