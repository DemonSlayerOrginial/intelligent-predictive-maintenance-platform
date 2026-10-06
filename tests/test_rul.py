import pandas as pd

from src.models.train_streaming_models import add_rul_target


def test_rul_counts_down_to_failure():
    df = pd.DataFrame(
        {
            "machine_id": ["M1"] * 5,
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
            "failure": [0, 0, 0, 1, 0],
        }
    )
    out = add_rul_target(df)
    assert out.loc[0, "remaining_useful_life_hours"] == 3
    assert out.loc[2, "remaining_useful_life_hours"] == 1
    assert out.loc[3, "remaining_useful_life_hours"] == 0
    assert pd.isna(out.loc[4, "remaining_useful_life_hours"])
