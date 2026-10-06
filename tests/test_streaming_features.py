from datetime import datetime, timedelta, timezone

from src.streaming.online_features import OnlineFeatureStore


def event(hour: int, value: float) -> dict:
    return {
        "timestamp": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hour)).isoformat(),
        "machine_id": "M001",
        "temperature": value,
        "vibration": value / 10,
        "pressure": 50 - value / 20,
        "rpm": 1500 + value,
        "voltage": 230 - value / 100,
        "load": 0.5,
        "error_count": hour % 2,
        "hours_since_maintenance": hour,
    }


def test_online_features_warmup_and_delta():
    store = OnlineFeatureStore()
    output = None
    for hour in range(7):
        output = store.update(event(hour, 60 + hour))
    assert output is not None
    assert output["temperature_mean_6h"] == 63.5
    assert output["temperature_delta_6h"] == 6.0
    assert output["errors_6h"] == 3
    assert store.machine_count() == 1
