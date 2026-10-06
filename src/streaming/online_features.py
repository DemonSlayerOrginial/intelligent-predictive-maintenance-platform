from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Deque

BASE_SIGNALS = ["temperature", "vibration", "pressure", "rpm", "voltage", "load"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


@dataclass
class MachineWindow:
    maxlen: int = 24
    events: Deque[dict] = field(default_factory=lambda: deque(maxlen=24))

    def __post_init__(self) -> None:
        if self.events.maxlen != self.maxlen:
            self.events = deque(self.events, maxlen=self.maxlen)


class OnlineFeatureStore:
    """Stateful rolling features for accelerated hourly telemetry.

    The simulator advances its logical clock by one hour per emitted event, even if
    events are published much faster in wall-clock time. Therefore the 6-event and
    24-event windows correspond to the same 6h/24h definitions used offline.
    """

    def __init__(self, max_window: int = 24) -> None:
        self.max_window = max_window
        self._machines: dict[str, MachineWindow] = defaultdict(
            lambda: MachineWindow(maxlen=max_window)
        )

    def update(self, event: dict) -> dict | None:
        machine_id = str(event["machine_id"])
        window = self._machines[machine_id]
        window.events.append(event)
        history = list(window.events)
        if len(history) < 7:
            return None

        features: dict[str, float | int | str] = {
            "timestamp": event["timestamp"],
            "machine_id": machine_id,
            "temperature": float(event["temperature"]),
            "vibration": float(event["vibration"]),
            "pressure": float(event["pressure"]),
            "rpm": float(event["rpm"]),
            "voltage": float(event["voltage"]),
            "load": float(event["load"]),
            "error_count": int(event["error_count"]),
            "hours_since_maintenance": int(event["hours_since_maintenance"]),
        }

        for signal in BASE_SIGNALS:
            values_6 = [float(x[signal]) for x in history[-6:]]
            values_24 = [float(x[signal]) for x in history[-24:]]
            features[f"{signal}_mean_6h"] = _mean(values_6)
            features[f"{signal}_std_24h"] = _std(values_24)
            prior = history[-7]
            features[f"{signal}_delta_6h"] = float(event[signal]) - float(prior[signal])

        features["errors_24h"] = sum(int(x["error_count"]) for x in history[-24:])
        features["errors_6h"] = sum(int(x["error_count"]) for x in history[-6:])
        return features

    def machine_count(self) -> int:
        return len(self._machines)
