from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np


@dataclass
class MachineState:
    machine_id: str
    wear: float
    hours_since_maintenance: int
    machine_bias: float


class FleetSimulator:
    """Stateful industrial telemetry simulator for live Kafka demos."""

    def __init__(self, num_machines: int = 50, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.states: list[MachineState] = []
        for idx in range(num_machines):
            self.states.append(
                MachineState(
                    machine_id=f"M{idx:03d}",
                    wear=float(self.rng.uniform(0.05, 0.25)),
                    hours_since_maintenance=int(self.rng.integers(0, 240)),
                    machine_bias=float(self.rng.normal(0, 1)),
                )
            )
        self.logical_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._idx = 0

    def next_event(self) -> dict:
        state = self.states[self._idx]
        event_time = self.logical_time
        self._idx = (self._idx + 1) % len(self.states)

        load = float(np.clip(self.rng.normal(0.62, 0.16), 0.15, 1.0))
        state.hours_since_maintenance += 1
        state.wear += 0.00035 + 0.0008 * load + float(self.rng.normal(0, 0.00015))
        state.wear = max(state.wear, 0.0)

        temperature = 58 + 31 * state.wear + 9 * load + state.machine_bias + self.rng.normal(0, 1.8)
        vibration = 1.1 + 5.5 * state.wear + 2.0 * load + self.rng.normal(0, 0.35)
        pressure = 48 - 8.5 * state.wear - 2.0 * load + self.rng.normal(0, 0.9)
        rpm = 1400 + 1250 * load - 170 * state.wear + self.rng.normal(0, 45)
        voltage = 230 - 5.0 * state.wear + self.rng.normal(0, 1.5)
        error_count = int(self.rng.poisson(max(0.02, state.wear * 1.8)))

        risk_logit = (
            -9.4
            + 8.2 * state.wear
            + 0.055 * max(float(temperature) - 78, 0)
            + 0.50 * max(float(vibration) - 4.5, 0)
            + 0.13 * error_count
        )
        failure_prob = float(1 / (1 + np.exp(-risk_logit)))
        failed = int(self.rng.random() < failure_prob)

        event = {
            "timestamp": event_time.isoformat(),
            "machine_id": state.machine_id,
            "temperature": round(float(temperature), 3),
            "vibration": round(float(vibration), 3),
            "pressure": round(float(pressure), 3),
            "rpm": round(float(rpm), 3),
            "voltage": round(float(voltage), 3),
            "load": round(load, 4),
            "error_count": error_count,
            "hours_since_maintenance": state.hours_since_maintenance,
            "observed_failure": failed,
        }

        if failed:
            state.wear = float(self.rng.uniform(0.05, 0.16))
            state.hours_since_maintenance = 0
        elif state.hours_since_maintenance > int(self.rng.integers(550, 900)) and self.rng.random() < 0.006:
            state.wear = max(0.04, state.wear * float(self.rng.uniform(0.25, 0.45)))
            state.hours_since_maintenance = 0

        if self._idx == 0:
            self.logical_time += timedelta(hours=1)
        return event
