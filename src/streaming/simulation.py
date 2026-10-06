from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import numpy as np

@dataclass
class MachineState:
    machine_id: str
    wear: float
    hours_since_maintenance: int

class FleetSimulator:
    def __init__(self, num_machines=50, seed=42):
        self.rng = np.random.default_rng(seed)
        self.states = [MachineState(f"M{i:03d}", float(self.rng.uniform(.05,.25)), int(self.rng.integers(0,240))) for i in range(num_machines)]
        self.logical_time = datetime(2026,1,1,tzinfo=timezone.utc)
        self.i = 0

    def next_event(self):
        s = self.states[self.i]
        load = float(np.clip(self.rng.normal(.62,.16),.15,1))
        s.hours_since_maintenance += 1
        s.wear = max(0, s.wear + .00035 + .0008*load + self.rng.normal(0,.00015))
        temp = 58 + 31*s.wear + 9*load + self.rng.normal(0,1.8)
        vib = 1.1 + 5.5*s.wear + 2*load + self.rng.normal(0,.35)
        pressure = 48 - 8.5*s.wear - 2*load + self.rng.normal(0,.9)
        rpm = 1400 + 1250*load - 170*s.wear + self.rng.normal(0,45)
        voltage = 230 - 5*s.wear + self.rng.normal(0,1.5)
        errors = int(self.rng.poisson(max(.02,s.wear*1.8)))
        p = 1/(1+np.exp(-(-9.4+8.2*s.wear+.055*max(temp-78,0)+.5*max(vib-4.5,0)+.13*errors)))
        failed = int(self.rng.random() < p)
        event = dict(timestamp=self.logical_time.isoformat(), machine_id=s.machine_id, temperature=round(float(temp),3), vibration=round(float(vib),3), pressure=round(float(pressure),3), rpm=round(float(rpm),3), voltage=round(float(voltage),3), load=round(load,4), error_count=errors, hours_since_maintenance=s.hours_since_maintenance, observed_failure=failed)
        if failed:
            s.wear = float(self.rng.uniform(.05,.16)); s.hours_since_maintenance = 0
        self.i = (self.i+1)%len(self.states)
        if self.i == 0: self.logical_time += timedelta(hours=1)
        return event
