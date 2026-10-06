from collections import defaultdict, deque
from math import sqrt

SIGNALS = ["temperature", "vibration", "pressure", "rpm", "voltage", "load"]

def mean(v):
    return sum(v) / len(v)

def std(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return sqrt(sum((x-m)**2 for x in v)/(len(v)-1))

class OnlineFeatureStore:
    def __init__(self, max_window=24):
        self.max_window = max_window
        self.windows = defaultdict(lambda: deque(maxlen=max_window))

    def update(self, event):
        machine_id = str(event["machine_id"])
        self.windows[machine_id].append(event)
        h = list(self.windows[machine_id])
        if len(h) < 7:
            return None
        f = {
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
        for s in SIGNALS:
            v6 = [float(x[s]) for x in h[-6:]]
            v24 = [float(x[s]) for x in h[-24:]]
            f[f"{s}_mean_6h"] = mean(v6)
            f[f"{s}_std_24h"] = std(v24)
            f[f"{s}_delta_6h"] = float(event[s]) - float(h[-7][s])
        f["errors_24h"] = sum(int(x["error_count"]) for x in h[-24:])
        f["errors_6h"] = sum(int(x["error_count"]) for x in h[-6:])
        return f

    def machine_count(self):
        return len(self.windows)
