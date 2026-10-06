from src.streaming.simulation import FleetSimulator


def test_simulator_emits_required_fields():
    sim = FleetSimulator(num_machines=2, seed=7)
    event = sim.next_event()
    required = {
        "timestamp", "machine_id", "temperature", "vibration", "pressure", "rpm",
        "voltage", "load", "error_count", "hours_since_maintenance", "observed_failure"
    }
    assert required.issubset(event)
    assert 0.0 <= event["load"] <= 1.0
