from __future__ import annotations

import json
import os
import time

from src.streaming.simulation import FleetSimulator


def main() -> None:
    from confluent_kafka import Producer

    broker = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("TELEMETRY_TOPIC", "machine-telemetry")
    machines = int(os.getenv("SIMULATOR_MACHINES", "50"))
    events_per_second = float(os.getenv("SIMULATOR_EVENTS_PER_SECOND", "25"))
    simulator = FleetSimulator(num_machines=machines, seed=int(os.getenv("SIMULATOR_SEED", "42")))
    producer = Producer({"bootstrap.servers": broker, "client.id": "predictive-maintenance-simulator"})

    delay = 1.0 / max(events_per_second, 0.1)
    print(f"Publishing {events_per_second:g} events/s for {machines} machines to {topic}")
    while True:
        event = simulator.next_event()
        producer.produce(
            topic,
            key=event["machine_id"].encode(),
            value=json.dumps(event).encode(),
        )
        producer.poll(0)
        time.sleep(delay)


if __name__ == "__main__":
    main()
