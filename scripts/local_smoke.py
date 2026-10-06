from src.models.streaming_inference import StreamingModelBundle
from src.streaming.online_features import OnlineFeatureStore
from src.streaming.simulation import FleetSimulator


def main() -> None:
    simulator = FleetSimulator(num_machines=5, seed=11)
    store = OnlineFeatureStore()
    models = StreamingModelBundle()
    emitted = 0

    for _ in range(250):
        event = simulator.next_event()
        features = store.update(event)
        if features is None:
            continue
        prediction = models.predict(features)
        emitted += 1
        if emitted <= 8:
            print(
                features["machine_id"],
                f"failure={prediction['failure_probability']:.3f}",
                f"anomaly={prediction['anomaly_score']:.3f}",
                f"rul={prediction['remaining_useful_life_hours']:.1f}h",
                prediction["risk_level"],
            )
    print(f"Generated {emitted} online predictions without Kafka/storage dependencies.")


if __name__ == "__main__":
    main()
