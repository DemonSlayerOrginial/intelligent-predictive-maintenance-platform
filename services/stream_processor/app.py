from __future__ import annotations

import json
import os
import time

import httpx
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from src.streaming.contracts import build_prediction_event
from src.streaming.online_features import OnlineFeatureStore

CONSUMED = Counter("stream_events_consumed_total", "Telemetry events consumed")
PREDICTED = Counter("stream_predictions_total", "Predictions emitted", ["risk_level"])
ERRORS = Counter("stream_processing_errors_total", "Stream processing errors", ["stage"])
MACHINES = Gauge("stream_active_machines", "Machines with rolling state")
INFERENCE_LATENCY = Histogram("stream_inference_latency_seconds", "Inference call latency")


def main() -> None:
    from confluent_kafka import Consumer, Producer

    broker = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    telemetry_topic = os.getenv("TELEMETRY_TOPIC", "machine-telemetry")
    predictions_topic = os.getenv("PREDICTIONS_TOPIC", "machine-predictions")
    inference_url = os.getenv("INFERENCE_URL", "http://localhost:8001/predict")
    metrics_port = int(os.getenv("METRICS_PORT", "9102"))
    start_http_server(metrics_port)

    consumer = Consumer(
        {
            "bootstrap.servers": broker,
            "group.id": "stream-feature-processor",
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        }
    )
    producer = Producer({"bootstrap.servers": broker, "client.id": "prediction-producer"})
    consumer.subscribe([telemetry_topic])
    store = OnlineFeatureStore(max_window=24)
    client = httpx.Client(timeout=5.0)

    print(f"Consuming {telemetry_topic}; predictions -> {predictions_topic}")
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                ERRORS.labels(stage="kafka_consume").inc()
                continue
            try:
                event = json.loads(msg.value())
                CONSUMED.inc()
                features = store.update(event)
                MACHINES.set(store.machine_count())
                if features is None:
                    continue

                started = time.perf_counter()
                response = client.post(inference_url, json=features)
                INFERENCE_LATENCY.observe(time.perf_counter() - started)
                response.raise_for_status()
                prediction = response.json()
                output = build_prediction_event(features, prediction)
                producer.produce(
                    predictions_topic,
                    key=features["machine_id"].encode(),
                    value=json.dumps(output).encode(),
                )
                producer.poll(0)
                PREDICTED.labels(risk_level=prediction["risk_level"]).inc()
            except httpx.HTTPError as exc:
                ERRORS.labels(stage="inference").inc()
                print(f"Inference error: {exc}")
                time.sleep(0.5)
            except Exception as exc:
                ERRORS.labels(stage="processing").inc()
                print(f"Processing error: {exc}")
    finally:
        consumer.close()
        client.close()
