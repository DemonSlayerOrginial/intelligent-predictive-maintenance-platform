from __future__ import annotations

import json
import os
import time


def connect_postgres():
    import psycopg
    return psycopg.connect(os.getenv("DATABASE_URL", "postgresql://pmp:pmp@localhost:5432/pmp"), autocommit=True)


def connect_redis():
    import redis
    return redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)


def persist_prediction(conn, redis_client, event: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions (
                event_time, processed_at, machine_id, failure_probability,
                predicted_failure, anomaly_score, remaining_useful_life_hours,
                risk_level, features, model_versions
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
            """,
            (
                event["timestamp"], event["processed_at"], event["machine_id"],
                event["failure_probability"], event["predicted_failure"],
                event["anomaly_score"], event["remaining_useful_life_hours"],
                event["risk_level"], json.dumps(event["features"]),
                json.dumps(event.get("model_versions", {})),
            ),
        )
        if event["risk_level"] in {"warning", "critical"}:
            cur.execute(
                """
                INSERT INTO alerts (event_time, machine_id, severity, message, payload)
                VALUES (%s,%s,%s,%s,%s::jsonb)
                """,
                (
                    event["timestamp"], event["machine_id"], event["risk_level"],
                    f"{event['machine_id']} entered {event['risk_level']} risk state",
                    json.dumps(event),
                ),
            )
    redis_client.hset("machines:latest", event["machine_id"], json.dumps(event))
    redis_client.publish("predictions:live", json.dumps(event))


def main() -> None:
    from confluent_kafka import Consumer
    broker = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("PREDICTIONS_TOPIC", "machine-predictions")
    consumer = Consumer({"bootstrap.servers": broker,"group.id": "prediction-persistence","auto.offset.reset": "latest"})
    consumer.subscribe([topic])
    while True:
        try:
            conn = connect_postgres()
            redis_client = connect_redis()
            break
        except Exception as exc:
            print(f"Waiting for stores: {exc}")
            time.sleep(2)
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            try:
                persist_prediction(conn, redis_client, json.loads(msg.value()))
            except Exception as exc:
                print(f"Persistence error: {exc}")
                time.sleep(0.25)
    finally:
        consumer.close()
        conn.close()
