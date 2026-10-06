from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, make_asgi_app

REQUESTS = Counter("backend_http_requests_total", "Backend requests", ["route"])
LATENCY = Histogram("backend_request_latency_seconds", "Backend request latency", ["route"])


def _redis():
    import redis
    return redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)


def _postgres():
    import psycopg
    return psycopg.connect(os.getenv("DATABASE_URL", "postgresql://pmp:pmp@localhost:5432/pmp"), autocommit=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = _redis()
    app.state.db = _postgres()
    yield
    app.state.db.close()
    app.state.redis.close()


app = FastAPI(title="Predictive Maintenance Backend", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("DASHBOARD_ORIGIN", "http://localhost:5173"), "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/machines")
def machines() -> list[dict]:
    REQUESTS.labels(route="machines").inc()
    raw = app.state.redis.hgetall("machines:latest")
    data = [json.loads(value) for value in raw.values()]
    priority = {"critical": 0, "warning": 1, "healthy": 2}
    data.sort(key=lambda x: (priority.get(x.get("risk_level"), 9), x.get("machine_id", "")))
    return data


@app.get("/api/machines/{machine_id}/history")
def history(machine_id: str, limit: int = Query(120, ge=1, le=2000)) -> list[dict]:
    REQUESTS.labels(route="history").inc()
    with app.state.db.cursor() as cur:
        cur.execute(
            """
            SELECT event_time, failure_probability, anomaly_score,
                   remaining_useful_life_hours, risk_level, features
            FROM predictions
            WHERE machine_id=%s
            ORDER BY event_time DESC
            LIMIT %s
            """,
            (machine_id, limit),
        )
        rows = cur.fetchall()
    return [
        {
            "timestamp": row[0].isoformat(),
            "failure_probability": float(row[1]),
            "anomaly_score": float(row[2]),
            "remaining_useful_life_hours": float(row[3]),
            "risk_level": row[4],
            "features": row[5],
        }
        for row in reversed(rows)
    ]


@app.get("/api/alerts")
def alerts(limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
    REQUESTS.labels(route="alerts").inc()
    with app.state.db.cursor() as cur:
        cur.execute(
            """
            SELECT id, event_time, machine_id, severity, message, payload
            FROM alerts ORDER BY created_at DESC LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {"id": row[0],"timestamp": row[1].isoformat(),"machine_id": row[2],"severity": row[3],"message": row[4],"payload": row[5]}
        for row in rows
    ]


@app.get("/api/summary")
def summary() -> dict:
    machines = [json.loads(value) for value in app.state.redis.hgetall("machines:latest").values()]
    counts = {"healthy": 0, "warning": 0, "critical": 0}
    for machine in machines:
        level = machine.get("risk_level", "healthy")
        counts[level] = counts.get(level, 0) + 1
    return {
        "machines_online": len(machines),
        **counts,
        "average_failure_probability": (
            sum(float(x.get("failure_probability", 0)) for x in machines) / len(machines)
            if machines else 0.0
        ),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    pubsub = app.state.redis.pubsub()
    pubsub.subscribe("predictions:live")
    try:
        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("data"):
                await websocket.send_text(message["data"])
            else:
                await asyncio.sleep(0.15)
    except WebSocketDisconnect:
        pass
    finally:
        pubsub.close()
