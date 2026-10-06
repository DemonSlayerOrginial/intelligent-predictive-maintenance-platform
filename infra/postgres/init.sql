CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    event_time TIMESTAMPTZ NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL,
    machine_id TEXT NOT NULL,
    failure_probability DOUBLE PRECISION NOT NULL,
    predicted_failure BOOLEAN NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    remaining_useful_life_hours DOUBLE PRECISION NOT NULL,
    risk_level TEXT NOT NULL CHECK (risk_level IN ('healthy','warning','critical')),
    features JSONB NOT NULL,
    model_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_machine_time
    ON predictions(machine_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_time
    ON predictions(risk_level, event_time DESC);

CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    event_time TIMESTAMPTZ NOT NULL,
    machine_id TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('warning','critical')),
    message TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC);
