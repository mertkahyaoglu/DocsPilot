-- Per-user daily spend (USD, estimated from token usage). Shared by all replicas so a
-- budget cap cannot be dodged by hitting different instances.
CREATE TABLE usage_daily (
    user_id   TEXT NOT NULL,
    day       DATE NOT NULL,
    cost_usd  DOUBLE PRECISION NOT NULL DEFAULT 0,
    requests  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
