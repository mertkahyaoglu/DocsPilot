-- Human-confirmation state shared by all replicas and surviving deploys: a user can
-- approve on a different instance than the one that proposed the action.
CREATE TABLE pending_actions (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    payload        JSONB NOT NULL,
    expires_at     DOUBLE PRECISION NOT NULL,          -- unix seconds
    claimed_until  DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX pending_actions_user_idx ON pending_actions (user_id, created_at DESC);
CREATE INDEX pending_actions_expiry_idx ON pending_actions (expires_at);
