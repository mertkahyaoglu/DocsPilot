CREATE TABLE feedback (
    id          BIGSERIAL PRIMARY KEY,
    request_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- one vote per user per answer: changing your mind updates, double-clicks don't stack
    UNIQUE (request_id, user_id)
);
CREATE INDEX feedback_negative_idx ON feedback (created_at DESC) WHERE rating = -1;
