CREATE TABLE feedback (
    id          BIGSERIAL PRIMARY KEY,
    request_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX feedback_request_idx ON feedback (request_id);
CREATE INDEX feedback_negative_idx ON feedback (created_at DESC) WHERE rating = -1;
