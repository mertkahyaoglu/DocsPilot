CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id               BIGSERIAL PRIMARY KEY,
    path             TEXT NOT NULL UNIQUE,
    title            TEXT NOT NULL,
    category         TEXT NOT NULL DEFAULT '',
    updated          DATE,
    content_hash     TEXT NOT NULL,
    embedding_model  TEXT NOT NULL,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- vector(1024) matches voyage-3.5 / the offline hash embedder. Changing the embedding
-- model or dimension means a new migration + full re-ingest.
CREATE TABLE chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INT NOT NULL,
    heading_path TEXT NOT NULL,
    content      TEXT NOT NULL,
    token_count  INT NOT NULL,
    embedding    vector(1024) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS
                   (to_tsvector('english', heading_path || ' ' || content)) STORED,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);

CREATE TABLE tickets (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    title            TEXT NOT NULL,
    description      TEXT NOT NULL,
    severity         TEXT NOT NULL CHECK (severity IN ('low', 'normal', 'high')),
    status           TEXT NOT NULL DEFAULT 'open',
    idempotency_key  TEXT NOT NULL UNIQUE,
    fingerprint      TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX tickets_user_idx ON tickets (user_id, created_at DESC);
