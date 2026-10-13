from __future__ import annotations

import re

from pgvector import Vector
from psycopg_pool import AsyncConnectionPool

from docspilot.rag.store import keyword_terms
from docspilot.rag.types import ChunkDraft, RawDoc, SearchHit, StoredChunk

_COLS = "c.id, d.path, d.title, d.category, c.heading_path, c.content, d.updated"


def _chunk(row: tuple[object, ...]) -> StoredChunk:
    return StoredChunk(
        id=row[0],  # type: ignore[arg-type]
        path=row[1],  # type: ignore[arg-type]
        title=row[2],  # type: ignore[arg-type]
        category=row[3],  # type: ignore[arg-type]
        heading_path=row[4],  # type: ignore[arg-type]
        text=row[5],  # type: ignore[arg-type]
        updated=row[6],  # type: ignore[arg-type]
    )


class PostgresChunkStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def document_state(self) -> dict[str, tuple[str, str]]:
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT path, content_hash, embedding_model FROM documents")
            return {r[0]: (r[1], r[2]) for r in await cur.fetchall()}

    async def replace_document(
        self,
        doc: RawDoc,
        chunks: list[ChunkDraft],
        embeddings: list[list[float]],
        embedding_model: str,
    ) -> None:
        async with self._pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                INSERT INTO documents
                    (path, title, category, updated, content_hash, embedding_model)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (path) DO UPDATE SET
                    title = EXCLUDED.title, category = EXCLUDED.category,
                    updated = EXCLUDED.updated, content_hash = EXCLUDED.content_hash,
                    embedding_model = EXCLUDED.embedding_model, ingested_at = now()
                RETURNING id
                """,
                (doc.path, doc.title, doc.category, doc.updated, doc.content_hash, embedding_model),
            )
            row = await cur.fetchone()
            assert row is not None
            doc_id = row[0]
            await conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            async with conn.cursor() as c:
                await c.executemany(
                    "INSERT INTO chunks (document_id, ordinal, heading_path, content, token_count,"
                    " embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (doc_id, d.ordinal, d.heading_path, d.text, d.tokens, Vector(e))
                        for d, e in zip(chunks, embeddings, strict=True)
                    ],
                )

    async def delete_documents(self, paths: list[str]) -> None:
        if paths:
            async with self._pool.connection() as conn:
                await conn.execute("DELETE FROM documents WHERE path = ANY(%s)", (paths,))

    async def search_vector(self, embedding: list[float], k: int) -> list[SearchHit]:
        # Inner ORDER BY ... LIMIT on the bare table lets the HNSW index be used.
        sql = f"""
            SELECT {_COLS}, c.dist
            FROM (SELECT id, document_id, heading_path, content,
                         embedding <=> %s AS dist
                  FROM chunks ORDER BY embedding <=> %s LIMIT %s) c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.dist
        """  # noqa: S608 - columns are a constant
        vec = Vector(embedding)
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, (vec, vec, k))
            rows = await cur.fetchall()
        return [SearchHit(chunk=_chunk(r), score=1.0 - float(r[7])) for r in rows]

    async def search_keyword(self, query: str, k: int) -> list[SearchHit]:
        terms = [t for t in keyword_terms(query) if re.fullmatch(r"[a-z0-9]+", t)]
        if not terms:
            return []
        # OR semantics: natural-language questions rarely contain *all* doc terms.
        tsquery = " | ".join(terms)
        sql = f"""
            SELECT {_COLS}, ts_rank_cd(c.tsv, q) AS rank
            FROM chunks c
            JOIN documents d ON d.id = c.document_id,
                 to_tsquery('english', %s) q
            WHERE c.tsv @@ q
            ORDER BY rank DESC
            LIMIT %s
        """  # noqa: S608
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, (tsquery, k))
            rows = await cur.fetchall()
        return [SearchHit(chunk=_chunk(r), score=float(r[7])) for r in rows]

    async def count_chunks(self) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) FROM chunks")
            row = await cur.fetchone()
            return int(row[0]) if row else 0
