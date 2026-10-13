"""Real Postgres + pgvector. Skipped unless TEST_DATABASE_URL points at a server where
the test user may CREATE DATABASE (e.g. `docker compose up -d db`)."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import psycopg
import pytest

from docspilot.db import apply_migrations, create_pool
from docspilot.rag.chunking import chunk_markdown, parse_markdown_doc
from docspilot.rag.embeddings import HashEmbedder
from docspilot.rag.ingest import ingest_corpus
from docspilot.rag.pg_store import PostgresChunkStore
from docspilot.rag.rerank import NoopReranker
from docspilot.rag.retrieve import Retriever
from docspilot.tickets import IdempotencyConflictError
from docspilot.tickets_pg import PostgresTicketStore

ADMIN_DSN = os.environ.get("TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(ADMIN_DSN is None, reason="TEST_DATABASE_URL not set"),
]


@pytest.fixture
async def dsn() -> AsyncIterator[str]:
    assert ADMIN_DSN
    name = f"docspilot_test_{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
        await admin.execute(f'CREATE DATABASE "{name}"')  # type: ignore[arg-type]
    base, _, _ = ADMIN_DSN.rpartition("/")
    test_dsn = f"{base}/{name}"
    await apply_migrations(test_dsn, ROOT / "migrations")
    yield test_dsn
    async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')  # type: ignore[arg-type]


async def test_migrations_are_idempotent(dsn: str) -> None:
    assert await apply_migrations(dsn, ROOT / "migrations") == []


async def test_ingest_search_and_prune_against_pgvector(dsn: str, tmp_path: Path) -> None:
    pool = await create_pool(dsn)
    try:
        store, emb = PostgresChunkStore(pool), HashEmbedder()
        report = await ingest_corpus(ROOT / "corpus", store, emb)
        assert report.chunks_written > 50 and await store.count_chunks() == report.chunks_written

        again = await ingest_corpus(ROOT / "corpus", store, emb)
        assert again.added_or_updated == [] and again.unchanged == len(report.added_or_updated)

        retriever = Retriever(store, emb, NoopReranker())
        result = await retriever.search("how long do I have to request a refund")
        paths = [h.chunk.path for h in result.hits]
        assert "policies/refunds.md" in paths
        top = result.hits[0]
        assert top.vector_rank or top.keyword_rank  # provenance recorded

        # keyword path: exact token (webhook) with OR semantics across many words
        kw = await store.search_keyword("why was my webhook endpoint disabled after days", 5)
        assert kw and kw[0].chunk.path.startswith(("api/webhooks", "troubleshooting/"))
        assert await store.search_keyword("the of and", 5) == []

        # exact dotted identifier outranks documents that merely share generic words
        ident = await store.search_keyword("invoice.paid event", 3)
        assert ident[0].chunk.path == "api/webhooks.md"

        # pruning removes documents and (via cascade) their chunks
        empty = tmp_path / "empty"
        empty.mkdir()
        pruned = await ingest_corpus(empty, store, emb, allow_prune_all=True)
        assert len(pruned.deleted) == len(report.added_or_updated)
        assert await store.count_chunks() == 0
    finally:
        await pool.close()


async def test_replace_document_swaps_chunks_atomically(dsn: str) -> None:
    pool = await create_pool(dsn)
    try:
        store, emb = PostgresChunkStore(pool), HashEmbedder()
        v1 = parse_markdown_doc("a.md", "# A\n\n## S\n\n" + "alpha beta gamma. " * 40)
        v2 = parse_markdown_doc("a.md", "# A\n\n## S\n\n" + "delta epsilon zeta. " * 40)
        for doc in (v1, v2):
            chunks = chunk_markdown(doc)
            vecs = await emb.embed([c.embed_text for c in chunks], "document")
            await store.replace_document(doc, chunks, vecs, emb.name)
        texts = [h.chunk.text for h in await store.search_keyword("delta", 10)]
        assert texts and all("alpha" not in t for t in texts)
        assert await store.search_keyword("alpha", 10) == []
    finally:
        await pool.close()


async def test_ticket_store_idempotency_and_ownership(dsn: str) -> None:
    pool = await create_pool(dsn)
    try:
        tickets = PostgresTicketStore(pool)
        kwargs = dict(user_id="u1", title="t", description="d" * 12, severity="normal")
        a = await tickets.create(**kwargs, idempotency_key="k1")  # type: ignore[arg-type]
        b = await tickets.create(**kwargs, idempotency_key="k1")  # type: ignore[arg-type]
        assert a.id == b.id
        with pytest.raises(IdempotencyConflictError):
            await tickets.create(
                **{**kwargs, "title": "other"},  # type: ignore[arg-type]
                idempotency_key="k1",
            )
        assert (await tickets.get(a.id, user_id="u1")) is not None
        assert (await tickets.get(a.id, user_id="someone-else")) is None
    finally:
        await pool.close()


async def test_feedback_upsert_and_nul_safe(dsn: str) -> None:
    from docspilot.feedback import Feedback, PostgresFeedbackStore

    pool = await create_pool(dsn)
    try:
        store = PostgresFeedbackStore(pool)
        a = await store.add(Feedback("req-abcdef12", "u1", "up", ""))
        b = await store.add(Feedback("req-abcdef12", "u1", "down", "wrong plan"))
        assert a == b  # same row updated
        c = await store.add(Feedback("req-abcdef12", "u2", "up", ""))
        assert c != a
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT rating, comment FROM feedback WHERE user_id = 'u1'")
            assert await cur.fetchone() == (-1, "wrong plan")
    finally:
        await pool.close()


async def test_spend_store_add_is_atomic_and_returns_the_new_total(dsn: str) -> None:
    import asyncio

    from docspilot.budget import Budget, BudgetExceeded, PostgresSpendStore

    pool = await create_pool(dsn)
    try:
        store = PostgresSpendStore(pool)
        assert await store.spent_today("u1") == 0.0
        assert await store.add("u1", 0.25) == pytest.approx(0.25)
        assert await store.add("u1", 0.5) == pytest.approx(0.75)
        assert await store.add("u1", -0.25) == pytest.approx(0.5)

        # reservation race against the real database: only two 0.4 reservations fit under 1.0
        budget = Budget(store, user_daily_usd=1.0, global_daily_usd=0, reserve_usd=0.4)
        results = await asyncio.gather(
            *(budget.reserve("racer") for _ in range(6)), return_exceptions=True
        )
        assert sum(not isinstance(r, BaseException) for r in results) == 2
        assert all(isinstance(r, BudgetExceeded) for r in results if isinstance(r, BaseException))
        assert await store.spent_today("racer") == pytest.approx(0.8)
    finally:
        await pool.close()
