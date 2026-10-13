"""Real Postgres + pgvector. Skipped unless TEST_DATABASE_URL points at a server where
the test user may CREATE DATABASE (e.g. `docker compose up -d db`)."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
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


async def test_pending_store_shared_between_replicas_and_survives_restart(dsn: str) -> None:
    import time

    from anthropic.types import TextBlock, ToolUseBlock

    from docspilot.agent.loop import Agent, AgentConfig, AgentFinished, NeedsConfirmation
    from docspilot.agent.pending import PendingBusyError
    from docspilot.agent.pending_pg import PostgresPendingStore, decode, encode
    from docspilot.agent.tools import ToolContext, ToolRegistry, default_tools
    from docspilot.llm.fake import FakeLLM, Script
    from docspilot.llm.types import ToolCall
    from docspilot.tickets import InMemoryTicketStore

    pool_a, pool_b = await create_pool(dsn), await create_pool(dsn)
    try:
        replica_a, replica_b = PostgresPendingStore(pool_a), PostgresPendingStore(pool_b)
        args = {
            "title": "Stuck payout",
            "description": "In transit for 6 days.",
            "severity": "high",
        }

        # replica A runs the turn up to the human gate; blocks are real SDK objects
        agent_a = Agent(
            FakeLLM([Script(tool_calls=[ToolCall("w", "create_ticket", args)])]),
            ToolRegistry(default_tools()),
            replica_a,
            AgentConfig(model="m", system="s"),
        )
        tickets = InMemoryTicketStore()
        ctx = ToolContext("alice", tickets)
        events = [e async for e in agent_a.run([{"role": "user", "content": "open ticket"}], ctx)]
        needs = events[-1]
        assert isinstance(needs, NeedsConfirmation)

        # the assistant turn survives JSON with real SDK content blocks inside
        stored = await replica_b.claim(needs.pending_id, user_id="alice")
        assert stored is not None
        assert stored.calls[0].name == "create_ticket" and stored.awaiting == ["w"]
        stored.messages[-1]["content"] = [
            TextBlock(type="text", text="ok", citations=None),
            ToolUseBlock(type="tool_use", id="w", name="create_ticket", input=args),
        ]
        again = decode(encode(stored))
        assert again.messages[-1]["content"][1] == {
            "type": "tool_use",
            "id": "w",
            "name": "create_ticket",
            "input": args,
        }

        # claim exclusivity across replicas, owner scoping, and release persistence
        with pytest.raises(PendingBusyError):
            await replica_a.claim(needs.pending_id, user_id="alice")
        assert await replica_a.claim("nope" * 8, user_id="alice") is None
        await replica_b.release(stored)
        assert await replica_a.claim(needs.pending_id, user_id="mallory") is None

        # replica B (a different process) completes the approved action
        agent_b = Agent(
            FakeLLM([Script(text="Your ticket is open.")]),
            ToolRegistry(default_tools()),
            replica_b,
            AgentConfig(model="m", system="s"),
        )
        claimed = await replica_b.claim(needs.pending_id, user_id="alice")
        assert claimed is not None
        out = [e async for e in agent_b.resume(claimed, approve=True, ctx=ctx)]
        assert isinstance(out[-1], AgentFinished) and len(tickets._by_id) == 1
        assert await replica_a.claim(needs.pending_id, user_id="alice") is None  # deleted

        # expiry and per-user cap
        stale = claimed
        stale.id, stale.expires_at, stale.claimed_until = "stale-id-0123456789", time.time() - 1, 0
        await replica_a.put(stale)
        assert await replica_a.claim("stale-id-0123456789", user_id="alice") is None
    finally:
        await pool_a.close()
        await pool_b.close()


async def test_pending_release_and_outcome_survive_cancellation_and_use_the_db_clock(
    dsn: str,
) -> None:
    import asyncio
    import time

    from docspilot.agent.loop import Agent, AgentConfig, NeedsConfirmation
    from docspilot.agent.pending_pg import PostgresPendingStore
    from docspilot.agent.tools import ToolContext, ToolRegistry, default_tools
    from docspilot.llm.fake import FakeLLM, Script
    from docspilot.llm.types import ToolCall
    from docspilot.tickets import InMemoryTicketStore

    pool = await create_pool(dsn)
    try:
        store = PostgresPendingStore(pool, ttl_s=120)
        args = {
            "title": "Stuck payout",
            "description": "In transit for 6 days.",
            "severity": "high",
        }
        agent = Agent(
            FakeLLM(
                [Script(tool_calls=[ToolCall("w", "create_ticket", args)]), Script(text="done")]
            ),
            ToolRegistry(default_tools()),
            store,
            AgentConfig(model="m", system="s"),
            confirmation_ttl_s=120,
        )
        tickets = InMemoryTicketStore()
        ctx = ToolContext("alice", tickets)
        events = [e async for e in agent.run([{"role": "user", "content": "ticket"}], ctx)]
        needs = events[-1]
        assert isinstance(needs, NeedsConfirmation)

        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT expires_at - extract(epoch from now()) FROM pending_actions WHERE id = %s",
                (needs.pending_id,),
            )
            remaining = float((await cur.fetchone())[0])  # type: ignore[index]
        assert 100 < remaining <= 120  # DB clock + ttl, whatever the app clock says

        claimed = await store.claim(needs.pending_id, user_id="alice")
        assert claimed is not None

        executed = asyncio.Event()

        async def consume() -> None:
            # like the SSE wrapper: the generator is closed deterministically on teardown
            async with aclosing(agent.resume(claimed, approve=True, ctx=ctx)) as stream:
                async for event in stream:
                    if type(event).__name__ == "ToolEnd":
                        executed.set()
                        await asyncio.sleep(30)  # the client stalls, then goes away

        task = asyncio.create_task(consume())
        await asyncio.wait_for(executed.wait(), timeout=5)  # the write has happened
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.3)  # shielded bookkeeping finishes in the background

        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT claimed_until, payload->'completed'->'w' IS NOT NULL "
                "FROM pending_actions WHERE id = %s",
                (needs.pending_id,),
            )
            row = await cur.fetchone()
        assert row is not None
        claimed_until, outcome_saved = row
        assert claimed_until == 0 and outcome_saved  # released, and the write's outcome kept
        assert len(tickets._by_id) == 1

        # the retry replays the stored outcome to the model: no second execution
        again = await store.claim(needs.pending_id, user_id="alice")
        assert again is not None
        retry_agent = Agent(
            FakeLLM([Script(text="Your ticket is open.")]),
            ToolRegistry(default_tools()),
            store,
            AgentConfig(model="m", system="s"),
        )
        out = [e async for e in retry_agent.resume(again, approve=True, ctx=ctx)]
        assert not any(type(e).__name__ == "ToolStart" for e in out)
        assert type(out[-1]).__name__ == "AgentFinished" and len(tickets._by_id) == 1
        assert time.time() > 0
    finally:
        await pool.close()


async def test_vector_benchmark_cleans_up_its_schema_even_when_it_fails(dsn: str) -> None:
    import argparse

    import psycopg
    import pytest as _pytest

    from docspilot.bench import vector_bench

    args = argparse.Namespace(
        n=200, dim=3000, clusters=5, queries=5, k=5, seed=1, noise=1.0, dsn=dsn, out=None
    )
    # 3000 dims exceeds the IVFFlat limit (2000): the run fails midway, after creating data
    with _pytest.raises(Exception):  # noqa: B017
        await asyncio.to_thread(vector_bench.run, args)
    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        cur = await conn.execute("SELECT count(*) FROM pg_namespace WHERE nspname LIKE 'bench_%'")
        assert (await cur.fetchone()) == (0,)
