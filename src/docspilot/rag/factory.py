"""Assemble storage + embedder + reranker + retriever from settings."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from docspilot.config import Settings
from docspilot.logging import get_logger
from docspilot.rag.embeddings import Embedder, HashEmbedder, VoyageEmbedder
from docspilot.rag.ingest import ingest_corpus
from docspilot.rag.rerank import NoopReranker, Reranker, VoyageReranker
from docspilot.rag.retrieve import RetrievalConfig, Retriever
from docspilot.rag.store import ChunkStore, InMemoryChunkStore
from docspilot.tickets import InMemoryTicketStore, TicketStore

log = get_logger("rag")


@dataclass(slots=True)
class RagStack:
    store: ChunkStore
    embedder: Embedder
    reranker: Reranker
    retriever: Retriever
    tickets: TicketStore
    pool: Any | None  # psycopg AsyncConnectionPool when Postgres-backed


def _key(settings: Settings) -> str:
    # Settings validation guarantees presence when voyage is selected.
    if settings.voyage_api_key is None:  # pragma: no cover
        raise RuntimeError("VOYAGE_API_KEY missing")
    return settings.voyage_api_key.get_secret_value()


@asynccontextmanager
async def build_rag(settings: Settings, *, ingest_on_start: bool = True) -> AsyncIterator[RagStack]:
    # An ExitStack closes whatever was created even if a later step fails.
    async with AsyncExitStack() as stack:
        embedder: Embedder
        if settings.resolved_embedder == "voyage":
            voyage = VoyageEmbedder(_key(settings), model=settings.voyage_embed_model)
            stack.push_async_callback(voyage.aclose)
            embedder = voyage
        else:
            log.warning("using_hash_embedder", note="offline stand-in; retrieval quality is poor")
            embedder = HashEmbedder()

        reranker: Reranker = NoopReranker()
        if settings.resolved_rerank == "voyage":
            voyage_rr = VoyageReranker(_key(settings), model=settings.voyage_rerank_model)
            stack.push_async_callback(voyage_rr.aclose)
            reranker = voyage_rr

        pool = None
        store: ChunkStore
        tickets: TicketStore
        if settings.database_url:
            from docspilot.db import apply_migrations, create_pool
            from docspilot.rag.pg_store import PostgresChunkStore
            from docspilot.tickets_pg import PostgresTicketStore

            await apply_migrations(settings.database_url, settings.migrations_dir)
            pool = await create_pool(settings.database_url)
            stack.push_async_callback(pool.close)
            store = PostgresChunkStore(pool)
            tickets = PostgresTicketStore(pool)
        else:
            log.warning("using_in_memory_store", note="data is lost on restart")
            store = InMemoryChunkStore()
            tickets = InMemoryTicketStore()

        retriever = Retriever(
            store,
            embedder,
            reranker,
            RetrievalConfig(
                k_final=settings.retrieval_k,
                n_candidates=settings.retrieval_candidates,
                min_rerank_score=settings.rerank_min_score,
            ),
        )
        if (
            ingest_on_start
            and settings.auto_ingest
            and await asyncio.to_thread(settings.corpus_dir.is_dir)
        ):
            # Always incremental: cheap when nothing changed, and it is what heals a
            # partially failed earlier ingest or an embedding-model switch.
            await ingest_corpus(settings.corpus_dir, store, embedder)
        yield RagStack(store, embedder, reranker, retriever, tickets, pool)
