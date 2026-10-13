"""Postgres plumbing: connection pool and forward-only SQL migrations."""

from __future__ import annotations

import asyncio
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from docspilot.logging import get_logger

log = get_logger("db")
_LOCK_KEY = 727_001  # pg_advisory_lock id: serialises concurrent migrators (rolling deploys)


async def apply_migrations(dsn: str, directory: Path) -> list[str]:
    """Apply `NNN_name.sql` files not yet recorded. Each file runs in one transaction."""
    applied: list[str] = []
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        try:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            cur = await conn.execute("SELECT version FROM schema_migrations")
            done = {row[0] for row in await cur.fetchall()}
            files = await asyncio.to_thread(
                lambda: [
                    (p.name, p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.sql"))
                ]
            )
            for name, sql in files:
                if name in done:
                    continue
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s)", (name,)
                    )
                applied.append(name)
                log.info("migration_applied", version=name)
        finally:
            await conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
    return applied


async def create_pool(dsn: str, *, min_size: int = 1, max_size: int = 10) -> AsyncConnectionPool:
    # Migrations must have created the `vector` type before connections register it.
    pool = AsyncConnectionPool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        open=False,
        configure=register_vector_async,
        timeout=10,
    )
    await pool.open(wait=True)
    return pool
