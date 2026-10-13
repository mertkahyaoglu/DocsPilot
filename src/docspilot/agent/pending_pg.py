"""Postgres-backed PendingStore (same contract as the in-memory one).

Claiming is a single conditional UPDATE, so two replicas racing for the same
confirmation cannot both win. State changes made while a claim is held (resolved
outcomes) are written back on `release`, which is what makes a retry after a failed
follow-up call replay instead of re-executing."""

from __future__ import annotations

import dataclasses
import secrets
import time
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from docspilot.agent.pending import CLAIM_TIMEOUT_S, PendingAction, PendingBusyError
from docspilot.agent.tools import Source, ToolOutcome
from docspilot.llm.types import ToolCall, Usage

_NOW = "extract(epoch from now())"


def to_jsonable(obj: Any) -> Any:
    """SDK content blocks (pydantic) and our dataclasses -> plain JSON structures."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in obj]
    return obj


def encode(p: PendingAction) -> dict[str, Any]:
    data = to_jsonable(p)
    data.pop("claimed_until", None)  # a column of its own, DB-clock based
    return data  # type: ignore[no-any-return]


def decode(data: dict[str, Any]) -> PendingAction:
    def source(d: dict[str, Any]) -> Source:
        return Source(**d)

    return PendingAction(
        id=data["id"],
        user_id=data["user_id"],
        messages=data["messages"],
        calls=[ToolCall(**c) for c in data["calls"]],
        completed={
            k: ToolOutcome(v["content"], v["is_error"], tuple(source(s) for s in v["sources"]))
            for k, v in data["completed"].items()
        },
        awaiting=list(data["awaiting"]),
        steps_left=data["steps_left"],
        expires_at=data["expires_at"],
        usage=Usage(**data["usage"]),
        steps=data["steps"],
        retrieved={k: source(v) for k, v in data["retrieved"].items()},
        agent=data.get("agent", ""),
    )


class PostgresPendingStore:
    def __init__(
        self, pool: AsyncConnectionPool, *, ttl_s: float = 600.0, max_per_user: int = 10
    ) -> None:
        self._pool = pool
        self._ttl_s = ttl_s
        self._max_per_user = max_per_user

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def new_id(self) -> str:
        return secrets.token_urlsafe(24)

    async def put(self, pending: PendingAction) -> None:
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute(
                f"DELETE FROM pending_actions WHERE expires_at < {_NOW}"  # noqa: S608
            )
            # TTL is applied on the database clock: replicas with skewed clocks must
            # not shift when a confirmation expires.
            ttl = max(pending.expires_at - time.time(), 0.0)
            await conn.execute(
                "INSERT INTO pending_actions (id, user_id, payload, expires_at) "
                "VALUES (%s, %s, %s, extract(epoch from now()) + %s)",
                (pending.id, pending.user_id, Jsonb(encode(pending)), ttl),
            )
            # per-user cap: keep the newest N (a noisy caller evicts only itself)
            await conn.execute(
                "DELETE FROM pending_actions WHERE id IN ("
                "SELECT id FROM pending_actions WHERE user_id = %s "
                "ORDER BY created_at DESC OFFSET %s)",
                (pending.user_id, self._max_per_user),
            )

    async def claim(self, pending_id: str, *, user_id: str) -> PendingAction | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"UPDATE pending_actions SET claimed_until = {_NOW} + %s "  # noqa: S608
                f"WHERE id = %s AND user_id = %s AND expires_at > {_NOW} "
                f"AND claimed_until <= {_NOW} RETURNING payload",
                (CLAIM_TIMEOUT_S, pending_id, user_id),
            )
            row = await cur.fetchone()
            if row is not None:
                return decode(row[0])
            cur = await conn.execute(
                f"SELECT 1 FROM pending_actions WHERE id = %s AND user_id = %s "  # noqa: S608
                f"AND expires_at > {_NOW}",
                (pending_id, user_id),
            )
            if await cur.fetchone() is not None:
                raise PendingBusyError(pending_id)  # exists, owned, live: someone holds it
            return None

    async def checkpoint(self, pending: PendingAction) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE pending_actions SET payload = %s WHERE id = %s",
                (Jsonb(encode(pending)), pending.id),
            )

    async def release(self, pending: PendingAction) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE pending_actions SET payload = %s, claimed_until = 0 WHERE id = %s",
                (Jsonb(encode(pending)), pending.id),
            )

    async def delete(self, pending_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute("DELETE FROM pending_actions WHERE id = %s", (pending_id,))
