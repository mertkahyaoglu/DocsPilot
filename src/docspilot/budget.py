"""Spend caps. Request-rate limits do not bound cost: one user can send few, huge,
expensive requests. A budget is denominated in the thing you actually pay for."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Protocol

from psycopg_pool import AsyncConnectionPool

GLOBAL_KEY = "*"


class BudgetExceeded(Exception):
    def __init__(self, scope: str) -> None:
        super().__init__(scope)
        self.scope = scope  # "user" | "global"


class SpendStore(Protocol):
    async def spent_today(self, user_id: str) -> float: ...
    async def add(self, user_id: str, usd: float) -> None: ...


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class InMemorySpendStore:
    def __init__(self, today: Callable[[], str] = _today) -> None:
        self._today = today
        self._rows: dict[tuple[str, str], float] = {}

    async def spent_today(self, user_id: str) -> float:
        return self._rows.get((user_id, self._today()), 0.0)

    async def add(self, user_id: str, usd: float) -> None:
        for key in (user_id, GLOBAL_KEY):
            k = (key, self._today())
            self._rows[k] = self._rows.get(k, 0.0) + usd


class PostgresSpendStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def spent_today(self, user_id: str) -> float:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT cost_usd FROM usage_daily "
                "WHERE user_id = %s AND day = (now() AT TIME ZONE 'utc')::date",
                (user_id,),
            )
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0

    async def add(self, user_id: str, usd: float) -> None:
        async with self._pool.connection() as conn, conn.transaction():
            for key in (user_id, GLOBAL_KEY):
                await conn.execute(
                    "INSERT INTO usage_daily (user_id, day, cost_usd, requests) "
                    "VALUES (%s, (now() AT TIME ZONE 'utc')::date, %s, 1) "
                    "ON CONFLICT (user_id, day) DO UPDATE SET "
                    "cost_usd = usage_daily.cost_usd + EXCLUDED.cost_usd, "
                    "requests = usage_daily.requests + 1",
                    (key, usd),
                )


class Budget:
    """0 disables a cap. Checked *before* work starts; recorded after (a single request
    can overshoot a little: caps are a guard rail, not an exact meter)."""

    def __init__(
        self, store: SpendStore, *, user_daily_usd: float, global_daily_usd: float
    ) -> None:
        self._store = store
        self._user = user_daily_usd
        self._global = global_daily_usd

    async def check(self, user_id: str) -> None:
        if self._global > 0 and await self._store.spent_today(GLOBAL_KEY) >= self._global:
            raise BudgetExceeded("global")
        if self._user > 0 and await self._store.spent_today(user_id) >= self._user:
            raise BudgetExceeded("user")

    async def record(self, user_id: str, usd: float) -> None:
        if usd > 0:
            await self._store.add(user_id, usd)


class CostMeter:
    """Per-request accumulator, filled by InstrumentedLLM for *every* model call the
    request causes (agent steps, routing, triage...)."""

    def __init__(self) -> None:
        self.usd = 0.0


current_meter: ContextVar[CostMeter | None] = ContextVar("current_meter", default=None)
