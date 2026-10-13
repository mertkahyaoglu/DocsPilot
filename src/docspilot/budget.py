"""Spend caps. Request-rate limits do not bound cost: one user can send few, huge,
expensive requests. A budget is denominated in the thing you actually pay for.

Reserve-then-settle: at admission a worst-case estimate is *added* to the day's total
(atomically per key) and the caps are verified against the result, so concurrent
requests cannot all pass a check-then-act race. When the request ends the reservation is
replaced by the actual usage. A client that disconnects before any usage was observed
keeps its reservation (the provider still bills those tokens); a request that failed
before reaching the model is refunded."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Protocol

from psycopg_pool import AsyncConnectionPool

GLOBAL_KEY = "*"
RESERVED_IDS = frozenset({GLOBAL_KEY, "anonymous"})


class BudgetExceeded(Exception):
    def __init__(self, scope: str) -> None:
        super().__init__(scope)
        self.scope = scope  # "user" | "global"


class SpendStore(Protocol):
    async def spent_today(self, key: str) -> float: ...
    async def add(self, key: str, usd: float) -> float:
        """Atomically add (may be negative) and return the new total for today."""
        ...


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class InMemorySpendStore:
    def __init__(self, today: Callable[[], str] = _today) -> None:
        self._today = today
        self._rows: dict[tuple[str, str], float] = {}

    async def spent_today(self, key: str) -> float:
        return self._rows.get((key, self._today()), 0.0)

    async def add(self, key: str, usd: float) -> float:
        k = (key, self._today())
        self._rows[k] = self._rows.get(k, 0.0) + usd
        return self._rows[k]


class PostgresSpendStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def spent_today(self, key: str) -> float:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT cost_usd FROM usage_daily "
                "WHERE user_id = %s AND day = (now() AT TIME ZONE 'utc')::date",
                (key,),
            )
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0

    async def add(self, key: str, usd: float) -> float:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO usage_daily (user_id, day, cost_usd, requests) "
                "VALUES (%s, (now() AT TIME ZONE 'utc')::date, %s, 1) "
                "ON CONFLICT (user_id, day) DO UPDATE SET "
                "cost_usd = usage_daily.cost_usd + EXCLUDED.cost_usd, "
                "requests = usage_daily.requests "
                "+ CASE WHEN EXCLUDED.cost_usd > 0 THEN 1 ELSE 0 END "
                "RETURNING cost_usd",
                (key, usd),
            )
            row = await cur.fetchone()
            assert row is not None
            return float(row[0])


class Budget:
    """0 disables a cap (usage is still recorded)."""

    def __init__(
        self,
        store: SpendStore,
        *,
        user_daily_usd: float,
        global_daily_usd: float,
        reserve_usd: float = 0.25,
    ) -> None:
        self._store = store
        self._user = user_daily_usd
        self._global = global_daily_usd
        self._reserve = reserve_usd

    async def reserve(self, user_id: str) -> float:
        """Admit or raise BudgetExceeded. Returns the amount reserved (0 if caps are off)."""
        if self._user <= 0 and self._global <= 0:
            return 0.0
        r = self._reserve
        user_total = await self._store.add(user_id, r)
        if self._user > 0 and user_total > self._user:
            await self._store.add(user_id, -r)
            raise BudgetExceeded("user")
        global_total = await self._store.add(GLOBAL_KEY, r)
        if self._global > 0 and global_total > self._global:
            await self._store.add(GLOBAL_KEY, -r)
            await self._store.add(user_id, -r)
            raise BudgetExceeded("global")
        return r

    async def settle(
        self, user_id: str, reserved: float, actual: float, *, refund: bool = False
    ) -> None:
        """Replace the reservation with what was really spent."""
        if refund:
            delta = -reserved
        elif actual > 0:
            delta = actual - reserved
        else:
            delta = 0.0  # no usage observed (e.g. disconnect): keep the reservation
        if delta:
            await self._store.add(user_id, delta)
            await self._store.add(GLOBAL_KEY, delta)


class CostMeter:
    """Per-request accumulator, filled by InstrumentedLLM for *every* model call the
    request causes (agent steps, routing, triage...)."""

    def __init__(self) -> None:
        self.usd = 0.0


current_meter: ContextVar[CostMeter | None] = ContextVar("current_meter", default=None)
