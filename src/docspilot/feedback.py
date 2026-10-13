"""Online feedback. Thumbs are the cheapest production signal: each 👎 is a candidate
golden case (export it, label it, add it to evals/golden.jsonl)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from psycopg_pool import AsyncConnectionPool

Rating = Literal["up", "down"]


@dataclass(frozen=True, slots=True)
class Feedback:
    request_id: str
    user_id: str
    rating: Rating
    comment: str = ""


class FeedbackStore(Protocol):
    async def add(self, item: Feedback) -> int: ...


@dataclass
class InMemoryFeedbackStore:
    items: list[Feedback] = field(default_factory=list)

    async def add(self, item: Feedback) -> int:
        self.items.append(item)
        return len(self.items)


class PostgresFeedbackStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def add(self, item: Feedback) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO feedback (request_id, user_id, rating, comment) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (item.request_id, item.user_id, 1 if item.rating == "up" else -1, item.comment),
            )
            row = await cur.fetchone()
            assert row is not None
            return int(row[0])
