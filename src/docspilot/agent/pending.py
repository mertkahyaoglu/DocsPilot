"""Pending human confirmations. The server keeps the exact arguments it will run:
the client only says approve/reject, so approved arguments cannot be tampered with.

Lifecycle: awaiting -> (claim) -> resolved -> (turn finishes) -> deleted.
`resolved` keeps the executed outcomes, so if the follow-up model call fails and the
client retries the same confirmation, results are replayed and the side effect is not
executed twice."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from docspilot.agent.tools import Source, ToolOutcome
from docspilot.llm.types import ToolCall, Usage

CLAIM_TIMEOUT_S = 60.0


class PendingBusyError(Exception):
    """The confirmation is currently being processed by another request."""


@dataclass(slots=True)
class PendingAction:
    id: str
    user_id: str
    #: Conversation including the assistant turn that holds the tool_use blocks
    #: (raw content blocks, replayed unchanged).
    messages: list[dict[str, Any]]
    #: Every tool call of that turn, in the model's order.
    calls: list[ToolCall]
    #: Outcomes of calls that already ran (read tools, then the approved/declined writes).
    completed: dict[str, ToolOutcome]
    #: ids of calls waiting for a human decision. Empty once resolved.
    awaiting: list[str]
    steps_left: int
    expires_at: float
    usage: Usage = field(default_factory=Usage)  # carried so `done` reports the whole turn
    steps: int = 0
    retrieved: dict[str, Source] = field(default_factory=dict)
    agent: str = ""  # name of the (specialist) agent that paused
    claimed_until: float = 0.0


class PendingStore(Protocol):
    def new_id(self) -> str: ...
    async def put(self, pending: PendingAction) -> None: ...
    async def claim(self, pending_id: str, *, user_id: str) -> PendingAction | None:
        """Take exclusive processing rights. None if unknown, expired or not the owner's;
        raises PendingBusyError if another request holds it."""
        ...

    async def checkpoint(self, pending: PendingAction) -> None:
        """Persist progress (resolved outcomes) while keeping the claim."""
        ...

    async def release(self, pending: PendingAction) -> None:
        """Give the claim back (keeps the entry, e.g. after a failed follow-up call)."""
        ...

    async def delete(self, pending_id: str) -> None: ...


class InMemoryPendingStore:
    """Single-process store. Multi-replica deployments need a shared store (see Phase 7)."""

    def __init__(
        self, *, ttl_s: float = 600.0, max_items: int = 1000, max_per_user: int = 10
    ) -> None:
        self._items: dict[str, PendingAction] = {}
        self._ttl_s = ttl_s
        self._max = max_items
        self._max_per_user = max_per_user

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def new_id(self) -> str:
        return secrets.token_urlsafe(24)

    async def put(self, pending: PendingAction) -> None:
        now = time.time()
        self._items = {k: v for k, v in self._items.items() if v.expires_at > now}
        # Per-user cap first: one noisy caller must not evict everyone else's confirmations.
        mine = [k for k, v in self._items.items() if v.user_id == pending.user_id]
        while len(mine) >= self._max_per_user:
            self._items.pop(mine.pop(0))
        while len(self._items) >= self._max:
            self._items.pop(next(iter(self._items)))
        self._items[pending.id] = pending

    async def claim(self, pending_id: str, *, user_id: str) -> PendingAction | None:
        item = self._items.get(pending_id)
        now = time.time()
        if item is None or item.user_id != user_id or item.expires_at <= now:
            return None  # wrong owner is indistinguishable from unknown
        if item.claimed_until > now:
            raise PendingBusyError(pending_id)
        item.claimed_until = now + CLAIM_TIMEOUT_S
        return item

    async def checkpoint(self, pending: PendingAction) -> None:
        return None  # the object *is* the stored state

    async def release(self, pending: PendingAction) -> None:
        pending.claimed_until = 0.0

    async def delete(self, pending_id: str) -> None:
        self._items.pop(pending_id, None)
