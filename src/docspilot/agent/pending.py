"""Pending human confirmations. The server keeps the exact arguments it will run:
the client only says approve/reject, so approved arguments cannot be tampered with."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any, Protocol

from docspilot.agent.tools import ToolOutcome
from docspilot.llm.types import ToolCall


@dataclass(slots=True)
class PendingAction:
    id: str
    user_id: str
    #: Conversation including the assistant turn that holds the tool_use blocks
    #: (raw content blocks, replayed unchanged).
    messages: list[dict[str, Any]]
    #: Every tool call of that turn, in the model's order.
    calls: list[ToolCall]
    #: Outcomes of calls that already ran (read-only tools).
    completed: dict[str, ToolOutcome]
    #: ids of calls waiting for a human decision.
    awaiting: list[str]
    steps_left: int
    expires_at: float


class PendingStore(Protocol):
    def new_id(self) -> str: ...
    async def put(self, pending: PendingAction) -> None: ...
    async def pop(self, pending_id: str, *, user_id: str) -> PendingAction | None:
        """Atomically take (single use). None if unknown, expired or not the owner's."""
        ...


class InMemoryPendingStore:
    """Single-process store. Multi-replica deployments need a shared store (see Phase 7)."""

    def __init__(self, *, ttl_s: float = 600.0, max_items: int = 1000) -> None:
        self._items: dict[str, PendingAction] = {}
        self._ttl_s = ttl_s
        self._max = max_items

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def new_id(self) -> str:
        return secrets.token_urlsafe(24)

    async def put(self, pending: PendingAction) -> None:
        now = time.time()
        self._items = {k: v for k, v in self._items.items() if v.expires_at > now}
        while len(self._items) >= self._max:  # evict oldest
            self._items.pop(next(iter(self._items)))
        self._items[pending.id] = pending

    async def pop(self, pending_id: str, *, user_id: str) -> PendingAction | None:
        item = self._items.get(pending_id)
        if item is None or item.user_id != user_id:
            return None  # wrong owner is indistinguishable from unknown, and does not consume it
        del self._items[pending_id]
        return item if item.expires_at > time.time() else None
