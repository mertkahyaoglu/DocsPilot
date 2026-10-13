"""Support tickets: the one write-side effect the assistant can trigger."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["low", "normal", "high"]


class Ticket(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    user_id: str
    title: str
    description: str
    severity: Severity
    status: Literal["open", "in_progress", "resolved"] = "open"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TicketStore(Protocol):
    async def create(
        self,
        *,
        user_id: str,
        title: str,
        description: str,
        severity: Severity,
        idempotency_key: str,
    ) -> Ticket:
        """Create a ticket. Calling again with the same key returns the same ticket."""
        ...

    async def get(self, ticket_id: str, *, user_id: str) -> Ticket | None:
        """Only returns tickets owned by `user_id`."""
        ...


class InMemoryTicketStore:
    def __init__(self) -> None:
        self._by_id: dict[str, Ticket] = {}
        self._by_key: dict[str, str] = {}

    async def create(
        self,
        *,
        user_id: str,
        title: str,
        description: str,
        severity: Severity,
        idempotency_key: str,
    ) -> Ticket:
        existing = self._by_key.get(idempotency_key)
        if existing is not None:
            return self._by_id[existing]
        ticket = Ticket(
            id=f"TCK-{secrets.token_hex(4)}",
            user_id=user_id,
            title=title,
            description=description,
            severity=severity,
        )
        self._by_id[ticket.id] = ticket
        self._by_key[idempotency_key] = ticket.id
        return ticket

    async def get(self, ticket_id: str, *, user_id: str) -> Ticket | None:
        ticket = self._by_id.get(ticket_id)
        return ticket if ticket is not None and ticket.user_id == user_id else None
