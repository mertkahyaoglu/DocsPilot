from __future__ import annotations

import secrets

from psycopg_pool import AsyncConnectionPool

from docspilot.tickets import IdempotencyConflictError, Severity, Ticket


def _fingerprint(user_id: str, title: str, description: str, severity: str) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps([user_id, title, description, severity]).encode()).hexdigest()


_COLS = "id, user_id, title, description, severity, status, created_at"


class PostgresTicketStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create(
        self,
        *,
        user_id: str,
        title: str,
        description: str,
        severity: Severity,
        idempotency_key: str,
    ) -> Ticket:
        fp = _fingerprint(user_id, title, description, severity)
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                INSERT INTO tickets (id, user_id, title, description, severity,
                                     idempotency_key, fingerprint)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING {_COLS}
                """,  # noqa: S608
                (
                    f"TCK-{secrets.token_hex(4)}",
                    user_id,
                    title,
                    description,
                    severity,
                    idempotency_key,
                    fp,
                ),
            )
            row = await cur.fetchone()
            if row is None:  # key already used: return the original unless args differ
                cur = await conn.execute(
                    f"SELECT {_COLS}, fingerprint FROM tickets WHERE idempotency_key = %s",  # noqa: S608
                    (idempotency_key,),
                )
                existing = await cur.fetchone()
                assert existing is not None
                if existing[-1] != fp:
                    raise IdempotencyConflictError(idempotency_key)
                row = existing
        return _ticket(row)

    async def get(self, ticket_id: str, *, user_id: str) -> Ticket | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLS} FROM tickets WHERE id = %s AND user_id = %s",  # noqa: S608
                (ticket_id, user_id),
            )
            row = await cur.fetchone()
        return _ticket(row) if row else None


def _ticket(row: tuple[object, ...]) -> Ticket:
    return Ticket(
        id=row[0],
        user_id=row[1],
        title=row[2],
        description=row[3],
        severity=row[4],
        status=row[5],
        created_at=row[6],
    )
