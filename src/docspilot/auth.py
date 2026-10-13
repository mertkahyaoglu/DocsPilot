"""Authentication behind a tiny interface so Phase 8 can swap in OIDC/JWT.

API keys are stored as SHA-256 digests in memory and compared in constant time over
*every* configured key (no early exit that leaks which prefix matched)."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str


class Authenticator(Protocol):
    @property
    def required(self) -> bool: ...
    def authenticate(self, token: str | None) -> Principal | None: ...


def _digest(secret: str) -> bytes:
    return hashlib.sha256(secret.encode()).digest()


class ApiKeyAuthenticator:
    def __init__(self, keys: dict[str, str]) -> None:
        """keys: user_id -> raw API key."""
        self._digests = {user: _digest(key) for user, key in keys.items()}

    @property
    def required(self) -> bool:
        return bool(self._digests)

    def authenticate(self, token: str | None) -> Principal | None:
        if not self._digests:
            return Principal("anonymous")  # auth not configured (dev only; prod validates)
        if not token:
            return None
        candidate = _digest(token)
        match: str | None = None
        for user, digest in self._digests.items():
            if hmac.compare_digest(candidate, digest):
                match = user
        return Principal(match) if match else None


def parse_api_keys(spec: str) -> dict[str, str]:
    """'alice:key-a,bob:key-b' -> {'alice': 'key-a', 'bob': 'key-b'}"""
    keys: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        user, sep, key = part.partition(":")
        if not sep or not user.strip() or len(key.strip()) < 16:
            raise ValueError("API_KEYS entries must be 'user:key' with keys of 16+ characters")
        keys[user.strip()] = key.strip()
    return keys
