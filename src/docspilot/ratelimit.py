"""Token-bucket rate limiter (per key). In-memory: limits are per replica. For a global
limit put it at the gateway/CDN or back this with Redis; the interface is one method."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(
        self,
        *,
        per_minute: float,
        burst: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self._rate = per_minute / 60.0
        self._capacity = burst if burst is not None else max(per_minute, 1.0)
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._max_keys = max_keys

    @property
    def enabled(self) -> bool:
        return self._rate > 0

    def acquire(self, key: str, cost: float = 1.0) -> float:
        """Returns 0.0 if allowed, else the seconds to wait before retrying."""
        if not self.enabled:
            return 0.0
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self._max_keys:  # bound memory under key-spraying
                self._buckets.pop(next(iter(self._buckets)))
            bucket = self._buckets[key] = _Bucket(self._capacity, now)
        bucket.tokens = min(self._capacity, bucket.tokens + (now - bucket.updated) * self._rate)
        bucket.updated = now
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return 0.0
        return (cost - bucket.tokens) / self._rate
