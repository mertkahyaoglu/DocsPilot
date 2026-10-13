"""Exact-match response cache for first-turn, read-only questions.

What makes this safe enough to ship:
- key includes the prompt hash, model, effort and a *corpus version*: change any of
  them and old answers are unreachable (no stale-answer-after-deploy);
- only single-turn conversations (no personal context), and only turns that used
  nothing but `search_docs` (no ticket lookups, no writes, nothing user-specific);
- only complete answers (`end_turn`): never cache truncated/refused/errored turns;
- bounded: TTL + LRU, so staleness and memory have hard limits.
Semantic ("similar question") caching is deliberately not done: a near-miss that
returns the wrong policy confidently is worse than paying for the call."""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

_WS = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    text: str
    citations: tuple[dict[str, str], ...]
    stop_reason: str
    model: str


def normalize(question: str) -> str:
    return _WS.sub(" ", question.strip().lower()).rstrip("?!. ")


def cache_key(
    question: str, *, prompt_sha: str, model: str, effort: str | None, corpus_version: str
) -> str:
    raw = "\x1f".join([normalize(question), prompt_sha, model, effort or "", corpus_version])
    return hashlib.sha256(raw.encode()).hexdigest()


class ResponseCache:
    def __init__(
        self, *, ttl_s: float, max_items: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._ttl = ttl_s
        self._max = max_items
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, CachedAnswer]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def get(self, key: str) -> CachedAnswer | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if self._clock() - stored_at > self._ttl:
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: str, value: CachedAnswer) -> None:
        if not self.enabled:
            return
        self._items[key] = (self._clock(), value)
        self._items.move_to_end(key)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


def corpus_version(state: dict[str, tuple[str, str]]) -> str:
    """Stable fingerprint of what is indexed: (path, content hash, embedding model)."""
    h = hashlib.sha256()
    for path in sorted(state):
        content_hash, model = state[path]
        h.update(f"{path}\x1f{content_hash}\x1f{model}\x1e".encode())
    return h.hexdigest()[:16]
