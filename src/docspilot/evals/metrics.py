from __future__ import annotations

import math
from collections.abc import Sequence


def unique_in_order(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def recall_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Fraction of expected documents found in the top-k (documents, deduped)."""
    if not expected:
        raise ValueError("recall is undefined without expected documents")
    top = set(unique_in_order(ranked)[:k])
    return sum(1 for e in set(expected) if e in top) / len(set(expected))


def hit_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> float:
    top = set(unique_in_order(ranked)[:k])
    return 1.0 if any(e in top for e in expected) else 0.0


def reciprocal_rank(ranked: Sequence[str], expected: Sequence[str]) -> float:
    exp = set(expected)
    for rank, path in enumerate(unique_in_order(ranked), start=1):
        if path in exp:
            return 1.0 / rank
    return 0.0


def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a pass rate. With ~50 cases a single flip moves the rate 2pp:
    never read a 3pp difference between two runs as signal without this."""
    if total == 0:
        return (0.0, 1.0)
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
