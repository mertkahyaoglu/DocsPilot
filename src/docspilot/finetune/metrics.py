from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Report:
    n: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusions: dict[str, int]


def evaluate(y_true: Sequence[str], y_pred: Sequence[str]) -> Report:
    labels = sorted(set(y_true) | set(y_pred))
    per: dict[str, dict[str, float]] = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": float(tp + fn)}
    supported = [v["f1"] for v in per.values() if v["support"] > 0]
    wrong = Counter(f"{t}->{p}" for t, p in zip(y_true, y_pred, strict=True) if t != p)
    return Report(
        n=len(y_true),
        accuracy=sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / max(len(y_true), 1),
        macro_f1=sum(supported) / max(len(supported), 1),
        per_class=per,
        confusions=dict(wrong.most_common(8)),
    )
