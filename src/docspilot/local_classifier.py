"""Local intent classifiers produced by `docspilot.finetune`.

Two artefact kinds, chosen by `meta.json`:
- `tfidf`: TF-IDF + linear model exported as plain JSON and evaluated in pure Python
  (microseconds, no ML runtime in production, no pickle);
- `transformers`: a (LoRA-merged) sequence-classification checkpoint; needs torch +
  transformers at runtime and is imported lazily."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from docspilot.intents import INTENTS, Intent

_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def ngrams(tokens: list[str], n_max: int = 2) -> list[str]:
    out = list(tokens)
    for n in range(2, n_max + 1):
        out.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return out


class TfidfLinear:
    """Inference side of a TF-IDF (sublinear tf, L2-normalised) + linear classifier."""

    def __init__(
        self,
        vocab: dict[str, int],
        idf: list[float],
        coef: list[list[float]],
        intercept: list[float],
        labels: list[str],
        ngram_max: int = 2,
    ) -> None:
        self.vocab, self.idf, self.coef = vocab, idf, coef
        self.intercept, self.labels, self.ngram_max = intercept, labels, ngram_max

    def scores(self, text: str) -> list[float]:
        counts = Counter(g for g in ngrams(tokenize(text), self.ngram_max) if g in self.vocab)
        weights = {
            self.vocab[g]: (1.0 + math.log(c)) * self.idf[self.vocab[g]] for g, c in counts.items()
        }
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        out = []
        for row, b in zip(self.coef, self.intercept, strict=True):
            out.append(sum(row[i] * w / norm for i, w in weights.items()) + b)
        return out

    def predict(self, text: str) -> str:
        s = self.scores(text)
        return self.labels[max(range(len(s)), key=s.__getitem__)]

    def to_json(self) -> dict[str, Any]:
        return {
            "vocab": self.vocab,
            "idf": self.idf,
            "coef": self.coef,
            "intercept": self.intercept,
            "labels": self.labels,
            "ngram_max": self.ngram_max,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TfidfLinear:
        return cls(**data)


class LocalIntentClassifier:
    def __init__(self, name: str, predict: Any) -> None:
        self.name = name
        self._predict = predict  # str -> label (blocking)

    @classmethod
    def load(cls, directory: Path) -> LocalIntentClassifier:
        meta = json.loads((directory / "meta.json").read_text())
        kind = meta["kind"]
        if kind == "tfidf":
            model = TfidfLinear.from_json(json.loads((directory / "model.json").read_text()))
            return cls(f"local:tfidf:{directory.name}", model.predict)
        if kind == "transformers":
            return cls(f"local:transformers:{directory.name}", _load_transformers(directory))
        raise ValueError(f"unknown classifier kind {kind!r}")

    async def classify(self, message: str) -> Intent:
        # Inference is CPU-bound: keep it off the event loop.
        label = await asyncio.to_thread(self._predict, message)
        return label if label in INTENTS else "other"


def _load_transformers(directory: Path) -> Any:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    def predict(text: str) -> str:
        enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0]
        return str(id2label[int(logits.argmax())])

    return predict
