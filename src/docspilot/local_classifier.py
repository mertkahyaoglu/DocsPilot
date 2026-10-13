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

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        """(label, softmax probability). Probability 0.0 when no known feature matched:
        a document of unseen words has no evidence, whatever the intercept says."""
        feats = [g for g in ngrams(tokenize(text), self.ngram_max) if g in self.vocab]
        if not feats:
            return self.labels[0], 0.0
        s = self.scores(text)
        top = max(s)
        exps = [math.exp(v - top) for v in s]
        best = max(range(len(s)), key=s.__getitem__)
        return self.labels[best], exps[best] / sum(exps)

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


#: Below this the model is guessing: say "other" (which routes to the default specialist).
MIN_CONFIDENCE = 0.5
#: Misrouting a question to the ticketing specialist is a hard failure for the user
#: (it cannot search the docs), so that label needs stronger evidence.
MIN_TICKET_CONFIDENCE = 0.75


class LocalIntentClassifier:
    def __init__(
        self,
        name: str,
        predict: Any,
        *,
        min_confidence: float = MIN_CONFIDENCE,
        min_ticket_confidence: float = MIN_TICKET_CONFIDENCE,
    ) -> None:
        self.name = name
        self._predict = predict  # str -> (label, confidence)   (blocking)
        self._min = min_confidence
        self._min_ticket = min_ticket_confidence

    @classmethod
    def load(cls, directory: Path) -> LocalIntentClassifier:
        meta = json.loads((directory / "meta.json").read_text())
        kind = meta["kind"]
        if kind == "tfidf":
            model = TfidfLinear.from_json(json.loads((directory / "model.json").read_text()))
            return cls(f"local:tfidf:{directory.name}", model.predict_with_confidence)
        if kind == "transformers":
            return cls(f"local:transformers:{directory.name}", _load_transformers(directory))
        raise ValueError(f"unknown classifier kind {kind!r}")

    def raw_label(self, message: str) -> str:
        """The model's argmax with no confidence handling (for offline evaluation)."""
        return str(self._predict(message)[0])

    async def classify(self, message: str) -> Intent:
        # Inference is CPU-bound: keep it off the event loop.
        label, confidence = await asyncio.to_thread(self._predict, message)
        known = next((i for i in INTENTS if i == label), None)
        if known is None or confidence < self._min:
            return "other"
        if known == "ticket_request" and confidence < self._min_ticket:
            return "question"
        return known


def _load_transformers(directory: Path) -> Any:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    def predict(text: str) -> tuple[str, float]:
        enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits[0], dim=-1)
        best = int(probs.argmax())
        return str(id2label[best]), float(probs[best])

    return predict
