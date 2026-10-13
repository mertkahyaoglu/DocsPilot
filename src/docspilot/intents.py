"""Intent classification behind an interface, so the *same* decision can be made by a
prompted LLM, a fine-tuned local model, or a rules baseline, and compared on one eval set."""

from __future__ import annotations

from typing import Literal, Protocol

from docspilot.llm.types import LLM
from docspilot.triage import triage

Intent = Literal["question", "ticket_request", "complaint", "smalltalk", "other"]
INTENTS: tuple[Intent, ...] = ("question", "ticket_request", "complaint", "smalltalk", "other")


class IntentClassifier(Protocol):
    name: str

    async def classify(self, message: str) -> Intent: ...


class LLMIntentClassifier:
    """The prompted small-model classifier (structured output): the default."""

    def __init__(self, llm: LLM, *, system: str, model: str, purpose: str = "route") -> None:
        self.name = f"llm:{model}"
        self._llm, self._system, self._model, self._purpose = llm, system, model, purpose

    async def classify(self, message: str) -> Intent:
        result, _ = await triage(
            self._llm,
            message=message,
            system=self._system,
            model=self._model,
            purpose=self._purpose,
        )
        return result.intent
