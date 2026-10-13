"""Optional front door: a classifier labels the first message; chit-chat is answered by
the small model with no tools, everything else goes to the full agent.

This is a *trade*, not a free win: every non-chit-chat request pays one extra
classification round trip (cost and, for an LLM classifier, ~0.3-1s). It only pays off
when a meaningful share of traffic is easy. Measure (evals + `docspilot_route_total`)
before enabling. It fails open: any routing error sends the request to the full agent."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Literal

from docspilot.agent.loop import AgentEvent, AgentFinished
from docspilot.intents import Intent, IntentClassifier
from docspilot.llm.types import LLM, LLMRequest, LLMResult, TextDelta, Usage
from docspilot.logging import get_logger

log = get_logger("routing")
Route = Literal["agent", "smalltalk"]

MAX_SMALLTALK_CHARS = 160


async def choose_route(classifier: IntentClassifier, *, message: str) -> Route:
    try:
        intent = await classifier.classify(message)
    except Exception:
        log.warning("routing_failed_open", classifier=classifier.name, exc_info=True)
        return "agent"
    return route_for(intent, message)


def route_for(intent: Intent, message: str) -> Route:
    # Only the unambiguous case is diverted: classified small talk AND short. A long
    # message that mixes a greeting with a real request ("thanks! also open a ticket
    # about...") must reach the agent, which has the tools.
    if intent == "smalltalk" and len(message) <= MAX_SMALLTALK_CHARS:
        return "smalltalk"
    return "agent"


async def smalltalk_events(
    llm: LLM, *, message: str, system: str, model: str
) -> AsyncGenerator[AgentEvent, None]:
    request = LLMRequest(
        model=model,
        system=system,
        messages=[{"role": "user", "content": message}],
        max_tokens=512,
        effort="low",
        purpose="smalltalk",
    )
    result: LLMResult | None = None
    async for event in llm.stream(request):
        if isinstance(event, TextDelta):
            yield event
        elif isinstance(event, LLMResult):
            result = event
    assert result is not None
    reason = "refusal" if result.refused else (result.stop_reason or "end_turn")
    yield AgentFinished(reason, result.usage or Usage(), 1, result.model, text=result.text)
