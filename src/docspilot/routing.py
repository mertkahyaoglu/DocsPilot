"""Optional front door: a small model classifies the first message; chit-chat is
answered by the small model with no tools, everything else goes to the full agent.

This is a *trade*, not a free win: every non-chit-chat request pays one extra small-
model round trip (cost and ~0.3-1s latency). It only pays off when a meaningful share
of traffic is easy. Measure (evals + `docspilot_route_total`) before enabling.
It fails open: any routing error sends the request to the full agent."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Literal

from docspilot.agent.loop import AgentEvent, AgentFinished
from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLM, LLMRequest, LLMResult, TextDelta, Usage
from docspilot.logging import get_logger
from docspilot.triage import Triage, TriageError, triage

log = get_logger("routing")
Route = Literal["agent", "smalltalk"]


async def choose_route(llm: LLM, *, message: str, triage_prompt: str, model: str) -> Route:
    try:
        result, _ = await triage(
            llm, message=message, system=triage_prompt, model=model, purpose="route"
        )
    except (TriageError, LLMError):
        log.warning("routing_failed_open")
        return "agent"
    except Exception:
        log.exception("routing_crashed_failed_open")
        return "agent"
    return route_for(result, message)


MAX_SMALLTALK_CHARS = 160


def route_for(t: Triage, message: str) -> Route:
    # Only the unambiguous case is diverted: classified small talk AND short. A long
    # message that mixes a greeting with a real request ("thanks! also open a ticket
    # about...") must reach the agent, which has the tools.
    if t.intent == "smalltalk" and len(message) <= MAX_SMALLTALK_CHARS:
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
