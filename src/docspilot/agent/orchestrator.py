"""Multi-agent: a router in front of specialist agents with *different privileges*.

The main reason to split here is not capability, it is **security by separation**
(the "dual LLM" pattern):

- the **docs** specialist reads untrusted documents but has no write tools, so a
  successful injection in a document has nothing dangerous to call;
- the **tickets** specialist can act (behind the human gate) but never sees retrieved
  documents, so there is no untrusted text in its context.

Everything else (smaller prompts, a cheaper model for the simple task) is a bonus; the
cost is an extra routing hop and handoff mistakes. Whether it wins on quality is an
empirical question: `evals answers --architecture multi` runs the same golden set."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any

from opentelemetry import trace

from docspilot.agent.loop import Agent, AgentEvent
from docspilot.agent.pending import PendingAction
from docspilot.agent.tools import ToolContext
from docspilot.intents import IntentClassifier
from docspilot.logging import get_logger

log = get_logger("orchestrator")

_AFFIRMATIVE = re.compile(
    r"^\s*(yes|yep|yeah|y|ok|okay|sure|please|please do|do it|go ahead|open it|"
    r"yes please|sounds good|that works|create it|file it)[\s.!]*$",
    re.I,
)


def is_ticket_followup(history: list[dict[str, Any]]) -> bool:
    """A bare 'yes' after the assistant offered a ticket belongs to the tickets agent."""
    if len(history) < 2 or history[-1].get("role") != "user":
        return False
    previous = history[-2]
    if previous.get("role") != "assistant" or not isinstance(previous.get("content"), str):
        return False
    return bool(
        _AFFIRMATIVE.match(str(history[-1]["content"])) and "ticket" in previous["content"].lower()
    )


class Orchestrator:
    """Same `run`/`resume` surface as `Agent`, so the API layer does not care."""

    def __init__(
        self, *, classifier: IntentClassifier, specialists: dict[str, Agent], default: str = "docs"
    ) -> None:
        if default not in specialists or "tickets" not in specialists:
            raise ValueError("specialists must include 'docs' and 'tickets'")
        self._classifier = classifier
        self._agents = specialists
        self._default = default

    async def pick(self, history: list[dict[str, Any]]) -> str:
        if is_ticket_followup(history):
            return "tickets"
        message = str(history[-1].get("content", ""))
        try:
            intent = await self._classifier.classify(message)
        except Exception:
            log.warning("specialist_routing_failed", classifier=self._classifier.name)
            return self._default
        return "tickets" if intent == "ticket_request" else self._default

    async def run(
        self, history: list[dict[str, Any]], ctx: ToolContext
    ) -> AsyncGenerator[AgentEvent, None]:
        name = await self.pick(history)
        trace.get_current_span().set_attribute("docspilot.specialist", name)
        log.info("specialist_selected", specialist=name)
        async for event in self._agents[name].run(history, ctx):
            yield event

    async def resume(
        self, pending: PendingAction, *, approve: bool, ctx: ToolContext
    ) -> AsyncGenerator[AgentEvent, None]:
        agent = self._agents.get(pending.agent) or self._agents["tickets"]
        async for event in agent.resume(pending, approve=approve, ctx=ctx):
            yield event
