"""Multi-agent: a router in front of specialist agents with *different privileges*.

The main reason to split here is not capability, it is **privilege separation**
(the "dual LLM" pattern):

- the **docs** specialist reads untrusted documents but has no write tools, so a
  successful injection in a document has nothing dangerous to call (this holds
  unconditionally: tool names are fixed per specialist);
- the **tickets** specialist can act (behind the human gate) but has no retrieval tool,
  so no *retrieved document text* is ever placed in its context by a tool. Residual: the
  conversation history it receives is client-supplied and may contain earlier docs-agent
  answers, i.e. model output derived from documents. That text is bounded by the human
  confirmation of every write, not by isolation.

Everything else (smaller prompts, a cheaper model for the simple task) is a bonus; the
cost is an extra routing hop and handoff mistakes. Whether it wins on quality is an
empirical question: `evals answers --architecture multi` runs the same golden set."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any

from opentelemetry import trace

from docspilot.agent.loop import Agent, AgentEvent, ToolStart
from docspilot.agent.pending import PendingAction
from docspilot.agent.tools import ToolContext
from docspilot.intents import IntentClassifier
from docspilot.llm.types import StreamReset, TextDelta
from docspilot.logging import get_logger

log = get_logger("orchestrator")

#: tool the ticketing specialist calls when the request is really a docs question
HANDOFF_TOOL = "hand_off_to_docs"

_AFFIRMATIVE = re.compile(
    r"^\s*(yes|yep|yeah|y|ok|okay|sure|please|please do|do it|go ahead|open it|"
    r"yes please|sounds good|that works|create it|file it)[\s.!]*$",
    re.I,
)
_OFFER = re.compile(
    r"\b(shall i|should i|would you like|do you want|want me to|can i|may i|if you'd like)\b",
    re.I,
)
_TICKET_TOPIC = re.compile(r"\b(ticket|escalat\w*|support team|human agent)\b", re.I)
_NEW_QUESTION = re.compile(
    r"^\s*(what|how|why|when|where|which|who|can|could|do|does|is|are)\b", re.I
)


def is_ticket_followup(history: list[dict[str, Any]]) -> bool:
    """Is the user answering something the *ticketing* conversation just asked?

    True for a bare "yes" after an offer to open a ticket, and for a plain statement
    answering the ticketing specialist's clarifying question (otherwise the details would
    be classified alone as a "question" and land on the specialist that cannot write).
    A message that is itself a question is classified normally."""
    if len(history) < 2 or history[-1].get("role") != "user":
        return False
    previous = history[-2]
    if previous.get("role") != "assistant" or not isinstance(previous.get("content"), str):
        return False
    prev = str(previous["content"])
    # the ticket conversation may be a few turns old: the ticketing specialist's clarifying
    # question ("which invoice?") need not itself say the word "ticket"
    recent = " ".join(
        str(m.get("content", "")) for m in history[-4:-1] if isinstance(m.get("content"), str)
    )
    if not _TICKET_TOPIC.search(recent):
        return False
    asked = prev.rstrip().endswith("?") or bool(_OFFER.search(prev))
    if not asked:
        return False  # "...see the ticket policy." then "ok": not a ticket conversation
    user = str(history[-1]["content"]).strip()
    if _AFFIRMATIVE.match(user):
        return True
    return not (user.endswith("?") or _NEW_QUESTION.match(user))


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
        emitted_text = False
        stream = self._agents[name].run(history, ctx)
        handed_off = False
        try:
            async for event in stream:
                if isinstance(event, ToolStart) and event.name == HANDOFF_TOOL:
                    handed_off = True  # the ticketing agent decided this is a docs question
                    break
                if isinstance(event, TextDelta):
                    emitted_text = True
                yield event
        finally:
            await stream.aclose()
        if handed_off:
            log.info("specialist_handoff", from_=name, to=self._default)
            if emitted_text:
                yield StreamReset()  # discard anything the wrong specialist already said
            async for event in self._agents[self._default].run(history, ctx):
                yield event

    async def resume(
        self, pending: PendingAction, *, approve: bool, ctx: ToolContext
    ) -> AsyncGenerator[AgentEvent, None]:
        agent = self._agents.get(pending.agent) or self._agents["tickets"]
        async for event in agent.resume(pending, approve=approve, ctx=ctx):
            yield event
