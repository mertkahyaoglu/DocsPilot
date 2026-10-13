from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from docspilot.agent.loop import AgentFinished, NeedsConfirmation
from docspilot.agent.tools import ToolContext
from docspilot.experiments.langgraph_router import build_graph
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.types import ToolCall
from docspilot.tickets import InMemoryTicketStore
from tests.test_phase8_auth_and_agents import StubClassifier, build_multi

TICKET = {"title": "Payout stuck", "description": "In transit for 6 days.", "severity": "high"}


async def run_both(intent: str, scripts: list[Script]) -> tuple[list, list]:
    hist = [{"role": "user", "content": "anything"}]
    hand = await build_multi(FakeLLM(scripts), StubClassifier(intent))
    graph = build_graph(await build_multi(FakeLLM(scripts), StubClassifier(intent)))
    ctx = ToolContext("u", InMemoryTicketStore())
    hand_events = [e async for e in hand.run(hist, ctx)]
    out = await graph.ainvoke({"history": hist, "ctx": ToolContext("u", InMemoryTicketStore())})
    return hand_events, out["events"]


async def test_graph_is_behaviourally_equivalent_for_the_docs_route() -> None:
    hand, graph = await run_both("question", [Script(text="Plain answer.")])
    assert isinstance(hand[-1], AgentFinished) and isinstance(graph[-1], AgentFinished)
    assert hand[-1].text == graph[-1].text == "Plain answer."


async def test_graph_is_behaviourally_equivalent_for_the_ticket_route() -> None:
    scripts = [Script(tool_calls=[ToolCall("w", "create_ticket", TICKET)])]
    hand, graph = await run_both("ticket_request", scripts)
    assert isinstance(hand[-1], NeedsConfirmation) and isinstance(graph[-1], NeedsConfirmation)
    assert hand[-1].items[0].summary == graph[-1].items[0].summary
