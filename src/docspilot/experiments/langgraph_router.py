"""The router + specialists from `agent/orchestrator.py`, re-expressed as a LangGraph
StateGraph (optional extra: `uv sync --extra langgraph`).

Why this file exists: the hand-rolled version came first so the mechanics are visible.
This shows what a graph framework buys *for this shape of problem*, honestly:

  + explicit topology you can draw/inspect, per-node checkpointing, resumable runs,
    built-in streaming of node updates, a place to hang human-in-the-loop interrupts;
  - another state model to learn, our SSE/budget/guard/trace plumbing still has to be
    wired in, errors surface through the framework, and for a 2-way router the graph is
    no shorter than `Orchestrator`.

Rule of thumb: reach for a graph engine when you need durable multi-step workflows with
branching, retries and persistence across processes. A router is not that."""

from __future__ import annotations

from typing import Any, TypedDict

from docspilot.agent.loop import AgentEvent
from docspilot.agent.orchestrator import Orchestrator


class State(TypedDict, total=False):
    history: list[dict[str, Any]]
    ctx: Any
    route: str
    events: list[AgentEvent]


def build_graph(orchestrator: Orchestrator) -> Any:
    from langgraph.graph import END, StateGraph

    async def classify(state: State) -> State:
        return {"route": await orchestrator.pick(state["history"])}

    def specialist(name: str) -> Any:
        async def node(state: State) -> State:
            agent = orchestrator._agents[name]
            return {"events": [e async for e in agent.run(state["history"], state["ctx"])]}

        return node

    graph = StateGraph(State)
    graph.add_node("classify", classify)
    graph.add_node("docs", specialist("docs"))
    graph.add_node("tickets", specialist("tickets"))
    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", lambda s: s["route"], {"docs": "docs", "tickets": "tickets"}
    )
    graph.add_edge("docs", END)
    graph.add_edge("tickets", END)
    return graph.compile()
