from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from docspilot.agent.loop import (
    DECLINED,
    Agent,
    AgentConfig,
    AgentEvent,
    AgentFinished,
    NeedsConfirmation,
    ToolEnd,
    ToolStart,
)
from docspilot.agent.pending import InMemoryPendingStore, PendingAction, PendingBusyError
from docspilot.agent.tools import (
    Tool,
    ToolContext,
    ToolError,
    ToolInput,
    ToolRegistry,
    default_tools,
)
from docspilot.llm.errors import LLMError
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.types import LLMRequest, ToolCall
from docspilot.tickets import IdempotencyConflictError, InMemoryTicketStore

USER = "u1"


def build(
    scripts: list[Script] | Any, tools: list[Tool] | None = None, **cfg: Any
) -> tuple[Agent, FakeLLM, InMemoryTicketStore, InMemoryPendingStore]:
    llm = FakeLLM(scripts)
    tickets = InMemoryTicketStore()
    pending = InMemoryPendingStore()
    agent = Agent(
        llm,
        ToolRegistry(tools if tools is not None else default_tools()),
        pending,
        AgentConfig(model="m", system="sys", **cfg),
    )
    return agent, llm, tickets, pending


def ctx(tickets: InMemoryTicketStore) -> ToolContext:
    return ToolContext(user_id=USER, tickets=tickets)


async def collect(gen: Any) -> list[AgentEvent]:
    return [e async for e in gen]


def call(id_: str, name: str, **input_: Any) -> ToolCall:
    return ToolCall(id=id_, name=name, input=input_)


HISTORY = [{"role": "user", "content": "hello"}]
TICKET_ARGS = {
    "title": "Cannot change currency",
    "description": "Currency dropdown is disabled on the billing page.",
    "severity": "normal",
}


async def test_plain_answer_single_step() -> None:
    agent, llm, tickets, _ = build([Script(text="hi there")])
    events = await collect(agent.run(HISTORY, ctx(tickets)))
    finished = events[-1]
    assert isinstance(finished, AgentFinished)
    assert (finished.stop_reason, finished.steps, finished.text) == ("end_turn", 1, "hi there")
    assert llm.requests[0].tools  # tools are advertised


async def test_read_tool_error_is_fed_back_and_assistant_blocks_replayed_unchanged() -> None:
    agent, llm, tickets, _ = build(
        [
            Script(tool_calls=[call("t1", "get_ticket", ticket_id="TCK-deadbeef")]),
            Script(text="I could not find that ticket."),
        ]
    )
    events = await collect(agent.run(HISTORY, ctx(tickets)))
    assert [type(e).__name__ for e in events if isinstance(e, ToolStart | ToolEnd)] == [
        "ToolStart",
        "ToolEnd",
    ]
    second = llm.requests[1].messages
    assert second[-1]["role"] == "user"
    block = second[-1]["content"][0]
    assert block["tool_use_id"] == "t1" and block["is_error"] is True
    # the assistant turn holds the exact content blocks the model produced
    assert second[-2]["role"] == "assistant"
    assert second[-2]["content"][0]["type"] == "tool_use"
    assert isinstance(events[-1], AgentFinished) and events[-1].steps == 2


async def test_parallel_calls_run_concurrently_and_return_in_one_message() -> None:
    class Slow(Tool):
        name = "slow"
        description = "sleeps"
        input_model = ToolInput

        async def run(self, args: Any, ctx: ToolContext) -> dict[str, Any]:
            await asyncio.sleep(0.2)
            return {"ok": True}

    agent, llm, tickets, _ = build(
        [
            Script(tool_calls=[call("a", "slow"), call("b", "slow"), call("c", "slow")]),
            Script(text="done"),
        ],
        tools=[Slow()],
    )
    started = time.perf_counter()
    await collect(agent.run(HISTORY, ctx(tickets)))
    assert time.perf_counter() - started < 0.5  # 3 x 0.2s serially would be 0.6s
    results = llm.requests[1].messages[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]


async def test_unknown_tool_and_invalid_args_become_error_results() -> None:
    agent, llm, tickets, _ = build(
        [
            Script(
                tool_calls=[
                    call("x", "drop_database"),
                    call("y", "get_ticket", ticket_id="not-an-id"),
                ]
            ),
            Script(text="sorry"),
        ]
    )
    await collect(agent.run(HISTORY, ctx(tickets)))
    results = llm.requests[1].messages[-1]["content"]
    assert all(r["is_error"] for r in results)
    assert "Unknown tool" in results[0]["content"]
    assert "ticket_id" in results[1]["content"]


async def test_tool_exceptions_are_contained_and_not_leaked() -> None:
    class Broken(Tool):
        name = "broken"
        description = "always fails"
        input_model = ToolInput

        async def run(self, args: Any, ctx: ToolContext) -> dict[str, Any]:
            raise RuntimeError("db password is hunter2")

    class Expected(Tool):
        name = "expected"
        description = "fails politely"
        input_model = ToolInput

        async def run(self, args: Any, ctx: ToolContext) -> dict[str, Any]:
            raise ToolError("Nothing matches that.")

    agent, llm, tickets, _ = build(
        [Script(tool_calls=[call("a", "broken"), call("b", "expected")]), Script(text="ok")],
        tools=[Broken(), Expected()],
    )
    await collect(agent.run(HISTORY, ctx(tickets)))
    a, b = llm.requests[1].messages[-1]["content"]
    assert "hunter2" not in a["content"] and a["is_error"]
    assert b["content"] == "Nothing matches that."


async def test_tool_timeout() -> None:
    class Hang(Tool):
        name = "hang"
        description = "never returns"
        input_model = ToolInput

        async def run(self, args: Any, ctx: ToolContext) -> dict[str, Any]:
            await asyncio.sleep(10)
            return {}

    agent, llm, tickets, _ = build(
        [Script(tool_calls=[call("h", "hang")]), Script(text="ok")],
        tools=[Hang()],
        tool_timeout_s=0.05,
    )
    await collect(agent.run(HISTORY, ctx(tickets)))
    result = llm.requests[1].messages[-1]["content"][0]
    assert result["is_error"] and "timed out" in result["content"]


async def test_step_budget_stops_runaway_loops() -> None:
    forever = Script(tool_calls=[call("t", "get_ticket", ticket_id="TCK-00000000")])
    agent, llm, tickets, _ = build([forever], max_steps=3)
    events = await collect(agent.run(HISTORY, ctx(tickets)))
    finished = events[-1]
    assert isinstance(finished, AgentFinished)
    assert finished.stop_reason == "max_steps" and len(llm.requests) == 3


async def test_write_tool_pauses_then_runs_once_on_approval() -> None:
    agent, llm, tickets, pending = build(
        [
            Script(
                text="Opening a ticket.", tool_calls=[call("w1", "create_ticket", **TICKET_ARGS)]
            ),
            Script(text="Done: your ticket is open."),
        ]
    )
    events = await collect(agent.run(HISTORY, ctx(tickets)))
    needs = events[-1]
    assert isinstance(needs, NeedsConfirmation)
    assert needs.items[0].tool == "create_ticket"
    assert "normal-severity" in needs.items[0].summary
    assert len(llm.requests) == 1  # model is NOT called again until a human decides
    assert tickets._by_id == {}  # and nothing was created

    taken = await pending.claim(needs.pending_id, user_id=USER)
    assert taken is not None
    with pytest.raises(PendingBusyError):  # concurrent confirm while in flight
        await pending.claim(needs.pending_id, user_id=USER)

    resumed = await collect(agent.resume(taken, approve=True, ctx=ctx(tickets)))
    assert isinstance(resumed[-1], AgentFinished) and resumed[-1].text.startswith("Done")
    assert len(tickets._by_id) == 1
    tool_result = llm.requests[1].messages[-1]["content"][0]
    assert tool_result["tool_use_id"] == "w1" and "TCK-" in tool_result["content"]
    # finished turn => entry gone: replaying the confirmation finds nothing
    assert await pending.claim(needs.pending_id, user_id=USER) is None


async def test_approval_on_last_step_still_gets_an_answer() -> None:
    agent, _, tickets, pending = build(
        [Script(tool_calls=[call("w1", "create_ticket", **TICKET_ARGS)]), Script(text="Opened.")],
        max_steps=1,
    )
    events = await collect(agent.run(HISTORY, ctx(tickets)))
    needs = events[-1]
    assert isinstance(needs, NeedsConfirmation)
    taken = await pending.claim(needs.pending_id, user_id=USER)
    assert taken is not None
    resumed = await collect(agent.resume(taken, approve=True, ctx=ctx(tickets)))
    last = resumed[-1]
    assert isinstance(last, AgentFinished)
    assert (last.stop_reason, last.text) == ("end_turn", "Opened.")
    assert len(tickets._by_id) == 1


async def test_failed_followup_keeps_outcome_and_retry_does_not_duplicate() -> None:
    boom = LLMError("overloaded", kind="overloaded", status_code=529)
    agent, llm, tickets, pending = build(
        [
            Script(tool_calls=[call("w1", "create_ticket", **TICKET_ARGS)]),
            Script(error=boom),  # follow-up model call fails after the ticket was created
            Script(text="Your ticket is open."),
        ]
    )
    events = await collect(agent.run(HISTORY, ctx(tickets)))
    needs = events[-1]
    assert isinstance(needs, NeedsConfirmation)

    taken = await pending.claim(needs.pending_id, user_id=USER)
    assert taken is not None
    with pytest.raises(LLMError):
        await collect(agent.resume(taken, approve=True, ctx=ctx(tickets)))
    assert len(tickets._by_id) == 1  # side effect happened exactly once

    retry = await pending.claim(needs.pending_id, user_id=USER)  # entry survived, claim released
    assert retry is not None and retry.awaiting == []
    final = await collect(agent.resume(retry, approve=True, ctx=ctx(tickets)))
    assert isinstance(final[-1], AgentFinished) and final[-1].text == "Your ticket is open."
    assert len(tickets._by_id) == 1  # replayed, not re-executed
    assert not any(isinstance(e, ToolStart) for e in final)
    assert "TCK-" in llm.requests[2].messages[-1]["content"][0]["content"]


async def test_usage_accumulates_across_pause() -> None:
    agent, _, tickets, pending = build(
        [Script(tool_calls=[call("w", "create_ticket", **TICKET_ARGS)]), Script(text="ok")]
    )
    needs = (await collect(agent.run(HISTORY, ctx(tickets))))[-1]
    assert isinstance(needs, NeedsConfirmation)
    taken = await pending.claim(needs.pending_id, user_id=USER)
    assert taken is not None
    done = (await collect(agent.resume(taken, approve=True, ctx=ctx(tickets))))[-1]
    assert isinstance(done, AgentFinished)
    assert done.usage.input_tokens == 20 and done.steps == 2  # 10 per scripted turn


async def test_write_tool_declined_never_executes() -> None:
    agent, llm, tickets, pending = build(
        [
            Script(tool_calls=[call("w1", "create_ticket", **TICKET_ARGS)]),
            Script(text="Okay, no ticket."),
        ]
    )
    events = await collect(agent.run(HISTORY, ctx(tickets)))
    needs = events[-1]
    assert isinstance(needs, NeedsConfirmation)
    taken = await pending.claim(needs.pending_id, user_id=USER)
    assert taken is not None
    await collect(agent.resume(taken, approve=False, ctx=ctx(tickets)))
    assert tickets._by_id == {}
    assert llm.requests[1].messages[-1]["content"][0]["content"] == DECLINED


async def test_mixed_turn_runs_reads_before_pausing() -> None:
    agent, llm, tickets, pending = build(
        [
            Script(
                tool_calls=[
                    call("r", "get_ticket", ticket_id="TCK-00000000"),
                    call("w", "create_ticket", **TICKET_ARGS),
                ]
            ),
            Script(text="ok"),
        ]
    )
    events = await collect(agent.run(HISTORY, ctx(tickets)))
    assert any(isinstance(e, ToolEnd) and e.id == "r" for e in events)
    needs = events[-1]
    assert isinstance(needs, NeedsConfirmation)
    taken = await pending.claim(needs.pending_id, user_id=USER)
    assert taken is not None and "r" in taken.completed and taken.awaiting == ["w"]
    await collect(agent.resume(taken, approve=True, ctx=ctx(tickets)))
    results = llm.requests[1].messages[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["r", "w"]  # original order


def _item(id_: str, user: str = USER, expires: float | None = None) -> PendingAction:
    return PendingAction(
        id_, user, [], [], {}, [], 1, expires if expires is not None else time.time() + 60
    )


async def test_pending_store_owner_expiry_and_capacity() -> None:
    store = InMemoryPendingStore(ttl_s=60, max_items=3, max_per_user=2)

    await store.put(_item("a"))
    assert await store.claim("a", user_id="someone-else") is None
    assert await store.claim("a", user_id=USER) is not None

    await store.put(_item("old", expires=time.time() - 1))
    assert await store.claim("old", user_id=USER) is None

    # per-user cap evicts that user's oldest, never another user's entry
    await store.put(_item("other", user="u2"))
    for n in ("1", "2", "3"):
        await store.put(_item(n))
    assert await store.claim("1", user_id=USER) is None
    assert await store.claim("3", user_id=USER) is not None
    assert await store.claim("other", user_id="u2") is not None


async def test_idempotency_key_reuse_with_different_args_is_rejected() -> None:
    tickets = InMemoryTicketStore()
    first = await tickets.create(
        user_id=USER, title="t", description="d", severity="low", idempotency_key="k"
    )
    same = await tickets.create(
        user_id=USER, title="t", description="d", severity="low", idempotency_key="k"
    )
    assert same.id == first.id
    with pytest.raises(IdempotencyConflictError):
        await tickets.create(
            user_id=USER, title="other", description="d", severity="low", idempotency_key="k"
        )


def test_tool_specs_are_strict_and_deterministic() -> None:
    specs = ToolRegistry(default_tools()).specs()
    assert [s["name"] for s in specs] == ["create_ticket", "get_ticket"]
    for spec in specs:
        assert spec["strict"] is True
        assert spec["input_schema"]["additionalProperties"] is False
    with pytest.raises(ValueError):
        ToolRegistry([*default_tools(), *default_tools()])


async def test_request_carries_system_effort_and_budget() -> None:
    seen: list[LLMRequest] = []

    def script(req: LLMRequest) -> Script:
        seen.append(req)
        return Script(text="x")

    agent, _, tickets, _ = build(script, effort="low", max_tokens=999)
    await collect(agent.run(HISTORY, ctx(tickets)))
    assert (seen[0].system, seen[0].effort, seen[0].max_tokens) == ("sys", "low", 999)
