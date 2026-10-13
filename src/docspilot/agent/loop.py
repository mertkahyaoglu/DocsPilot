"""The agent loop. ~100 lines on purpose: this is what frameworks wrap.

while the model asks for tools: run them, feed results back, repeat, within a step budget.
Differences from the toy version: read tools run concurrently, errors become tool
results the model can react to, and write tools pause for human approval.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from docspilot.agent.pending import PendingAction, PendingStore
from docspilot.agent.tools import (
    Source,
    Tool,
    ToolContext,
    ToolInput,
    ToolOutcome,
    ToolRegistry,
)
from docspilot.config import Effort
from docspilot.llm.types import (
    LLM,
    LLMRequest,
    LLMResult,
    StreamReset,
    TextDelta,
    ToolCall,
    Usage,
)
from docspilot.logging import get_logger

log = get_logger("agent")

DECLINED = "The user declined this action. Do not retry it; acknowledge and offer alternatives."


@dataclass(frozen=True, slots=True)
class ToolStart:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolEnd:
    id: str
    name: str
    is_error: bool
    duration_ms: float


@dataclass(frozen=True, slots=True)
class ConfirmItem:
    call_id: str
    tool: str
    arguments: dict[str, Any]
    summary: str


@dataclass(frozen=True, slots=True)
class NeedsConfirmation:
    pending_id: str
    items: list[ConfirmItem]
    expires_in_s: int


@dataclass(frozen=True, slots=True)
class AgentFinished:
    stop_reason: str  # end_turn | max_tokens | refusal | max_steps
    usage: Usage
    steps: int
    model: str
    text: str = ""
    served_by_fallback: bool = False
    #: Sources the answer cites, restricted to documents actually retrieved this turn.
    citations: tuple[Source, ...] = ()
    #: Everything retrieved this turn (cited or not), for evals and debugging.
    retrieved: tuple[Source, ...] = ()


_BRACKET = re.compile(r"\[([^\[\]\n]*?\.md[^\[\]\n]*)\]")


def extract_citations(text: str, retrieved: dict[str, Source]) -> tuple[Source, ...]:
    """Citations are model output, so they are untrusted: keep only paths that were
    really retrieved this turn. A hallucinated path never reaches the user as a source."""
    out: dict[str, Source] = {}
    for group in _BRACKET.findall(text):
        # Tolerate "[a.md, b.md]", "[a.md#section]", "[source: a.md]".
        for part in re.split(r"[;,]", group):
            path = part.strip().removeprefix("source:").removeprefix("doc:").strip()
            path = path.split("#", 1)[0].strip()
            if path in retrieved:
                out.setdefault(path, retrieved[path])
    return tuple(out.values())


AgentEvent = TextDelta | StreamReset | ToolStart | ToolEnd | NeedsConfirmation | AgentFinished


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str
    system: str
    max_tokens: int = 16000
    effort: Effort | None = None
    max_steps: int = 6
    tool_timeout_s: float = 15.0
    prompt_ref: str = ""  # for logs/traces only
    prompt_sha: str = ""


def _tool_result_block(call_id: str, outcome: ToolOutcome) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": outcome.content,
    }
    if outcome.is_error:
        block["is_error"] = True
    return block


class Agent:
    def __init__(
        self,
        llm: LLM,
        registry: ToolRegistry,
        pending: PendingStore,
        config: AgentConfig,
        *,
        confirmation_ttl_s: int = 600,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._pending = pending
        self._config = config
        self._ttl = confirmation_ttl_s

    # -- public ---------------------------------------------------------------

    def run(
        self, history: list[dict[str, Any]], ctx: ToolContext
    ) -> AsyncGenerator[AgentEvent, None]:
        return self._loop(list(history), ctx, steps_left=self._config.max_steps)

    async def resume(
        self, pending: PendingAction, *, approve: bool, ctx: ToolContext
    ) -> AsyncGenerator[AgentEvent, None]:
        """Continue after a human decision on the paused write call(s).

        `pending` must be claimed by the caller. Safe to call again on the same entry
        after a failure: already-resolved calls are replayed, never re-executed."""
        if pending.awaiting:
            waiting = [c for c in pending.calls if c.id in pending.awaiting]
            for call in waiting:
                if not approve:
                    pending.completed[call.id] = ToolOutcome(DECLINED)
                    continue
                parsed = self._registry.parse(call.name, call.input)
                if isinstance(parsed, ToolOutcome):
                    pending.completed[call.id] = parsed
                    continue
                tool, args = parsed
                yield ToolStart(call.id, call.name, call.input)
                started = time.perf_counter()
                # Same key on every retry of this confirmation => at-most-once side effect
                # even if we crashed after executing but before recording the outcome.
                keyed = ctx.with_key(f"{pending.id}:{call.id}")
                outcome = await self._registry.execute(
                    tool, args, keyed, timeout_s=self._config.tool_timeout_s
                )
                pending.completed[call.id] = outcome
                yield ToolEnd(
                    call.id,
                    call.name,
                    outcome.is_error,
                    round((time.perf_counter() - started) * 1000, 1),
                )
            pending.awaiting = []  # resolved: a retry replays instead of re-executing

        messages = [*pending.messages, self._results_message(pending.calls, pending.completed)]
        finished = False
        try:
            # The pause already spent its step; always leave one for the model to answer.
            async with aclosing(
                self._loop(
                    messages,
                    ctx,
                    steps_left=max(pending.steps_left, 1),
                    usage=pending.usage,
                    steps=pending.steps,
                    retrieved=pending.retrieved,
                )
            ) as stream:
                async for event in stream:
                    if isinstance(event, AgentFinished | NeedsConfirmation):
                        finished = True
                    yield event
        finally:
            if finished:
                await self._pending.delete(pending.id)
            else:  # failed or disconnected: keep the resolved entry so the client can retry
                await self._pending.release(pending)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _results_message(calls: list[ToolCall], outcomes: dict[str, ToolOutcome]) -> dict[str, Any]:
        # One user message with *all* results: splitting them teaches the model to stop
        # making parallel calls.
        return {
            "role": "user",
            "content": [_tool_result_block(c.id, outcomes[c.id]) for c in calls],
        }

    async def _loop(
        self,
        messages: list[dict[str, Any]],
        ctx: ToolContext,
        *,
        steps_left: int,
        usage: Usage | None = None,
        steps: int = 0,
        retrieved: dict[str, Source] | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        usage = usage or Usage()
        retrieved = dict(retrieved or {})
        cfg = self._config
        while steps_left > 0:
            steps += 1
            steps_left -= 1
            request = LLMRequest(
                model=cfg.model,
                system=cfg.system,
                messages=messages,
                tools=self._registry.specs() or None,
                max_tokens=cfg.max_tokens,
                effort=cfg.effort,
            )
            result: LLMResult | None = None
            async with aclosing(self._llm.stream(request)) as llm_stream:
                async for event in llm_stream:
                    if isinstance(event, LLMResult):
                        result = event
                    else:
                        yield event
            assert result is not None
            usage += result.usage
            log.info(
                "llm_call",
                step=steps,
                model=result.model,
                prompt=cfg.prompt_ref,
                prompt_sha=cfg.prompt_sha,
                stop_reason=result.stop_reason,
                served_by_fallback=result.served_by_fallback,
                provider_request_id=result.request_id,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cache_read_input_tokens=result.usage.cache_read_input_tokens,
                cache_creation_input_tokens=result.usage.cache_creation_input_tokens,
            )

            if result.stop_reason != "tool_use" or not result.tool_calls:
                reason = "refusal" if result.refused else (result.stop_reason or "end_turn")
                yield AgentFinished(
                    reason,
                    usage,
                    steps,
                    result.model,
                    text=result.text,
                    served_by_fallback=result.served_by_fallback,
                    citations=extract_citations(result.text, retrieved),
                    retrieved=tuple(retrieved.values()),
                )
                return

            messages.append({"role": "assistant", "content": result.content})
            outcomes, awaiting = {}, []
            runnable: list[tuple[ToolCall, Tool, ToolInput]] = []
            for call in result.tool_calls:
                parsed = self._registry.parse(call.name, call.input)
                if isinstance(parsed, ToolOutcome):
                    outcomes[call.id] = parsed  # bad call: error goes back to the model
                    continue
                tool, args = parsed
                if tool.requires_confirmation:
                    awaiting.append(call.id)
                else:
                    runnable.append((call, tool, args))

            for call, _, _ in runnable:
                yield ToolStart(call.id, call.name, call.input)
            for call, outcome, ms in await self._run_concurrently(runnable, ctx):
                outcomes[call.id] = outcome
                for src in outcome.sources:
                    prev = retrieved.get(src.path)
                    if prev is not None and src.text and src.text not in prev.text:
                        src = Source(
                            src.path, src.title, src.section, prev.text + "\n\n" + src.text
                        )
                    elif prev is not None:
                        src = prev
                    retrieved[src.path] = src
                yield ToolEnd(call.id, call.name, outcome.is_error, ms)

            if awaiting:
                yield await self._pause(
                    messages, result, outcomes, awaiting, ctx, steps_left, usage, steps, retrieved
                )
                return
            messages.append(self._results_message(result.tool_calls, outcomes))

        log.warning("agent_step_budget_exhausted", steps=steps)
        yield AgentFinished("max_steps", usage, steps, cfg.model)

    async def _run_concurrently(
        self, runnable: list[tuple[ToolCall, Tool, ToolInput]], ctx: ToolContext
    ) -> list[tuple[ToolCall, ToolOutcome, float]]:
        async def one(
            call: ToolCall, tool: Tool, args: ToolInput
        ) -> tuple[ToolCall, ToolOutcome, float]:
            started = time.perf_counter()
            outcome = await self._registry.execute(
                tool, args, ctx, timeout_s=self._config.tool_timeout_s
            )
            return call, outcome, round((time.perf_counter() - started) * 1000, 1)

        return list(await asyncio.gather(*(one(c, t, a) for c, t, a in runnable)))

    async def _pause(
        self,
        messages: list[dict[str, Any]],
        result: LLMResult,
        outcomes: dict[str, ToolOutcome],
        awaiting: list[str],
        ctx: ToolContext,
        steps_left: int,
        usage: Usage,
        steps: int,
        retrieved: dict[str, Source],
    ) -> NeedsConfirmation:
        pending = PendingAction(
            id=self._pending.new_id(),
            user_id=ctx.user_id,
            messages=messages,
            calls=result.tool_calls,
            completed=outcomes,
            awaiting=awaiting,
            steps_left=steps_left,
            expires_at=time.time() + self._ttl,
            usage=usage,
            steps=steps,
            retrieved=retrieved,
        )
        await self._pending.put(pending)
        items = []
        for call in result.tool_calls:
            if call.id in awaiting:
                parsed = self._registry.parse(call.name, call.input)
                assert not isinstance(parsed, ToolOutcome)  # validated above
                tool, args = parsed
                items.append(ConfirmItem(call.id, call.name, call.input, tool.summarize(args)))
        return NeedsConfirmation(pending.id, items, self._ttl)
