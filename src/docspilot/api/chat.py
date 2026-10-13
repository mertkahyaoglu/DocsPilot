from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, StringConstraints
from sse_starlette.sse import EventSourceResponse

from docspilot.agent.loop import (
    Agent,
    AgentEvent,
    AgentFinished,
    NeedsConfirmation,
    ToolEnd,
    ToolStart,
)
from docspilot.agent.pending import PendingBusyError, PendingStore
from docspilot.agent.tools import ToolContext
from docspilot.api.deps import current_user_id, enforce_budget
from docspilot.api.schemas import ChatRequest
from docspilot.budget import Budget, CostMeter, current_meter
from docspilot.cache import CachedAnswer, CorpusVersionProvider, ResponseCache, cache_key
from docspilot.config import Settings
from docspilot.degraded import degraded_answer
from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLM, StreamReset, TextDelta, Usage
from docspilot.logging import get_logger
from docspilot.pricing import cost_usd
from docspilot.prompts import PromptRegistry
from docspilot.routing import choose_route, smalltalk_events
from docspilot.safety.output_guard import SAFE_MESSAGE, StreamingGuard, check_output
from docspilot.telemetry import Telemetry
from docspilot.tickets import TicketStore

router = APIRouter(prefix="/v1")
log = get_logger("chat")

_ERROR_REFUSAL = {
    "code": "refusal",
    "retryable": False,
    "message": "I can't help with that request.",
}
_ERROR_STEPS = {
    "code": "step_limit",
    "retryable": False,
    "message": "That took too many steps to complete. Please try a simpler request.",
}
# Tools whose results are the same for every user: a turn that used only these is cacheable.
_CACHEABLE_TOOLS = frozenset({"search_docs"})

# Settlement must finish even when the client has gone: the task is shielded and kept
# referenced here so it cannot be garbage collected mid-flight.
_BACKGROUND: set[asyncio.Task[None]] = set()


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def _validate_limits(body: ChatRequest, settings: Settings) -> None:
    if len(body.messages) > settings.max_history_messages:
        raise HTTPException(422, f"too many messages (max {settings.max_history_messages})")
    for m in body.messages:
        # Users are capped tightly; prior assistant turns are our own (bounded by
        # max_tokens) and echoed back by the client, so they get a looser cap.
        limit = settings.max_message_chars if m.role == "user" else settings.max_assistant_chars
        if len(m.content) > limit:
            raise HTTPException(422, f"{m.role} message too long (max {limit} chars)")


@dataclass(slots=True)
class CacheSlot:
    """Where a finished turn may be stored. `allowed` is flipped off by code paths that
    must never populate the shared cache (e.g. small-model routed replies)."""

    key: str
    model: str = ""  # the model the key was built for: answers from any other are not stored
    allowed: bool = True


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _strings(v)]
    return []


async def _settle_detached(settle: Callable[[str], Awaitable[None]], outcome: str) -> None:
    task = asyncio.ensure_future(settle(outcome))
    _BACKGROUND.add(task)

    def _done(t: asyncio.Task[None]) -> None:
        _BACKGROUND.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.error("budget_settle_failed", error=repr(t.exception()))

    task.add_done_callback(_done)
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.shield(task)  # we may be mid-cancellation; the task still finishes


async def _to_sse(
    events: AsyncGenerator[AgentEvent, None],
    request_id: str,
    tel: Telemetry,
    *,
    cache: ResponseCache | None = None,
    slot: CacheSlot | None = None,
    allowed_hosts: tuple[str, ...] | None = None,
    settle: Callable[[str], Awaitable[None]] | None = None,
    drop_pending: Callable[[str], Awaitable[None]] | None = None,
    degrade: Callable[[], Awaitable[tuple[str, list[dict[str, str]]] | None]] | None = None,
) -> AsyncIterator[dict[str, str]]:
    """Translate agent events to SSE. Always ends the stream with done|error.

    Tracing note: the root span is *not* held as the current context across `yield`
    (generators can be finalized from another task on disconnect). It is activated
    around each step of the inner stream only, so child spans still parent correctly.

    Safety note: text deltas pass through a StreamingGuard, so content that would be
    blocked is never sent in the first place (a `replace` event only follows)."""
    m = tel.metrics
    started = time.perf_counter()
    outcome = "error"
    text = ""
    tools: set[str] = set()
    search_ok = 0
    search_failed = 0
    sent_chars = 0
    step_start_sent = 0  # chars shown before the current model step began
    guard = StreamingGuard(allowed_hosts) if allowed_hosts else None
    root = tel.start("chat.request", **{"docspilot.request_id": request_id})
    m.in_flight.inc()
    it = events.__aiter__()

    def flush_tail() -> str:
        return guard.finish() if guard else ""

    try:
        while True:
            try:
                with tel.activate(root):
                    event = await it.__anext__()
            except StopAsyncIteration:
                break
            if isinstance(event, TextDelta):
                text += event.text
                out = guard.feed(event.text) if guard else event.text
                if out:
                    sent_chars += len(out)
                    yield _sse("delta", {"text": out})
            elif isinstance(event, StreamReset):
                # The current step's text is being discarded (failover): so is the guard's
                # held tail, and the user has seen nothing of this step.
                text = ""
                sent_chars = step_start_sent
                guard = StreamingGuard(allowed_hosts) if allowed_hosts else None
                yield _sse("reset", {})
            elif isinstance(event, ToolStart):
                tools.add(event.name)
                yield _sse("tool_start", {"id": event.id, "name": event.name})
            elif isinstance(event, ToolEnd):
                step_start_sent = sent_chars
                if event.name == "search_docs":
                    search_failed += event.is_error
                    search_ok += not event.is_error
                yield _sse(
                    "tool_end",
                    {
                        "id": event.id,
                        "name": event.name,
                        "is_error": event.is_error,
                        "duration_ms": event.duration_ms,
                    },
                )
            elif isinstance(event, NeedsConfirmation):
                tail = flush_tail()
                if tail:
                    sent_chars += len(tail)
                    yield _sse("delta", {"text": tail})
                # What the user is asked to approve is model-authored too: same checks.
                args_text = "\n".join(s for i in event.items for s in _strings(i.arguments))
                verdict = check_output(args_text, allowed_hosts) if allowed_hosts else None
                if (guard and guard.blocked) or (verdict and not verdict.ok):
                    outcome = "output_blocked"
                    reasons = (guard.result.reasons if guard and guard.blocked else ()) + (
                        verdict.reasons if verdict and not verdict.ok else ()
                    )
                    for reason in reasons:
                        m.output_blocked.labels(reason.split(":")[0]).inc()
                    log.warning("output_blocked", reasons=list(reasons), where="confirmation")
                    if drop_pending is not None:
                        await drop_pending(event.pending_id)
                    yield _sse("replace", {"text": SAFE_MESSAGE})
                    yield _sse("done", {"stop_reason": "output_blocked", "request_id": request_id})
                    return
                outcome = "awaiting_confirmation"
                yield _sse(
                    "confirmation_required",
                    {
                        "pending_id": event.pending_id,
                        "expires_in_s": event.expires_in_s,
                        "items": [
                            {"tool": i.tool, "summary": i.summary, "arguments": i.arguments}
                            for i in event.items
                        ],
                    },
                )
                yield _sse(
                    "done", {"stop_reason": "awaiting_confirmation", "request_id": request_id}
                )
                return
            elif isinstance(event, AgentFinished):
                cost = cost_usd(event.model, event.usage)
                root.set_attributes(
                    {
                        "docspilot.stop_reason": event.stop_reason,
                        "docspilot.steps": event.steps,
                        "docspilot.citations": len(event.citations),
                        "docspilot.cost_usd": -1.0 if cost is None else cost,
                    }
                )
                tel.content(root, "answer", event.text)
                log.info(
                    "chat_done",
                    stop_reason=event.stop_reason,
                    steps=event.steps,
                    model=event.model,
                    cost_usd=None if cost is None else round(cost, 6),
                    latency_s=round(time.perf_counter() - started, 3),
                    tools=sorted(tools),
                    flags=list(event.flags),
                    **asdict(event.usage),
                )
                tail = flush_tail()
                if tail:
                    sent_chars += len(tail)
                    yield _sse("delta", {"text": tail})
                if guard and guard.blocked:
                    outcome = "output_blocked"
                    for reason in guard.result.reasons:
                        m.output_blocked.labels(reason.split(":")[0]).inc()
                    log.warning("output_blocked", reasons=list(guard.result.reasons))
                    yield _sse("replace", {"text": SAFE_MESSAGE})
                    yield _sse(
                        "done",
                        {
                            "stop_reason": "output_blocked",
                            "usage": asdict(event.usage),
                            "model": event.model,
                            "steps": event.steps,
                            "request_id": request_id,
                            "cached": False,
                        },
                    )
                elif event.stop_reason == "refusal":
                    outcome = "refusal"
                    yield _sse("error", _ERROR_REFUSAL)
                elif event.stop_reason == "max_steps":
                    outcome = "step_limit"
                    yield _sse("error", _ERROR_STEPS)
                else:
                    outcome = "ok"
                    citations = [
                        {"path": c.path, "title": c.title, "section": c.section}
                        for c in event.citations
                    ]
                    if citations:
                        yield _sse("citations", {"sources": citations})
                    cacheable = (
                        cache is not None
                        and slot is not None
                        and slot.allowed
                        and event.stop_reason == "end_turn"
                        and bool(event.text.strip())
                        # grounded in a *healthy* search of the docs, and nothing else:
                        and tools == _CACHEABLE_TOOLS
                        and search_ok > 0
                        and search_failed == 0
                        and not event.flags
                        and not event.served_by_fallback
                        and event.model == slot.model  # never store a failover answer
                    )
                    if cacheable:
                        assert cache is not None and slot is not None
                        cache.put(
                            slot.key,
                            CachedAnswer(
                                text=event.text,
                                citations=tuple(citations),
                                stop_reason=event.stop_reason,
                                model=event.model,
                            ),
                        )
                    yield _sse(
                        "done",
                        {
                            "stop_reason": event.stop_reason,
                            "usage": asdict(event.usage),
                            "model": event.model,
                            "steps": event.steps,
                            "request_id": request_id,
                            "cached": False,
                        },
                    )
    except LLMError as exc:
        outcome = f"error_{exc.kind}"
        log.warning(
            "llm_error",
            kind=exc.kind,
            status=exc.status_code,
            provider_request_id=exc.request_id,
            detail=str(exc),
        )
        root.record_exception(exc)
        fallback = None
        if degrade is not None and exc.retryable and sent_chars == 0:
            # Generation is down but nothing has been shown yet: answer from retrieval.
            try:
                fallback = await degrade()
            except Exception:
                log.exception("degraded_mode_failed")
        if fallback is not None and (
            not allowed_hosts or check_output(fallback[0], allowed_hosts).ok
        ):
            text_out, cites = fallback
            outcome = "degraded"
            m.degraded.inc()
            log.warning("degraded_answer", cause=exc.kind, sources=len(cites))
            yield _sse("delta", {"text": text_out})
            yield _sse("citations", {"sources": cites})
            yield _sse(
                "done",
                {
                    "stop_reason": "degraded",
                    "usage": asdict(Usage()),
                    "model": None,
                    "steps": 0,
                    "request_id": request_id,
                    "cached": False,
                    "degraded": True,
                },
            )
        else:
            yield _sse(
                "error",
                {"code": exc.kind, "retryable": exc.retryable, "message": exc.public_message},
            )
    except (GeneratorExit, asyncio.CancelledError):
        outcome = "cancelled"  # client disconnected: not an error
        raise
    except Exception as exc:
        # HTTP 200 is already on the wire: never let a stream die silently.
        outcome = "error_unexpected"
        log.exception("chat_unexpected_error")
        root.record_exception(exc)
        err = LLMError.from_unexpected(exc)
        yield _sse("error", {"code": err.kind, "retryable": False, "message": err.public_message})
    finally:
        # Synchronous bookkeeping first: it must survive any cancellation below.
        m.in_flight.dec()
        m.requests.labels(outcome).inc()
        root.set_attribute("docspilot.outcome", outcome)
        if outcome.startswith("error"):
            root.set_status(Status(StatusCode.ERROR, outcome))
        root.end()
        with contextlib.suppress(BaseException):
            await events.aclose()  # deterministic cleanup of the inner generators
        if settle is not None:
            # Spend is real even if the client left or the turn failed midway.
            await _settle_detached(settle, outcome)


@router.post("/chat")
async def chat(
    body: ChatRequest, request: Request, user_id: Annotated[str, Depends(current_user_id)]
) -> EventSourceResponse:
    settings: Settings = request.app.state.settings
    agent: Agent = request.app.state.agent
    tickets: TicketStore = request.app.state.tickets
    tel: Telemetry = request.app.state.telemetry
    cache: ResponseCache = request.app.state.response_cache
    request_id: str = request.state.request_id
    _validate_limits(body, settings)
    first_turn = len(body.messages) == 1
    question = body.messages[0].content

    # A cache hit costs nothing, so it is served before any budget is reserved.
    slot: CacheSlot | None = None
    if cache.enabled and first_turn:
        versions: CorpusVersionProvider = request.app.state.corpus_versions
        key = cache_key(
            question,
            prompt_sha=request.app.state.answer_prompt.sha,
            model=settings.model_answer,
            effort=settings.effort_answer,
            corpus_version=await versions.get(),
        )
        hit = cache.get(key)
        tel.metrics.cache.labels("hit" if hit else "miss").inc()
        if hit is not None:
            return EventSourceResponse(_replay(hit, request_id, tel))
        slot = CacheSlot(key, model=settings.model_answer)

    reserved = await enforce_budget(request, user_id)
    try:
        meter = CostMeter()
        current_meter.set(meter)  # InstrumentedLLM adds every model call of this request
        budget: Budget = request.app.state.budget

        async def settle(outcome: str) -> None:
            await budget.settle(
                user_id,
                reserved,
                meter.usd,
                # Nothing was consumed (error, degraded, blocked before the model): give the
                # reservation back. A disconnect keeps it: the provider may still bill.
                refund=outcome != "cancelled" and meter.usd == 0,
            )

        history = [m.model_dump() for m in body.messages]
        events: AsyncGenerator[AgentEvent, None]
        if settings.routing and first_turn:
            events = _routed(request, settings, question, history, user_id, tickets, tel, slot)
        else:
            events = agent.run(history, ToolContext(user_id=user_id, tickets=tickets))
        hosts = settings.allowed_host_list if settings.output_guard else None
        degrade = None
        if settings.degraded_mode and first_turn:
            retriever = request.app.state.rag.retriever

            async def degrade() -> tuple[str, list[dict[str, str]]] | None:
                return await degraded_answer(retriever, question, settings.injection_policy)

        return EventSourceResponse(
            _to_sse(
                events,
                request_id,
                tel,
                cache=cache,
                slot=slot,
                allowed_hosts=hosts,
                settle=settle,
                drop_pending=request.app.state.pending.delete,
                degrade=degrade,
            )
        )
    except BaseException:
        await budget.settle(user_id, reserved, 0.0, refund=True)  # never leak a reservation
        raise


async def _replay(
    hit: CachedAnswer, request_id: str, tel: Telemetry
) -> AsyncIterator[dict[str, str]]:
    tel.metrics.requests.labels("cache_hit").inc()
    span = tel.start(
        "chat.cache_hit",
        **{"docspilot.request_id": request_id, "docspilot.cached_model": hit.model},
    )
    with tel.activate(span):
        log.info("chat_cache_hit", model=hit.model, chars=len(hit.text))
    span.end()
    yield _sse("delta", {"text": hit.text})
    if hit.citations:
        yield _sse("citations", {"sources": list(hit.citations)})
    yield _sse(
        "done",
        {
            "stop_reason": hit.stop_reason,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            "model": hit.model,
            "steps": 0,
            "request_id": request_id,
            "cached": True,
        },
    )


async def _routed(
    request: Request,
    settings: Settings,
    question: str,
    history: list[dict[str, Any]],
    user_id: str,
    tickets: TicketStore,
    tel: Telemetry,
    slot: CacheSlot | None,
) -> AsyncGenerator[AgentEvent, None]:
    llm: LLM = request.app.state.llm
    prompts: PromptRegistry = request.app.state.prompts_registry
    route = await choose_route(request.app.state.intent_classifier, message=question)
    tel.metrics.route.labels(route).inc()
    if route == "smalltalk":
        if slot is not None:
            slot.allowed = False  # a misrouted question must not poison the shared cache
        async for event in smalltalk_events(
            llm,
            message=question,
            system=prompts.get("smalltalk").text,
            model=settings.model_light,
        ):
            yield event
        return
    agent: Agent = request.app.state.agent
    async for event in agent.run(history, ToolContext(user_id=user_id, tickets=tickets)):
        yield event


class ConfirmRequest(BaseModel):
    pending_id: Annotated[str, StringConstraints(pattern=re.compile(r"^[A-Za-z0-9_-]{16,64}$"))]
    approve: bool


@router.post("/chat/confirm")
async def confirm(
    body: ConfirmRequest, request: Request, user_id: Annotated[str, Depends(current_user_id)]
) -> EventSourceResponse:
    agent: Agent = request.app.state.agent
    store: PendingStore = request.app.state.pending
    tickets: TicketStore = request.app.state.tickets
    tel: Telemetry = request.app.state.telemetry
    settings: Settings = request.app.state.settings
    # Exclusive claim: a double-click or concurrent retry cannot run the action twice.
    try:
        pending = await store.claim(body.pending_id, user_id=user_id)
    except PendingBusyError:
        raise HTTPException(409, "this confirmation is already being processed") from None
    if pending is None:
        raise HTTPException(404, "pending action not found or expired")
    try:
        reserved = await enforce_budget(request, user_id)
    except BaseException:
        await store.release(pending)  # over budget: the confirmation stays retryable
        raise
    meter = CostMeter()
    current_meter.set(meter)
    budget: Budget = request.app.state.budget

    async def settle(outcome: str) -> None:
        await budget.settle(
            user_id, reserved, meter.usd, refund=outcome != "cancelled" and meter.usd == 0
        )

    log.info("confirmation", approve=body.approve, tools=[c.name for c in pending.calls])
    events = agent.resume(
        pending, approve=body.approve, ctx=ToolContext(user_id=user_id, tickets=tickets)
    )
    hosts = settings.allowed_host_list if settings.output_guard else None
    return EventSourceResponse(
        _to_sse(
            events,
            request.state.request_id,
            tel,
            allowed_hosts=hosts,
            settle=settle,
            drop_pending=store.delete,
        )
    )
