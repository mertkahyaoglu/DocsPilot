from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator
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
from docspilot.api.deps import current_user_id
from docspilot.api.schemas import ChatRequest
from docspilot.cache import CachedAnswer, CorpusVersionProvider, ResponseCache, cache_key
from docspilot.config import Settings
from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLM, StreamReset, TextDelta
from docspilot.logging import get_logger
from docspilot.pricing import cost_usd
from docspilot.prompts import PromptRegistry
from docspilot.routing import choose_route, smalltalk_events
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
    allowed: bool = True


async def _to_sse(
    events: AsyncGenerator[AgentEvent, None],
    request_id: str,
    tel: Telemetry,
    *,
    cache: ResponseCache | None = None,
    slot: CacheSlot | None = None,
) -> AsyncIterator[dict[str, str]]:
    """Translate agent events to SSE. Always ends the stream with done|error.

    Tracing note: the root span is *not* held as the current context across `yield`
    (generators can be finalized from another task on disconnect). It is activated
    around each step of the inner stream only, so child spans still parent correctly."""
    m = tel.metrics
    started = time.perf_counter()
    outcome = "error"
    text = ""
    tools: set[str] = set()
    search_ok = 0
    search_failed = 0
    root = tel.start("chat.request", **{"docspilot.request_id": request_id})
    m.in_flight.inc()
    it = events.__aiter__()
    try:
        while True:
            try:
                with tel.activate(root):
                    event = await it.__anext__()
            except StopAsyncIteration:
                break
            if isinstance(event, TextDelta):
                text += event.text
                yield _sse("delta", {"text": event.text})
            elif isinstance(event, StreamReset):
                text = ""
                yield _sse("reset", {})
            elif isinstance(event, ToolStart):
                tools.add(event.name)
                yield _sse("tool_start", {"id": event.id, "name": event.name})
            elif isinstance(event, ToolEnd):
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
                if event.stop_reason == "refusal":
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
        yield _sse(
            "error", {"code": exc.kind, "retryable": exc.retryable, "message": exc.public_message}
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
        with contextlib.suppress(Exception):
            await events.aclose()  # deterministic cleanup of the inner generators
        m.in_flight.dec()
        m.requests.labels(outcome).inc()
        root.set_attribute("docspilot.outcome", outcome)
        if outcome.startswith("error"):
            root.set_status(Status(StatusCode.ERROR, outcome))
        root.end()


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
    history = [m.model_dump() for m in body.messages]
    first_turn = len(body.messages) == 1
    question = body.messages[0].content

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
        slot = CacheSlot(key)

    events: AsyncGenerator[AgentEvent, None]
    if settings.routing and first_turn:
        events = _routed(request, settings, question, history, user_id, tickets, tel, slot)
    else:
        events = agent.run(history, ToolContext(user_id=user_id, tickets=tickets))
    return EventSourceResponse(_to_sse(events, request_id, tel, cache=cache, slot=slot))


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
    route = await choose_route(
        llm,
        message=question,
        triage_prompt=prompts.get("triage").text,
        model=settings.model_light,
    )
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
    # Exclusive claim: a double-click or concurrent retry cannot run the action twice.
    try:
        pending = await store.claim(body.pending_id, user_id=user_id)
    except PendingBusyError:
        raise HTTPException(409, "this confirmation is already being processed") from None
    if pending is None:
        raise HTTPException(404, "pending action not found or expired")
    log.info("confirmation", approve=body.approve, tools=[c.name for c in pending.calls])
    events = agent.resume(
        pending, approve=body.approve, ctx=ToolContext(user_id=user_id, tickets=tickets)
    )
    return EventSourceResponse(_to_sse(events, request.state.request_id, tel))
