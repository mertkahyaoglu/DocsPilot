from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
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
from docspilot.cache import CachedAnswer, ResponseCache, cache_key
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


async def _to_sse(
    events: AsyncGenerator[AgentEvent, None],
    request_id: str,
    tel: Telemetry,
    *,
    cache: ResponseCache | None = None,
    cache_k: str | None = None,
) -> AsyncIterator[dict[str, str]]:
    """Translate agent events to SSE. Always ends the stream with done|error."""
    m = tel.metrics
    started = time.perf_counter()
    outcome = "error"
    text = ""
    tools: set[str] = set()
    citations: list[dict[str, str]] = []
    m.in_flight.inc()
    try:
        with tel.span("chat.request", **{"docspilot.request_id": request_id}) as span:
            try:
                async with aclosing(events) as stream:
                    async for event in stream:
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
                                        {
                                            "tool": i.tool,
                                            "summary": i.summary,
                                            "arguments": i.arguments,
                                        }
                                        for i in event.items
                                    ],
                                },
                            )
                            yield _sse(
                                "done",
                                {"stop_reason": "awaiting_confirmation", "request_id": request_id},
                            )
                            return
                        elif isinstance(event, AgentFinished):
                            cost = cost_usd(event.model, event.usage)
                            span.set_attributes(
                                {
                                    "docspilot.stop_reason": event.stop_reason,
                                    "docspilot.steps": event.steps,
                                    "docspilot.citations": len(event.citations),
                                    "docspilot.cost_usd": -1.0 if cost is None else cost,
                                }
                            )
                            tel.content(span, "answer", event.text)
                            log.info(
                                "chat_done",
                                stop_reason=event.stop_reason,
                                steps=event.steps,
                                model=event.model,
                                cost_usd=None if cost is None else round(cost, 6),
                                latency_s=round(time.perf_counter() - started, 3),
                                tools=sorted(tools),
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
                                if (
                                    cache is not None
                                    and cache_k is not None
                                    and event.stop_reason == "end_turn"
                                    and tools <= _CACHEABLE_TOOLS
                                    and event.text.strip()
                                ):
                                    cache.put(
                                        cache_k,
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
                yield _sse(
                    "error",
                    {"code": exc.kind, "retryable": exc.retryable, "message": exc.public_message},
                )
            except Exception as exc:
                # HTTP 200 is already on the wire: never let a stream die silently.
                outcome = "error_unexpected"
                log.exception("chat_unexpected_error")
                err = LLMError.from_unexpected(exc)
                yield _sse(
                    "error", {"code": err.kind, "retryable": False, "message": err.public_message}
                )
            span.set_attribute("docspilot.outcome", outcome)
    finally:
        m.in_flight.dec()
        m.requests.labels(outcome).inc()


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

    cache_k = None
    if cache.enabled and first_turn:
        cache_k = cache_key(
            question,
            prompt_sha=request.app.state.answer_prompt.sha,
            model=settings.model_answer,
            effort=settings.effort_answer,
            corpus_version=request.app.state.corpus_version,
        )
        hit = cache.get(cache_k)
        tel.metrics.cache.labels("hit" if hit else "miss").inc()
        if hit is not None:
            return EventSourceResponse(_replay(hit, request_id, tel))

    events: AsyncGenerator[AgentEvent, None]
    if settings.routing and first_turn:
        events = _routed(request, settings, question, history, user_id, tickets, tel)
    else:
        events = agent.run(history, ToolContext(user_id=user_id, tickets=tickets))
    return EventSourceResponse(_to_sse(events, request_id, tel, cache=cache, cache_k=cache_k))


async def _replay(
    hit: CachedAnswer, request_id: str, tel: Telemetry
) -> AsyncIterator[dict[str, str]]:
    tel.metrics.requests.labels("cache_hit").inc()
    log.info("chat_cache_hit", model=hit.model, chars=len(hit.text))
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
