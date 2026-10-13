from __future__ import annotations

import json
import re
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
from docspilot.agent.pending import PendingStore
from docspilot.agent.tools import ToolContext
from docspilot.api.deps import current_user_id
from docspilot.api.schemas import ChatRequest
from docspilot.config import Settings
from docspilot.llm.errors import LLMError
from docspilot.llm.types import StreamReset, TextDelta
from docspilot.logging import get_logger
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


async def _to_sse(events: AsyncGenerator[AgentEvent, None]) -> AsyncIterator[dict[str, str]]:
    """Translate agent events to SSE. Always ends the stream with done|error."""
    try:
        async with aclosing(events) as stream:
            async for event in stream:
                if isinstance(event, TextDelta):
                    yield _sse("delta", {"text": event.text})
                elif isinstance(event, StreamReset):
                    yield _sse("reset", {})
                elif isinstance(event, ToolStart):
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
                    yield _sse("done", {"stop_reason": "awaiting_confirmation"})
                    return
                elif isinstance(event, AgentFinished):
                    if event.stop_reason == "refusal":
                        yield _sse("error", _ERROR_REFUSAL)
                    elif event.stop_reason == "max_steps":
                        yield _sse("error", _ERROR_STEPS)
                    else:
                        yield _sse(
                            "done",
                            {
                                "stop_reason": event.stop_reason,
                                "usage": asdict(event.usage),
                                "model": event.model,
                                "steps": event.steps,
                            },
                        )
    except LLMError as exc:
        log.warning(
            "llm_error",
            kind=exc.kind,
            status=exc.status_code,
            provider_request_id=exc.request_id,
            detail=str(exc),
        )
        yield _sse(
            "error", {"code": exc.kind, "retryable": exc.retryable, "message": exc.public_message}
        )
    except Exception as exc:
        # HTTP 200 is already on the wire: never let a stream die silently.
        log.exception("chat_unexpected_error")
        err = LLMError.from_unexpected(exc)
        yield _sse("error", {"code": err.kind, "retryable": False, "message": err.public_message})


@router.post("/chat")
async def chat(
    body: ChatRequest, request: Request, user_id: Annotated[str, Depends(current_user_id)]
) -> EventSourceResponse:
    settings: Settings = request.app.state.settings
    agent: Agent = request.app.state.agent
    tickets: TicketStore = request.app.state.tickets
    _validate_limits(body, settings)
    history = [m.model_dump() for m in body.messages]
    events = agent.run(history, ToolContext(user_id=user_id, tickets=tickets))
    return EventSourceResponse(_to_sse(events))


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
    # Single use: popping first means a double-click cannot run the action twice.
    pending = await store.pop(body.pending_id, user_id=user_id)
    if pending is None:
        raise HTTPException(404, "pending action not found or expired")
    log.info("confirmation", approve=body.approve, tools=[c.name for c in pending.calls])
    events = agent.resume(
        pending, approve=body.approve, ctx=ToolContext(user_id=user_id, tickets=tickets)
    )
    return EventSourceResponse(_to_sse(events))
