from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from docspilot.api.schemas import ChatRequest
from docspilot.config import Settings
from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLM, LLMRequest, LLMResult, StreamReset, TextDelta
from docspilot.logging import get_logger
from docspilot.prompts import Prompt

router = APIRouter(prefix="/v1")
log = get_logger("chat")


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


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> EventSourceResponse:
    settings: Settings = request.app.state.settings
    llm: LLM = request.app.state.llm
    prompt: Prompt = request.app.state.prompts.get("answer", settings.answer_prompt_version)
    _validate_limits(body, settings)

    llm_request = LLMRequest(
        model=settings.model_answer,
        system=prompt.text,
        messages=[m.model_dump() for m in body.messages],
        max_tokens=settings.max_tokens,
        effort=settings.effort_answer,
    )

    async def events() -> AsyncIterator[dict[str, str]]:
        try:
            # aclosing: on client disconnect the SDK stream is closed deterministically.
            async with aclosing(llm.stream(llm_request)) as stream:
                async for event in stream:
                    if isinstance(event, TextDelta):
                        yield _sse("delta", {"text": event.text})
                    elif isinstance(event, StreamReset):
                        yield _sse("reset", {})
                    elif isinstance(event, LLMResult):
                        log.info(
                            "llm_call",
                            model=event.model,
                            prompt=prompt.ref,
                            prompt_sha=prompt.sha,
                            stop_reason=event.stop_reason,
                            served_by_fallback=event.served_by_fallback,
                            provider_request_id=event.request_id,
                            **asdict(event.usage),
                        )
                        if event.refused:
                            yield _sse(
                                "error",
                                {
                                    "code": "refusal",
                                    "retryable": False,
                                    "message": "I can't help with that request.",
                                },
                            )
                            return
                        yield _sse(
                            "done",
                            {
                                "stop_reason": event.stop_reason,
                                "usage": asdict(event.usage),
                                "model": event.model,
                                "prompt": prompt.ref,
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
                "error",
                {"code": exc.kind, "retryable": exc.retryable, "message": exc.public_message},
            )
        except Exception as exc:
            # HTTP 200 is already on the wire: never let a stream die silently.
            log.exception("chat_unexpected_error")
            err = LLMError.from_unexpected(exc)
            yield _sse(
                "error",
                {"code": err.kind, "retryable": False, "message": err.public_message},
            )

    return EventSourceResponse(events())
