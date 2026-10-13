from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from docspilot.api.schemas import ChatRequest
from docspilot.config import Settings
from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLM, LLMRequest, LLMResult, TextDelta
from docspilot.logging import get_logger
from docspilot.prompts import Prompt

router = APIRouter(prefix="/v1")
log = get_logger("chat")


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> EventSourceResponse:
    settings: Settings = request.app.state.settings
    llm: LLM = request.app.state.llm
    prompt: Prompt = request.app.state.prompts.get("answer", settings.answer_prompt_version)

    if len(body.messages) > settings.max_history_messages:
        raise HTTPException(422, f"too many messages (max {settings.max_history_messages})")
    if any(len(m.content) > settings.max_message_chars for m in body.messages):
        raise HTTPException(422, f"message too long (max {settings.max_message_chars} chars)")

    llm_request = LLMRequest(
        model=settings.model_answer,
        system=prompt.text,
        messages=[m.model_dump() for m in body.messages],
        max_tokens=settings.max_tokens,
        effort=settings.effort_answer,
    )

    async def events() -> AsyncIterator[dict[str, str]]:
        try:
            async for event in llm.stream(llm_request):
                if isinstance(event, TextDelta):
                    yield _sse("delta", {"text": event.text})
                elif isinstance(event, LLMResult):
                    log.info(
                        "llm_call",
                        model=event.model,
                        prompt=prompt.ref,
                        prompt_sha=prompt.sha,
                        stop_reason=event.stop_reason,
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

    return EventSourceResponse(events())
