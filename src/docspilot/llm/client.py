"""Anthropic adapter. Retries/backoff for 408/409/429/5xx are delegated to the SDK
(`max_retries`); we only add: timeouts, error taxonomy, and result normalization."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import anthropic

from docspilot.config import FALLBACK_CAPABLE_MODELS, Settings
from docspilot.llm.errors import LLMError
from docspilot.llm.types import (
    LLMEvent,
    LLMRequest,
    LLMResult,
    StreamReset,
    TextDelta,
    ToolCall,
    Usage,
)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def build_anthropic_client(settings: Settings) -> anthropic.AsyncAnthropic:
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    return anthropic.AsyncAnthropic(
        api_key=key,  # None -> SDK resolves env / profile credentials itself
        timeout=anthropic.Timeout(settings.llm_timeout_s, connect=settings.llm_connect_timeout_s),
        max_retries=settings.llm_max_retries,
    )


class AnthropicLLM:
    def __init__(self, client: anthropic.AsyncAnthropic, *, refusal_fallback: bool = True) -> None:
        self._client = client
        self._refusal_fallback = refusal_fallback

    async def stream(self, request: LLMRequest) -> AsyncGenerator[LLMEvent, None]:
        kwargs = self._build_kwargs(request)
        use_fallback = self._refusal_fallback and request.model in FALLBACK_CAPABLE_MODELS
        try:
            manager: Any
            if use_fallback:
                manager = self._client.beta.messages.stream(
                    **kwargs, betas=[_FALLBACK_BETA], fallbacks="default"
                )
            else:
                manager = self._client.messages.stream(**kwargs)
            async with manager as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield TextDelta(event.delta.text)
                    elif (
                        event.type == "content_block_start"
                        and event.content_block.type == "fallback"
                    ):
                        # Refused model's partial text must not be shown with the rescue's.
                        yield StreamReset()
                final = await stream.get_final_message()
                request_id = stream.request_id
        except anthropic.APIError as exc:
            raise LLMError.from_anthropic(exc) from exc
        yield _to_result(final, request_id)

    @staticmethod
    def _build_kwargs(request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": request.messages,
        }
        if request.system is not None:
            kwargs["system"] = request.system
        if request.tools:
            kwargs["tools"] = request.tools
        output_config: dict[str, Any] = {}
        if request.effort is not None:
            output_config["effort"] = request.effort
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if output_config:
            kwargs["output_config"] = output_config
        if request.cache:
            kwargs["cache_control"] = {"type": "ephemeral"}
        # Deliberately absent: temperature/top_p (rejected by current models) and
        # `thinking` (adaptive by default; effort is the control knob).
        return kwargs


def _to_result(message: Any, request_id: str | None = None) -> LLMResult:
    blocks = list(message.content)
    # Only the final model's text counts: drop everything before the last fallback marker.
    last_fallback = max((i for i, b in enumerate(blocks) if b.type == "fallback"), default=-1)
    text = "".join(b.text for b in blocks[last_fallback + 1 :] if b.type == "text")
    u = message.usage
    iterations = getattr(u, "iterations", None) or []
    fell_back = last_fallback >= 0 or any(
        getattr(it, "type", None) == "fallback_message" for it in iterations
    )
    tool_calls = [
        ToolCall(id=b.id, name=b.name, input=dict(b.input))
        for b in blocks[last_fallback + 1 :]
        if b.type == "tool_use"
    ]
    usage = Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )
    return LLMResult(
        text=text,
        content=blocks,
        stop_reason=message.stop_reason,
        usage=usage,
        model=message.model,
        tool_calls=tool_calls,
        request_id=request_id or getattr(message, "_request_id", None),
        served_by_fallback=fell_back,
    )
