"""Cloudflare Workers AI adapter (OpenAI-compatible chat completions endpoint).

Same `LLM` interface as the Anthropic adapter, so the agent loop, evals, failover and
instrumentation work unchanged. What does *not* carry over, by nature of the provider:
prompt caching (`cache`), `effort`, thinking blocks and the safety-classifier refusal
fallback are ignored; tool-calling quality depends on the chosen open model.

Internally conversations use Anthropic-style content blocks (that is what the agent
loop replays). This adapter translates them to/from OpenAI chat messages."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from docspilot.llm.errors import ErrorKind, LLMError
from docspilot.llm.types import LLMEvent, LLMRequest, LLMResult, TextDelta, ToolCall, Usage
from docspilot.logging import get_logger

log = get_logger("cloudflare_llm")

API = "https://api.cloudflare.com/client/v4"
_FINISH = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
_RETRY_STATUS = {429, 500, 502, 503, 504}
INVALID_JSON_KEY = "__invalid_json__"


def to_openai_messages(
    messages: list[dict[str, Any]], system: str | list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        text = (
            system
            if isinstance(system, str)
            else "\n\n".join(b.get("text", "") for b in system if b.get("type") == "text")
        )
        out.append({"role": "system", "content": text})
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            out.append({"role": "system", "content": str(content)})
            continue
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            texts = [b["text"] for b in content if _kind(b) == "text"]
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                }
                for b in content
                if _kind(b) == "tool_use"
            ]
            # "" not null: Workers AI's schema rejects a null content next to tool_calls
            entry: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
        else:  # user turn made of tool_result and/or text blocks
            for b in content:
                if _kind(b) == "tool_result":
                    body = (
                        b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                    )
                    if b.get("is_error"):
                        body = f"ERROR: {body}"
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body})
                elif _kind(b) == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


def _kind(block: Any) -> str:
    return str(block.get("type")) if isinstance(block, dict) else str(getattr(block, "type", ""))


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],  # `strict` is Anthropic-only: dropped
            },
        }
        for t in tools
    ]


def _is_quota(status: int, body: str) -> bool:
    """The free tier's *daily* allocation: backoff cannot fix it before the next reset."""
    return status == 429 and "daily" in body.lower() and "allocation" in body.lower()


def _status_kind(status: int) -> ErrorKind:
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limited"
    if status in (408, 504):
        return "timeout"
    if status >= 500:
        return "server"
    if status >= 400:
        return "bad_request"
    return "unknown"


class CloudflareLLM:
    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 60.0,
        connect_timeout_s: float = 5.0,
        max_retries: int = 2,
    ) -> None:
        self._url = f"{API}/accounts/{account_id}/ai/v1/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_token}"
        }  # per request: works with any client
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s)
        )
        self._owns_client = client is None
        self._max_retries = max_retries

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": to_openai_messages(request.messages, request.system),
            "max_tokens": request.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            payload["tools"] = to_openai_tools(request.tools)
        if request.json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.json_schema},
            }
        return payload

    async def stream(self, request: LLMRequest) -> AsyncGenerator[LLMEvent, None]:
        payload = self._payload(request)
        attempt = 0
        while True:
            emitted = False
            try:
                async with self._client.stream(
                    "POST", self._url, json=payload, headers=self._headers
                ) as res:
                    if res.status_code != 200:
                        body = (await res.aread()).decode(errors="replace")[:300]
                        raise LLMError(
                            f"cloudflare {res.status_code}: {body}",
                            kind=_status_kind(res.status_code),
                            status_code=res.status_code,
                            request_id=res.headers.get("cf-ray"),
                            retry_futile=_is_quota(res.status_code, body),
                        )
                    text_parts: list[str] = []
                    calls: dict[int, dict[str, Any]] = {}
                    usage = Usage()
                    finish: str | None = None
                    served_model = request.model
                    async for line in res.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if "error" in chunk and "choices" not in chunk:
                            raise LLMError(
                                f"cloudflare stream error: {str(chunk['error'])[:300]}",
                                kind="server",
                            )
                        served_model = chunk.get("model") or served_model
                        u = chunk.get("usage") or {}
                        # Workers AI reports per-chunk (not cumulative) usage: sum it.
                        usage = usage + Usage(
                            input_tokens=int(u.get("prompt_tokens") or 0),
                            output_tokens=int(u.get("completion_tokens") or 0),
                        )
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            raw_piece = delta.get("content")
                            # Workers AI can emit a bare number ("30") as a JSON number
                            piece = "" if raw_piece is None else str(raw_piece)
                            if piece:
                                emitted = True
                                text_parts.append(piece)
                                yield TextDelta(piece)
                            for tc in delta.get("tool_calls") or []:
                                slot = calls.setdefault(
                                    int(tc.get("index", 0)), {"id": None, "name": "", "args": ""}
                                )
                                slot["id"] = tc.get("id") or slot["id"]
                                fn = tc.get("function") or {}
                                slot["name"] = fn.get("name") or slot["name"]
                                slot["args"] += fn.get("arguments") or ""
                            finish = choice.get("finish_reason") or finish
                    yield self._result(
                        text_parts, calls, usage, finish, served_model, res.headers.get("cf-ray")
                    )
                    return
            except LLMError as exc:
                if exc.retry_futile or not (
                    exc.retryable and not emitted and attempt < self._max_retries
                ):
                    raise
                error: LLMError = exc
            except httpx.TimeoutException as exc:
                if emitted or attempt >= self._max_retries:
                    raise LLMError(str(exc) or "timeout", kind="timeout") from exc
                error = LLMError("timeout", kind="timeout")
            except httpx.TransportError as exc:
                if emitted or attempt >= self._max_retries:
                    raise LLMError(str(exc) or "connection error", kind="connection") from exc
                error = LLMError("connection error", kind="connection")
            attempt += 1
            delay = 0.5 * 2 ** (attempt - 1) + random.uniform(0, 0.25)  # noqa: S311
            log.warning("cloudflare_retry", attempt=attempt, kind=error.kind)
            await asyncio.sleep(delay)

    @staticmethod
    def _result(
        text_parts: list[str],
        calls: dict[int, dict[str, Any]],
        usage: Usage,
        finish: str | None,
        model: str,
        request_id: str | None,
    ) -> LLMResult:
        text = "".join(text_parts)
        tool_calls: list[ToolCall] = []
        blocks: list[Any] = []
        if text:
            blocks.append({"type": "text", "text": text})
        for i, idx in enumerate(sorted(calls)):
            slot = calls[idx]
            raw = slot["args"] or "{}"
            try:
                parsed = json.loads(raw)
                args = parsed if isinstance(parsed, dict) else {INVALID_JSON_KEY: raw[:500]}
            except json.JSONDecodeError:
                # Surfaces as a validation error the model can read and correct.
                args = {INVALID_JSON_KEY: raw[:500]}
            call_id = slot["id"] or f"call_{i}"
            tool_calls.append(ToolCall(id=call_id, name=slot["name"], input=args))
            blocks.append({"type": "tool_use", "id": call_id, "name": slot["name"], "input": args})
        if tool_calls:
            stop = "tool_use"
        elif finish == "content_filter":
            stop = "refusal"
        else:
            stop = _FINISH.get(finish or "stop", "end_turn")
        return LLMResult(
            text=text,
            content=blocks,
            stop_reason=stop,
            usage=usage,
            model=model,
            tool_calls=tool_calls,
            request_id=request_id,
        )
