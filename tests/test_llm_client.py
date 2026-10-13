"""AnthropicLLM against a fake SDK client: no network, but exercises real wrapper logic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from docspilot.llm.client import AnthropicLLM
from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLMRequest, LLMResult, StreamReset, TextDelta, complete


def text_event(text: str) -> Any:
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
    )


def fallback_event() -> Any:
    return SimpleNamespace(
        type="content_block_start", content_block=SimpleNamespace(type="fallback")
    )


class FakeStream:
    request_id = "req_stream"

    def __init__(
        self,
        events: list[Any],
        final: Any,
        *,
        enter_error: Exception | None = None,
        mid_error: Exception | None = None,
    ) -> None:
        self._events, self._final = events, final
        self._enter_error, self._mid_error = enter_error, mid_error

    async def __aenter__(self) -> FakeStream:
        if self._enter_error:
            raise self._enter_error
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        for e in self._events:
            yield e
        if self._mid_error:
            raise self._mid_error

    async def get_final_message(self) -> Any:
        return self._final


def message(
    text: str = "hello", stop_reason: str = "end_turn", blocks: list[Any] | None = None
) -> Any:
    blocks = blocks if blocks is not None else [SimpleNamespace(type="text", text=text)]
    usage = SimpleNamespace(
        input_tokens=11,
        output_tokens=7,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=3,
        iterations=None,
    )
    return SimpleNamespace(
        content=blocks, stop_reason=stop_reason, usage=usage, model="claude-opus-5-5"
    )


class FakeSDK:
    def __init__(self, stream: FakeStream) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._stream = stream
        self.messages = SimpleNamespace(stream=self._make("stable"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._make("beta")))

    def _make(self, label: str) -> Any:
        def stream(**kwargs: Any) -> FakeStream:
            self.calls.append((label, kwargs))
            return self._stream

        return stream


REQ = LLMRequest(
    model="claude-opus-5-5",
    messages=[{"role": "user", "content": "hi"}],
    system="sys",
    effort="medium",
)


def llm_for(stream: FakeStream, *, fallback: bool = False) -> tuple[AnthropicLLM, FakeSDK]:
    sdk = FakeSDK(stream)
    return AnthropicLLM(sdk, refusal_fallback=fallback), sdk  # type: ignore[arg-type]


async def test_streams_text_then_result_with_normalized_usage() -> None:
    llm, _ = llm_for(FakeStream([text_event("hel"), text_event("lo")], message()))
    events = [e async for e in llm.stream(REQ)]
    assert events[:2] == [TextDelta("hel"), TextDelta("lo")]
    result = events[2]
    assert isinstance(result, LLMResult)
    assert result.text == "hello"
    assert result.usage.cache_read_input_tokens == 0  # None normalized
    assert result.usage.cache_creation_input_tokens == 3
    assert result.request_id == "req_stream"  # taken from the stream, not the snapshot


async def test_request_shape_uses_fallback_beta_only_for_capable_models() -> None:
    llm, sdk = llm_for(FakeStream([], message()), fallback=True)
    await complete(llm, REQ)
    label, kwargs = sdk.calls[-1]
    assert label == "beta"
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert kwargs["output_config"] == {"effort": "medium"}
    assert "temperature" not in kwargs and "thinking" not in kwargs

    haiku = LLMRequest(model="claude-haiku-5-5", messages=REQ.messages)
    await complete(llm, haiku)
    assert sdk.calls[-1][0] == "stable"

    llm_off, sdk_off = llm_for(FakeStream([], message()), fallback=False)
    await complete(llm_off, REQ)
    assert sdk_off.calls[-1][0] == "stable"


async def test_tool_calls_extracted_and_raw_blocks_preserved() -> None:
    blocks = [
        SimpleNamespace(type="text", text="let me look"),
        SimpleNamespace(type="tool_use", id="tu_1", name="search", input={"q": "x"}),
    ]
    llm, _ = llm_for(FakeStream([], message(stop_reason="tool_use", blocks=blocks)))
    result = await complete(llm, REQ)
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].input == {"q": "x"}
    assert result.content == blocks  # unchanged, for replay


async def test_fallback_resets_stream_and_keeps_only_rescue_text() -> None:
    final_blocks = [
        SimpleNamespace(type="text", text="I cannot ass"),
        SimpleNamespace(type="fallback"),
        SimpleNamespace(type="text", text="Sure, here is"),
    ]
    events = [text_event("I cannot ass"), fallback_event(), text_event("Sure, here is")]
    llm, _ = llm_for(FakeStream(events, message(blocks=final_blocks)), fallback=True)
    out = [e async for e in llm.stream(REQ)]
    assert out[:3] == [TextDelta("I cannot ass"), StreamReset(), TextDelta("Sure, here is")]
    result = out[3]
    assert isinstance(result, LLMResult)
    assert result.text == "Sure, here is"
    assert result.served_by_fallback is True


def _status_error(
    cls: type[anthropic.APIStatusError], status: int, body: object = None
) -> anthropic.APIStatusError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req, headers={"request-id": "req_err"})
    return cls("nope", response=resp, body=body)


@pytest.mark.parametrize(
    ("exc", "kind", "retryable"),
    [
        (_status_error(anthropic.RateLimitError, 429), "rate_limited", True),
        (_status_error(anthropic.AuthenticationError, 401), "auth", False),
        (_status_error(anthropic.NotFoundError, 404), "not_found", False),
        (_status_error(anthropic.BadRequestError, 400), "bad_request", False),
        (_status_error(anthropic.InternalServerError, 500), "server", True),
        (anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")), "timeout", True),
        (
            anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")),
            "connection",
            True,
        ),
        # Errors sent *inside* a 200 SSE stream: classified from the body.
        (
            _status_error(
                anthropic.APIStatusError,
                200,
                {"type": "error", "error": {"type": "overloaded_error", "message": "x"}},
            ),
            "overloaded",
            True,
        ),
        (
            _status_error(
                anthropic.APIStatusError,
                200,
                {"type": "error", "error": {"type": "rate_limit_error", "message": "x"}},
            ),
            "rate_limited",
            True,
        ),
        (_status_error(anthropic.APIStatusError, 200, None), "unknown", False),
    ],
)
@pytest.mark.parametrize("where", ["enter", "mid_stream"])
async def test_error_taxonomy(
    exc: anthropic.APIError, kind: str, retryable: bool, where: str
) -> None:
    stream = FakeStream(
        [text_event("par")],
        None,
        enter_error=exc if where == "enter" else None,
        mid_error=exc if where == "mid_stream" else None,
    )
    llm, _ = llm_for(stream)
    with pytest.raises(LLMError) as info:
        await complete(llm, REQ)
    assert info.value.kind == kind
    assert info.value.retryable is retryable
    assert "nope" not in info.value.public_message
