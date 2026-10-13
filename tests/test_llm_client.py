"""AnthropicLLM against a fake SDK client: no network, but exercises real wrapper logic."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from docspilot.llm.client import AnthropicLLM
from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLMRequest, LLMResult, TextDelta, complete


class FakeStream:
    def __init__(self, chunks: list[str], final: Any, error: Exception | None = None) -> None:
        self._chunks, self._final, self._error = chunks, final, error

    async def __aenter__(self) -> FakeStream:
        if self._error:
            raise self._error
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def text_stream(self):  # type: ignore[no-untyped-def]
        async def gen():  # type: ignore[no-untyped-def]
            for c in self._chunks:
                yield c

        return gen()

    async def get_final_message(self) -> Any:
        return self._final


def message(text: str = "hello", stop_reason: str = "end_turn", extra: list[Any] | None = None):  # type: ignore[no-untyped-def]
    blocks = [SimpleNamespace(type="text", text=text), *(extra or [])]
    usage = SimpleNamespace(
        input_tokens=11,
        output_tokens=7,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=3,
    )
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        model="claude-opus-5-5",
        _request_id="req_1",
    )


class FakeSDK:
    def __init__(self, stream: FakeStream) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._stream = stream
        self.messages = SimpleNamespace(stream=self._make("stable"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._make("beta")))

    def _make(self, label: str):  # type: ignore[no-untyped-def]
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


async def test_streams_text_then_result_with_normalized_usage() -> None:
    sdk = FakeSDK(FakeStream(["hel", "lo"], message()))
    llm = AnthropicLLM(sdk, refusal_fallback=True)  # type: ignore[arg-type]
    events = [e async for e in llm.stream(REQ)]
    assert events[:2] == [TextDelta("hel"), TextDelta("lo")]
    result = events[2]
    assert isinstance(result, LLMResult)
    assert result.text == "hello"
    assert result.usage.cache_read_input_tokens == 0  # None normalized
    assert result.usage.cache_creation_input_tokens == 3
    assert result.request_id == "req_1"


async def test_request_shape_uses_fallback_beta_only_for_capable_models() -> None:
    sdk = FakeSDK(FakeStream([], message()))
    await complete(AnthropicLLM(sdk, refusal_fallback=True), REQ)  # type: ignore[arg-type]
    label, kwargs = sdk.calls[-1]
    assert label == "beta"
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert kwargs["output_config"] == {"effort": "medium"}
    assert "temperature" not in kwargs and "thinking" not in kwargs

    haiku = LLMRequest(model="claude-haiku-5-5", messages=REQ.messages)
    await complete(AnthropicLLM(sdk, refusal_fallback=True), haiku)  # type: ignore[arg-type]
    assert sdk.calls[-1][0] == "stable"

    await complete(AnthropicLLM(sdk, refusal_fallback=False), REQ)  # type: ignore[arg-type]
    assert sdk.calls[-1][0] == "stable"


async def test_tool_calls_and_fallback_marker_extracted() -> None:
    tool = SimpleNamespace(type="tool_use", id="tu_1", name="search", input={"q": "x"})
    fb = SimpleNamespace(type="fallback")
    sdk = FakeSDK(FakeStream([], message(stop_reason="tool_use", extra=[tool, fb])))
    result = await complete(AnthropicLLM(sdk, refusal_fallback=False), REQ)  # type: ignore[arg-type]
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].input == {"q": "x"}
    assert result.served_by_fallback is True
    assert len(result.content) == 3  # raw blocks preserved for replay


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req, headers={"request-id": "req_err"})
    return cls("nope", response=resp, body=None)


@pytest.mark.parametrize(
    ("exc", "kind", "retryable"),
    [
        (_status_error(anthropic.RateLimitError, 429), "rate_limited", True),
        (_status_error(anthropic.AuthenticationError, 401), "auth", False),
        (_status_error(anthropic.NotFoundError, 404), "not_found", False),
        (_status_error(anthropic.BadRequestError, 400), "bad_request", False),
        (_status_error(anthropic.InternalServerError, 500), "server", True),
        (
            anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")),
            "timeout",
            True,
        ),
        (
            anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")),
            "connection",
            True,
        ),
    ],
)
async def test_error_taxonomy(exc: anthropic.APIError, kind: str, retryable: bool) -> None:
    sdk = FakeSDK(FakeStream([], None, error=exc))
    llm = AnthropicLLM(sdk, refusal_fallback=False)  # type: ignore[arg-type]
    with pytest.raises(LLMError) as info:
        await complete(llm, REQ)
    assert info.value.kind == kind
    assert info.value.retryable is retryable
    assert "nope" not in info.value.public_message
