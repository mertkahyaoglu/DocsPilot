from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from docspilot.config import Settings
from docspilot.llm.cloudflare import (
    INVALID_JSON_KEY,
    CloudflareLLM,
    to_openai_messages,
    to_openai_tools,
)
from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLMRequest, TextDelta, complete
from docspilot.pricing import cost_usd
from docspilot.rag.embeddings import DIM, CloudflareEmbedder, VoyageError
from docspilot.rag.rerank import CloudflareReranker
from docspilot.rag.types import SearchHit, StoredChunk


def sse(*chunks: dict[str, Any] | str) -> bytes:
    lines = [f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks]
    return "".join(lines).encode()


def chunk(delta: dict[str, Any], finish: str | None = None, **usage: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": "@cf/test/model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        out["usage"] = usage
    return out


def llm_with(handler: Any, **kw: Any) -> CloudflareLLM:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return CloudflareLLM("acct", "token", client=client, **kw)


REQ = LLMRequest(
    model="@cf/test/model",
    messages=[{"role": "user", "content": "hi"}],
    system="sys",
    max_tokens=50,
)


# ---- message / tool translation ------------------------------------------------------


def test_anthropic_style_history_becomes_openai_messages() -> None:
    history = [
        {"role": "user", "content": "refund window?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "search_docs",
                    "input": {"query": "refund"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": '{"results": []}'},
                {"type": "tool_result", "tool_use_id": "t2", "content": "boom", "is_error": True},
            ],
        },
    ]
    out = to_openai_messages(history, "SYSTEM")
    assert out[0] == {"role": "system", "content": "SYSTEM"}
    assert out[2]["tool_calls"][0] == {
        "id": "t1",
        "type": "function",
        "function": {"name": "search_docs", "arguments": '{"query": "refund"}'},
    }
    assert out[2]["content"] == "Let me check."
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"results": []}'}
    assert out[4]["content"] == "ERROR: boom"  # the error flag survives translation


def test_tool_only_assistant_turn_has_empty_string_content_not_null() -> None:
    out = to_openai_messages(
        [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "a", "name": "n", "input": {}}],
            },
        ],
        None,
    )
    assert out[1]["content"] == ""  # Workers AI rejects null next to tool_calls


def test_tool_specs_drop_anthropic_only_fields() -> None:
    spec = {
        "name": "t",
        "description": "d",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}},
    }
    (tool,) = to_openai_tools([spec])
    assert tool == {
        "type": "function",
        "function": {
            "name": "t",
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        },
    }


# ---- streaming ---------------------------------------------------------------------------


async def test_text_stream_sums_per_chunk_usage_and_maps_finish_reason() -> None:
    body = sse(
        chunk({"role": "assistant", "content": ""}, prompt_tokens=100, completion_tokens=0),
        chunk({"content": "Hel"}, completion_tokens=2),
        chunk({"content": 30}, completion_tokens=1),  # bare JSON number, seen in the wild
        chunk({"content": "!"}, "stop", completion_tokens=1),
        "[DONE]",
    )
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, content=body, headers={"cf-ray": "ray123"})

    events = [e async for e in llm_with(handler).stream(REQ)]
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hel", "30", "!"]
    result = events[-1]
    assert result.text == "Hel30!" and result.stop_reason == "end_turn"  # type: ignore[union-attr]
    assert (result.usage.input_tokens, result.usage.output_tokens) == (100, 4)  # type: ignore[union-attr]
    assert result.request_id == "ray123"  # type: ignore[union-attr]
    assert seen["stream"] is True and seen["auth"] == "Bearer token"
    assert seen["messages"][0] == {"role": "system", "content": "sys"}
    assert "tools" not in seen and "temperature" not in seen


async def test_streamed_tool_calls_are_assembled_and_replayable() -> None:
    body = sse(
        chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "search_docs"}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"query": "re'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'fund"}'}}]}, "tool_calls"),
        chunk(
            {"tool_calls": [{"index": 1, "function": {"name": "get_ticket", "arguments": "{}"}}]}
        ),
        "[DONE]",
    )
    result = await complete(llm_with(lambda r: httpx.Response(200, content=body)), REQ)
    assert result.stop_reason == "tool_use"
    assert [(c.id, c.name, c.input) for c in result.tool_calls] == [
        ("c1", "search_docs", {"query": "refund"}),
        ("call_1", "get_ticket", {}),  # missing id is synthesised
    ]
    # content blocks are what the agent loop replays: they must round-trip through the converter
    replay = to_openai_messages([{"role": "assistant", "content": result.content}], None)
    assert [c["function"]["name"] for c in replay[0]["tool_calls"]] == ["search_docs", "get_ticket"]


async def test_invalid_tool_arguments_surface_as_a_correctable_input() -> None:
    body = sse(
        chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "c", "function": {"name": "t", "arguments": "{oops"}}
                ]
            }
        ),
        "[DONE]",
    )
    result = await complete(llm_with(lambda r: httpx.Response(200, content=body)), REQ)
    assert (
        INVALID_JSON_KEY in result.tool_calls[0].input
    )  # strict schema validation then rejects it


async def test_structured_output_and_tools_are_forwarded() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=sse(chunk({"content": "{}"}, "stop"), "[DONE]"))

    req = LLMRequest(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        json_schema={"type": "object"},
        tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
    )
    await complete(llm_with(handler), req)
    assert seen["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert seen["tools"][0]["function"]["name"] == "t"


# ---- errors and retries -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (401, "auth", False),
        (404, "not_found", False),
        (400, "bad_request", False),
        (429, "rate_limited", True),
        (500, "server", True),
        (504, "timeout", True),
    ],
)
async def test_http_errors_map_to_the_shared_taxonomy(
    status: int, kind: str, retryable: bool
) -> None:
    llm = llm_with(lambda r: httpx.Response(status, json={"errors": ["x"]}), max_retries=0)
    with pytest.raises(LLMError) as info:
        await complete(llm, REQ)
    assert info.value.kind == kind and info.value.retryable is retryable
    assert info.value.status_code == status


async def test_retries_transient_errors_before_any_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("docspilot.llm.cloudflare.asyncio.sleep", no_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={})
        return httpx.Response(200, content=sse(chunk({"content": "ok"}, "stop"), "[DONE]"))

    assert (await complete(llm_with(handler), REQ)).text == "ok"
    assert calls["n"] == 3


async def test_no_retry_once_text_has_been_streamed() -> None:
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self) -> Any:
            yield sse(chunk({"content": "partial"}))
            raise httpx.ReadError("connection reset")

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, stream=Broken())

    with pytest.raises(LLMError) as info:
        await complete(llm_with(handler), REQ)
    assert info.value.kind == "connection" and calls["n"] == 1  # would duplicate shown text


async def test_stream_embedded_error_is_raised() -> None:
    body = sse({"error": {"message": "model exploded"}})
    with pytest.raises(LLMError) as info:
        await complete(llm_with(lambda r: httpx.Response(200, content=body), max_retries=0), REQ)
    assert info.value.kind == "server"


# ---- embeddings / rerank -------------------------------------------------------------------


async def test_cloudflare_embedder_batches_and_validates_shape() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["text"]
        calls.append(len(texts))
        return httpx.Response(
            200, json={"result": {"data": [[0.1] * DIM for _ in texts]}, "success": True}
        )

    client = httpx.AsyncClient(
        base_url="https://cf.test/ai/run", transport=httpx.MockTransport(handler)
    )
    emb = CloudflareEmbedder("a", "t", client=client, batch_size=2)
    out = await emb.embed(["a", "b", "c"], "document")
    assert len(out) == 3 and calls == [2, 1] and emb.name == "cloudflare:@cf/baai/bge-m3"

    bad = httpx.AsyncClient(
        base_url="https://cf.test/ai/run",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"result": {"data": [[0.1] * 5]}})
        ),
    )
    with pytest.raises(VoyageError, match="shape"):
        await CloudflareEmbedder("a", "t", client=bad).embed(["x"], "query")


async def test_cloudflare_reranker_maps_ids_and_orders_by_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["top_k"] == 2 and len(body["contexts"]) == 3
        return httpx.Response(
            200,
            json={"result": {"response": [{"id": 2, "score": 0.4}, {"id": 0, "score": 0.9}]}},
        )

    hits = [
        SearchHit(StoredChunk(i, f"{i}.md", "T", "c", "T > S", f"t{i}", None), score=0.0)
        for i in range(3)
    ]
    client = httpx.AsyncClient(
        base_url="https://cf.test/ai/run", transport=httpx.MockTransport(handler)
    )
    out = await CloudflareReranker("a", "t", client=client).rerank("q", hits, top_k=2)
    assert [h.chunk.id for h in out] == [0, 2] and out[0].rerank_score == 0.9


# ---- config / pricing -----------------------------------------------------------------------


def test_cloudflare_backend_requires_credentials_and_picks_workers_ai_models() -> None:
    with pytest.raises(ValidationError, match="CLOUDFLARE"):
        Settings(llm_backend="cloudflare", _env_file=None)  # type: ignore[call-arg]
    s = Settings(
        llm_backend="cloudflare",
        cloudflare_account_id="acct",
        cloudflare_api_token=SecretStr("tok"),
        _env_file=None,  # type: ignore[call-arg]
    )
    assert (
        s.model_answer.startswith("@cf/")
        and s.model_fallback
        and s.model_fallback.startswith("@cf/")
    )
    assert s.model_light.startswith("@cf/") and s.model_tickets.startswith("@cf/")
    assert s.resolved_embedder == "cloudflare" and s.resolved_rerank == "cloudflare"


def test_explicit_model_choice_is_not_overridden_and_voyage_wins_auto() -> None:
    s = Settings(
        llm_backend="cloudflare",
        cloudflare_account_id="acct",
        cloudflare_api_token=SecretStr("tok"),
        model_answer="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        voyage_api_key=SecretStr("pa-x"),
        _env_file=None,  # type: ignore[call-arg]
    )
    assert s.model_answer == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    assert s.resolved_embedder == "voyage"


def test_strict_env_needs_no_anthropic_key_with_cloudflare() -> None:
    s = Settings(
        env="staging",
        llm_backend="cloudflare",
        cloudflare_account_id="acct",
        cloudflare_api_token=SecretStr("tok"),
        database_url="postgresql://x/y",
        api_keys=SecretStr("alice:alice-secret-key-0001"),
        _env_file=None,  # type: ignore[call-arg]
    )
    assert s.strict and s.anthropic_api_key is None


def test_workers_ai_models_are_priced() -> None:
    from docspilot.llm.types import Usage

    cost = cost_usd("@cf/openai/gpt-oss-120b", Usage(1_000_000, 1_000_000))
    assert cost == pytest.approx(0.35 + 0.75)


async def test_daily_quota_exhaustion_is_not_retried() -> None:
    calls = {"n": 0}
    body = {
        "errors": [{"message": "you have used up your daily free allocation of 10,000 neurons"}]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json=body)

    with pytest.raises(LLMError) as info:
        await complete(llm_with(handler, max_retries=3), REQ)
    assert info.value.kind == "rate_limited" and info.value.retry_futile
    assert calls["n"] == 1  # a transient 429 is retried, this one cannot succeed
