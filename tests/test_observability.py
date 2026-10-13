from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from docspilot.api.app import create_app
from docspilot.cache import CachedAnswer, ResponseCache, cache_key, corpus_version, normalize
from docspilot.config import Settings
from docspilot.llm.errors import LLMError
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.types import LLMRequest, ToolCall
from docspilot.logging import _add_trace_ids
from docspilot.redact import redact
from tests.test_chat_api import parse_sse

SEARCH = ToolCall("s1", "search_docs", {"query": "refund window days"})
ANSWER = "30 days [policies/refunds.md]"
BODY = {"messages": [{"role": "user", "content": "How long is the refund window?"}]}


def base_script(req: LLMRequest) -> Script:
    """Answer path: search once, then answer. Triage/route calls get JSON."""
    if req.purpose in ("route", "triage"):
        return Script(
            text=json.dumps(
                {
                    "intent": "question",
                    "urgency": "low",
                    "product_area": "billing",
                    "summary": "asks about refunds",
                }
            )
        )
    last = req.messages[-1]
    has_results = last["role"] == "user" and isinstance(last["content"], list)
    if has_results:
        return Script(text=ANSWER)
    return Script(tool_calls=[SEARCH])


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


async def make_client(
    settings: Settings,
    llm: FakeLLM,
    exporter: InMemorySpanExporter | None = None,
    **overrides: Any,
) -> tuple[FastAPI, AsyncIterator[httpx.AsyncClient]]:
    cfg = settings.model_copy(update=overrides) if overrides else settings
    app = create_app(cfg, llm, span_exporter=exporter)

    async def gen() -> AsyncIterator[httpx.AsyncClient]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                yield c

    return app, gen()


async def open_client(
    settings: Settings, llm: FakeLLM, exporter: InMemorySpanExporter | None = None, **o: Any
) -> tuple[FastAPI, httpx.AsyncClient, Any]:
    app, gen = await make_client(settings, llm, exporter, **o)
    client = await gen.__anext__()
    return app, client, gen


async def close(gen: Any) -> None:
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


# ---- tracing ---------------------------------------------------------------------


async def test_one_trace_per_chat_with_llm_tool_and_retrieval_spans(
    settings: Settings, exporter: InMemorySpanExporter
) -> None:
    llm = FakeLLM(base_script)
    _, client, gen = await open_client(settings, llm, exporter)
    res = await client.post("/v1/chat", json=BODY)
    assert parse_sse(res.text)[-1][0] == "done"
    await close(gen)

    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert {"chat.request", "llm.answer", "tool.search_docs", "retrieval.search"} <= set(names)
    assert len({s.context.trace_id for s in spans}) == 1  # one trace for the whole turn
    by_name = {s.name: s for s in spans if s.name != "llm.answer"}
    root = by_name["chat.request"]
    assert root.parent is None
    assert by_name["tool.search_docs"].parent.span_id == root.context.span_id
    assert by_name["retrieval.search"].parent.span_id == by_name["tool.search_docs"].context.span_id

    llm_spans = [s for s in spans if s.name == "llm.answer"]
    assert len(llm_spans) == 2  # tool-use step + final answer
    attrs = llm_spans[-1].attributes
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-opus-5-5"
    assert attrs["gen_ai.usage.input_tokens"] == 10 and attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["docspilot.cost_usd"] > 0
    assert attrs["docspilot.cache_requested"] is True


async def test_content_is_not_traced_by_default_and_redacted_when_enabled(
    settings: Settings,
) -> None:
    secret_q = {"messages": [{"role": "user", "content": "mail me at jane@corp.example refund?"}]}

    off = InMemorySpanExporter()
    _, client, gen = await open_client(settings, FakeLLM(base_script), off)
    await client.post("/v1/chat", json=secret_q)
    await close(gen)
    dumped = json.dumps([dict(s.attributes) for s in off.get_finished_spans()], default=str)
    assert "jane@corp.example" not in dumped and "docspilot.answer" not in dumped

    on = InMemorySpanExporter()
    _, client, gen = await open_client(settings, FakeLLM(base_script), on, trace_content=True)
    await client.post("/v1/chat", json=secret_q)
    await close(gen)
    dumped = json.dumps([dict(s.attributes) for s in on.get_finished_spans()], default=str)
    assert "jane@corp.example" not in dumped  # redacted even when enabled
    assert "docspilot.answer" in dumped


async def test_provider_errors_mark_the_span_and_the_metric(
    settings: Settings, exporter: InMemorySpanExporter
) -> None:
    err = LLMError("boom", kind="overloaded", status_code=529)
    _, client, gen = await open_client(settings, FakeLLM([Script(error=err)]), exporter)
    res = await client.post("/v1/chat", json=BODY)
    assert parse_sse(res.text)[-1][1]["code"] == "overloaded"
    metrics = (await client.get("/metrics")).text
    await close(gen)

    llm_span = next(s for s in exporter.get_finished_spans() if s.name == "llm.answer")
    assert llm_span.status.status_code.name == "ERROR"
    assert llm_span.attributes["docspilot.error_kind"] == "overloaded"
    assert 'outcome="error_overloaded"' in metrics
    assert 'docspilot_chat_requests_total{outcome="error_overloaded"} 1.0' in metrics


def test_log_lines_carry_trace_ids(exporter: InMemorySpanExporter) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    assert "trace_id" not in _add_trace_ids(None, "", {})
    with tracer.start_as_current_span("x") as span:
        out = _add_trace_ids(None, "", {})
        assert out["trace_id"] == format(span.get_span_context().trace_id, "032x")


# ---- metrics ---------------------------------------------------------------------


async def test_metrics_endpoint_reports_tokens_cost_tools_and_retrieval(
    settings: Settings,
) -> None:
    _, client, gen = await open_client(settings, FakeLLM(base_script))
    await client.post("/v1/chat", json=BODY)
    text = (await client.get("/metrics")).text
    await close(gen)
    for series in (
        'docspilot_llm_requests_total{model="claude-opus-5-5",outcome="ok",purpose="answer"} 2.0',
        'docspilot_llm_tokens_total{kind="input",model="claude-opus-5-5",purpose="answer"} 20.0',
        'docspilot_llm_tokens_total{kind="output",model="claude-opus-5-5",purpose="answer"} 10.0',
        'docspilot_tool_calls_total{outcome="ok",tool="search_docs"} 1.0',
        "docspilot_llm_cost_usd_total",
        'docspilot_retrieval_latency_seconds_count{stage="vector"} 1.0',
        'docspilot_chat_requests_total{outcome="ok"} 1.0',
        "docspilot_chat_in_flight 0.0",
    ):
        assert series in text, series


async def test_metrics_can_be_disabled(settings: Settings) -> None:
    _, client, gen = await open_client(settings, FakeLLM(), metrics_enabled=False)
    assert (await client.get("/metrics")).status_code == 404
    await close(gen)


# ---- prompt caching --------------------------------------------------------------


async def test_cache_prefix_is_identical_across_requests(settings: Settings) -> None:
    """Prompt caching is a prefix match: tools + system must be byte-identical between
    users and requests, or every call is a cache write instead of a read."""
    llm = FakeLLM(base_script)
    _, client, gen = await open_client(settings, llm, response_cache_ttl_s=0)
    await client.post("/v1/chat", json=BODY)
    await client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "something else entirely"}]}
    )
    await close(gen)
    answer_reqs = [r for r in llm.requests if r.purpose == "answer"]
    assert len(answer_reqs) == 4
    assert all(r.cache for r in answer_reqs)
    assert len({json.dumps(r.tools, sort_keys=False) for r in answer_reqs}) == 1
    assert len({r.system for r in answer_reqs}) == 1


def test_anthropic_adapter_sets_top_level_cache_control() -> None:
    from docspilot.llm.client import AnthropicLLM

    kwargs = AnthropicLLM._build_kwargs(
        LLMRequest(model="m", messages=[{"role": "user", "content": "x"}], cache=True)
    )
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    plain = AnthropicLLM._build_kwargs(LLMRequest(model="m", messages=[]))
    assert "cache_control" not in plain


# ---- response cache --------------------------------------------------------------


async def test_identical_first_turn_question_is_served_from_cache(settings: Settings) -> None:
    llm = FakeLLM(base_script)
    _, client, gen = await open_client(settings, llm)
    first = parse_sse((await client.post("/v1/chat", json=BODY)).text)
    calls_after_first = len(llm.requests)
    second = parse_sse(
        (
            await client.post(
                "/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "  how long is the REFUND window?? "}]
                },
            )
        ).text
    )
    metrics = (await client.get("/metrics")).text
    await close(gen)

    assert len(llm.requests) == calls_after_first  # no model call at all
    assert first[-1][1]["cached"] is False and second[-1][1]["cached"] is True
    assert second[-1][1]["usage"]["output_tokens"] == 0
    assert "".join(d["text"] for e, d in second if e == "delta") == ANSWER
    assert [d for e, d in second if e == "citations"] == [d for e, d in first if e == "citations"]
    assert 'docspilot_cache_requests_total{result="hit"} 1.0' in metrics


async def test_only_single_turn_read_only_complete_answers_are_cached(
    settings: Settings,
) -> None:
    def script(req: LLMRequest) -> Script:
        last = req.messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            return Script(text="Your ticket is open.")
        return Script(tool_calls=[ToolCall("g", "get_ticket", {"ticket_id": "TCK-00000000"})])

    llm = FakeLLM(script)
    app, client, gen = await open_client(settings, llm)
    await client.post("/v1/chat", json=BODY)  # used get_ticket: user-specific
    assert len(app.state.response_cache) == 0

    multi = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "refund window?"},
        ]
    }
    await client.post("/v1/chat", json=multi)
    assert len(app.state.response_cache) == 0
    await close(gen)


def test_cache_key_changes_with_prompt_model_and_corpus() -> None:
    base = dict(prompt_sha="a", model="m", effort="medium", corpus_version="v1")
    k = cache_key("Refund window?", **base)
    assert k == cache_key("  refund   WINDOW ", **base)  # normalised
    assert k != cache_key("refund window", **{**base, "prompt_sha": "b"})
    assert k != cache_key("refund window", **{**base, "model": "other"})
    assert k != cache_key("refund window", **{**base, "corpus_version": "v2"})
    assert normalize("Hi there?!") == "hi there"
    assert corpus_version({"a.md": ("h1", "m")}) != corpus_version({"a.md": ("h2", "m")})


def test_cache_ttl_and_lru() -> None:
    now = [0.0]
    cache = ResponseCache(ttl_s=10, max_items=2, clock=lambda: now[0])
    ans = CachedAnswer("t", (), "end_turn", "m")
    cache.put("a", ans)
    cache.put("b", ans)
    assert cache.get("a") is ans  # touch a: b becomes least recently used
    cache.put("c", ans)
    assert cache.get("b") is None and cache.get("a") is ans and cache.get("c") is ans
    now[0] = 11.0
    assert cache.get("a") is None  # expired
    assert ResponseCache(ttl_s=0, max_items=5).enabled is False


# ---- routing ---------------------------------------------------------------------


def routing_script(intent: str):  # type: ignore[no-untyped-def]
    def script(req: LLMRequest) -> Script:
        if req.purpose == "route":
            return Script(
                text=json.dumps(
                    {"intent": intent, "urgency": "low", "product_area": "other", "summary": "s"}
                )
            )
        if req.purpose == "smalltalk":
            return Script(text="Hi! Happy to help with Acme Billing.")
        return base_script(req)

    return script


async def test_routing_diverts_only_smalltalk_to_the_small_model(settings: Settings) -> None:
    llm = FakeLLM(routing_script("smalltalk"))
    _, client, gen = await open_client(settings, llm, routing=True, response_cache_ttl_s=0)
    res = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi!"}]})
    metrics = (await client.get("/metrics")).text
    await close(gen)
    events = parse_sse(res.text)
    assert events[-1][1]["model"] == "claude-haiku-5-5"
    purposes = [r.purpose for r in llm.requests]
    assert purposes == ["route", "smalltalk"]
    assert llm.requests[1].tools is None and llm.requests[1].model == "claude-haiku-5-5"
    assert 'docspilot_route_total{route="smalltalk"} 1.0' in metrics


async def test_routing_sends_real_questions_to_the_agent_and_fails_open(
    settings: Settings,
) -> None:
    llm = FakeLLM(routing_script("question"))
    _, client, gen = await open_client(settings, llm, routing=True, response_cache_ttl_s=0)
    res = await client.post("/v1/chat", json=BODY)
    await close(gen)
    assert parse_sse(res.text)[-1][1]["model"] == "claude-opus-5-5"
    assert llm.requests[0].purpose == "route"

    def broken(req: LLMRequest) -> Script:
        if req.purpose == "route":
            return Script(error=LLMError("down", kind="overloaded"))
        return base_script(req)

    _, client, gen = await open_client(
        settings, FakeLLM(broken), routing=True, response_cache_ttl_s=0
    )
    res = await client.post("/v1/chat", json=BODY)
    await close(gen)
    assert parse_sse(res.text)[-1][0] == "done"  # routing failure never fails the request


async def test_follow_up_turns_are_never_routed(settings: Settings) -> None:
    llm = FakeLLM(routing_script("smalltalk"))
    _, client, gen = await open_client(settings, llm, routing=True, response_cache_ttl_s=0)
    follow_up = {
        "messages": [
            {"role": "user", "content": "my payout is stuck"},
            {"role": "assistant", "content": "Want me to open a ticket?"},
            {"role": "user", "content": "yes please"},
        ]
    }
    await client.post("/v1/chat", json=follow_up)
    await close(gen)
    assert "route" not in [r.purpose for r in llm.requests]


# ---- redaction -------------------------------------------------------------------


def test_redaction() -> None:
    text = (
        "mail jane.doe@corp.example, key sk-ant-api03-abcdefghijklmnop, "
        "card 4242 4242 4242 4242, order 1234567890123 (not a card), call +1 (415) 555-0100"
    )
    out = redact(text)
    assert "jane.doe" not in out and "sk-ant" not in out and "4242" not in out
    assert "[email]" in out and "[secret]" in out and "[card]" in out and "[phone]" in out
    assert "1234567890123" in out  # fails Luhn: left alone, not mangled
