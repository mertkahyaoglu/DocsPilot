from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

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


async def test_user_specific_tool_turns_are_not_cached(settings: Settings) -> None:
    def script(req: LLMRequest) -> Script:
        last = req.messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            return Script(text="No such ticket.")
        return Script(tool_calls=[ToolCall("g", "get_ticket", {"ticket_id": "TCK-00000000"})])

    app, client, gen = await open_client(settings, FakeLLM(script))
    await client.post("/v1/chat", json=BODY)
    await close(gen)
    assert len(app.state.response_cache) == 0


async def test_follow_up_turns_are_not_cached_even_when_search_only(settings: Settings) -> None:
    app, client, gen = await open_client(settings, FakeLLM(base_script))
    multi = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "How long is the refund window?"},
        ]
    }
    res = await client.post("/v1/chat", json=multi)
    assert parse_sse(res.text)[-1][0] == "done"
    # same script, first turn: this *would* be cached, so the multi-turn rule is what held
    await client.post("/v1/chat", json=BODY)
    await close(gen)
    assert len(app.state.response_cache) == 1


async def test_answers_built_on_failed_or_degraded_retrieval_are_never_cached(
    settings: Settings,
) -> None:
    app, client, gen = await open_client(settings, FakeLLM(base_script))
    store = app.state.rag.store
    real_vec, real_kw = store.search_vector, store.search_keyword

    async def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("db blip")

    # 1) one leg down: results are usable but degraded
    store.search_vector = boom
    await client.post("/v1/chat", json=BODY)
    assert len(app.state.response_cache) == 0
    # 2) both legs down: the tool errors, the model apologises
    store.search_keyword = boom
    res = await client.post("/v1/chat", json=BODY)
    assert parse_sse(res.text)[-1][0] == "done"
    assert len(app.state.response_cache) == 0
    # 3) healthy again: now it is cacheable
    store.search_vector, store.search_keyword = real_vec, real_kw
    await client.post("/v1/chat", json=BODY)
    assert len(app.state.response_cache) == 1
    await close(gen)


async def test_answers_without_any_search_are_not_cached(settings: Settings) -> None:
    app, client, gen = await open_client(settings, FakeLLM([Script(text="I think it is 30 days.")]))
    await client.post("/v1/chat", json=BODY)
    await close(gen)
    assert len(app.state.response_cache) == 0


async def test_corpus_version_provider_picks_up_reingest_without_restart() -> None:
    from docspilot.cache import CorpusVersionProvider

    now = [0.0]
    state = {"a.md": ("h1", "m")}

    async def fetch() -> dict[str, tuple[str, str]]:
        return dict(state)

    provider = CorpusVersionProvider(fetch, refresh_s=30, clock=lambda: now[0])
    v1 = await provider.get()
    state["a.md"] = ("h2", "m")  # another process re-ingested
    assert await provider.get() == v1  # within the refresh window
    now[0] = 31.0
    assert await provider.get() != v1


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


# ---- review round: lifecycle, safety, config --------------------------------------


async def test_routed_smalltalk_never_populates_the_shared_cache(settings: Settings) -> None:
    app, client, gen = await open_client(
        settings, FakeLLM(routing_script("smalltalk")), routing=True
    )
    # a (mis)classified short product question answered by the small model, no tools
    res = await client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "refund window?"}]}
    )
    assert parse_sse(res.text)[-1][1]["model"] == "claude-haiku-5-5"
    assert len(app.state.response_cache) == 0
    await close(gen)


async def test_long_or_mixed_messages_are_not_diverted_to_smalltalk(settings: Settings) -> None:
    llm = FakeLLM(routing_script("smalltalk"))
    _, client, gen = await open_client(settings, llm, routing=True, response_cache_ttl_s=0)
    long_msg = "thanks! " + "also, please open a ticket about my payout being stuck. " * 5
    await client.post("/v1/chat", json={"messages": [{"role": "user", "content": long_msg}]})
    await close(gen)
    assert "smalltalk" not in [r.purpose for r in llm.requests]


async def test_routing_fails_open_on_unexpected_exceptions(settings: Settings) -> None:
    def script(req: LLMRequest) -> Script:
        if req.purpose == "route":
            raise RuntimeError("router bug")
        return base_script(req)

    _, client, gen = await open_client(
        settings, FakeLLM(script), routing=True, response_cache_ttl_s=0
    )
    res = await client.post("/v1/chat", json=BODY)
    await close(gen)
    assert parse_sse(res.text)[-1][0] == "done"


async def test_cancellation_is_not_an_error_and_spans_still_end(
    exporter: InMemorySpanExporter,
) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from docspilot.llm.instrumented import InstrumentedLLM
    from docspilot.telemetry import Metrics, Telemetry

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tel = Telemetry(Metrics(), provider.get_tracer("t"))
    llm = InstrumentedLLM(FakeLLM([Script(text="a long streamed answer " * 5)]), tel)

    gen = llm.stream(LLMRequest(model="claude-opus-5-5", messages=[]))
    await gen.__anext__()  # first delta, then the client "disconnects"
    await gen.aclose()

    series = {
        (s.labels["outcome"]): s.value
        for metric in tel.metrics.registry.collect()
        if metric.name == "docspilot_llm_requests"
        for s in metric.samples
        if s.name.endswith("_total")
    }
    assert series == {"cancelled": 1.0}
    (span,) = exporter.get_finished_spans()
    assert span.attributes["docspilot.outcome"] == "cancelled"


async def test_sse_wrapper_counts_cancellation_and_releases_in_flight(
    exporter: InMemorySpanExporter,
) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from docspilot.api.chat import _to_sse
    from docspilot.llm.types import TextDelta
    from docspilot.telemetry import Metrics, Telemetry

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tel = Telemetry(Metrics(), provider.get_tracer("t"))

    async def events():  # type: ignore[no-untyped-def]
        for i in range(100):
            yield TextDelta(f"chunk{i}")

    out = _to_sse(events(), "req-1234567890", tel)
    await out.__anext__()
    assert tel.metrics.in_flight._value.get() == 1.0
    await out.aclose()  # disconnect
    assert tel.metrics.in_flight._value.get() == 0.0
    assert tel.metrics.requests.labels("cancelled")._value.get() == 1.0
    assert tel.metrics.requests.labels("error")._value.get() == 0.0
    (root,) = exporter.get_finished_spans()
    assert root.name == "chat.request" and root.attributes["docspilot.outcome"] == "cancelled"


async def test_error_outcome_sets_root_span_error_once(
    settings: Settings, exporter: InMemorySpanExporter
) -> None:
    err = LLMError("boom", kind="overloaded", status_code=529)
    _, client, gen = await open_client(settings, FakeLLM([Script(error=err)]), exporter)
    await client.post("/v1/chat", json=BODY)
    await close(gen)
    root = next(s for s in exporter.get_finished_spans() if s.name == "chat.request")
    assert root.status.status_code.name == "ERROR"
    assert root.attributes["docspilot.outcome"] == "error_overloaded"
    llm_span = next(s for s in exporter.get_finished_spans() if s.name == "llm.answer")
    assert len([e for e in llm_span.events if e.name == "exception"]) == 1


async def test_cache_hits_are_visible_in_traces(
    settings: Settings, exporter: InMemorySpanExporter
) -> None:
    _, client, gen = await open_client(settings, FakeLLM(base_script), exporter)
    await client.post("/v1/chat", json=BODY)
    exporter.clear()
    await client.post("/v1/chat", json=BODY)
    await close(gen)
    assert [s.name for s in exporter.get_finished_spans()] == ["chat.cache_hit"]


async def test_fallback_and_unpriced_models_are_counted() -> None:
    from docspilot.llm.instrumented import InstrumentedLLM
    from docspilot.llm.types import LLMResult, Usage
    from docspilot.telemetry import Telemetry

    class Rescued:
        async def stream(self, request):  # type: ignore[no-untyped-def]
            yield LLMResult(
                "ok", [], "end_turn", Usage(10, 5), "claude-opus-4-8", served_by_fallback=True
            )

    class Mystery:
        async def stream(self, request):  # type: ignore[no-untyped-def]
            yield LLMResult("ok", [], "end_turn", Usage(10, 5), "claude-mystery-9")

    tel = Telemetry.noop()
    for inner in (Rescued(), Mystery()):
        async for _ in InstrumentedLLM(inner, tel).stream(  # type: ignore[arg-type]
            LLMRequest(model="claude-fable-5-1", messages=[])
        ):
            pass
    m = tel.metrics
    assert m.llm_fallback.labels("claude-fable-5-1", "claude-opus-4-8")._value.get() == 1.0
    assert m.llm_unpriced.labels("claude-mystery-9")._value.get() == 1.0


def test_otlp_headers_are_url_decoded_and_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    from docspilot import telemetry

    seen: dict[str, Any] = {}

    class FakeExporter:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        def export(self, spans: Any) -> Any:  # pragma: no cover
            return None

        def shutdown(self) -> None: ...

        def force_flush(self, timeout_millis: int = 0) -> bool:
            return True

    monkeypatch.setattr(telemetry, "OTLPSpanExporter", FakeExporter)
    cfg = Settings(
        otlp_endpoint="http://collector:4318/",
        otlp_headers="authorization=Basic%20abc123, x-team = core",
        _env_file=None,  # type: ignore[call-arg]
    )
    provider = telemetry.setup_tracing(cfg)
    assert provider is not None
    assert seen["headers"] == {"authorization": "Basic abc123", "x-team": "core"}
    assert seen["endpoint"] == "http://collector:4318/v1/traces"
    provider.shutdown()


async def test_metrics_endpoint_requires_the_token_when_configured(settings: Settings) -> None:
    _, client, gen = await open_client(settings, FakeLLM(), metrics_token=SecretStr("s3cret"))
    assert (await client.get("/metrics")).status_code == 401
    assert (
        await client.get("/metrics", headers={"authorization": "Bearer wrong"})
    ).status_code == 401
    ok = await client.get("/metrics", headers={"authorization": "Bearer s3cret"})
    assert ok.status_code == 200 and "docspilot_" in ok.text
    await close(gen)


def test_metrics_route_is_absent_in_prod_without_a_token(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")

    def paths(**kw: Any) -> set[str]:
        cfg = Settings(
            env="prod",
            database_url="postgresql://x/y",
            api_keys="alice:alice-secret-key-0001",  # type: ignore[arg-type]
            _env_file=None,
            **kw,
        )
        return {getattr(r, "path", "") for r in create_app(cfg, FakeLLM()).routes}

    assert "/metrics" not in paths()
    assert "/metrics" in paths(metrics_token="t")


def test_redaction_covers_common_token_shapes() -> None:
    cases = {
        "sk_live_51HxabcDEF123456789": "sk_live",
        "whsec_abcdefghijklmnop1234": "whsec_",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789": "ghp_",
        "AKIAIOSFODNN7EXAMPLE": "AKIA",
        "xoxb-123456789012-abcdefghijkl": "xoxb",
        "authorization: bearer abcdefghijklmnop": "abcdefghijklmnop",
        "Authorization: Basic dXNlcjpwYXNzd29yZDEyMw==": "dXNlcjpw",
        "postgres://admin:hunter2pass@db.example.com/x": "hunter2pass",
        "iban GB82 WEST 1234 5698 7654 32": "WEST 1234",
        "ssn 078-05-1120": "078-05-1120",
    }
    for text, leak in cases.items():
        assert leak not in redact(text), text
    assert (
        redact("invoice INV-2024-0042 total $1,240.00") == "invoice INV-2024-0042 total $1,240.00"
    )
