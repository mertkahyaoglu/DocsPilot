from __future__ import annotations

import httpx
import pytest

from docspilot.config import Settings
from docspilot.llm.errors import LLMError
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.resilient import CircuitBreaker, ResilientLLM
from docspilot.llm.types import (
    LLMRequest,
    LLMResult,
    StreamReset,
    TextDelta,
    complete,
)
from docspilot.telemetry import Metrics
from tests.test_chat_api import parse_sse
from tests.test_observability import base_script, close, open_client

DOWN = LLMError("overloaded", kind="overloaded", status_code=529)
REQ = LLMRequest(model="claude-opus-5-5", messages=[{"role": "user", "content": "hi"}])


def test_circuit_breaker_opens_probes_once_and_recovers() -> None:
    now = [0.0]
    cb = CircuitBreaker(threshold=3, recovery_s=10, clock=lambda: now[0])
    assert cb.state == "closed" and cb.allow()
    for _ in range(3):
        cb.record_failure()
    assert cb.state == "open" and not cb.allow()
    now[0] = 10.0
    assert cb.state == "half_open"
    assert cb.allow() and not cb.allow()  # exactly one probe in flight
    cb.record_failure()  # probe failed: straight back to open
    assert cb.state == "open"
    now[0] = 25.0
    assert cb.allow()
    cb.record_success()
    assert cb.state == "closed" and cb.allow()


class Scripted:
    """Inner LLM whose behaviour depends on the model it is asked for."""

    def __init__(self, plan: dict[str, Script]) -> None:
        self.plan = plan
        self.calls: list[str] = []

    async def stream(self, request: LLMRequest):  # type: ignore[no-untyped-def]
        self.calls.append(request.model)
        script = self.plan[request.model]
        async for e in FakeLLM([script]).stream(request):
            yield e


async def test_failover_to_secondary_model_and_metrics() -> None:
    inner = Scripted(
        {"claude-opus-5-5": Script(error=DOWN), "claude-sonnet-5-5": Script(text="from sonnet")}
    )
    metrics = Metrics()
    llm = ResilientLLM(inner, fallback_model="claude-sonnet-5-5", metrics=metrics)  # type: ignore[arg-type]
    result = await complete(llm, REQ)
    assert result.text == "from sonnet" and result.model == "claude-sonnet-5-5"
    assert inner.calls == ["claude-opus-5-5", "claude-sonnet-5-5"]
    assert metrics.llm_failover.labels("claude-opus-5-5", "claude-sonnet-5-5")._value.get() == 1.0


async def test_partial_output_from_the_failed_model_is_reset() -> None:
    class Dies:
        async def stream(self, request: LLMRequest):  # type: ignore[no-untyped-def]
            if request.model == "claude-opus-5-5":
                yield TextDelta("half an ans")
                raise DOWN
            yield TextDelta("complete answer")
            yield LLMResult(
                "complete answer", [], "end_turn", FakeLLM()._scripts[0].usage, request.model
            )  # type: ignore[attr-defined]

    llm = ResilientLLM(Dies(), fallback_model="claude-sonnet-5-5")  # type: ignore[arg-type]
    events = [e async for e in llm.stream(REQ)]
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["TextDelta", "StreamReset", "TextDelta", "LLMResult"]
    assert isinstance(events[1], StreamReset)


async def test_non_retryable_errors_do_not_fail_over() -> None:
    bad = LLMError("bad request", kind="bad_request", status_code=400)
    inner = Scripted({"claude-opus-5-5": Script(error=bad), "claude-sonnet-5-5": Script(text="x")})
    llm = ResilientLLM(inner, fallback_model="claude-sonnet-5-5")  # type: ignore[arg-type]
    with pytest.raises(LLMError) as info:
        await complete(llm, REQ)
    assert info.value.kind == "bad_request" and inner.calls == ["claude-opus-5-5"]


async def test_only_the_answer_path_fails_over() -> None:
    inner = Scripted(
        {"claude-haiku-5-5": Script(error=DOWN), "claude-sonnet-5-5": Script(text="x")}
    )
    llm = ResilientLLM(inner, fallback_model="claude-sonnet-5-5")  # type: ignore[arg-type]
    triage = LLMRequest(model="claude-haiku-5-5", messages=[], purpose="triage")
    with pytest.raises(LLMError):
        await complete(llm, triage)
    assert inner.calls == ["claude-haiku-5-5"]


async def test_open_breaker_skips_the_primary_until_recovery() -> None:
    now = [0.0]
    inner = Scripted(
        {"claude-opus-5-5": Script(error=DOWN), "claude-sonnet-5-5": Script(text="ok")}
    )
    llm = ResilientLLM(
        inner,  # type: ignore[arg-type]
        fallback_model="claude-sonnet-5-5",
        threshold=2,
        recovery_s=30,
        clock=lambda: now[0],
    )
    for _ in range(2):
        await complete(llm, REQ)  # two failures trip the breaker
    inner.calls.clear()
    await complete(llm, REQ)
    assert inner.calls == ["claude-sonnet-5-5"]  # primary not even attempted
    assert llm.states()["claude-opus-5-5"] == "open"

    now[0] = 31.0
    inner.plan["claude-opus-5-5"] = Script(text="recovered")
    inner.calls.clear()
    assert (await complete(llm, REQ)).text == "recovered"  # half-open probe succeeded
    assert inner.calls == ["claude-opus-5-5"] and llm.states()["claude-opus-5-5"] == "closed"


async def test_everything_down_raises_the_last_error() -> None:
    inner = Scripted(
        {"claude-opus-5-5": Script(error=DOWN), "claude-sonnet-5-5": Script(error=DOWN)}
    )
    llm = ResilientLLM(inner, fallback_model="claude-sonnet-5-5")  # type: ignore[arg-type]
    with pytest.raises(LLMError):
        await complete(llm, REQ)


# ---- degraded mode through the API -----------------------------------------------


def all_down(req: LLMRequest) -> Script:
    return Script(error=DOWN)


async def test_degraded_mode_serves_retrieval_only_answers(settings: Settings) -> None:
    app, client, gen = await open_client(
        settings, FakeLLM(all_down), degraded_mode=True, model_fallback=None
    )
    res = await client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "refund window days?"}]}
    )
    events = parse_sse(res.text)
    names = [e for e, _ in events]
    assert names == ["delta", "citations", "done"]
    text = events[0][1]["text"]
    assert "can't generate an answer right now" in text and "[policies/refunds" in text
    done = events[-1][1]
    assert done["degraded"] is True and done["stop_reason"] == "degraded" and done["model"] is None
    cites = events[1][1]["sources"]
    assert cites and all(c["path"].endswith(".md") for c in cites)
    metrics = (await client.get("/metrics")).text
    await close(gen)
    assert "docspilot_degraded_answers_total 1.0" in metrics
    assert len(app.state.response_cache) == 0  # degraded answers are never cached


async def test_degraded_mode_is_off_for_follow_ups_and_when_disabled(settings: Settings) -> None:
    _, client, gen = await open_client(settings, FakeLLM(all_down), degraded_mode=True)
    follow = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "refund window?"},
        ]
    }
    res = await client.post("/v1/chat", json=follow)
    assert parse_sse(res.text)[-1][0] == "error"
    await close(gen)

    _, client, gen = await open_client(settings, FakeLLM(all_down), degraded_mode=False)
    res = await client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "refund window?"}]}
    )
    assert parse_sse(res.text)[-1][0] == "error"
    await close(gen)


async def test_degraded_mode_does_not_replace_an_answer_already_started(
    settings: Settings,
) -> None:
    def script(req: LLMRequest) -> Script:
        last = req.messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            return Script(error=DOWN)  # dies on the step after the tool results
        return Script(
            text="Let me check the docs. ",
            tool_calls=[
                __import__("docspilot.llm.types", fromlist=["ToolCall"]).ToolCall(
                    "s", "search_docs", {"query": "refund window"}
                )
            ],
        )

    _, client, gen = await open_client(
        settings, FakeLLM(script), degraded_mode=True, model_fallback=None
    )
    res = await client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "refund window?"}]}
    )
    events = parse_sse(res.text)
    await close(gen)
    assert events[-1][0] == "error"  # text was already shown: no confusing second answer


async def test_failover_is_wired_into_the_app_and_visible_in_readyz(settings: Settings) -> None:
    def script(req: LLMRequest) -> Script:
        if req.model == "claude-opus-5-5":
            return Script(error=DOWN)
        return base_script(req)

    # the real wiring: App -> Resilient -> Instrumented -> (fake) provider

    from docspilot.api.app import create_app

    app = create_app(settings.model_copy(update={"response_cache_ttl_s": 0}), FakeLLM(script))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            body = {"messages": [{"role": "user", "content": "refund window?"}]}
            res = await client.post("/v1/chat", json=body)
            done = parse_sse(res.text)[-1][1]
            assert done["model"] == "claude-sonnet-5-5"  # served by the secondary
            ready = (await client.get("/readyz")).json()
            assert ready["status"] == "ready" and "claude-opus-5-5" in ready["breakers"]
            version = (await client.get("/version")).json()
            assert version["version"] and "git_sha" in version
