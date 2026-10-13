from __future__ import annotations

import json

import httpx

from docspilot.llm.errors import LLMError
from docspilot.llm.fake import FakeLLM, Script


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        event, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((event, json.loads(data)))
    return events


async def test_healthz(client: httpx.AsyncClient) -> None:
    res = await client.get("/healthz")
    assert res.status_code == 200
    assert res.headers["x-request-id"]


async def test_request_id_is_propagated(client: httpx.AsyncClient) -> None:
    res = await client.get("/healthz", headers={"x-request-id": "abcdef123456"})
    assert res.headers["x-request-id"] == "abcdef123456"
    res = await client.get("/healthz", headers={"x-request-id": "bad id with spaces"})
    assert res.headers["x-request-id"] != "bad id with spaces"


async def test_chat_streams_deltas_then_done(client: httpx.AsyncClient, fake_llm: FakeLLM) -> None:
    res = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert res.status_code == 200
    events = parse_sse(res.text)
    assert [e for e, _ in events][-1] == "done"
    assert "".join(d["text"] for e, d in events if e == "delta") == "ok"
    done = events[-1][1]
    assert done["usage"]["input_tokens"] == 10
    assert done["prompt"] == "answer@v1"
    # the request actually carried the system prompt, effort and the conversation
    sent = fake_llm.requests[0]
    assert sent.system and "DocsPilot" in sent.system
    assert sent.effort == "medium"
    assert sent.messages == [{"role": "user", "content": "hi"}]


async def test_chat_maps_provider_errors_to_error_event(app, client: httpx.AsyncClient) -> None:
    app.state.llm = FakeLLM([Script(error=LLMError("boom", kind="overloaded", status_code=529))])
    res = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    events = parse_sse(res.text)
    assert events == [
        (
            "error",
            {
                "code": "overloaded",
                "retryable": True,
                "message": "The assistant is temporarily overloaded. Please retry shortly.",
            },
        )
    ]
    assert "boom" not in res.text  # provider detail never leaks to clients


async def test_chat_refusal_becomes_error_event(app, client: httpx.AsyncClient) -> None:
    app.state.llm = FakeLLM([Script(text="", stop_reason="refusal")])
    res = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "x"}]})
    events = parse_sse(res.text)
    assert events[-1][0] == "error" and events[-1][1]["code"] == "refusal"


async def test_chat_validates_conversation_shape(client: httpx.AsyncClient) -> None:
    bad = [
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "hi"}]},
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ]
        },
    ]
    for payload in bad:
        res = await client.post("/v1/chat", json=payload)
        assert res.status_code == 422, payload


async def test_chat_enforces_configured_limits(client: httpx.AsyncClient) -> None:
    long = "x" * 4001
    res = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": long}]})
    assert res.status_code == 422
