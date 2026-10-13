from __future__ import annotations

import json

import httpx

from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.types import ToolCall
from tests.test_chat_api import parse_sse

CREATE = ToolCall(
    id="w1",
    name="create_ticket",
    input={
        "title": "Cannot change currency",
        "description": "Currency dropdown is disabled on the billing page.",
        "severity": "normal",
    },
)
BODY = {"messages": [{"role": "user", "content": "please open a ticket"}]}


async def test_confirmation_round_trip(app, use_llm, client: httpx.AsyncClient) -> None:  # type: ignore[no-untyped-def]
    llm = FakeLLM([Script(tool_calls=[CREATE]), Script(text="Ticket created.")])
    use_llm(llm)

    res = await client.post("/v1/chat", json=BODY)
    events = parse_sse(res.text)
    assert [e for e, _ in events] == ["confirmation_required", "done"]
    required = events[0][1]
    assert required["items"][0]["tool"] == "create_ticket"
    assert required["items"][0]["arguments"]["title"] == "Cannot change currency"
    assert app.state.tickets._by_id == {}

    confirm = {"pending_id": required["pending_id"], "approve": True}
    res2 = await client.post("/v1/chat/confirm", json=confirm)
    events2 = parse_sse(res2.text)
    assert [e for e, _ in events2][-1] == "done"
    assert "".join(d["text"] for e, d in events2 if e == "delta") == "Ticket created."
    assert len(app.state.tickets._by_id) == 1

    # single use: replaying the confirmation is a 404, not a second ticket
    res3 = await client.post("/v1/chat/confirm", json=confirm)
    assert res3.status_code == 404
    assert len(app.state.tickets._by_id) == 1


async def test_reject_creates_nothing(app, use_llm, client: httpx.AsyncClient) -> None:  # type: ignore[no-untyped-def]
    use_llm(FakeLLM([Script(tool_calls=[CREATE]), Script(text="No problem.")]))
    res = await client.post("/v1/chat", json=BODY)
    pid = parse_sse(res.text)[0][1]["pending_id"]
    res2 = await client.post("/v1/chat/confirm", json={"pending_id": pid, "approve": False})
    assert res2.status_code == 200
    assert app.state.tickets._by_id == {}


async def test_confirm_rejects_malformed_and_unknown_ids(client: httpx.AsyncClient) -> None:
    res = await client.post("/v1/chat/confirm", json={"pending_id": "../x", "approve": True})
    assert res.status_code == 422
    res = await client.post("/v1/chat/confirm", json={"pending_id": "A" * 32, "approve": True})
    assert res.status_code == 404


async def test_triage_uses_light_model_and_schema(use_llm, client: httpx.AsyncClient) -> None:  # type: ignore[no-untyped-def]
    payload = {
        "intent": "ticket_request",
        "urgency": "high",
        "product_area": "payments",
        "summary": "Customer was double charged.",
    }
    llm = FakeLLM([Script(text=json.dumps(payload))])
    use_llm(llm)
    res = await client.post("/v1/triage", json={"message": "I was charged twice!!"})
    assert res.status_code == 200
    assert res.json()["triage"] == payload
    sent = llm.requests[0]
    assert sent.model == "claude-haiku-5-5" and sent.effort == "low"
    assert sent.json_schema and sent.json_schema["additionalProperties"] is False
    assert sent.tools is None


async def test_triage_bad_output_is_502(use_llm, client: httpx.AsyncClient) -> None:  # type: ignore[no-untyped-def]
    use_llm(FakeLLM([Script(text='{"intent": "banana"}')]))
    res = await client.post("/v1/triage", json={"message": "hi"})
    assert res.status_code == 502
    use_llm(FakeLLM([Script(text="", stop_reason="refusal")]))
    res = await client.post("/v1/triage", json={"message": "hi"})
    assert res.status_code == 502
