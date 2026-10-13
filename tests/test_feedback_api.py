from __future__ import annotations

import httpx

from tests.test_chat_api import parse_sse


async def test_done_event_carries_request_id_and_feedback_is_stored(
    app,
    client: httpx.AsyncClient,  # type: ignore[no-untyped-def]
) -> None:
    res = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    done = parse_sse(res.text)[-1][1]
    assert done["request_id"] == res.headers["x-request-id"]

    fb = await client.post(
        "/v1/feedback",
        json={"request_id": done["request_id"], "rating": "down", "comment": "  wrong plan  "},
    )
    assert fb.status_code == 201
    stored = app.state.feedback.items[0]
    assert (stored.rating, stored.comment, stored.user_id) == ("down", "wrong plan", "anonymous")
    assert stored.request_id == res.headers["x-request-id"]


async def test_feedback_validation(client: httpx.AsyncClient) -> None:
    bad = [
        {"request_id": "x", "rating": "up"},  # id too short
        {"request_id": "abcdefgh12", "rating": "meh"},
        {"request_id": "abcdefgh12", "rating": "up", "comment": "x" * 501},
        {"request_id": "bad id with spaces!", "rating": "up"},
    ]
    for payload in bad:
        assert (await client.post("/v1/feedback", json=payload)).status_code == 422, payload


async def test_feedback_is_one_vote_per_answer_and_strips_nul(
    app,
    client: httpx.AsyncClient,  # type: ignore[no-untyped-def]
) -> None:
    rid = "req-abcdef12"
    await client.post("/v1/feedback", json={"request_id": rid, "rating": "up"})
    res = await client.post(
        "/v1/feedback", json={"request_id": rid, "rating": "down", "comment": "bad\u0000data"}
    )
    assert res.status_code == 201
    items = app.state.feedback.items
    assert len(items) == 1  # double votes replace, they do not stack
    assert (items[0].rating, items[0].comment) == ("down", "baddata")
