from __future__ import annotations

import json

import httpx
import pytest

from docspilot.rag.embeddings import DIM, VoyageEmbedder, VoyageError
from docspilot.rag.rerank import VoyageReranker
from docspilot.rag.types import SearchHit, StoredChunk


def client_for(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="https://voyage.test/v1", transport=handler)


async def test_embeddings_are_batched_and_reordered() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        # return rows reversed to prove we sort by index
        data = [
            {"index": i, "embedding": [float(len(body["input"][i]))] * DIM}
            for i in reversed(range(len(body["input"])))
        ]
        return httpx.Response(200, json={"data": data})

    emb = VoyageEmbedder("k", client=client_for(httpx.MockTransport(handler)), batch_size=2)
    out = await emb.embed(["a", "bb", "ccc"], "document")
    assert [v[0] for v in out] == [1.0, 2.0, 3.0]
    assert [len(c["input"]) for c in calls] == [2, 1]
    assert calls[0]["input_type"] == "document" and calls[0]["model"] == "voyage-3.5"


async def test_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("docspilot.rag.embeddings.asyncio.sleep", _no_sleep)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, json={})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * DIM}]})

    emb = VoyageEmbedder("k", client=client_for(httpx.MockTransport(handler)))
    await emb.embed(["x"], "query")
    assert attempts["n"] == 3


async def test_client_errors_are_not_retried_and_wrong_shape_rejected() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(401, json={})

    emb = VoyageEmbedder("k", client=client_for(httpx.MockTransport(handler)))
    with pytest.raises(VoyageError):
        await emb.embed(["x"], "query")
    assert attempts["n"] == 1

    def bad_shape(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 3}]})

    emb2 = VoyageEmbedder("k", client=client_for(httpx.MockTransport(bad_shape)))
    with pytest.raises(VoyageError, match="shape"):
        await emb2.embed(["x"], "query")


async def test_reranker_orders_by_relevance_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "rerank-2.5" and body["top_k"] == 2
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                ]
            },
        )

    hits = [
        SearchHit(StoredChunk(i, f"{i}.md", "T", "c", "T > S", f"t{i}", None), score=0.0)
        for i in range(3)
    ]
    rr = VoyageReranker("k", client=client_for(httpx.MockTransport(handler)))
    out = await rr.rerank("q", hits, top_k=2)
    assert [h.chunk.id for h in out] == [2, 0]
    assert out[0].rerank_score == 0.9


async def _no_sleep(_: float) -> None:
    return None
