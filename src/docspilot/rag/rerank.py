"""Rerankers: a cross-encoder reads (query, passage) together, so it is far more
accurate than embedding similarity but too slow to run over the whole corpus.
Pattern: cheap recall (vector + keyword) -> small candidate set -> rerank."""

from __future__ import annotations

from typing import Protocol

import httpx

from docspilot.rag.embeddings import VoyageError, _post_with_retries
from docspilot.rag.types import SearchHit


class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...


class NoopReranker:
    name = "none"

    async def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        return hits[:top_k]


class VoyageReranker:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "rerank-2.5",
        base_url: str = "https://api.voyageai.com/v1",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = f"voyage:{model}"
        self._model = model
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(15.0, connect=5.0),
        )

    async def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        data = await _post_with_retries(
            self._client,
            "/rerank",
            {
                "query": query,
                "documents": [h.chunk.heading_path + "\n\n" + h.chunk.text for h in hits],
                "model": self._model,
                "top_k": min(top_k, len(hits)),
            },
        )
        ranked: list[SearchHit] = []
        try:
            for row in data["data"]:
                hit = hits[row["index"]]
                hit.rerank_score = float(row["relevance_score"])
                hit.score = hit.rerank_score
                ranked.append(hit)
        except (KeyError, IndexError, TypeError) as exc:
            raise VoyageError("unexpected rerank response shape") from exc
        return ranked

    async def aclose(self) -> None:
        await self._client.aclose()


class CloudflareReranker:
    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        model: str = "@cf/baai/bge-reranker-base",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = f"cloudflare:{model}"
        self._model = model
        self._client = client or httpx.AsyncClient(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run",
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=httpx.Timeout(15.0, connect=5.0),
        )

    async def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        data = await _post_with_retries(
            self._client,
            f"/{self._model}",
            {
                "query": query,
                "contexts": [{"text": h.chunk.heading_path + "\n\n" + h.chunk.text} for h in hits],
                "top_k": min(top_k, len(hits)),
            },
        )
        ranked: list[SearchHit] = []
        try:
            for row in (data.get("result") or {})["response"]:
                hit = hits[row["id"]]
                hit.rerank_score = float(row["score"])
                hit.score = hit.rerank_score
                ranked.append(hit)
        except (KeyError, IndexError, TypeError) as exc:
            raise VoyageError("unexpected rerank response shape") from exc
        return sorted(ranked, key=lambda h: -(h.rerank_score or 0.0))

    async def aclose(self) -> None:
        await self._client.aclose()
