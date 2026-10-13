"""Hybrid retrieval: vector + keyword in parallel, Reciprocal Rank Fusion, rerank.

Why hybrid: embeddings capture meaning but blur exact tokens (error codes, plan names,
"HTTP 429"); keyword search is the opposite. RRF merges the two rankings without
needing their scores to be comparable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from docspilot.rag.embeddings import Embedder
from docspilot.rag.rerank import Reranker
from docspilot.rag.store import ChunkStore
from docspilot.rag.types import SearchHit

RRF_K = 60


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    k_vector: int = 40
    k_keyword: int = 40
    n_candidates: int = 20  # fused candidates sent to the reranker
    k_final: int = 5
    #: Drop reranked hits below this relevance. None = keep all. Must be calibrated on
    #: the eval set (Phase 4): scores are model-specific and not probabilities.
    min_rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    hits: list[SearchHit]
    n_vector: int
    n_keyword: int
    n_candidates: int


def reciprocal_rank_fusion(
    vector: list[SearchHit], keyword: list[SearchHit], *, k: int = RRF_K
) -> list[SearchHit]:
    merged: dict[int, SearchHit] = {}
    for rank, hit in enumerate(vector, start=1):
        hit.vector_rank = rank
        merged[hit.chunk.id] = hit
    for rank, hit in enumerate(keyword, start=1):
        existing = merged.get(hit.chunk.id)
        if existing is None:
            hit.keyword_rank = rank
            merged[hit.chunk.id] = hit
        else:
            existing.keyword_rank = rank
    for hit in merged.values():
        score = 0.0
        if hit.vector_rank:
            score += 1 / (k + hit.vector_rank)
        if hit.keyword_rank:
            score += 1 / (k + hit.keyword_rank)
        hit.fused_score = hit.score = score
    return sorted(merged.values(), key=lambda h: -(h.fused_score or 0.0))


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        reranker: Reranker,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._cfg = config or RetrievalConfig()

    async def search(self, query: str) -> RetrievalResult:
        cfg = self._cfg
        (embedding,) = await self._embedder.embed([query], "query")
        vector, keyword = await asyncio.gather(
            self._store.search_vector(embedding, cfg.k_vector),
            self._store.search_keyword(query, cfg.k_keyword),
        )
        fused = reciprocal_rank_fusion(vector, keyword)
        candidates = fused[: cfg.n_candidates]
        final = await self._reranker.rerank(query, candidates, cfg.k_final)
        if cfg.min_rerank_score is not None:
            final = [
                h for h in final if h.rerank_score is None or h.rerank_score >= cfg.min_rerank_score
            ]
        return RetrievalResult(final, len(vector), len(keyword), len(candidates))
