"""Hybrid retrieval: vector + keyword in parallel, Reciprocal Rank Fusion, rerank.

Why hybrid: embeddings capture meaning but blur exact tokens (error codes, plan names,
"HTTP 429"); keyword search is the opposite. RRF merges the two rankings without
needing their scores to be comparable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from docspilot.logging import get_logger
from docspilot.rag.embeddings import Embedder
from docspilot.rag.rerank import Reranker
from docspilot.rag.store import ChunkStore
from docspilot.rag.types import SearchHit

RRF_K = 60
log = get_logger("retrieve")


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
        """Each leg may fail independently: losing the embedding provider degrades to
        keyword-only, a bad keyword query degrades to vector-only, a rerank outage to the
        fused order. Only losing *everything* is an error."""
        cfg = self._cfg
        vector_task = asyncio.create_task(self._vector(query))
        keyword_task = asyncio.create_task(self._store.search_keyword(query, cfg.k_keyword))
        vector_res, keyword_res = await asyncio.gather(
            vector_task, keyword_task, return_exceptions=True
        )
        if isinstance(vector_res, BaseException) and isinstance(keyword_res, BaseException):
            raise vector_res
        vector = [] if isinstance(vector_res, BaseException) else vector_res
        keyword = [] if isinstance(keyword_res, BaseException) else keyword_res
        for leg, res in (("vector", vector_res), ("keyword", keyword_res)):
            if isinstance(res, BaseException):
                log.warning("retrieval_leg_failed", leg=leg, error=repr(res))

        fused = reciprocal_rank_fusion(vector, keyword)
        candidates = fused[: cfg.n_candidates]
        try:
            final = await self._reranker.rerank(query, candidates, cfg.k_final)
        except Exception as exc:
            log.warning("rerank_failed_using_fused_order", error=repr(exc))
            final = candidates[: cfg.k_final]
        if cfg.min_rerank_score is not None:
            final = [
                h for h in final if h.rerank_score is None or h.rerank_score >= cfg.min_rerank_score
            ]
        return RetrievalResult(final, len(vector), len(keyword), len(candidates))

    async def _vector(self, query: str) -> list[SearchHit]:
        (embedding,) = await self._embedder.embed([query], "query")
        return await self._store.search_vector(embedding, self._cfg.k_vector)
