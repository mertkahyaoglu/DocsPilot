"""Retrieval evals: no LLM, deterministic, cheap. Run on every PR.

Answers are only as good as the passages they are built from, so measure retrieval
separately: when an answer is wrong, this tells you which half is broken."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from docspilot.evals.dataset import GoldenCase
from docspilot.evals.metrics import (
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
    unique_in_order,
)
from docspilot.rag.embeddings import Embedder
from docspilot.rag.rerank import NoopReranker, Reranker
from docspilot.rag.retrieve import RetrievalConfig, Retriever
from docspilot.rag.store import ChunkStore

Strategy = Callable[[str], Awaitable[list[str]]]  # query -> ranked document paths


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    n: int
    recall: float
    hit: float
    mrr: float
    k: int
    per_case: dict[str, float]  # case id -> recall@k (for pinpointing failures)

    def as_dict(self) -> dict[str, float]:
        return {f"recall@{self.k}": self.recall, f"hit@{self.k}": self.hit, "mrr": self.mrr}


async def evaluate(strategy: Strategy, cases: list[GoldenCase], k: int) -> RetrievalMetrics:
    targets = [c for c in cases if c.has_retrieval_target]
    recalls, hits, rrs = [], [], []
    per_case: dict[str, float] = {}
    for case in targets:
        # Every rung is compared at the same unit: the top-k *documents*. Chunk lists are
        # deduped first, otherwise a rung that returns several chunks per document is
        # silently handicapped against one that returns many documents.
        ranked = unique_in_order(await strategy(case.question))[:k]
        r = recall_at_k(ranked, case.expected_sources, k)
        recalls.append(r)
        hits.append(hit_at_k(ranked, case.expected_sources, k))
        rrs.append(reciprocal_rank(ranked, case.expected_sources))
        per_case[case.id] = r
    return RetrievalMetrics(len(targets), mean(recalls), mean(hits), mean(rrs), k, per_case)


def strategies(
    store: ChunkStore, embedder: Embedder, reranker: Reranker, *, k: int, candidates: int
) -> dict[str, Strategy]:
    """The ablation ladder: each rung adds one idea, so you can see what it buys."""

    async def vector_only(q: str) -> list[str]:
        (emb,) = await embedder.embed([q], "query")
        return [h.chunk.path for h in await store.search_vector(emb, 40)]

    async def keyword_only(q: str) -> list[str]:
        return [h.chunk.path for h in await store.search_keyword(q, 40)]

    # `hybrid` ranks documents (all fused candidates). `hybrid@chunks` is the production
    # setting: only the top-k *chunks* reach the model, which may cover fewer documents.
    hybrid = Retriever(
        store,
        embedder,
        NoopReranker(),
        RetrievalConfig(k_final=candidates, n_candidates=candidates),
    )
    hybrid_chunks = Retriever(
        store, embedder, NoopReranker(), RetrievalConfig(k_final=k, n_candidates=candidates)
    )

    async def hybrid_fn(q: str) -> list[str]:
        return [h.chunk.path for h in (await hybrid.search(q)).hits]

    async def hybrid_chunks_fn(q: str) -> list[str]:
        return [h.chunk.path for h in (await hybrid_chunks.search(q)).hits]

    ladder: dict[str, Strategy] = {
        "vector": vector_only,
        "keyword": keyword_only,
        "hybrid": hybrid_fn,
        "hybrid@chunks": hybrid_chunks_fn,
    }
    if not isinstance(reranker, NoopReranker):
        full = Retriever(
            store, embedder, reranker, RetrievalConfig(k_final=candidates, n_candidates=candidates)
        )

        async def hybrid_rerank(q: str) -> list[str]:
            return [h.chunk.path for h in (await full.search(q)).hits]

        ladder["hybrid+rerank"] = hybrid_rerank
    return ladder


async def run_retrieval_eval(
    cases: list[GoldenCase],
    store: ChunkStore,
    embedder: Embedder,
    reranker: Reranker,
    *,
    k: int = 5,
    candidates: int = 20,
) -> dict[str, RetrievalMetrics]:
    return {
        name: await evaluate(strategy, cases, k)
        for name, strategy in strategies(
            store, embedder, reranker, k=k, candidates=candidates
        ).items()
    }
