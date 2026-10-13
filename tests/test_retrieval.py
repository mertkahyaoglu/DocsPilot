from __future__ import annotations

from pathlib import Path

import pytest

from docspilot.rag.embeddings import HashEmbedder
from docspilot.rag.ingest import ingest_corpus
from docspilot.rag.rerank import NoopReranker
from docspilot.rag.retrieve import RetrievalConfig, Retriever, reciprocal_rank_fusion
from docspilot.rag.store import InMemoryChunkStore
from docspilot.rag.types import SearchHit, StoredChunk

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


def hit(id_: int, path: str = "a.md") -> SearchHit:
    return SearchHit(
        chunk=StoredChunk(id_, path, "T", "c", "T > S", f"text {id_}", None), score=0.0
    )


def test_rrf_rewards_agreement_between_retrievers() -> None:
    vector = [hit(1), hit(2), hit(3)]
    keyword = [hit(3), hit(4)]
    fused = reciprocal_rank_fusion(vector, keyword)
    ids = [h.chunk.id for h in fused]
    assert ids[0] == 3  # found by both beats rank-1 in only one list
    assert set(ids) == {1, 2, 3, 4}
    top = fused[0]
    assert (top.vector_rank, top.keyword_rank) == (3, 1)
    assert top.fused_score and top.fused_score > fused[-1].fused_score  # type: ignore[operator]


@pytest.fixture(scope="module")
async def retriever() -> Retriever:
    store, embedder = InMemoryChunkStore(), HashEmbedder()
    report = await ingest_corpus(CORPUS, store, embedder)
    assert report.chunks_written > 50
    return Retriever(store, embedder, NoopReranker(), RetrievalConfig(k_final=5))


@pytest.mark.parametrize(
    ("query", "expected_path"),
    [
        ("how long do I have to request a refund", "policies/refunds.md"),
        ("why was my webhook endpoint disabled", "troubleshooting/legacy-webhook-notes.md"),
        ("what are the API rate limits per plan", "api/rate-limits.md"),
        ("Growth plan monthly price", "billing/pricing-and-plans.md"),
    ],
)
async def test_known_questions_retrieve_the_right_document(
    retriever: Retriever, query: str, expected_path: str
) -> None:
    result = await retriever.search(query)
    assert expected_path in [h.chunk.path for h in result.hits]


async def test_empty_or_stopword_only_query_does_not_crash(retriever: Retriever) -> None:
    result = await retriever.search("the of and")
    assert result.n_keyword == 0
    assert len(result.hits) <= 5
