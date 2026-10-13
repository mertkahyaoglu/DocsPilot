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


async def test_dotted_identifiers_match_whole_and_in_parts() -> None:
    from docspilot.rag.store import keyword_terms

    terms = keyword_terms("when does the invoice.paid event fire?")
    assert "invoice.paid" in terms and "invoice" in terms and "paid" in terms
    assert keyword_terms("résumé généré") == ["résumé", "généré"]  # unicode survives


async def test_retrieval_degrades_instead_of_failing() -> None:
    class BrokenEmbedder(HashEmbedder):
        async def embed(self, texts, input_type):  # type: ignore[no-untyped-def]
            raise RuntimeError("provider down")

    class BrokenReranker(NoopReranker):
        async def rerank(self, query, hits, top_k):  # type: ignore[no-untyped-def]
            raise RuntimeError("rerank down")

    store = InMemoryChunkStore()
    await ingest_corpus(CORPUS, store, HashEmbedder())
    r = Retriever(store, BrokenEmbedder(), BrokenReranker(), RetrievalConfig(k_final=3))
    result = await r.search("webhook endpoint disabled after failures")
    assert result.n_vector == 0 and result.n_keyword > 0  # keyword-only
    assert 0 < len(result.hits) <= 3  # fused order, no rerank


async def test_total_retrieval_failure_is_an_error() -> None:
    import pytest

    class Dead(InMemoryChunkStore):
        async def search_vector(self, embedding, k):  # type: ignore[no-untyped-def]
            raise RuntimeError("db down")

        async def search_keyword(self, query, k):  # type: ignore[no-untyped-def]
            raise RuntimeError("db down")

    r = Retriever(Dead(), HashEmbedder(), NoopReranker())
    with pytest.raises(RuntimeError):
        await r.search("anything")
