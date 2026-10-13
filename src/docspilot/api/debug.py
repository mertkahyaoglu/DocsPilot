"""Retrieval debugging. Inspect retrieval on its own, without the model in the way:
most "bad answers" are retrieval bugs. Mounted only outside prod."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from docspilot.api.deps import current_user_id
from docspilot.rag.factory import RagStack

router = APIRouter(prefix="/v1/debug", tags=["debug"], dependencies=[Depends(current_user_id)])


@router.get("/search")
async def debug_search(
    request: Request, q: Annotated[str, Query(min_length=2, max_length=300)]
) -> dict[str, Any]:
    rag: RagStack = request.app.state.rag
    result = await rag.retriever.search(q)
    return {
        "query": q,
        "embedder": rag.embedder.name,
        "reranker": rag.reranker.name,
        "vector_candidates": result.n_vector,
        "keyword_candidates": result.n_keyword,
        "fused_candidates": result.n_candidates,
        "hits": [
            {
                "path": h.chunk.path,
                "section": h.chunk.heading_path,
                "updated": h.chunk.updated.isoformat() if h.chunk.updated else None,
                "score": round(h.score, 4),
                "vector_rank": h.vector_rank,
                "keyword_rank": h.keyword_rank,
                "fused_score": round(h.fused_score, 5) if h.fused_score else None,
                "rerank_score": h.rerank_score,
                "preview": h.chunk.text[:160],
            }
            for h in result.hits
        ],
    }
