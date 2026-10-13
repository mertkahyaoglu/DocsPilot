from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from docspilot.rag.types import ChunkDraft, RawDoc, SearchHit, StoredChunk

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by can do does for from how i if in is it my of on or our so that "
    "the their there this to was we what when where which who why will with you your".split()
)


def keyword_terms(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for w in _WORD.findall(text.lower()):
        if len(w) > 1 and w not in _STOP:
            seen.setdefault(w)
    return list(seen)[:24]


class ChunkStore(Protocol):
    async def document_state(self) -> dict[str, tuple[str, str]]:
        """path -> (content_hash, embedding_model)"""
        ...

    async def replace_document(
        self,
        doc: RawDoc,
        chunks: list[ChunkDraft],
        embeddings: list[list[float]],
        embedding_model: str,
    ) -> None: ...

    async def delete_documents(self, paths: list[str]) -> None: ...
    async def search_vector(self, embedding: list[float], k: int) -> list[SearchHit]: ...
    async def search_keyword(self, query: str, k: int) -> list[SearchHit]: ...
    async def count_chunks(self) -> int: ...


class InMemoryChunkStore:
    """Brute-force cosine + BM25. For tests and key-less local dev; same contract as pgvector."""

    def __init__(self) -> None:
        self._docs: dict[str, tuple[RawDoc, str]] = {}
        self._chunks: dict[str, list[tuple[StoredChunk, list[float], list[str]]]] = {}
        self._next_id = 1

    async def document_state(self) -> dict[str, tuple[str, str]]:
        return {p: (d.content_hash, model) for p, (d, model) in self._docs.items()}

    async def replace_document(
        self,
        doc: RawDoc,
        chunks: list[ChunkDraft],
        embeddings: list[list[float]],
        embedding_model: str,
    ) -> None:
        self._docs[doc.path] = (doc, embedding_model)
        rows = []
        for draft, emb in zip(chunks, embeddings, strict=True):
            stored = StoredChunk(
                id=self._next_id,
                path=doc.path,
                title=doc.title,
                category=doc.category,
                heading_path=draft.heading_path,
                text=draft.text,
                updated=doc.updated,
            )
            self._next_id += 1
            rows.append(
                (stored, emb, _WORD.findall((draft.heading_path + " " + draft.text).lower()))
            )
        self._chunks[doc.path] = rows

    async def delete_documents(self, paths: list[str]) -> None:
        for p in paths:
            self._docs.pop(p, None)
            self._chunks.pop(p, None)

    def _all(self) -> list[tuple[StoredChunk, list[float], list[str]]]:
        return [row for rows in self._chunks.values() for row in rows]

    async def search_vector(self, embedding: list[float], k: int) -> list[SearchHit]:
        scored = [
            (sum(a * b for a, b in zip(embedding, emb, strict=True)), chunk)
            for chunk, emb, _ in self._all()
        ]
        scored.sort(key=lambda x: -x[0])
        return [SearchHit(chunk=c, score=s) for s, c in scored[:k]]

    async def search_keyword(self, query: str, k: int) -> list[SearchHit]:
        terms = keyword_terms(query)
        rows = self._all()
        if not terms or not rows:
            return []
        n = len(rows)
        avg_len = sum(len(toks) for _, _, toks in rows) / n
        df = {t: sum(1 for _, _, toks in rows if t in toks) for t in terms}
        scored: list[tuple[float, StoredChunk]] = []
        for chunk, _, toks in rows:
            tf = Counter(toks)
            score = 0.0
            for t in terms:
                if tf[t] == 0:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(toks) / avg_len))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: -x[0])
        return [SearchHit(chunk=c, score=s) for s, c in scored[:k]]

    async def count_chunks(self) -> int:
        return sum(len(r) for r in self._chunks.values())
