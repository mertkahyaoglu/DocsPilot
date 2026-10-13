from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True, slots=True)
class RawDoc:
    path: str  # corpus-relative, posix style: the stable id used in citations
    title: str
    category: str
    updated: date | None
    body: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    ordinal: int
    heading_path: str  # "Refund Policy > Eligibility"
    text: str
    tokens: int

    @property
    def embed_text(self) -> str:
        """What gets embedded: the text plus its heading context, so a chunk that says
        'within 30 days' still knows it is about refunds."""
        return f"{self.heading_path}\n\n{self.text}"


@dataclass(frozen=True, slots=True)
class StoredChunk:
    id: int
    path: str
    title: str
    category: str
    heading_path: str
    text: str
    updated: date | None


@dataclass(slots=True)
class SearchHit:
    chunk: StoredChunk
    score: float
    # Debug trail: where this hit ranked in each stage (1-based), None if absent.
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    extra: dict[str, float] = field(default_factory=dict)
