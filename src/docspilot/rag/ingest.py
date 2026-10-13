"""Ingestion: corpus dir -> chunks -> embeddings -> store. Idempotent and incremental:
unchanged documents are skipped (content hash + embedding model), removed files are
pruned, so it is safe to run on every deploy."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from docspilot.logging import get_logger
from docspilot.rag.chunking import chunk_markdown, parse_markdown_doc
from docspilot.rag.embeddings import Embedder
from docspilot.rag.store import ChunkStore

log = get_logger("ingest")


@dataclass(frozen=True, slots=True)
class IngestReport:
    added_or_updated: list[str]
    unchanged: int
    deleted: list[str]
    chunks_written: int


def _read_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    return [
        (f.relative_to(corpus_dir).as_posix(), f.read_text(encoding="utf-8"))
        for f in sorted(corpus_dir.rglob("*.md"))
    ]


async def ingest_corpus(corpus_dir: Path, store: ChunkStore, embedder: Embedder) -> IngestReport:
    state = await store.document_state()
    seen: set[str] = set()
    changed: list[str] = []
    unchanged = 0
    chunks_written = 0

    for rel, raw in await asyncio.to_thread(_read_corpus, corpus_dir):
        seen.add(rel)
        doc = parse_markdown_doc(rel, raw)
        if state.get(rel) == (doc.content_hash, embedder.name):
            unchanged += 1
            continue
        chunks = chunk_markdown(doc)
        if not chunks:
            log.warning("ingest_empty_document", path=rel)
            continue
        embeddings = await embedder.embed([c.embed_text for c in chunks], "document")
        await store.replace_document(doc, chunks, embeddings, embedder.name)
        changed.append(rel)
        chunks_written += len(chunks)

    stale = sorted(set(state) - seen)
    if stale:
        await store.delete_documents(stale)
    report = IngestReport(changed, unchanged, stale, chunks_written)
    log.info(
        "ingest_done",
        updated=len(changed),
        unchanged=unchanged,
        deleted=len(stale),
        chunks=chunks_written,
    )
    return report


async def _cli(corpus: Path) -> None:
    from docspilot.config import get_settings
    from docspilot.logging import configure_logging
    from docspilot.rag.factory import build_rag

    settings = get_settings()
    configure_logging(settings)
    async with build_rag(settings, ingest_on_start=False) as rag:
        report = await ingest_corpus(corpus, rag.store, rag.embedder)
        print(
            f"updated={len(report.added_or_updated)} unchanged={report.unchanged} "
            f"deleted={len(report.deleted)} chunks={report.chunks_written}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the docs corpus.")
    parser.add_argument("corpus", type=Path, nargs="?", default=None)
    args = parser.parse_args()
    from docspilot.config import get_settings

    asyncio.run(_cli(args.corpus or get_settings().corpus_dir))


if __name__ == "__main__":
    main()
