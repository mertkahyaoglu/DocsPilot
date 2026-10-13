from __future__ import annotations

from pathlib import Path

from docspilot.rag.embeddings import HashEmbedder
from docspilot.rag.ingest import ingest_corpus
from docspilot.rag.store import InMemoryChunkStore

BODY = "# {t}\n\n## Section\n\n" + "Some meaningful sentence about billing. " * 20


def write(root: Path, rel: str, title: str) -> None:
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"---\ntitle: {title}\nupdated: 2025-01-01\n---\n\n" + BODY.format(t=title))


async def test_ingest_is_incremental_and_prunes(tmp_path: Path) -> None:
    store, emb = InMemoryChunkStore(), HashEmbedder()
    write(tmp_path, "a/one.md", "One")
    write(tmp_path, "b/two.md", "Two")

    first = await ingest_corpus(tmp_path, store, emb)
    assert sorted(first.added_or_updated) == ["a/one.md", "b/two.md"] and first.unchanged == 0
    assert await store.count_chunks() >= 2

    again = await ingest_corpus(tmp_path, store, emb)
    assert again.added_or_updated == [] and again.unchanged == 2 and again.chunks_written == 0

    write(tmp_path, "a/one.md", "One changed")
    changed = await ingest_corpus(tmp_path, store, emb)
    assert changed.added_or_updated == ["a/one.md"] and changed.unchanged == 1

    (tmp_path / "b/two.md").unlink()
    pruned = await ingest_corpus(tmp_path, store, emb)
    assert pruned.deleted == ["b/two.md"]
    assert list(await store.document_state()) == ["a/one.md"]


async def test_embedding_model_change_forces_reembed(tmp_path: Path) -> None:
    class Other(HashEmbedder):
        name = "other-model"

    store = InMemoryChunkStore()
    write(tmp_path, "a.md", "A")
    await ingest_corpus(tmp_path, store, HashEmbedder())
    report = await ingest_corpus(tmp_path, store, Other())
    assert report.added_or_updated == ["a.md"]
