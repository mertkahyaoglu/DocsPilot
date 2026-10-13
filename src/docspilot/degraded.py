"""Degraded mode: when no model can answer, retrieval still can.

Showing the most relevant documentation passages (clearly labelled) is far better
than an error page for a docs assistant, costs no tokens, and cannot hallucinate."""

from __future__ import annotations

from docspilot.rag.retrieve import Retriever
from docspilot.safety.injection import Policy, neutralize
from docspilot.safety.spotlight import defang

NOTICE = (
    "The assistant can't generate an answer right now. These documentation sections look "
    "most relevant to your question:"
)


async def degraded_answer(
    retriever: Retriever, question: str, policy: Policy, *, max_items: int = 3
) -> tuple[str, list[dict[str, str]]] | None:
    result = await retriever.search(question)
    if not result.hits:
        return None
    lines = [NOTICE, ""]
    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    for hit in result.hits:
        c = hit.chunk
        if c.path in seen:
            continue
        seen.add(c.path)
        # Same hygiene as the tool path: excerpts, titles and headings are document text.
        excerpt, _ = neutralize(c.text, policy)
        excerpt = " ".join(excerpt.split())[:300]
        title = defang(neutralize(c.title, policy)[0])[:200]
        section = defang(neutralize(c.heading_path, policy)[0])[:300]
        lines.append(f"- {title} > {section.split(' > ')[-1]} [{c.path}]")
        lines.append(f"  {excerpt}")
        citations.append({"path": c.path, "title": title, "section": section})
        if len(citations) >= max_items:
            break
    return "\n".join(lines), citations
