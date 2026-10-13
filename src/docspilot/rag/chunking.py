"""Markdown-aware chunking.

Why not fixed-size windows: docs have structure. Splitting on headings keeps a chunk
about one thing and gives it a meaningful label; splitting mid-table or mid-code-block
produces unusable fragments. Token counts are an approximation (words x 1.3): good
enough to size chunks, never used for billing.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Any

import yaml

from docspilot.rag.types import ChunkDraft, RawDoc

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


class _FenceTracker:
    """CommonMark-ish fence state: a fence closes only with the same character and at
    least as many of them, so ~~~ inside ``` (or a 4-backtick fence) is handled."""

    def __init__(self) -> None:
        self._open: tuple[str, int] | None = None

    def feed(self, line: str) -> bool:
        """Consume a line; return True if the line is inside (or delimits) a fence."""
        m = _FENCE.match(line)
        if self._open is None:
            if m:
                self._open = (m.group(1)[0], len(m.group(1)))
                return True
            return False
        if m and m.group(1)[0] == self._open[0] and len(m.group(1)) >= self._open[1]:
            if not line.strip().strip(self._open[0]):  # closing fences carry no info string
                self._open = None
        return True


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.3))


def parse_markdown_doc(path: str, raw: str) -> RawDoc:
    meta: dict[str, Any] = {}
    body = raw
    match = _FRONT_MATTER.match(raw)
    if match:
        loaded = yaml.safe_load(match.group(1))
        meta = loaded if isinstance(loaded, dict) else {}
        body = raw[match.end() :]
    updated = meta.get("updated")
    if isinstance(updated, str):
        try:
            updated = date.fromisoformat(updated)
        except ValueError:
            updated = None
    title = str(meta.get("title") or _first_h1(body) or path)
    # HTML comments are invisible to readers, so they are not part of the document.
    # (They are also a classic place to hide instructions aimed at the model.)
    body = _COMMENT.sub("", body)
    return RawDoc(
        path=path,
        title=title,
        category=str(meta.get("category") or (path.split("/")[0] if "/" in path else "")),
        updated=updated if isinstance(updated, date) else None,
        body=body.strip(),
        content_hash=hashlib.sha256(raw.encode()).hexdigest(),
    )


def _first_h1(body: str) -> str | None:
    for line in body.splitlines():
        m = _HEADING.match(line)
        if m and len(m.group(1)) == 1:
            return m.group(2)
    return None


def _blocks(lines: list[str]) -> list[str]:
    """Split a section into atomic blocks: paragraphs, lists, tables, fenced code."""
    blocks: list[str] = []
    cur: list[str] = []
    fence = _FenceTracker()
    for line in lines:
        in_fence = fence.feed(line)
        if not in_fence and not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _split_oversized(block: str, max_tokens: int) -> list[str]:
    """A single block bigger than max_tokens (huge code block / table): split by lines."""
    if estimate_tokens(block) <= max_tokens:
        return [block]
    parts: list[str] = []
    cur: list[str] = []
    for line in block.splitlines():
        if estimate_tokens(line) > max_tokens:  # one enormous line: fall back to words
            if cur:
                parts.append("\n".join(cur))
                cur = []
            words = line.split()
            step = max(1, int(max_tokens / 1.3))
            parts.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
            continue
        cur.append(line)
        if estimate_tokens("\n".join(cur)) >= max_tokens:
            parts.append("\n".join(cur))
            cur = []
    if cur:
        parts.append("\n".join(cur))
    return parts


def chunk_markdown(
    doc: RawDoc,
    *,
    target_tokens: int = 350,
    max_tokens: int = 500,
    min_tokens: int = 60,
    overlap_tokens: int = 40,
) -> list[ChunkDraft]:
    # 1. sections by heading
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    path = doc.title
    cur: list[str] = []
    fence = _FenceTracker()
    for line in doc.body.splitlines():
        in_fence = fence.feed(line)
        m = None if in_fence else _HEADING.match(line)
        if m is None:
            cur.append(line)
            continue
        sections.append((path, cur))
        level, title = len(m.group(1)), m.group(2)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        names = [t for lvl, t in stack if not (lvl == 1 and t == doc.title)]
        path = " > ".join([doc.title, *names])
        cur = []
    sections.append((path, cur))

    # 2. blocks per section, dropping empty sections, merging tiny ones forward
    prepared: list[tuple[str, list[str]]] = []
    carry_path: str | None = None
    carry: list[str] = []
    for heading_path, lines in sections:
        blocks = [b for blk in _blocks(lines) for b in _split_oversized(blk, max_tokens)]
        if not blocks:
            continue
        if carry and heading_path != doc.title:
            # Merged forward under the earlier label: keep this section's own heading
            # in the text so it is not lost.
            blocks = [heading_path.split(" > ")[-1], *blocks]
        blocks = carry + blocks
        heading_path = carry_path or heading_path
        if sum(estimate_tokens(b) for b in blocks) < min_tokens:
            carry, carry_path = blocks, heading_path
            continue
        carry, carry_path = [], None
        prepared.append((heading_path, blocks))
    if carry:  # trailing tiny section: attach to the previous chunk group or keep alone
        if (
            prepared
            and sum(estimate_tokens(b) for b in prepared[-1][1])
            + sum(estimate_tokens(b) for b in carry)
            <= max_tokens
        ):
            prepared[-1] = (prepared[-1][0], prepared[-1][1] + carry)
        else:
            prepared.append((carry_path or doc.title, carry))

    # 3. pack blocks into chunks of ~target tokens, with block-level overlap
    chunks: list[ChunkDraft] = []
    for heading_path, blocks in prepared:
        window: list[str] = []
        size = 0
        fresh = 0  # blocks in the window that are not just carried-over overlap
        for block in blocks:
            t = estimate_tokens(block)
            if window and size + t > target_tokens:
                if fresh:
                    chunks.append(_make(len(chunks), heading_path, window))
                    window, size = _overlap_tail(window, overlap_tokens)
                    fresh = 0
                if window and size + t > target_tokens:
                    window, size = [], 0  # overlap leaves no room: never emit overlap alone
            window.append(block)
            size += t
            fresh += 1
        if fresh:
            chunks.append(_make(len(chunks), heading_path, window))
    return chunks


def _overlap_tail(window: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    tail: list[str] = []
    size = 0
    for block in reversed(window):
        t = estimate_tokens(block)
        if size + t > overlap_tokens:
            break
        tail.insert(0, block)
        size += t
    return tail, size


def _make(ordinal: int, heading_path: str, blocks: list[str]) -> ChunkDraft:
    text = "\n\n".join(blocks).strip()
    return ChunkDraft(ordinal, heading_path, text, estimate_tokens(text))
