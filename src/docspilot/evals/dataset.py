from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "answerable",
    "multi_hop",
    "contradiction",
    "unanswerable",
    "injection",
    "ticket_intent",
    "smalltalk",
]
Behavior = Literal["answer", "abstain", "create_ticket", "chitchat"]


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str = Field(min_length=1)
    category: Category
    expected_sources: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    #: Strings whose presence proves harmful compliance (exfiltration tokens, wrong computed
    #: totals). NOT for positive-claim phrases: a correct denial or warning quotes those.
    must_not: list[str] = Field(default_factory=list)
    #: create_ticket cases: substrings the proposed ticket must carry (ids, amounts).
    ticket_must_contain: list[str] = Field(default_factory=list)
    expected_behavior: Behavior
    notes: str = ""

    @property
    def has_retrieval_target(self) -> bool:
        return bool(self.expected_sources)


class DatasetError(Exception):
    pass


def load_golden(path: Path, *, corpus_dir: Path | None = None) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(GoldenCase.model_validate(json.loads(line)))
        except ValueError as exc:
            raise DatasetError(f"{path}:{n}: {exc}") from exc
    dupes = [i for i, c in Counter(c.id for c in cases).items() if c > 1]
    if dupes:
        raise DatasetError(f"duplicate case ids: {dupes}")
    if corpus_dir is not None:
        missing = sorted(
            {s for c in cases for s in c.expected_sources if not (corpus_dir / s).is_file()}
        )
        if missing:
            raise DatasetError(f"expected_sources not found in corpus: {missing}")
    return cases
