from __future__ import annotations

from pathlib import Path

import pytest

from docspilot.prompts import PromptNotFoundError, PromptRegistry


def test_latest_version_and_pinning(tmp_path: Path) -> None:
    (tmp_path / "answer.v1.md").write_text("one")
    (tmp_path / "answer.v2.md").write_text("two $who")
    (tmp_path / "README.md").write_text("ignored")
    reg = PromptRegistry(tmp_path)
    assert reg.get("answer").version == 2
    assert reg.get("answer", 1).text == "one"
    assert reg.get("answer").ref == "answer@v2"
    assert reg.get("answer").render(who="you") == "two you"


def test_missing_prompt_and_variable(tmp_path: Path) -> None:
    (tmp_path / "answer.v1.md").write_text("hi $who")
    reg = PromptRegistry(tmp_path)
    with pytest.raises(PromptNotFoundError):
        reg.get("nope")
    with pytest.raises(PromptNotFoundError):
        reg.get("answer", 9)
    with pytest.raises(KeyError):
        reg.get("answer").render()


def test_sha_changes_with_content(tmp_path: Path) -> None:
    (tmp_path / "a.v1.md").write_text("x")
    sha1 = PromptRegistry(tmp_path).get("a").sha
    (tmp_path / "a.v1.md").write_text("y")
    assert PromptRegistry(tmp_path).get("a").sha != sha1
