from __future__ import annotations

import pytest
from pydantic import ValidationError

from docspilot.config import Settings


def test_prod_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(env="prod", _env_file=None)  # type: ignore[call-arg]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Settings(env="prod", _env_file=None).anthropic_api_key is not None  # type: ignore[call-arg]


def test_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCSPILOT_MODEL_ANSWER", "claude-sonnet-5-5")
    assert Settings(_env_file=None).model_answer == "claude-sonnet-5-5"  # type: ignore[call-arg]


def test_prod_rejects_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with pytest.raises(ValidationError):
        Settings(env="prod", llm_backend="fake", _env_file=None)  # type: ignore[call-arg]


def test_blank_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    with pytest.raises(ValidationError):
        Settings(env="prod", _env_file=None)  # type: ignore[call-arg]
    assert Settings(_env_file=None).anthropic_api_key is None  # type: ignore[call-arg]
