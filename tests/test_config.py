from __future__ import annotations

import pytest
from pydantic import ValidationError

from docspilot.config import Settings

PROD = {
    "env": "prod",
    "database_url": "postgresql://x/y",
    "api_keys": "alice:alice-secret-key-0001",
    "_env_file": None,
}


def test_prod_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    with pytest.raises(ValidationError):
        Settings(**PROD)  # type: ignore[arg-type]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Settings(**PROD).anthropic_api_key is not None  # type: ignore[arg-type]


def test_prod_requires_database_and_real_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="embedder"):
        Settings(**PROD)  # type: ignore[arg-type]
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        Settings(env="prod", _env_file=None)  # type: ignore[call-arg]
    assert Settings(**PROD).resolved_embedder == "voyage"  # type: ignore[arg-type]


def test_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCSPILOT_MODEL_ANSWER", "claude-sonnet-5-5")
    assert Settings(_env_file=None).model_answer == "claude-sonnet-5-5"  # type: ignore[call-arg]


def test_prod_rejects_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    with pytest.raises(ValidationError):
        Settings(**PROD, llm_backend="fake")  # type: ignore[arg-type]


def test_blank_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    with pytest.raises(ValidationError):
        Settings(**PROD)  # type: ignore[arg-type]
    assert Settings(_env_file=None).anthropic_api_key is None  # type: ignore[call-arg]


def test_voyage_selection_and_min_score_need_their_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="VOYAGE_API_KEY"):
        Settings(embedder="voyage", _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="reranker"):
        Settings(rerank_min_score=0.3, _env_file=None)  # type: ignore[call-arg]
