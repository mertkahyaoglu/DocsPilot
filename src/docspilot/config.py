"""Typed, validated configuration. Single source of truth for env vars."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Models for which the server-side refusal fallback is supported (Claude API only).
FALLBACK_CAPABLE_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCSPILOT_",
        env_file=".env",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # The SDK also reads ANTHROPIC_API_KEY itself; we surface it for fail-fast validation.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # "fake" = canned offline replies (UI demos / no API key). Never allowed in prod.
    llm_backend: Literal["anthropic", "fake"] = "anthropic"

    # Main answer model vs. cheap model for classification / judging.
    model_answer: str = "claude-opus-5-5"
    model_light: str = "claude-haiku-5-5"
    effort_answer: Effort = "medium"
    max_tokens: int = Field(default=16000, ge=256)

    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_connect_timeout_s: float = Field(default=5.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=6)
    refusal_fallback: bool = True

    prompts_dir: Path = Path("prompts")
    web_dir: Path = Path("web")
    answer_prompt_version: int | None = None  # None = latest

    max_history_messages: int = Field(default=20, ge=1)
    max_message_chars: int = Field(default=4000, ge=1)

    @model_validator(mode="after")
    def _require_key_in_prod(self) -> Settings:
        if self.env == "prod" and self.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required when DOCSPILOT_ENV=prod")
        if self.env == "prod" and self.llm_backend != "anthropic":
            raise ValueError("DOCSPILOT_LLM_BACKEND must be 'anthropic' in prod")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
