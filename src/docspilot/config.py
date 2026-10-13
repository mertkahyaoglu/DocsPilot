"""Typed, validated configuration. Single source of truth for env vars."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]
# src/docspilot/config.py -> repo root. Absolute, so the CWD does not matter.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Models for which the server-side refusal fallback is supported (Claude API only).
FALLBACK_CAPABLE_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCSPILOT_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # The SDK also reads ANTHROPIC_API_KEY itself; we surface it for fail-fast validation.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # "fake" = canned offline replies (UI demos / no API key). Never allowed in prod.
    llm_backend: Literal["anthropic", "fake"] = "anthropic"

    # --- safety ---
    #: "user:key,user2:key2" (keys >= 16 chars). Required in prod; unset in dev = anonymous.
    api_keys: SecretStr | None = Field(default=None, validation_alias="DOCSPILOT_API_KEYS")
    injection_policy: Literal["off", "annotate", "redact"] = "redact"
    #: Hosts the answer may link to / mention (subdomains included), comma separated.
    allowed_hosts: str = "acmebilling.example"
    output_guard: bool = True
    rate_limit_per_min: float = Field(default=30.0, ge=0)  # 0 disables
    rate_limit_burst: float = Field(default=10.0, ge=1)
    user_daily_budget_usd: float = Field(default=5.0, ge=0)  # 0 disables
    global_daily_budget_usd: float = Field(default=100.0, ge=0)

    # --- observability ---
    otlp_endpoint: str | None = Field(default=None, validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otlp_headers: str | None = Field(default=None, validation_alias="OTEL_EXPORTER_OTLP_HEADERS")
    service_name: str = "docspilot"
    trace_content: bool = False  # prompts/answers in traces: redacted + truncated, opt-in
    metrics_enabled: bool = True
    #: Bearer token required for /metrics. In prod the endpoint stays off without one.
    metrics_token: SecretStr | None = None

    # --- cost controls ---
    prompt_caching: bool = True
    response_cache_ttl_s: int = Field(default=3600, ge=0)  # 0 disables
    response_cache_max_items: int = Field(default=500, ge=1)
    routing: bool = False  # small-model triage in front of the agent (see LEARN.md)

    # --- storage & retrieval ---
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")
    migrations_dir: Path = PROJECT_ROOT / "migrations"
    corpus_dir: Path = PROJECT_ROOT / "corpus"
    voyage_api_key: SecretStr | None = Field(default=None, validation_alias="VOYAGE_API_KEY")
    embedder: Literal["auto", "voyage", "hash"] = "auto"  # auto: voyage if key else hash
    voyage_embed_model: str = "voyage-3.5"
    rerank: Literal["auto", "voyage", "none"] = "auto"
    voyage_rerank_model: str = "rerank-2.5"
    retrieval_k: int = Field(default=5, ge=1, le=20)
    retrieval_candidates: int = Field(default=20, ge=1, le=100)
    rerank_min_score: float | None = None
    auto_ingest: bool = True  # ingest at startup when the index is empty / in-memory

    # Main answer model vs. cheap model for classification / judging.
    model_answer: str = "claude-opus-5-5"
    model_light: str = "claude-haiku-5-5"
    model_judge: str = "claude-sonnet-5-5"  # evals only; judge quality matters more than cost
    effort_answer: Effort = "medium"
    max_tokens: int = Field(default=16000, ge=256)

    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_connect_timeout_s: float = Field(default=5.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=6)
    refusal_fallback: bool = True

    prompts_dir: Path = PROJECT_ROOT / "prompts"
    web_dir: Path = PROJECT_ROOT / "web"
    answer_prompt_version: int | None = None  # None = latest

    max_agent_steps: int = Field(default=6, ge=1, le=20)
    tool_timeout_s: float = Field(default=15.0, gt=0)
    confirmation_ttl_s: int = Field(default=600, ge=30)

    max_history_messages: int = Field(default=20, ge=1)
    max_message_chars: int = Field(default=4000, ge=1)  # user messages
    max_assistant_chars: int = Field(default=20_000, ge=1)  # prior answers echoed back
    max_body_bytes: int = Field(default=256 * 1024, ge=1024)

    @property
    def allowed_host_list(self) -> tuple[str, ...]:
        return tuple(h.strip().lower() for h in self.allowed_hosts.split(",") if h.strip())

    @property
    def resolved_embedder(self) -> Literal["voyage", "hash"]:
        if self.embedder == "auto":
            return "voyage" if self.voyage_api_key else "hash"
        return self.embedder

    @property
    def resolved_rerank(self) -> Literal["voyage", "none"]:
        if self.rerank == "auto":
            return "voyage" if self.voyage_api_key else "none"
        return self.rerank

    @field_validator(
        "anthropic_api_key",
        "voyage_api_key",
        "database_url",
        "otlp_endpoint",
        "metrics_token",
        "api_keys",
        mode="before",
    )
    @classmethod
    def _blank_key_is_none(cls, v: object) -> object:
        # `.env.example` ships `ANTHROPIC_API_KEY=`: blank must not satisfy fail-fast.
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @model_validator(mode="after")
    def _require_key_in_prod(self) -> Settings:
        if self.env == "prod" and self.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required when DOCSPILOT_ENV=prod")
        if self.env == "prod" and self.llm_backend != "anthropic":
            raise ValueError("DOCSPILOT_LLM_BACKEND must be 'anthropic' in prod")
        if self.env == "prod" and self.database_url is None:
            raise ValueError("DATABASE_URL is required when DOCSPILOT_ENV=prod")
        if self.env == "prod" and self.resolved_embedder == "hash":
            raise ValueError("a real embedder (VOYAGE_API_KEY) is required in prod")
        if self.env == "prod" and self.api_keys is None:
            raise ValueError("DOCSPILOT_API_KEYS is required when DOCSPILOT_ENV=prod")
        if self.env == "prod" and self.injection_policy == "off":
            raise ValueError("DOCSPILOT_INJECTION_POLICY must not be 'off' in prod")
        if "voyage" in (self.embedder, self.rerank) and self.voyage_api_key is None:
            raise ValueError("VOYAGE_API_KEY is required for embedder/rerank=voyage")
        if self.rerank_min_score is not None and self.resolved_rerank == "none":
            raise ValueError("DOCSPILOT_RERANK_MIN_SCORE needs a reranker (it is a rerank score)")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
