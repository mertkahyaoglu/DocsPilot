"""Build the model client the same way for the API, evals and experiments."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from docspilot.config import Settings
from docspilot.llm.client import AnthropicLLM, build_anthropic_client
from docspilot.llm.cloudflare import CloudflareLLM
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.types import LLM


def create_llm(settings: Settings) -> tuple[LLM, Callable[[], Awaitable[None]]]:
    """Returns (llm, async closer)."""
    if settings.llm_backend == "fake":
        canned = Script(text="(offline fake model) Configure a real backend for real answers.")

        async def _noop() -> None:
            return None

        return FakeLLM([canned]), _noop
    if settings.llm_backend == "cloudflare":
        assert settings.cloudflare_account_id and settings.cloudflare_api_token
        cf = CloudflareLLM(
            settings.cloudflare_account_id,
            settings.cloudflare_api_token.get_secret_value(),
            timeout_s=settings.llm_timeout_s,
            connect_timeout_s=settings.llm_connect_timeout_s,
            max_retries=settings.llm_max_retries,
        )
        return cf, cf.aclose
    client = build_anthropic_client(settings)
    return AnthropicLLM(client, refusal_fallback=settings.refusal_fallback), client.close
