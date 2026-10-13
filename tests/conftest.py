from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from docspilot.api.app import create_app
from docspilot.config import Settings
from docspilot.llm.fake import FakeLLM

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        env="test",
        prompts_dir=ROOT / "prompts",
        web_dir=ROOT / "web",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def app(settings: Settings, fake_llm: FakeLLM) -> FastAPI:
    return create_app(settings, fake_llm)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    # httpx's ASGITransport does not run lifespan; drive it explicitly.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture
def use_llm(app: FastAPI):  # type: ignore[no-untyped-def]
    """Swap the model behind a running app (the agent captured the original at startup)."""

    def _set(llm: object) -> None:
        app.state.llm = llm
        app.state.agent._llm = llm

    return _set
