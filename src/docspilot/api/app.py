from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from docspilot import __version__
from docspilot.api.chat import router as chat_router
from docspilot.api.middleware import RequestContextMiddleware
from docspilot.config import Settings, get_settings
from docspilot.llm.client import AnthropicLLM, build_anthropic_client
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.types import LLM
from docspilot.logging import configure_logging, get_logger
from docspilot.prompts import PromptRegistry


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    """App factory: dependencies are injectable so tests never touch the network."""
    settings = settings or get_settings()
    configure_logging(settings)
    log = get_logger("app")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.prompts = PromptRegistry(settings.prompts_dir)
        app.state.prompts.get("answer", settings.answer_prompt_version)  # fail fast
        client = None
        if llm is None and settings.llm_backend == "fake":
            app.state.llm = FakeLLM(
                [Script(text="(offline fake model) You said hello to DocsPilot.")]
            )
        elif llm is None:
            client = build_anthropic_client(settings)
            app.state.llm = AnthropicLLM(client, refusal_fallback=settings.refusal_fallback)
        else:
            app.state.llm = llm
        log.info("startup", env=settings.env, version=__version__, model=settings.model_answer)
        yield
        if client is not None:
            await client.close()

    app = FastAPI(title="DocsPilot", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(chat_router)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    if settings.web_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
    return app
