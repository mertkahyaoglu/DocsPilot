from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from docspilot import __version__
from docspilot.agent.loop import Agent, AgentConfig
from docspilot.agent.pending import InMemoryPendingStore
from docspilot.agent.tools import ToolRegistry, default_tools
from docspilot.api.chat import router as chat_router
from docspilot.api.debug import router as debug_router
from docspilot.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from docspilot.api.triage import router as triage_router
from docspilot.config import Settings, get_settings
from docspilot.llm.client import AnthropicLLM, build_anthropic_client
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.types import LLM
from docspilot.logging import configure_logging, get_logger
from docspilot.prompts import PromptRegistry
from docspilot.rag.factory import build_rag


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    """App factory: dependencies are injectable so tests never touch the network."""
    settings = settings or get_settings()
    configure_logging(settings)
    log = get_logger("app")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            app.state.settings = settings
            registry = PromptRegistry(settings.prompts_dir)
            answer_prompt = registry.get("answer", settings.answer_prompt_version)  # fail fast
            registry.get("triage")
            app.state.prompts_registry = registry

            if llm is None and settings.llm_backend == "fake":
                canned = Script(text="(offline fake model) Set ANTHROPIC_API_KEY for real answers.")
                app.state.llm = FakeLLM([canned])
            elif llm is None:
                client = build_anthropic_client(settings)
                stack.push_async_callback(client.close)
                app.state.llm = AnthropicLLM(client, refusal_fallback=settings.refusal_fallback)
            else:
                app.state.llm = llm

            rag = await stack.enter_async_context(build_rag(settings))
            app.state.rag = rag
            app.state.tickets = rag.tickets
            app.state.pending = InMemoryPendingStore(ttl_s=settings.confirmation_ttl_s)
            app.state.agent = Agent(
                app.state.llm,
                ToolRegistry(default_tools(rag.retriever)),
                app.state.pending,
                AgentConfig(
                    model=settings.model_answer,
                    system=answer_prompt.text,
                    max_tokens=settings.max_tokens,
                    effort=settings.effort_answer,
                    max_steps=settings.max_agent_steps,
                    tool_timeout_s=settings.tool_timeout_s,
                    prompt_ref=answer_prompt.ref,
                    prompt_sha=answer_prompt.sha,
                ),
                confirmation_ttl_s=settings.confirmation_ttl_s,
            )
            log.info(
                "startup",
                env=settings.env,
                version=__version__,
                model=settings.model_answer,
                prompt=answer_prompt.ref,
                embedder=rag.embedder.name,
                reranker=rag.reranker.name,
                store="postgres" if rag.pool else "memory",
            )
            yield

    prod = settings.env == "prod"
    app = FastAPI(
        title="DocsPilot",
        version=__version__,
        lifespan=lifespan,
        docs_url=None if prod else "/docs",
        redoc_url=None,
        openapi_url=None if prod else "/openapi.json",
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(RequestContextMiddleware)  # outermost: sees every response, incl. 413
    app.include_router(chat_router)
    app.include_router(triage_router)
    if not prod:
        app.include_router(debug_router)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up. Must not touch dependencies."""
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        """Readiness: can this instance serve? (index reachable and non-empty)"""
        try:
            chunks = await app.state.rag.store.count_chunks()
        except Exception:
            log.exception("readiness_check_failed")
            return JSONResponse({"status": "unavailable"}, status_code=503)
        if chunks == 0:
            return JSONResponse({"status": "empty_index"}, status_code=503)
        current = app.state.rag.embedder.name
        state = await app.state.rag.store.document_state()
        stale = sorted(p for p, (_, model) in state.items() if model != current)
        if stale:  # vectors from another model are incomparable: retrieval would degrade silently
            return JSONResponse(
                {"status": "embedding_model_mismatch", "documents": len(stale)}, status_code=503
            )
        return JSONResponse({"status": "ready", "chunks": chunks})

    if settings.web_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
    else:
        log.warning("web_dir_missing", path=str(settings.web_dir))
    return app
