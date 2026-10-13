from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanExporter
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from docspilot import __version__
from docspilot.agent.factory import build_agent
from docspilot.agent.pending import InMemoryPendingStore
from docspilot.api.chat import router as chat_router
from docspilot.api.debug import router as debug_router
from docspilot.api.feedback import router as feedback_router
from docspilot.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from docspilot.api.triage import router as triage_router
from docspilot.cache import CorpusVersionProvider, ResponseCache
from docspilot.config import Settings, get_settings
from docspilot.llm.client import AnthropicLLM, build_anthropic_client
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.instrumented import InstrumentedLLM
from docspilot.llm.types import LLM
from docspilot.logging import configure_logging, get_logger
from docspilot.prompts import PromptRegistry
from docspilot.rag.factory import build_rag
from docspilot.telemetry import Metrics, Telemetry, setup_tracing


def create_app(
    settings: Settings | None = None,
    llm: LLM | None = None,
    span_exporter: SpanExporter | None = None,
) -> FastAPI:
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
            if settings.routing:
                registry.get("smalltalk")  # fail at startup, not on the first request
            app.state.prompts_registry = registry

            provider = setup_tracing(settings, span_exporter)
            if provider is not None:
                stack.callback(provider.shutdown)  # flush buffered spans on exit
            tracer = provider.get_tracer("docspilot") if provider else trace.get_tracer("docspilot")
            telemetry = Telemetry(Metrics(), tracer, trace_content=settings.trace_content)
            app.state.telemetry = telemetry

            base_llm: LLM
            if llm is None and settings.llm_backend == "fake":
                canned = Script(text="(offline fake model) Set ANTHROPIC_API_KEY for real answers.")
                base_llm = FakeLLM([canned])
            elif llm is None:
                client = build_anthropic_client(settings)
                stack.push_async_callback(client.close)
                base_llm = AnthropicLLM(client, refusal_fallback=settings.refusal_fallback)
            else:
                base_llm = llm
            # Every model call (agent, triage, routing, ...) goes through this wrapper.
            app.state.llm = InstrumentedLLM(base_llm, telemetry)
            app.state.answer_prompt = answer_prompt

            rag = await stack.enter_async_context(build_rag(settings))
            app.state.rag = rag
            app.state.tickets = rag.tickets
            app.state.feedback = rag.feedback
            app.state.pending = InMemoryPendingStore(ttl_s=settings.confirmation_ttl_s)
            rag.retriever.attach_telemetry(telemetry)
            app.state.agent = build_agent(
                settings,
                llm=app.state.llm,
                retriever=rag.retriever,
                pending=app.state.pending,
                answer_prompt=answer_prompt,
                telemetry=telemetry,
            )
            app.state.response_cache = ResponseCache(
                ttl_s=settings.response_cache_ttl_s, max_items=settings.response_cache_max_items
            )
            app.state.corpus_versions = CorpusVersionProvider(rag.store.document_state)
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
    app.include_router(feedback_router)
    if not prod:
        app.include_router(debug_router)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up. Must not touch dependencies."""
        return {"status": "ok"}

    token = settings.metrics_token.get_secret_value() if settings.metrics_token else None
    if settings.metrics_enabled and (not prod or token):

        @app.get("/metrics", include_in_schema=False)
        async def metrics(request: Request) -> Response:
            if token is not None:
                supplied = request.headers.get("authorization", "")
                if not secrets.compare_digest(supplied, f"Bearer {token}"):
                    return Response(status_code=401, headers={"WWW-Authenticate": "Bearer"})
            return Response(
                generate_latest(app.state.telemetry.metrics.registry),
                media_type=CONTENT_TYPE_LATEST,
            )

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
