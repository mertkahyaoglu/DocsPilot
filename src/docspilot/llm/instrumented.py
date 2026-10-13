"""Decorator that makes *every* model call observable, whichever feature made it
(agent steps, triage, judge, routing): one span + metrics + one structured log line."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import asdict

from opentelemetry.trace import Status, StatusCode

from docspilot.budget import current_meter
from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLM, LLMEvent, LLMRequest, LLMResult, TextDelta
from docspilot.logging import get_logger
from docspilot.pricing import cost_or_ceiling
from docspilot.telemetry import Telemetry, record_llm_usage

log = get_logger("llm")


class InstrumentedLLM:
    def __init__(self, inner: LLM, telemetry: Telemetry) -> None:
        self._inner = inner
        self._tel = telemetry

    async def stream(self, request: LLMRequest) -> AsyncGenerator[LLMEvent, None]:
        tel, m = self._tel, self._tel.metrics
        requested, purpose = request.model, request.purpose
        served = requested
        started = time.perf_counter()
        first_token_at: float | None = None
        outcome = "error"
        # Not a *current* span: it stays open across yields and may be finalized from
        # another context (client disconnect); explicit end() in `finally` is safe.
        span = tel.start(
            f"llm.{purpose}",
            **{
                "gen_ai.system": "anthropic",
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": requested,
                "gen_ai.request.max_tokens": request.max_tokens,
                "docspilot.purpose": purpose,
                "docspilot.effort": request.effort,
                "docspilot.tools": len(request.tools or []),
                "docspilot.cache_requested": request.cache,
            },
        )
        try:
            async for event in self._inner.stream(request):
                if isinstance(event, TextDelta) and first_token_at is None:
                    first_token_at = time.perf_counter()
                if isinstance(event, LLMResult):
                    latency = time.perf_counter() - started
                    served = event.model or requested
                    outcome = "refusal" if event.refused else "ok"
                    cost = record_llm_usage(
                        m, model=served, purpose=purpose, usage=event.usage, latency_s=latency
                    )
                    meter = current_meter.get()
                    if meter is not None:
                        meter.usd += cost_or_ceiling(served, event.usage)
                    if event.served_by_fallback:
                        m.llm_fallback.labels(requested, served).inc()
                    u = event.usage
                    span.set_attributes(
                        {
                            "gen_ai.response.model": served,
                            "gen_ai.response.finish_reasons": [event.stop_reason or ""],
                            "gen_ai.usage.input_tokens": u.input_tokens,
                            "gen_ai.usage.output_tokens": u.output_tokens,
                            "docspilot.cache_read_tokens": u.cache_read_input_tokens,
                            "docspilot.cache_write_tokens": u.cache_creation_input_tokens,
                            "docspilot.served_by_fallback": event.served_by_fallback,
                            "docspilot.provider_request_id": event.request_id or "",
                            "docspilot.cost_usd": cost if cost is not None else -1.0,
                        }
                    )
                    if first_token_at is not None:
                        ttft = first_token_at - started
                        m.llm_ttft.labels(served, purpose).observe(ttft)
                        span.set_attribute("docspilot.ttft_s", round(ttft, 3))
                    log.info(
                        "llm_call",
                        purpose=purpose,
                        model=served,
                        stop_reason=event.stop_reason,
                        latency_s=round(latency, 3),
                        cost_usd=None if cost is None else round(cost, 6),
                        served_by_fallback=event.served_by_fallback,
                        provider_request_id=event.request_id,
                        **asdict(u),
                    )
                yield event
        except LLMError as exc:
            outcome = f"error_{exc.kind}"
            span.set_attribute("docspilot.error_kind", exc.kind)
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, exc.kind))
            raise
        except (GeneratorExit, asyncio.CancelledError):
            outcome = "cancelled"  # client went away: not an error, excluded from alerts
            raise
        except Exception as exc:
            outcome = "error_unexpected"
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            m.llm_requests.labels(served, purpose, outcome).inc()
            if outcome.startswith("error"):
                m.llm_latency.labels(served, purpose).observe(time.perf_counter() - started)
            span.set_attribute("docspilot.outcome", outcome)
            span.end()
