"""Tracing (OpenTelemetry) and metrics (Prometheus) behind one small facade.

Principles:
- Spans follow the GenAI semantic conventions where they exist, so any OTel backend
  (Jaeger, Langfuse, Honeycomb, Grafana Tempo) renders LLM calls meaningfully.
- Prompt/answer content is NOT recorded unless explicitly enabled, and is redacted
  and truncated when it is. Token counts, model, cost and latency are always safe.
- Everything degrades to no-ops: telemetry must never break a request.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Status, StatusCode
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from docspilot.config import Settings
from docspilot.llm.types import Usage
from docspilot.pricing import cost_usd
from docspilot.redact import redact

_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)


class Metrics:
    """One instance per app, own registry: no global state, tests stay isolated."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.llm_requests = Counter(
            "docspilot_llm_requests_total", "LLM calls", ["model", "purpose", "outcome"], registry=r
        )
        self.llm_tokens = Counter(
            "docspilot_llm_tokens_total", "Tokens", ["model", "purpose", "kind"], registry=r
        )
        self.llm_cost = Counter(
            "docspilot_llm_cost_usd_total",
            "Estimated spend in USD",
            ["model", "purpose"],
            registry=r,
        )
        self.llm_latency = Histogram(
            "docspilot_llm_latency_seconds",
            "End-to-end LLM call latency",
            ["model", "purpose"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.llm_ttft = Histogram(
            "docspilot_llm_time_to_first_token_seconds",
            "Time to first streamed token",
            ["model", "purpose"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.tool_calls = Counter(
            "docspilot_tool_calls_total", "Tool executions", ["tool", "outcome"], registry=r
        )
        self.tool_latency = Histogram(
            "docspilot_tool_latency_seconds", "Tool latency", ["tool"], registry=r
        )
        self.retrieval_latency = Histogram(
            "docspilot_retrieval_latency_seconds",
            "Retrieval latency by stage",
            ["stage"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
            registry=r,
        )
        self.retrieval_degraded = Counter(
            "docspilot_retrieval_degraded_total",
            "Retrievals that lost a leg",
            ["leg"],
            registry=r,
        )
        self.cache = Counter(
            "docspilot_cache_requests_total", "Response cache lookups", ["result"], registry=r
        )
        self.route = Counter("docspilot_route_total", "Routing decisions", ["route"], registry=r)
        self.requests = Counter(
            "docspilot_chat_requests_total", "Chat turns", ["outcome"], registry=r
        )
        self.in_flight = Gauge("docspilot_chat_in_flight", "Chat streams in progress", registry=r)


@dataclass(slots=True)
class Telemetry:
    metrics: Metrics
    tracer: trace.Tracer
    trace_content: bool = False
    content_max_chars: int = 2000

    @classmethod
    def noop(cls) -> Telemetry:
        return cls(Metrics(), trace.get_tracer("docspilot"))

    @contextlib.contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        with self.tracer.start_as_current_span(
            name, attributes={k: v for k, v in attributes.items() if v is not None}
        ) as span:
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise

    def content(self, span: Span, key: str, value: str) -> None:
        """Opt-in, redacted, truncated. Never called with credentials by design."""
        if self.trace_content:
            span.set_attribute(f"docspilot.{key}", redact(value)[: self.content_max_chars])


def record_llm_usage(
    metrics: Metrics, *, model: str, purpose: str, usage: Usage, latency_s: float
) -> float | None:
    m, p = model, purpose
    metrics.llm_tokens.labels(m, p, "input").inc(usage.input_tokens)
    metrics.llm_tokens.labels(m, p, "output").inc(usage.output_tokens)
    metrics.llm_tokens.labels(m, p, "cache_read").inc(usage.cache_read_input_tokens)
    metrics.llm_tokens.labels(m, p, "cache_write").inc(usage.cache_creation_input_tokens)
    metrics.llm_latency.labels(m, p).observe(latency_s)
    cost = cost_usd(model, usage)
    if cost is not None:
        metrics.llm_cost.labels(m, p).inc(cost)
    return cost


def setup_tracing(
    settings: Settings, exporter: SpanExporter | None = None
) -> TracerProvider | None:
    """Install a provider only when an exporter is configured (or injected by tests).
    Without one, spans are still created but cost nothing and go nowhere."""
    if exporter is None and not settings.otlp_endpoint:
        return None
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": settings.service_name, "deployment.environment": settings.env}
        )
    )
    if exporter is None:
        headers = dict(
            pair.split("=", 1) for pair in (settings.otlp_headers or "").split(",") if "=" in pair
        )
        endpoint = (settings.otlp_endpoint or "").rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint += "/v1/traces"
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or None)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider
