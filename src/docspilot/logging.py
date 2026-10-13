"""Structured logging. JSON in prod, pretty console in dev. Request id via contextvars."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from opentelemetry import trace
from structlog.typing import Processor

from docspilot.config import Settings
from docspilot.redact import redact


def _add_trace_ids(
    _: Any, __: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


_STRUCTURAL = frozenset(
    {"event", "level", "timestamp", "trace_id", "span_id", "request_id", "method", "path"}
)


def _redact_values(
    _: Any, __: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Logs outlive requests and travel to third parties: scrub free-text values (error
    details, queries, exception text) of emails, secrets, card/phone/IBAN numbers."""
    for key, value in event_dict.items():
        if key not in _STRUCTURAL:
            event_dict[key] = _scrub(value)
    return event_dict


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list | tuple):
        return [_scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    return value


def configure_logging(settings: Settings) -> None:
    level = logging.getLevelName(settings.log_level.upper())
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_trace_ids,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json or settings.env == "prod"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        # exc_info is rendered to text *before* scrubbing so tracebacks are covered too
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            _redact_values,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
