"""Structured logging. JSON in prod, pretty console in dev. Request id via contextvars."""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.typing import Processor

from docspilot.config import Settings


def configure_logging(settings: Settings) -> None:
    level = logging.getLevelName(settings.log_level.upper())
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json or settings.env == "prod"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
