"""Structured logging for certilab-agentic-rag using structlog.

Usage::

    from app.logging import get_logger
    logger = get_logger(__name__)
    logger.info("pipeline.route", route="structured", question_length=42)

PII safety: callers must only bind safe kwargs (lengths, counts, durations,
routes, model names). Question text, answers, certificate codes, customer IDs,
names, API keys, and tokens must NEVER be logged.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    """Configure structlog for the application.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR). Unknown values
            fall back to INFO.
        json: If True, emit machine-readable JSON lines (production). If
            False, emit coloured console output (development).
    """

    numeric_level = getattr(logging, level.upper(), None)
    if numeric_level is None:
        numeric_level = logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )

    shared_processors: list[structlog.types.EventProcessor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to *name*.

    Args:
        name: Dotted module name, e.g. ``"pipeline.deterministic"``.
    """

    return structlog.get_logger(name)
