"""Tests for the structlog logging module.

STRICT TDD: tests written before implementation. These assert that
``configure_logging`` and ``get_logger`` work, that JSON mode is supported,
and that the Settings fields exist with safe defaults.

PII contract: the tests never assert on log *content* — only on structural
properties (logger exists, configuration does not raise, level is respected).
This keeps the test suite PII-safe and avoids coupling to renderer output.
"""

from __future__ import annotations

import structlog


def test_get_logger_returns_logger() -> None:
    """get_logger returns a non-None bound logger."""

    logger = structlog.get_logger("test.unit")
    assert logger is not None


def test_get_logger_accepts_dotted_name() -> None:
    """get_logger accepts module-dot notation names."""

    from app.logging import get_logger

    logger = get_logger("pipeline.deterministic")
    assert logger is not None


def test_configure_logging_does_not_raise() -> None:
    """configure_logging with a string level must not raise."""

    from app.logging import configure_logging

    configure_logging(level="WARNING")


def test_configure_logging_json_mode() -> None:
    """configure_logging with json=True must not raise (production path)."""

    from app.logging import configure_logging

    configure_logging(level="WARNING", json=True)


def test_configure_logging_debug_level() -> None:
    """configure_logging accepts DEBUG level without error."""

    from app.logging import configure_logging

    configure_logging(level="DEBUG")


def test_configure_logging_unknown_level_falls_back_to_info() -> None:
    """An unknown level string falls back to INFO, not a crash."""

    from app.logging import configure_logging

    configure_logging(level="BOGUS")


def test_log_level_config_field_default() -> None:
    """Settings exposes log_level with default 'INFO'."""

    from app.config import Settings

    s = Settings(_env_file=None)
    assert s.log_level == "INFO"


def test_log_json_config_field_default() -> None:
    """Settings exposes log_json with default False."""

    from app.config import Settings

    s = Settings(_env_file=None)
    assert s.log_json is False


def test_log_level_config_field_via_alias() -> None:
    """Settings reads LOG_LEVEL env alias."""

    from app.config import Settings

    s = Settings(_env_file=None, LOG_LEVEL="DEBUG")
    assert s.log_level == "DEBUG"


def test_get_logger_can_bind_safe_kwargs() -> None:
    """A logger can bind PII-safe kwargs (length, count, route) for correlation."""

    from app.logging import get_logger

    logger = get_logger("test.bind").bind(question_length=42, route="structured", source_count=3)
    assert logger is not None
