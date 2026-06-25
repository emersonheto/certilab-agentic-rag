"""Tests for new Settings fields added in PR 2 (config + env vars).

Strict TDD: tests written before implementation. These assert that the four
new configurable knobs exist on ``Settings`` with the expected defaults so the
hardcoded magic numbers across the codebase can be replaced.
"""

from __future__ import annotations

from app.config import Settings


def test_settings_openai_temperature_default() -> None:
    """``openai_temperature`` defaults to 0.0 (deterministic)."""
    settings = Settings(_env_file=None)
    assert settings.openai_temperature == 0.0


def test_settings_web_search_max_results_default() -> None:
    """``web_search_max_results`` defaults to 3."""
    settings = Settings(_env_file=None)
    assert settings.web_search_max_results == 3


def test_settings_qdrant_upsert_batch_size_default() -> None:
    """``qdrant_upsert_batch_size`` defaults to 200 (Qdrant payload limit safe)."""
    settings = Settings(_env_file=None)
    assert settings.qdrant_upsert_batch_size == 200


def test_settings_embedding_batch_size_default() -> None:
    """``embedding_batch_size`` defaults to 100 (OpenAI API batch limit safe)."""
    settings = Settings(_env_file=None)
    assert settings.embedding_batch_size == 100
