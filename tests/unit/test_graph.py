from __future__ import annotations

import sys

from app.config import Settings
from app.graph import build_rag_pipeline


def test_mock_mode_pipeline_builds_without_importing_qdrant_or_pymysql() -> None:
    """APP_MODE=mock must never import qdrant-client or pymysql.

    This guarantees the hermetic test suite runs without Docker, MySQL,
    or any real infrastructure dependencies.
    """
    settings = Settings(_env_file=None)
    assert settings.app_mode == "mock"

    # Remove any pre-existing imports from other tests
    for mod in list(sys.modules):
        if mod.startswith("qdrant_client") or mod.startswith("pymysql"):
            del sys.modules[mod]

    pipeline = build_rag_pipeline(settings)

    assert pipeline is not None
    assert "qdrant_client" not in sys.modules
    assert "pymysql" not in sys.modules


def test_mock_mode_pipeline_returns_valid_ask_response() -> None:
    """The mock pipeline must produce a valid AskResponse for a simple question."""

    from app.domain.models import Role
    from app.security.access_control import Principal

    settings = Settings(_env_file=None)
    pipeline = build_rag_pipeline(settings)

    admin = Principal(role=Role.ADMIN, customer_id=None, user_id=1)
    response = pipeline.ask("¿Cuántos certificados hay?", admin)

    assert response.route == "structured"
    assert response.answer


def test_mock_mode_uses_in_memory_index_not_qdrant() -> None:
    """The mock pipeline's semantic retriever must use an InMemoryVectorIndex."""

    from app.ingestion.indexer import InMemoryVectorIndex

    settings = Settings(_env_file=None)
    pipeline = build_rag_pipeline(settings)

    assert isinstance(pipeline._semantic._index, InMemoryVectorIndex)
