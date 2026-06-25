from __future__ import annotations

import sys

import pytest

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


def test_structured_retrieval_narrows_to_mentioned_customer_for_admin() -> None:
    """Admin asking about a specific customer should only return that customer's sources."""
    from app.domain.models import Role
    from app.graph import CertilabRagPipeline
    from app.security.access_control import Principal

    settings = Settings(_env_file=None)
    pipeline = CertilabRagPipeline(
        data_dir=settings.data_dir,
        top_k=settings.default_top_k,
        settings=settings,
    )
    admin = Principal(role=Role.ADMIN, customer_id=None, user_id=1)

    # Admin asks about customer 101
    response = pipeline.ask("dame los del cliente 101", admin)

    # Route should be structured because "cliente" triggers domain routing
    assert response.route == "structured"
    # All sources should belong to customer 101
    for source in response.sources:
        assert source.customer_id == 101, (
            f"Expected customer 101, got {source.customer_id}"
        )


def test_structured_retrieval_narrows_to_mentioned_customer_202_for_admin() -> None:
    """Admin asking about customer 202 should return only that customer's sources."""
    from app.domain.models import Role
    from app.graph import CertilabRagPipeline
    from app.security.access_control import Principal

    settings = Settings(_env_file=None)
    pipeline = CertilabRagPipeline(
        data_dir=settings.data_dir,
        top_k=settings.default_top_k,
        settings=settings,
    )
    admin = Principal(role=Role.ADMIN, customer_id=None, user_id=1)

    response = pipeline.ask("dame los del cliente 202", admin)

    assert response.route == "structured"
    for source in response.sources:
        assert source.customer_id == 202, (
            f"Expected customer 202, got {source.customer_id}"
        )


def test_structured_retrieval_cannot_escalate_client_scope() -> None:
    """Client 101 asking about customer 202 must NOT escalate — stay at 101."""
    from app.domain.models import Role
    from app.graph import CertilabRagPipeline
    from app.security.access_control import Principal

    settings = Settings(_env_file=None)
    pipeline = CertilabRagPipeline(
        data_dir=settings.data_dir,
        top_k=settings.default_top_k,
        settings=settings,
    )
    client = Principal(role=Role.CLIENT, customer_id=101, user_id=2)

    # Client 101 asks about customer 202 — should not escalate
    response = pipeline.ask("dame los del cliente 202", client)

    assert response.route == "structured"
    # All sources should belong to customer 101 (client's own scope)
    for source in response.sources:
        assert source.customer_id == 101, (
            f"Expected customer 101 (no escalation), got {source.customer_id}"
        )


# ── LangGraph pipeline parity (PR 3) ─────────────────────────────────────────


def test_langgraph_ask_off_topic_greeting() -> None:
    """LangGraph pipeline must short-circuit off-topic greetings."""
    pytest.importorskip("langgraph")
    from pathlib import Path

    from app.domain.models import Role
    from app.graph_langgraph import CertilabLangGraphPipeline
    from app.security.access_control import Principal

    settings = Settings(_env_file=None, data_dir=Path("data"), graph_engine="langgraph")
    pipeline = CertilabLangGraphPipeline(Path("data"), settings=settings)
    admin = Principal(role=Role.ADMIN, customer_id=None, user_id=1)

    response = pipeline.ask("hola", admin)
    assert response.route == "off_topic"
    assert response.sources == []


def test_langgraph_ask_narrows_scope_for_admin() -> None:
    """LangGraph pipeline narrows admin scope when a specific customer is mentioned."""
    pytest.importorskip("langgraph")
    from pathlib import Path

    from app.domain.models import Role
    from app.graph_langgraph import CertilabLangGraphPipeline
    from app.security.access_control import Principal

    settings = Settings(_env_file=None, data_dir=Path("data"), graph_engine="langgraph")
    pipeline = CertilabLangGraphPipeline(Path("data"), settings=settings)
    admin = Principal(role=Role.ADMIN, customer_id=None, user_id=1)

    response = pipeline.ask("dame los del cliente 101", admin)
    assert response.route != "off_topic"
    assert len(response.sources) > 0, "Expected sources from customer 101"
    for source in response.sources:
        assert source.customer_id == 101, (
            f"Expected customer 101, got {source.customer_id} — scope leakage"
        )


def test_langgraph_ask_off_topic_math() -> None:
    """LangGraph pipeline catches pure math expressions as off-topic."""
    pytest.importorskip("langgraph")
    from pathlib import Path

    from app.domain.models import Role
    from app.graph_langgraph import CertilabLangGraphPipeline
    from app.security.access_control import Principal

    settings = Settings(_env_file=None, data_dir=Path("data"), graph_engine="langgraph")
    pipeline = CertilabLangGraphPipeline(Path("data"), settings=settings)
    admin = Principal(role=Role.ADMIN, customer_id=None, user_id=1)

    response = pipeline.ask("2+2", admin)
    assert response.route == "off_topic"
