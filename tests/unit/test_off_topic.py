from __future__ import annotations

import pytest

from app.graph import _check_off_topic
from app.schemas import AskResponse, RouteName


# ── RouteName / AskResponse schema ──────────────────────────────────────

def test_ask_response_accepts_off_topic_route() -> None:
    """AskResponse.route must accept the new "off_topic" value."""
    response = AskResponse(
        answer="Test off-topic response",
        route="off_topic",
        sources=[],
    )
    assert response.route == "off_topic"
    assert response.answer == "Test off-topic response"
    assert response.sources == []


def test_ask_response_with_off_topic_and_sources() -> None:
    """An off_topic response should allow empty sources list (no domain data)."""
    response = AskResponse(
        answer="I can only help with certificate questions.",
        route="off_topic",
        sources=[],
    )
    assert response.route == "off_topic"
    assert len(response.sources) == 0


def test_route_name_import_includes_off_topic() -> None:
    """RouteName type alias must include "off_topic"."""
    route: RouteName = "off_topic"
    assert route == "off_topic"
    assert "structured" in RouteName.__args__
    assert "off_topic" in RouteName.__args__


# ── _check_off_topic pure function ──────────────────────────────────────

@pytest.mark.parametrize("question,expected_off_topic", [
    # Greetings
    ("hola", True),
    ("hi", True),
    ("hello", True),
    ("buenos días", True),
    ("buenas tardes", True),
    ("buenas noches", True),
    ("buen día", True),
    ("saludos", True),
    ("hey", True),
    # Off-domain topics
    ("qué es el clima hoy", True),
    ("tiempo atmosférico", True),
    ("qué clima hace hoy en Lima", True),
    ("deporte", True),
    ("fútbol", True),
    ("noticias", True),
    ("política", True),
    # Pure math
    ("2 + 2", True),
    ("1 + 2 + 3", True),
    ("10 / 2", True),
    ("  5 + 3  ", True),
    # Mixed messages — domain keywords override
    ("hola, cuántos certificados hay?", False),
    ("buenos días, el cliente 3 tiene certificados?", False),
    ("hey, dame el estado del certificado LMB-004.26-1", False),
    # Real domain questions
    ("cuántos certificados hay", False),
    ("dame los del cliente 8", False),
    ("estado del certificado LMB-004.26-1", False),
])
def test_check_off_topic(question: str, expected_off_topic: bool) -> None:
    """_check_off_topic correctly identifies off-topic questions."""
    assert _check_off_topic(question) is expected_off_topic


# ── Pipeline integration (CertilabRagPipeline directly) ──────────────────

def test_ask_returns_off_topic_response_for_greeting() -> None:
    """ask('hola') must return an off-topic response with empty sources."""
    from app.config import Settings
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

    response = pipeline.ask("hola", admin)

    assert response.route == "off_topic"
    assert response.answer == (
        "Hola, soy el asistente de Certilab. Solo puedo ayudarte con consultas "
        "sobre certificados, estados, clientes y documentos del sistema. "
        "¿En qué te puedo ayudar?"
    )
    assert response.sources == []


def test_ask_still_routes_real_questions() -> None:
    """ask('¿Cuántos certificados hay?') must still route normally."""
    from app.config import Settings
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

    response = pipeline.ask("¿Cuántos certificados hay?", admin)

    assert response.route != "off_topic"
    assert response.answer
    assert len(response.sources) > 0
