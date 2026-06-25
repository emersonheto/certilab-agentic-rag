"""Tests for ``app.pipeline.guards``.

Strict TDD: tests written before implementation.
"""

from __future__ import annotations

import pytest

from app.pipeline.guards import OFF_TOPIC_ANSWER, _check_off_topic


# ── Import verification ──────────────────────────────────────────────────

def test_import_no_circular() -> None:
    """Importing from app.pipeline.guards does not cause circular imports."""
    from app.pipeline.guards import OFF_TOPIC_ANSWER as _  # noqa: F811
    assert "Certilab" in OFF_TOPIC_ANSWER


def test_old_and_new_imports_coexist() -> None:
    """Both old (app.graph) and new (app.pipeline.guards) import paths work."""
    from app.graph import _check_off_topic as old_check  # noqa: F811
    from app.pipeline.guards import _check_off_topic as new_check  # noqa: F811
    # Both return the same results (they are separate function objects but must behave identically)
    assert old_check("hola") is True
    assert new_check("hola") is True
    assert old_check("cuántos certificados hay") is False
    assert new_check("cuántos certificados hay") is False


# ── _check_off_topic ─────────────────────────────────────────────────────

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
    """_check_off_topic correctly identifies off-topic questions via new import."""
    assert _check_off_topic(question) is expected_off_topic


# ── OFF_TOPIC_ANSWER ─────────────────────────────────────────────────────

def test_off_topic_answer_is_configured() -> None:
    """OFF_TOPIC_ANSWER is a non-empty string."""
    assert isinstance(OFF_TOPIC_ANSWER, str)
    assert len(OFF_TOPIC_ANSWER) > 0
    assert "Certilab" in OFF_TOPIC_ANSWER
