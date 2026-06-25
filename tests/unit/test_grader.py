"""Tests for app/retrieval/grader.py — keyword expansion and routing."""

import pytest

from app.retrieval.grader import Route, route_question


# ─── Task 1.3: Routing keyword expansion ─────────────────────────────────────

@pytest.mark.parametrize("question,expected_route", [
    # New SEMANTIC keywords
    ("hay algún problema con las muestras", "semantic"),
    ("resultados de análisis de agua", "semantic"),
    ("métodos de calibración utilizados", "semantic"),
    ("análisis de laboratorio", "semantic"),
    ("el ensayo de calibración", "semantic"),
    ("método de ensayo", "semantic"),
    ("resultado del análisis", "semantic"),
    # New WEB_SEARCH keywords
    ("últimas normas ISO para laboratorios", "web_search"),
    ("regulaciones INACAL 2024", "web_search"),
    ("resolución ministerial sobre certificados", "web_search"),
    ("el nuevo decreto del gobierno", "web_search"),
    ("reglamento de la norma", "web_search"),
    # Existing keywords still work
    ("cuántos certificados hay", "structured"),
    ("resumen del procedimiento técnico", "semantic"),
    ("qué hay en internet sobre calibración", "web_search"),
])
def test_routing_keyword_expansion(question: str, expected_route: Route) -> None:
    """route_question must route new keywords to the correct route."""
    result = route_question(question)
    assert result == expected_route
