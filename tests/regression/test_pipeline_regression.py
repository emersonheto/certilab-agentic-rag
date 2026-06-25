"""Parametrized regression tests covering all 8 pipeline quality bugs.

These tests run the full pipeline in mock mode (no external services) and
verify that previously-fixed bugs stay fixed.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.domain.models import Role
from app.graph import CertilabRagPipeline
from app.retrieval.grader import route_question
from app.retrieval.query import CERTIFICATE_CODE_RE
from app.security.access_control import Principal
from app.security.payload_sanitizer import sanitize_text_for_external_payload


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline() -> CertilabRagPipeline:
    """Build a deterministic mock pipeline with test data."""
    settings = Settings(_env_file=None)
    return CertilabRagPipeline(
        data_dir=settings.data_dir,
        top_k=settings.default_top_k,
        settings=settings,
    )


@pytest.fixture(scope="module")
def admin() -> Principal:
    """Admin-privilege principal with global scope."""
    return Principal(role=Role.ADMIN, customer_id=None, user_id=1)


# ── BUG-1: Scope leakage ─────────────────────────────────────────────────

# When a question names a specific customer, ALL returned sources must belong
# ONLY to that customer. No cross-customer data leaks.


@pytest.mark.parametrize("question,expected_customer_id", [
    ("dame los certificados del cliente 101", 101),
    ("qué tiene el cliente 202", 202),
])
def test_bug1_no_scope_leakage(
    pipeline: CertilabRagPipeline,
    admin: Principal,
    question: str,
    expected_customer_id: int,
) -> None:
    """Every returned source must belong to the target customer only."""
    response = pipeline.ask(question, admin)
    assert response.route != "off_topic"
    assert len(response.sources) > 0, "Expected at least one source from the target customer"
    for source in response.sources:
        assert source.customer_id == expected_customer_id, (
            f"Expected customer {expected_customer_id}, got {source.customer_id} "
            f"— scope leakage detected"
        )


# ── BUG-2: Sanitizer precision ───────────────────────────────────────────

# The sanitizer MUST redact "cliente 3" (with digits) but MUST NOT redact
# bare plural "clientes" (without digits).


@pytest.mark.parametrize("text", [
    "clientes con más certificados",
    "los clientes del sistema",
    "el cliente principal",
])
def test_bug2_sanitizer_preserves_generic_words(text: str) -> None:
    """Generic plural 'clientes' without digits must NOT be redacted."""
    result = sanitize_text_for_external_payload(text)
    assert "REDACTED" not in result, (
        f"Expected '{text}' to remain unchanged, but got '{result}'"
    )


# ── BUG-4 + BUG-5: Routing keyword expansion ─────────────────────────────

# BUG-4: Semantic keywords ("muestras", "análisis", "agua") must route to
#   "semantic", not "structured" or "combined".
# BUG-5: Web-search keywords ("normas", "INACAL", "regulaciones") must route
#   to "web_search".


@pytest.mark.parametrize("question,expected_route", [
    ("hay algún problema con las muestras", "semantic"),
    ("resultados de análisis de agua", "semantic"),
    ("últimas normas ISO para laboratorios", "web_search"),
    ("regulaciones INACAL 2024", "web_search"),
])
def test_bug4_bug5_routing_keywords(question: str, expected_route: str) -> None:
    """route_question must route new keywords to the correct route."""
    route = route_question(question)
    assert route == expected_route, (
        f"Expected route '{expected_route}' for {question!r}, got '{route}'"
    )


# ── BUG-6: Certificate code regex expansion ──────────────────────────────

# The regex must match ALL three cert code formats:
#   CERT-XXXX-XXX, LMB-XXX.XX-X, T-XXX.XX-X


@pytest.mark.parametrize("code", [
    "CERT-2025-001",
    "LMB-004.26-1",
    "T-023.24-1",
])
def test_bug6_cert_code_regex(code: str) -> None:
    """CERTIFICATE_CODE_RE must match all three cert code formats."""
    assert CERTIFICATE_CODE_RE.search(code), f"Regex failed to match {code!r}"


# ── BUG-8: Early-exit off-topic detection ────────────────────────────────

# Pure off-topic (greetings, off-domain, math) must short-circuit to
# "off_topic" with empty sources.


@pytest.mark.parametrize("question", [
    "hola",
    "buenos días",
    "qué es el clima",
    "2+2",
])
def test_bug8_off_topic_early_exit(
    pipeline: CertilabRagPipeline,
    admin: Principal,
    question: str,
) -> None:
    """Pure off-topic questions must be caught by early-exit."""
    response = pipeline.ask(question, admin)
    assert response.route == "off_topic", (
        f"Expected 'off_topic' route for {question!r}, got '{response.route}'"
    )
    assert response.sources == [], (
        f"Expected empty sources for off-topic, got {response.sources}"
    )


# BUG-8 mixed message guard: messages containing BOTH greeting AND domain
# intent MUST NOT be caught by the early-exit — they proceed to routing.


@pytest.mark.parametrize("question", [
    "hola, cuántos certificados hay?",
    "buenos días, el cliente 3 tiene algún certificado pendiente?",
])
def test_bug8_mixed_message_goes_to_pipeline(
    pipeline: CertilabRagPipeline,
    admin: Principal,
    question: str,
) -> None:
    """Mixed messages (greeting + domain intent) must NOT trigger early-exit."""
    response = pipeline.ask(question, admin)
    assert response.route != "off_topic", (
        f"Mixed message {question!r} was incorrectly classified as off-topic"
    )
