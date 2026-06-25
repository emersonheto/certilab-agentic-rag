"""Tests for ``app.pipeline.utils``.

Strict TDD: tests written before implementation.
"""

from __future__ import annotations

from app.domain.models import RetrievedSource, Role
from app.pipeline.utils import (
    _certificate_count,
    _dedupe_sources,
    _elapsed_ms,
    _scope_attributes,
    _source_id,
    _to_schema,
    _web_sources,
)
from app.schemas import Source
from app.security.access_control import AccessScope
from app.tools.web_search import WebSearchResult


# ── Import verification ──────────────────────────────────────────────────

def test_import_no_circular() -> None:
    """Importing from app.pipeline.utils does not cause circular imports."""
    from app.pipeline.utils import _dedupe_sources, _to_schema  # noqa: F811
    assert callable(_dedupe_sources)
    assert callable(_to_schema)


# ── _source_id ───────────────────────────────────────────────────────────

def test_source_id_with_code() -> None:
    """Returns source_type:code when code is present."""
    source = RetrievedSource(
        certificate_id=1, code="CERT-001", customer_id=101,
        source_type="metadata", path="", snippet="test",
    )
    assert _source_id(source) == "metadata:CERT-001"


def test_source_id_without_code() -> None:
    """Returns just source_type when code is None."""
    source = RetrievedSource(
        certificate_id=None, code=None, customer_id=None,
        source_type="web_search", path="https://example.com", snippet="test",
    )
    assert _source_id(source) == "web_search"


# ── _to_schema ───────────────────────────────────────────────────────────

def test_to_schema_converts_retrieved_source() -> None:
    source = RetrievedSource(
        certificate_id=1, code="CERT-001", customer_id=101,
        source_type="metadata", path="data/mock/certificates.json",
        snippet="Certificado activo.",
    )
    schema = _to_schema(source)
    assert isinstance(schema, Source)
    assert schema.certificate_id == 1
    assert schema.code == "CERT-001"
    assert schema.customer_id == 101
    assert schema.source_type == "metadata"
    assert schema.source_id == "metadata:CERT-001"
    assert schema.snippet == "Certificado activo."


# ── _dedupe_sources ──────────────────────────────────────────────────────

def test_dedupe_removes_duplicates() -> None:
    sources = [
        RetrievedSource(certificate_id=1, code="CERT-001", customer_id=101,
                        source_type="metadata", path="a", snippet="Same"),
        RetrievedSource(certificate_id=1, code="CERT-001", customer_id=101,
                        source_type="metadata", path="a", snippet="Same"),
    ]
    result = _dedupe_sources(sources)
    assert len(result) == 1


def test_dedupe_preserves_unique() -> None:
    sources = [
        RetrievedSource(certificate_id=1, code="CERT-001", customer_id=101,
                        source_type="metadata", path="a", snippet="First"),
        RetrievedSource(certificate_id=2, code="CERT-002", customer_id=102,
                        source_type="pdf_text", path="b", snippet="Second"),
    ]
    result = _dedupe_sources(sources)
    assert len(result) == 2


def test_dedupe_empty_list() -> None:
    assert _dedupe_sources([]) == []


# ── _web_sources ─────────────────────────────────────────────────────────

def test_web_sources_converts_results() -> None:
    results = [
        WebSearchResult(title="Test", url="https://example.com/1",
                        snippet="First result"),
        WebSearchResult(title="Second", url="https://example.com/2",
                        snippet="Second result"),
    ]
    sources = _web_sources(results)
    assert len(sources) == 2
    assert sources[0].source_type == "web_search"
    assert sources[0].certificate_id is None
    assert sources[1].snippet == "Second: Second result"


def test_web_sources_fallback_score() -> None:
    results = [
        WebSearchResult(title="Fallback", url="", snippet="not available",
                        source_note="fallback"),
    ]
    sources = _web_sources(results)
    assert sources[0].score == 0.0


def test_web_sources_regular_score() -> None:
    results = [
        WebSearchResult(title="Normal", url="https://example.com",
                        snippet="result", source_note="tavily"),
    ]
    sources = _web_sources(results)
    assert sources[0].score == 1.0


# ── _elapsed_ms ──────────────────────────────────────────────────────────

def test_elapsed_ms_returns_positive_float() -> None:
    import time
    start = time.perf_counter()
    time.sleep(0.001)
    elapsed = _elapsed_ms(start)
    assert isinstance(elapsed, float)
    assert elapsed > 0


# ── _scope_attributes ───────────────────────────────────────────────────

def test_scope_attributes_global() -> None:
    scope = AccessScope(role=Role.ADMIN, customer_id=None)
    attrs = _scope_attributes(scope)
    assert attrs["rag.role"] == "admin"
    assert attrs["rag.has_customer_scope"] is False


def test_scope_attributes_with_customer() -> None:
    scope = AccessScope(role=Role.CLIENT, customer_id=101)
    attrs = _scope_attributes(scope)
    assert attrs["rag.role"] == "client"
    assert attrs["rag.has_customer_scope"] is True


# ── _certificate_count ───────────────────────────────────────────────────

def test_certificate_count_unique() -> None:
    sources = [
        RetrievedSource(certificate_id=1, code="CERT-001", customer_id=101,
                        source_type="metadata", path="a", snippet="x"),
        RetrievedSource(certificate_id=2, code="CERT-002", customer_id=102,
                        source_type="metadata", path="b", snippet="y"),
    ]
    assert _certificate_count(sources) == 2


def test_certificate_count_duplicates() -> None:
    sources = [
        RetrievedSource(certificate_id=1, code="CERT-001", customer_id=101,
                        source_type="metadata", path="a", snippet="x"),
        RetrievedSource(certificate_id=1, code="CERT-001", customer_id=101,
                        source_type="pdf_text", path="b", snippet="y"),
    ]
    assert _certificate_count(sources) == 1


def test_certificate_count_none_ids() -> None:
    sources = [
        RetrievedSource(certificate_id=None, code=None, customer_id=None,
                        source_type="web_search", path="", snippet="x"),
    ]
    assert _certificate_count(sources) == 0
