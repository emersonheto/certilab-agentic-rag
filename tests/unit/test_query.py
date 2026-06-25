"""Tests for app/retrieval/query.py — extract_customer_id and cert code regex."""

import re

import pytest

from app.retrieval.query import CERTIFICATE_CODE_RE, extract_certificate_code


# ─── Task 1.1: extract_customer_id ───────────────────────────────────────────

@pytest.mark.parametrize("question,expected", [
    ("dame los certificados del cliente 8", 8),
    ("cuántos tiene el cliente 101", 101),
    ("qué tiene el customer 202", 202),
    ("certificados del client 3", 3),
    ("cuántos certificados hay en total", None),
    ("hola, buenos días", None),
    ("el laboratorio principal", None),
])
def test_extract_customer_id(question: str, expected: int | None) -> None:
    """extract_customer_id must extract integer ID from known patterns and return None otherwise."""
    from app.retrieval.query import extract_customer_id

    result = extract_customer_id(question)
    assert result == expected


# ─── Task 1.1: Cert code regex expansion ─────────────────────────────────────

CERT_CODE_CASES: list[tuple[str, bool]] = [
    ("CERT-2025-001", True),
    ("LMB-004.26-1", True),
    ("T-023.24-1", True),
    ("no hay código aquí", False),
    ("el cliente 3 tiene certificados", False),
]


@pytest.mark.parametrize("text,should_match", CERT_CODE_CASES)
def test_cert_code_regex_matches_expanded_formats(text: str, should_match: bool) -> None:
    """CERTIFICATE_CODE_RE must match all three cert code formats and reject non-codes."""
    result = bool(CERTIFICATE_CODE_RE.search(text))
    assert result == should_match


@pytest.mark.parametrize("text,expected_code", [
    ("CERT-2025-001", "CERT-2025-001"),
    ("LMB-004.26-1", "LMB-004.26-1"),
    ("T-023.24-1", "T-023.24-1"),
])
def test_extract_certificate_code_returns_matched_code(text: str, expected_code: str) -> None:
    """extract_certificate_code must return the matched code in uppercase."""
    result = extract_certificate_code(text)
    assert result == expected_code
