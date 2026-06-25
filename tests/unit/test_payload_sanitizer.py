"""Tests for app/security/payload_sanitizer.py — regex precision fixes."""

import re

import pytest

from app.security.payload_sanitizer import (
    CERTIFICATE_CODE_PATTERN,
    CUSTOMER_IDENTIFIER_PATTERN,
    REDACTED_CERTIFICATE,
    REDACTED_CUSTOMER,
    sanitize_text_for_external_payload,
)


# ─── Task 1.2: Customer identifier pattern precision ─────────────────────────

@pytest.mark.parametrize("text,should_redact", [
    ("cliente 3", True),
    ("cliente 101", True),
    ("clientes con más certificados", False),
    ("los clientes del sistema", False),
    ("el cliente principal", False),  # no digit
    ("customer 42", True),
    ("client 99", True),
])
def test_customer_pattern_precision(text: str, should_redact: bool) -> None:
    """CUSTOMER_IDENTIFIER_PATTERN must only match when digits follow the keyword."""
    result = bool(CUSTOMER_IDENTIFIER_PATTERN.search(text))
    assert result == should_redact


@pytest.mark.parametrize("text,expected", [
    ("cliente 3", "[REDACTED_CUSTOMER]"),
    ("el cliente 101 tiene certificados", f"el {REDACTED_CUSTOMER} tiene certificados"),
    ("clientes con más certificados", "clientes con más certificados"),
    ("los clientes del sistema", "los clientes del sistema"),
])
def test_sanitize_customer_pattern(text: str, expected: str) -> None:
    """sanitize_text_for_external_payload must redact customer IDs correctly."""
    result = sanitize_text_for_external_payload(text)
    assert result == expected


# ─── Task 1.2/1.1: Cert code pattern expansion ───────────────────────────────

@pytest.mark.parametrize("text,should_redact", [
    ("CERT-2025-001", True),
    ("LMB-004.26-1", True),
    ("T-023.24-1", True),
    ("no hay código aquí", False),
    ("el cliente 3 tiene certificados", False),
])
def test_cert_code_pattern_expanded(text: str, should_redact: bool) -> None:
    """CERTIFICATE_CODE_PATTERN must match expanded cert code formats."""
    result = bool(CERTIFICATE_CODE_PATTERN.search(text))
    assert result == should_redact


@pytest.mark.parametrize("text,expected", [
    ("cert LMB-004.26-1 encontrado", f"cert {REDACTED_CERTIFICATE} encontrado"),
    ("T-023.24-1 está listo", f"{REDACTED_CERTIFICATE} está listo"),
])
def test_sanitize_cert_code_expanded(text: str, expected: str) -> None:
    """sanitize_text_for_external_payload must redact expanded cert codes."""
    result = sanitize_text_for_external_payload(text)
    assert result == expected
