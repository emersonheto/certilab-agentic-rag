from __future__ import annotations

import pytest

from app.tools.mysql_connector import (
    ALLOWED_SELECT_QUERIES,
    SENSITIVE_COLUMN_NAMES,
    MySQLCertificateConnector,
)


def test_sensitive_column_names_include_ruc_email_phone() -> None:
    assert "ruc" in SENSITIVE_COLUMN_NAMES
    assert "email" in SENSITIVE_COLUMN_NAMES
    assert "phone" in SENSITIVE_COLUMN_NAMES
    assert "password" in SENSITIVE_COLUMN_NAMES
    assert "plain_password" in SENSITIVE_COLUMN_NAMES
    assert "remember_token" in SENSITIVE_COLUMN_NAMES


def test_certificate_query_uses_real_column_names() -> None:
    query = ALLOWED_SELECT_QUERIES["certificates"]
    assert "issue_date" in query
    assert "pdf_document_path" in query
    assert "emitted_at" not in query
    assert "pdf_path" not in query


def test_customer_query_uses_real_column_name_company_name() -> None:
    query = ALLOWED_SELECT_QUERIES["customers"]
    assert "company_name" in query
    # PII columns must NOT be in the customer query
    for sensitive in ("ruc", "email", "phone", "password"):
        assert sensitive not in query


def test_history_query_uses_real_table_and_column_names() -> None:
    query = ALLOWED_SELECT_QUERIES["histories"]
    assert "certification_histories" in query
    assert "action" in query
    assert "created_at" in query
    assert "event" not in query
    assert "occurred_at" not in query


def test_validate_query_rejects_ruc_column() -> None:
    evil_query = "SELECT id, ruc FROM customers"
    with pytest.raises(ValueError, match="[Ss]ensitive"):
        MySQLCertificateConnector._validate_query(evil_query)


def test_validate_query_rejects_email_column() -> None:
    evil_query = "SELECT id, email FROM users"
    with pytest.raises(ValueError, match="[Ss]ensitive"):
        MySQLCertificateConnector._validate_query(evil_query)


def test_validate_query_rejects_phone_column() -> None:
    evil_query = "SELECT id, phone FROM customers"
    with pytest.raises(ValueError, match="[Ss]ensitive"):
        MySQLCertificateConnector._validate_query(evil_query)


def test_all_allowlisted_queries_pass_validation() -> None:
    for query in ALLOWED_SELECT_QUERIES.values():
        MySQLCertificateConnector._validate_query(query)


def test_no_select_star_in_any_query() -> None:
    for query in ALLOWED_SELECT_QUERIES.values():
        assert "select *" not in query.lower()
