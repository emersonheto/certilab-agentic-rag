from __future__ import annotations

from datetime import date
from typing import Any

from app.ingestion.mysql_loader import MySQLLoader


class _FakeConnector:
    """Minimal stand-in for MySQLCertificateConnector returning canned rows."""

    def __init__(
        self,
        certificate_rows: list[dict[str, object]] | None = None,
        customer_rows: list[dict[str, object]] | None = None,
        history_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self._certificate_rows = certificate_rows or []
        self._customer_rows = customer_rows or []
        self._history_rows = history_rows or []

    def fetch_certificates(self, customer_id: int | None = None) -> list[dict[str, object]]:
        return list(self._certificate_rows)

    def fetch_customers(self) -> list[dict[str, object]]:
        return list(self._customer_rows)

    def fetch_histories(self, certificate_id: int | None = None) -> list[dict[str, object]]:
        return list(self._history_rows)


def _sample_certificate_row(**overrides: Any) -> dict[str, object]:
    row: dict[str, object] = {
        "id": 1,
        "code": "CERT-2025-001",
        "case_number": "CASO-100",
        "request_number": "REQ-200",
        "document_type": "calibracion",
        "issue_date": date(2025, 1, 18),
        "service_date": date(2025, 1, 15),
        "customer_id": 101,
        "user_id": 9001,
        "pdf_document_path": "certificates/CERT-2025-001.pdf",
        "qr_code": "QR-abc123",
        "status": "emitido",
    }
    row.update(overrides)
    return row


def _sample_customer_row(**overrides: Any) -> dict[str, object]:
    row: dict[str, object] = {
        "id": 101,
        "company_name": "Laboratorio Andino",
    }
    row.update(overrides)
    return row


def _sample_history_row(**overrides: Any) -> dict[str, object]:
    row: dict[str, object] = {
        "id": 1,
        "certification_id": 1,
        "user_id": 9001,
        "action": "created",
        "ip": "192.168.1.1",
        "user_agent": "Mozilla/5.0",
        "created_at": date(2025, 1, 17),
    }
    row.update(overrides)
    return row


def test_mysql_loader_maps_real_columns_to_canonical_fields() -> None:
    connector = _FakeConnector(
        certificate_rows=[_sample_certificate_row()],
        customer_rows=[_sample_customer_row()],
        history_rows=[_sample_history_row()],
    )
    loader = MySQLLoader(connector)

    customers, certificates, histories = loader.load()

    assert len(certificates) == 1
    cert = certificates[0]
    assert cert.code == "CERT-2025-001"
    assert cert.customer_id == 101
    assert cert.status == "emitido"
    assert cert.emitted_at == date(2025, 1, 18)
    assert cert.pdf_path == "certificates/CERT-2025-001.pdf"
    assert cert.document_type == "calibracion"
    assert cert.case_number == "CASO-100"
    assert cert.user_id == 9001
    assert cert.qr_code == "QR-abc123"


def test_mysql_loader_maps_customer_company_name_to_name() -> None:
    connector = _FakeConnector(customer_rows=[_sample_customer_row(company_name="Clínica Norte")])
    loader = MySQLLoader(connector)

    customers, _, _ = loader.load()

    assert len(customers) == 1
    assert customers[0].id == 101
    assert customers[0].name == "Clínica Norte"


def test_mysql_loader_maps_history_action_to_event() -> None:
    connector = _FakeConnector(history_rows=[_sample_history_row(action="emitted", created_at=date(2025, 2, 5))])
    loader = MySQLLoader(connector)

    _, _, histories = loader.load()

    assert len(histories) == 1
    assert histories[0].certificate_id == 1
    assert histories[0].event == "emitted"
    assert histories[0].occurred_at == date(2025, 2, 5)


def test_pii_columns_ruc_email_phone_never_reach_domain_models() -> None:
    """Even if raw rows somehow contained PII columns, the loader must not surface them."""

    connector = _FakeConnector(
        certificate_rows=[_sample_certificate_row()],
        customer_rows=[_sample_customer_row(ruc="12345678-9", email="leak@example.com", phone="+56912345678")],
        history_rows=[],
    )
    loader = MySQLLoader(connector)

    customers, certificates, histories = loader.load()

    customer = customers[0]
    # Customer dataclass has no ruc/email/phone fields — verify they're absent
    assert not hasattr(customer, "ruc")
    assert not hasattr(customer, "email")
    assert not hasattr(customer, "phone")
    assert not hasattr(customer, "password")

    # No PII leaks into name
    assert customer.name == "Laboratorio Andino"


def test_mysql_loader_returns_empty_lists_when_no_data() -> None:
    connector = _FakeConnector()
    loader = MySQLLoader(connector)

    customers, certificates, histories = loader.load()

    assert customers == []
    assert certificates == []
    assert histories == []
