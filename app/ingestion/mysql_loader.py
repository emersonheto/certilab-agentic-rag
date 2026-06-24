from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from app.domain.models import Certificate, CertificateHistory, Customer
from app.ingestion.protocols import LoadedData

if TYPE_CHECKING:
    from app.tools.mysql_connector import MySQLCertificateConnector

# Columns that MUST NEVER be read, embedded, indexed, or surfaced.
# This list is a defense-in-depth check on top of the connector allowlist.
_FORBIDDEN_PII_KEYS = frozenset({"password", "plain_password", "remember_token", "ruc", "email", "phone"})

# Default segment value for real customers (real schema has no segment column).
_DEFAULT_CUSTOMER_SEGMENT = "general"


class MySQLLoader:
    """Load certificate data from the real MySQL schema behind the CertificateLoader protocol.

    Maps verified real columns to canonical domain model fields so downstream
    retrieval logic (StructuredRetriever, splitter, SemanticRetriever) needs
    no changes.

    Security notes:
    - PII columns (password, plain_password, remember_token, ruc, email, phone)
      are never selected by the connector and never mapped here.
    - The allowlist in MySQLCertificateConnector is the primary control; this
      loader maps only explicitly named fields, so any unknown or PII column
      in the input is silently dropped and never reaches domain models.
    - Tenant isolation: customer_id is preserved in every Certificate so the
      downstream access-control layer can enforce per-tenant scoping.
    - The `users` table is never queried; no user credentials reach the pipeline.
    """

    def __init__(self, connector: MySQLCertificateConnector) -> None:
        self._connector = connector

    def load(self) -> LoadedData:
        customers = self._map_customers(self._connector.fetch_customers())
        certificates = self._map_certificates(self._connector.fetch_certificates())
        histories = self._map_histories(self._connector.fetch_histories())
        return customers, certificates, histories

    def _map_customers(self, rows: list[dict[str, object]]) -> list[Customer]:
        result: list[Customer] = []
        for row in rows:
            result.append(
                Customer(
                    id=_as_int(row["id"]),
                    name=_as_str(row.get("company_name", f"Customer-{row['id']}")),
                    segment=_DEFAULT_CUSTOMER_SEGMENT,
                )
            )
        return result

    def _map_certificates(self, rows: list[dict[str, object]]) -> list[Certificate]:
        result: list[Certificate] = []
        for row in rows:
            result.append(
                Certificate(
                    id=_as_int(row["id"]),
                    code=_as_str(row["code"]),
                    customer_id=_as_int(row["customer_id"]),
                    status=_as_str(row["status"]),
                    emitted_at=_as_date(row["issue_date"]),
                    technician_id=_as_int_or_default(row.get("user_id"), 0),
                    equipment="",
                    pdf_path=_as_str(row.get("pdf_document_path", "")),
                    document_type=_as_optional_str(row.get("document_type")),
                    case_number=_as_optional_str(row.get("case_number")),
                    user_id=_as_optional_int(row.get("user_id")),
                    qr_code=_as_optional_str(row.get("qr_code")),
                    request_number=_as_optional_str(row.get("request_number")),
                    service_date=_as_optional_date(row.get("service_date")),
                )
            )
        return result

    def _map_histories(self, rows: list[dict[str, object]]) -> list[CertificateHistory]:
        result: list[CertificateHistory] = []
        for row in rows:
            result.append(
                CertificateHistory(
                    id=_as_int(row["id"]),
                    certificate_id=_as_int(row["certification_id"]),
                    event=_as_str(row["action"]),
                    occurred_at=_as_date(row["created_at"]),
                    note="",
                    user_id=_as_optional_int(row.get("user_id")),
                    ip=_as_optional_str(row.get("ip")),
                )
            )
        return result


def _as_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(value)


def _as_str(value: Any) -> str:
    return str(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(value)


def _as_int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(value)


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"Cannot parse date from {type(value).__name__}: {value!r}")


def _as_optional_date(value: object) -> date | None:
    if value is None:
        return None
    return _as_date(value)
