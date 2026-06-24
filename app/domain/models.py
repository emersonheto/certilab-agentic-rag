from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    TECHNICIAN = "technician"
    CLIENT = "client"


@dataclass(frozen=True)
class Customer:
    id: int
    name: str
    segment: str


@dataclass(frozen=True)
class Certificate:
    id: int
    code: str
    customer_id: int
    status: str
    emitted_at: date
    technician_id: int
    equipment: str
    pdf_path: str
    document_type: str | None = None
    case_number: str | None = None
    user_id: int | None = None
    qr_code: str | None = None
    request_number: str | None = None
    service_date: date | None = None


@dataclass(frozen=True)
class CertificateHistory:
    id: int
    certificate_id: int
    event: str
    occurred_at: date
    note: str
    user_id: int | None = None
    ip: str | None = None


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    certificate_id: int
    certificate_code: str
    customer_id: int
    source_type: str
    path: str
    text: str


@dataclass(frozen=True)
class RetrievedSource:
    certificate_id: int | None
    code: str | None
    customer_id: int | None
    source_type: str
    path: str
    snippet: str
    score: float = 1.0
