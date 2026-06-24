import json
from datetime import date
from pathlib import Path
from typing import Any

from app.domain.models import Certificate, CertificateHistory, Customer
from app.ingestion.protocols import LoadedData


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def load_customers(data_dir: Path) -> list[Customer]:
    return [Customer(**item) for item in _read_json(data_dir / "mock" / "customers.json")]


def load_certificates(data_dir: Path) -> list[Certificate]:
    records = []
    for item in _read_json(data_dir / "mock" / "certificates.json"):
        records.append(Certificate(**{**item, "emitted_at": date.fromisoformat(item["emitted_at"])}))
    return records


def load_histories(data_dir: Path) -> list[CertificateHistory]:
    records = []
    for item in _read_json(data_dir / "mock" / "histories.json"):
        records.append(CertificateHistory(**{**item, "occurred_at": date.fromisoformat(item["occurred_at"])}))
    return records


def load_pdf_texts(data_dir: Path) -> dict[str, str]:
    """Load anonymized PDF text fixtures by filename."""

    pdf_dir = data_dir / "pdf_text"
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(pdf_dir.glob("*.txt"))}


class MockCertificateLoader:
    """Mock data loader that reads from JSON fixtures.

    Implements the CertificateLoader protocol so it is interchangeable with
    MySQLLoader in the pipeline factory.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def load(self) -> LoadedData:
        return load_customers(self._data_dir), load_certificates(self._data_dir), load_histories(self._data_dir)
