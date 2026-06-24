from __future__ import annotations

from typing import Protocol

from app.domain.models import Certificate, CertificateHistory, Customer

LoadedData = tuple[list[Customer], list[Certificate], list[CertificateHistory]]


class CertificateLoader(Protocol):
    """Abstraction for loading certificate data from any data source.

    Mock and real loaders implement this protocol so the pipeline factory
    can swap them without touching downstream retrieval logic.
    """

    def load(self) -> LoadedData:
        """Return customers, certificates, and histories in canonical form."""
