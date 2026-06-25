"""Pipeline protocols — shared interfaces for RAG pipeline implementations."""

from __future__ import annotations

from typing import Protocol

from app.security.access_control import Principal
from app.schemas import AskResponse


class RagPipeline(Protocol):
    """Protocol for all Certilab RAG pipeline implementations."""

    def ask(self, question: str, principal: Principal) -> AskResponse:
        """Answer a tenant-scoped RAG question."""

    def reindex(self) -> dict:
        """Reload data from source and re-index chunks.

        Returns ``{"indexed": N, "skipped": M}`` counts.
        Mock mode returns ``{"indexed": 0, "skipped": 0}``.
        """
