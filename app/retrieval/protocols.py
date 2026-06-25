from __future__ import annotations

from typing import Any, Protocol

from app.domain.models import DocumentChunk


class VectorIndex(Protocol):
    """Abstraction for a vector store that supports upsert and search.

    Both the in-memory mock index and the Qdrant-backed real index implement
    this protocol so the SemanticRetriever can work with either.

    Search results are filtered by ``allowed_customer_ids`` via Qdrant payload
    instead of pre-computed ID sets, eliminating the need for in-memory lookup
    dictionaries.
    """

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Insert or update document chunks into the index."""

    def search(
        self, query: str, allowed_customer_ids: set[int] | None, top_k: int
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for chunks matching *query*, filtered by *allowed_customer_ids*.

        ``allowed_customer_ids``:
        - ``None`` — global scope (ADMIN/TECHNICIAN): return all chunks.
        - ``set()`` — empty set: return no results (fail-closed).
        - ``{1, 2}`` — return only chunks whose payload ``customer_id`` is in the set.

        Returns ``(chunk_id, score, payload)`` triples sorted by descending
        score. The caller builds ``RetrievedSource`` from payload data.
        """
