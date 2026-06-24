from __future__ import annotations

from typing import Protocol

from app.domain.models import DocumentChunk


class VectorIndex(Protocol):
    """Abstraction for a vector store that supports upsert and search.

    Both the in-memory mock index and the Qdrant-backed real index implement
    this protocol so the SemanticRetriever can work with either.
    """

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Insert or update document chunks into the index."""

    def search(self, query: str, allowed_ids: set[str], top_k: int) -> list[tuple[str, float]]:
        """Search for chunks matching *query*, restricted to *allowed_ids*.

        Returns a list of ``(chunk_id, score)`` tuples sorted by descending
        score.
        """
