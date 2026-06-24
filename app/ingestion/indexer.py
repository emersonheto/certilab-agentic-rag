import math
import re
from collections import Counter

from app.domain.models import DocumentChunk
from app.retrieval.constants import DEFAULT_RETRIEVAL_TOP_K

TOKEN_RE = re.compile(r"[a-záéíóúñ0-9-]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize Spanish/technical text for deterministic local retrieval."""

    return [token.lower() for token in TOKEN_RE.findall(text)]


class InMemoryVectorIndex:
    """Small deterministic vector-ish index based on token cosine similarity.

    Implements the VectorIndex protocol with ``upsert`` and ``search`` methods
    so it is interchangeable with the Qdrant-backed index.
    """

    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[Counter[str]] = []
        if chunks:
            self.upsert(chunks)

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Add document chunks to the in-memory index."""

        self._chunks.extend(chunks)
        self._vectors.extend(Counter(tokenize(chunk.text)) for chunk in chunks)

    def search(self, query: str, allowed_ids: set[str], top_k: int = DEFAULT_RETRIEVAL_TOP_K) -> list[tuple[str, float]]:
        """Search for chunks matching *query*, restricted to *allowed_ids*.

        Returns ``(chunk_id, score)`` tuples sorted by descending score.
        """

        query_vector = Counter(tokenize(query))
        scored: list[tuple[str, float]] = []
        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            if chunk.id not in allowed_ids:
                continue
            score = _cosine(query_vector, vector)
            if score > 0:
                scored.append((chunk.id, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
