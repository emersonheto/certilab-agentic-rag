from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.domain.models import DocumentChunk


# --- Lightweight stand-ins for qdrant-client models (no qdrant-client import) ---


@dataclass
class FakePointStruct:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class FakeScoredPoint:
    id: str
    score: float
    payload: dict[str, Any]


@dataclass
class FakeQueryResponse:
    points: list[FakeScoredPoint]


@dataclass
class FakeFilter:
    should: list[Any] = field(default_factory=list)


@dataclass
class FakeHasIdCondition:
    has_id: list[str]


@dataclass
class FakeVectorParams:
    size: int
    distance: str


class FakeQdrantClient:
    """Minimal in-memory Qdrant stand-in for unit testing."""

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, FakePointStruct]] = {}
        self._vector_sizes: dict[str, int] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def create_collection(self, collection_name: str, vectors_config: FakeVectorParams) -> None:
        self._collections[collection_name] = {}
        self._vector_sizes[collection_name] = vectors_config.size

    def upsert(self, collection_name: str, points: list[FakePointStruct]) -> None:
        for point in points:
            self._collections[collection_name][point.id] = point

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        query_filter: FakeFilter | None = None,
        limit: int = 10,
    ) -> FakeQueryResponse:
        allowed_ids: set[str] | None = None
        if query_filter and query_filter.should:
            for cond in query_filter.should:
                if hasattr(cond, "has_id"):
                    if allowed_ids is None:
                        allowed_ids = set()
                    allowed_ids.update(cond.has_id)

        all_points = self._collections.get(collection_name, {})
        scored: list[FakeScoredPoint] = []
        for pid, point in all_points.items():
            if allowed_ids is not None and pid not in allowed_ids:
                continue
            score = _dot_product(query, point.vector)
            scored.append(FakeScoredPoint(id=pid, score=score, payload=point.payload))
        scored.sort(key=lambda p: p.score, reverse=True)
        return FakeQueryResponse(points=scored[:limit])


def _dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class FakeEmbeddingsProvider:
    """Deterministic embeddings: maps each unique text to a fixed vector."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            vector = [0.0] * self._dim
            for i, char in enumerate(text.encode()):
                vector[i % self._dim] += float(char) / 256.0
            self._cache[text] = vector
        return self._cache[text]

    @property
    def dimension(self) -> int:
        return self._dim


def _make_chunk(chunk_id: str, customer_id: int, code: str = "CERT-001") -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        certificate_id=1,
        certificate_code=code,
        customer_id=customer_id,
        source_type="pdf_text",
        path=f"data/pdf_text/{code}.txt",
        text=f"Technical content for {chunk_id} with measurement data.",
    )


@pytest.fixture()
def qdrant_index():
    """Build a QdrantVectorIndex backed by a FakeQdrantClient."""
    from app.retrieval.qdrant_index import QdrantVectorIndex

    client = FakeQdrantClient()
    embeddings = FakeEmbeddingsProvider(dim=8)
    index = QdrantVectorIndex(
        client=client,
        collection_name="test-collection",
        embedding_provider=embeddings,
        vector_size=8,
    )
    return index, client


def test_qdrant_index_creates_collection_idempotently(qdrant_index: tuple[Any, FakeQdrantClient]) -> None:
    index, client = qdrant_index
    assert client.collection_exists("test-collection")
    # Re-init should not error
    index._ensure_collection()
    assert client.collection_exists("test-collection")


def test_qdrant_index_tenant_scoped_search_only_returns_allowed_ids(qdrant_index: tuple[Any, FakeQdrantClient]) -> None:
    index, client = qdrant_index
    chunk_a1 = _make_chunk("chunk-a1", customer_id=101)
    chunk_a2 = _make_chunk("chunk-a2", customer_id=101)
    chunk_b1 = _make_chunk("chunk-b1", customer_id=202)

    index.upsert([chunk_a1, chunk_a2, chunk_b1])

    # Search restricted to tenant A's chunk IDs only
    results = index.search("Technical content", allowed_ids={"chunk-a1", "chunk-a2"}, top_k=10)

    result_ids = {chunk_id for chunk_id, _ in results}
    assert result_ids == {"chunk-a1", "chunk-a2"}
    assert "chunk-b1" not in result_ids


def test_qdrant_index_tenant_scoped_search_excludes_other_tenant(qdrant_index: tuple[Any, FakeQdrantClient]) -> None:
    index, client = qdrant_index
    chunk_b1 = _make_chunk("chunk-b1", customer_id=202)
    chunk_b2 = _make_chunk("chunk-b2", customer_id=202)

    index.upsert([chunk_b1, chunk_b2])

    # Search with empty allowed_ids returns nothing
    results = index.search("Technical content", allowed_ids=set(), top_k=10)
    assert results == []


def test_qdrant_index_stores_customer_id_in_payload(qdrant_index: tuple[Any, FakeQdrantClient]) -> None:
    index, client = qdrant_index
    chunk = _make_chunk("chunk-101", customer_id=101, code="CERT-2025-001")

    index.upsert([chunk])

    stored = client._collections["test-collection"]["chunk-101"]
    assert stored.payload["customer_id"] == 101
    assert stored.payload["code"] == "CERT-2025-001"


def test_qdrant_index_search_returns_chunk_id_and_score(qdrant_index: tuple[Any, FakeQdrantClient]) -> None:
    index, client = qdrant_index
    chunk = _make_chunk("chunk-x", customer_id=101)

    index.upsert([chunk])

    results = index.search("Technical content", allowed_ids={"chunk-x"}, top_k=5)

    assert len(results) == 1
    chunk_id, score = results[0]
    assert chunk_id == "chunk-x"
    assert isinstance(score, float)
    assert score > 0.0
