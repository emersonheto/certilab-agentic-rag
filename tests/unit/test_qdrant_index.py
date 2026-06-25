from __future__ import annotations

from dataclasses import dataclass
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
class FakeFieldCondition:
    key: str
    match: Any


@dataclass
class FakeMatchAny:
    any: list[Any]


@dataclass
class FakeFilter:
    must: list[Any] | None = None


@dataclass
class FakeVectorParams:
    size: int
    distance: str


@dataclass
class FakeCollectionInfo:
    points_count: int
    indexed_vectors_count: int


class FakeQdrantClient:
    """Minimal in-memory Qdrant stand-in for unit testing.

    Supports payload-based filtering via ``Filter(must=[FieldCondition(...)])``
    as used by the new ``QdrantVectorIndex.search()``.
    """

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, FakePointStruct]] = {}
        self._vector_sizes: dict[str, int] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def create_collection(self, collection_name: str, vectors_config: FakeVectorParams) -> None:
        self._collections[collection_name] = {}
        self._vector_sizes[collection_name] = vectors_config.size

    def get_collection(self, collection_name: str) -> FakeCollectionInfo:
        points = self._collections.get(collection_name, {})
        return FakeCollectionInfo(points_count=len(points), indexed_vectors_count=len(points))

    def retrieve(
        self, collection_name: str, ids: list[str], with_payload: bool = True, with_vectors: bool = True
    ) -> list[FakePointStruct]:
        collection = self._collections.get(collection_name, {})
        return [collection[pid] for pid in ids if pid in collection]

    def upsert(self, collection_name: str, points: list[FakePointStruct]) -> None:
        for point in points:
            self._collections[collection_name][point.id] = point

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        query_filter: FakeFilter | None = None,
        limit: int = 10,
        with_payload: bool = False,
    ) -> FakeQueryResponse:
        # Parse customer_id filter from the new Filter(must=[FieldCondition(...)]) format
        allowed_customer_ids: set[int] | None = None
        if query_filter is not None and query_filter.must:
            for cond in query_filter.must:
                if hasattr(cond, "key") and cond.key == "customer_id":
                    match = cond.match
                    if hasattr(match, "any"):
                        allowed_customer_ids = set(match.any)

        all_points = self._collections.get(collection_name, {})
        scored: list[FakeScoredPoint] = []
        for pid, point in all_points.items():
            # Apply customer_id filter if present
            if allowed_customer_ids is not None:
                point_customer_id = point.payload.get("customer_id")
                if point_customer_id not in allowed_customer_ids:
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
        self.call_count: int = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        if text not in self._cache:
            vector = [0.0] * self._dim
            for i, char in enumerate(text.encode()):
                vector[i % self._dim] += float(char) / 256.0
            self._cache[text] = vector
        return self._cache[text]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

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
    return index, client, embeddings


def test_qdrant_index_creates_collection_idempotently(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    index, client, _ = qdrant_index
    assert client.collection_exists("test-collection")
    # Re-init should not error
    index._ensure_collection()
    assert client.collection_exists("test-collection")


def test_qdrant_index_tenant_scoped_search_by_customer_id(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """Search restricted to customer_id 101 must return only that tenant's chunks."""
    index, client, _ = qdrant_index
    chunk_a1 = _make_chunk("chunk-a1", customer_id=101)
    chunk_a2 = _make_chunk("chunk-a2", customer_id=101)
    chunk_b1 = _make_chunk("chunk-b1", customer_id=202)

    index.upsert([chunk_a1, chunk_a2, chunk_b1])

    # Search restricted to customer 101
    results = index.search("Technical content", allowed_customer_ids={101}, top_k=10)

    result_ids = {chunk_id for chunk_id, _, _ in results}
    assert result_ids == {"chunk-a1", "chunk-a2"}
    assert "chunk-b1" not in result_ids
    # Results must include payload
    for chunk_id, score, payload in results:
        assert isinstance(score, float)
        assert payload["customer_id"] == 101


def test_qdrant_index_global_scope_returns_all_tenants(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """None = global scope (ADMIN/TECHNICIAN) — all chunks returned."""
    index, client, _ = qdrant_index
    chunk_a = _make_chunk("chunk-a", customer_id=101)
    chunk_b = _make_chunk("chunk-b", customer_id=202)

    index.upsert([chunk_a, chunk_b])

    results = index.search("Technical content", allowed_customer_ids=None, top_k=10)
    assert len(results) == 2
    result_ids = {chunk_id for chunk_id, _, _ in results}
    assert result_ids == {"chunk-a", "chunk-b"}


def test_qdrant_index_empty_customer_set_returns_empty(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """An empty allowed_customer_ids set must return no results (fail-closed)."""
    index, client, _ = qdrant_index
    chunk = _make_chunk("chunk-x", customer_id=101)
    index.upsert([chunk])

    results = index.search("Technical content", allowed_customer_ids=set(), top_k=10)
    assert results == []


def test_qdrant_index_search_returns_payload_triples(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """Each search result must be a (chunk_id, score, payload) triple."""
    index, client, _ = qdrant_index
    chunk = _make_chunk("chunk-x", customer_id=101, code="CERT-X")

    index.upsert([chunk])

    results = index.search("Technical content", allowed_customer_ids={101}, top_k=5)

    assert len(results) == 1
    chunk_id, score, payload = results[0]
    assert chunk_id == "chunk-x"
    assert isinstance(score, float)
    assert score > 0.0
    assert isinstance(payload, dict)
    assert payload["customer_id"] == 101
    assert payload["code"] == "CERT-X"
    assert payload["chunk_id"] == "chunk-x"


def test_qdrant_index_stores_customer_id_in_payload(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """Upsert must store customer_id and code in Qdrant point payload."""
    index, client, _ = qdrant_index
    chunk = _make_chunk("chunk-101", customer_id=101, code="CERT-2025-001")

    index.upsert([chunk])

    # Find the stored point by searching for it
    results = index.search("Technical content", allowed_customer_ids={101}, top_k=5)
    assert len(results) == 1
    _, _, payload = results[0]
    assert payload["customer_id"] == 101
    assert payload["code"] == "CERT-2025-001"


def test_qdrant_index_upsert_skips_existing_chunks(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """Idempotent upsert must skip chunks that already exist in Qdrant."""

    index, client, embeddings = qdrant_index
    chunk = _make_chunk("dup-chunk", customer_id=101)

    # First upsert — embeds and stores
    index.upsert([chunk])
    first_call_count = embeddings.call_count
    assert first_call_count > 0

    # Second upsert with the same chunk — must NOT call embed again
    index.upsert([chunk])

    # Collection must have the same number of points (no duplicates)
    assert len(client._collections["test-collection"]) == 1
    # Embed call count must NOT have grown (no re-embedding)
    assert embeddings.call_count == first_call_count


def test_qdrant_index_upsert_only_embeds_new_chunks(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """When mixing existing and new chunks, only new chunks should be embedded."""

    index, client, embeddings = qdrant_index

    existing = _make_chunk("existing", customer_id=101)
    index.upsert([existing])

    first_call_count = embeddings.call_count

    new_chunk = _make_chunk("new-chunk", customer_id=202)
    # Upsert existing + new together
    index.upsert([existing, new_chunk])

    # Embed call count should have grown by exactly 1 (only the new chunk)
    assert embeddings.call_count == first_call_count + 1
    # Both chunks must be in the collection
    assert len(client._collections["test-collection"]) == 2


def test_qdrant_index_upsert_empty_chunks(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """Upsert with empty list must be a no-op."""
    index, client, _ = qdrant_index
    index.upsert([])
    assert len(client._collections["test-collection"]) == 0


def test_qdrant_index_upsert_idempotent_stores_payload_on_first_call(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """When a chunk is skipped on the second upsert, its payload must still be in Qdrant."""
    index, client, _ = qdrant_index
    chunk = _make_chunk("payload-chunk", customer_id=303, code="CERT-TEST")

    index.upsert([chunk])
    results = index.search("Technical content", allowed_customer_ids={303}, top_k=5)
    assert len(results) == 1
    _, _, payload = results[0]
    assert payload["customer_id"] == 303
    assert payload["code"] == "CERT-TEST"

    # Second upsert skips it, but payload must remain
    index.upsert([chunk])
    results = index.search("Technical content", allowed_customer_ids={303}, top_k=5)
    assert len(results) == 1
    _, _, payload = results[0]
    assert payload["customer_id"] == 303


def test_qdrant_index_upsert_uses_batch_embedding(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """upsert must call embed_batch once for all new chunks instead of embed per chunk."""
    index, client, embeddings = qdrant_index

    chunks = [
        _make_chunk("batch-a", customer_id=101, code="CERT-A"),
        _make_chunk("batch-b", customer_id=102, code="CERT-B"),
        _make_chunk("batch-c", customer_id=103, code="CERT-C"),
    ]

    index.upsert(chunks)

    # All 3 should be in the collection
    assert len(client._collections["test-collection"]) == 3
    # call_count should equal number of texts embedded
    assert embeddings.call_count == 3


def test_qdrant_index_get_stats(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """get_stats must return total_points and indexed_vectors."""
    index, client, _ = qdrant_index

    stats = index.get_stats()
    assert stats["total_points"] == 0
    assert stats["indexed_vectors"] >= 0

    index.upsert([_make_chunk("stats-chunk", customer_id=101)])
    stats = index.get_stats()
    assert stats["total_points"] == 1


def test_qdrant_index_no_lookup_dicts_on_init() -> None:
    """QdrantVectorIndex no longer maintains in-memory lookup dicts."""
    from app.retrieval.qdrant_index import QdrantVectorIndex

    client = FakeQdrantClient()
    embeddings = FakeEmbeddingsProvider(dim=8)
    client.create_collection("fresh-collection", FakeVectorParams(size=8, distance="Cosine"))

    index = QdrantVectorIndex(
        client=client,
        collection_name="fresh-collection",
        embedding_provider=embeddings,
        vector_size=8,
    )

    # Verify lookup dicts do NOT exist
    assert not hasattr(index, "_chunk_to_uuid")
    assert not hasattr(index, "_uuid_to_chunk")


def test_qdrant_index_search_with_multiple_customer_ids(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """Search with multiple customer_ids must return chunks for all specified tenants."""
    index, client, _ = qdrant_index

    chunk_a = _make_chunk("multi-a", customer_id=101)
    chunk_b = _make_chunk("multi-b", customer_id=202)
    chunk_c = _make_chunk("multi-c", customer_id=303)

    index.upsert([chunk_a, chunk_b, chunk_c])

    results = index.search("Technical content", allowed_customer_ids={101, 303}, top_k=10)
    result_ids = {chunk_id for chunk_id, _, _ in results}
    assert result_ids == {"multi-a", "multi-c"}
    assert "multi-b" not in result_ids


def test_qdrant_index_search_payload_includes_chunk_id(qdrant_index: tuple[Any, FakeQdrantClient, FakeEmbeddingsProvider]) -> None:
    """chunk_id must be present in the returned payload for debugging."""
    index, client, _ = qdrant_index
    chunk = _make_chunk("debug-chunk", customer_id=101)

    index.upsert([chunk])

    results = index.search("Technical content", allowed_customer_ids={101}, top_k=5)
    assert len(results) == 1
    _, _, payload = results[0]
    assert payload.get("chunk_id") == "debug-chunk"
