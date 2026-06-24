from __future__ import annotations

import uuid

import pytest

from app.domain.models import DocumentChunk


class _SimpleEmbeddings:
    """Deterministic embedding provider for integration tests (no deps)."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for i, byte in enumerate(text.encode()):
            vector[i % self._dim] += float(byte) / 256.0
        return vector

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.mark.requires_qdrant
def test_qdrant_upsert_and_search_round_trip(qdrant_available: str | None) -> None:
    """Upsert chunks into a real Qdrant instance and verify search returns them."""

    if not qdrant_available:
        pytest.skip("Qdrant not reachable")

    from app.retrieval.qdrant_index import QdrantVectorIndex

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        pytest.skip("qdrant-client not installed")

    collection = f"itest-{uuid.uuid4().hex[:8]}"
    embeddings = _SimpleEmbeddings(dim=64)
    client = QdrantClient(url=qdrant_available)
    index = QdrantVectorIndex(
        client=client,
        collection_name=collection,
        embedding_provider=embeddings,
        vector_size=64,
    )

    chunk_a = DocumentChunk(
        id="itest-a",
        certificate_id=1,
        certificate_code="CERT-IT-001",
        customer_id=101,
        source_type="metadata",
        path="itest/a",
        text="calibracion balanza analitica temperatura",
    )
    chunk_b = DocumentChunk(
        id="itest-b",
        certificate_id=2,
        certificate_code="CERT-IT-002",
        customer_id=202,
        source_type="metadata",
        path="itest/b",
        text="manometro presion industrial",
    )
    index.upsert([chunk_a, chunk_b])

    results = index.search("calibracion balanza", allowed_ids={"itest-a"}, top_k=5)
    result_ids = {chunk_id for chunk_id, _ in results}
    assert "itest-a" in result_ids
    assert "itest-b" not in result_ids

    # Cleanup
    try:
        client.delete_collection(collection_name=collection)
    except Exception:
        pass


@pytest.mark.requires_qdrant
def test_qdrant_tenant_isolation_prevents_cross_tenant_leak(qdrant_available: str | None) -> None:
    """Verify that searching with tenant A's IDs never returns tenant B's data."""

    if not qdrant_available:
        pytest.skip("Qdrant not reachable")

    from app.retrieval.qdrant_index import QdrantVectorIndex

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        pytest.skip("qdrant-client not installed")

    collection = f"itest-iso-{uuid.uuid4().hex[:8]}"
    embeddings = _SimpleEmbeddings(dim=64)
    client = QdrantClient(url=qdrant_available)
    index = QdrantVectorIndex(
        client=client,
        collection_name=collection,
        embedding_provider=embeddings,
        vector_size=64,
    )

    chunk_a = DocumentChunk(
        id="iso-a",
        certificate_id=1,
        certificate_code="CERT-ISO-001",
        customer_id=101,
        source_type="metadata",
        path="iso/a",
        text="calibracion temperatura",
    )
    chunk_b = DocumentChunk(
        id="iso-b",
        certificate_id=2,
        certificate_code="CERT-ISO-002",
        customer_id=202,
        source_type="metadata",
        path="iso/b",
        text="calibracion temperatura",
    )
    index.upsert([chunk_a, chunk_b])

    # Search with only tenant A's chunk ID
    results = index.search("calibracion temperatura", allowed_ids={"iso-a"}, top_k=10)
    result_ids = {chunk_id for chunk_id, _ in results}
    assert result_ids == {"iso-a"}
    assert "iso-b" not in result_ids

    try:
        client.delete_collection(collection_name=collection)
    except Exception:
        pass
