from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from app.domain.models import DocumentChunk


class EmbeddingProviderProtocol(Protocol):
    """Minimal interface QdrantVectorIndex needs from an embedding provider."""

    def embed(self, text: str) -> list[float]: ...

    @property
    def dimension(self) -> int: ...


# --- Stand-in model types used when qdrant-client is not installed ---
# These mirror the attribute surface of qdrant_client.models equivalents so the
# index works identically with a real QdrantClient or an in-memory fake.


@dataclass
class _SimplePoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class _SimpleHasIdCondition:
    has_id: list[str]


@dataclass
class _SimpleFilter:
    should: list[Any] | None = None


@dataclass
class _SimpleVectorParams:
    size: int
    distance: str


def _make_point(point_id: str, vector: list[float], payload: dict[str, Any]) -> Any:
    """Create a point struct using qdrant-client types when available."""

    try:
        from qdrant_client.models import PointStruct

        return PointStruct(id=point_id, vector=vector, payload=payload)
    except ImportError:
        return _SimplePoint(id=point_id, vector=vector, payload=payload)


def _make_filter(allowed_ids: set[str]) -> Any:
    """Create a Qdrant filter restricting results to *allowed_ids*."""

    id_list = sorted(allowed_ids)
    try:
        from qdrant_client.models import Filter, HasIdCondition

        return Filter(should=[HasIdCondition(has_id=id_list)])  # type: ignore[arg-type]
    except ImportError:
        return _SimpleFilter(should=[_SimpleHasIdCondition(has_id=id_list)])


def _make_vector_params(size: int, distance: str) -> Any:
    try:
        from qdrant_client.models import Distance, VectorParams

        distance_enum = Distance.COSINE if distance.upper() == "COSINE" else Distance.DOT
        return VectorParams(size=size, distance=distance_enum)
    except ImportError:
        return _SimpleVectorParams(size=size, distance=distance)


class QdrantVectorIndex:
    """Qdrant-backed VectorIndex implementation with idempotent collection init.

    Stores tenant scope (customer_id) and certificate code in each point's
    payload. Retrieval is restricted to ``allowed_ids`` so tenant isolation
    is enforced at the index level.

    Collection initialization is idempotent: the collection is created only if
    it does not already exist, preserving any previously ingested data.

    Security notes:
    - PII columns never reach this index. The MySQLLoader maps only allowlisted
      text to DocumentChunk.text; the embedding provider only sees that
      sanitized text. PII (password, ruc, email, phone) is structurally
      excluded at the connector level.
    - Tenant isolation: every search call receives an ``allowed_ids`` set
      computed from the caller's AccessScope. Points outside this set are
      filtered server-side by Qdrant and never returned, preventing
      cross-tenant data leakage.
    - A search with an empty ``allowed_ids`` set returns no results, ensuring
      fail-closed behavior when no tenant scope is available.
    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embedding_provider: EmbeddingProviderProtocol,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._vector_size = vector_size
        self._distance = distance
        self._chunk_to_uuid: dict[str, str] = {}
        self._uuid_to_chunk: dict[str, str] = {}
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist (idempotent)."""

        if not self._client.collection_exists(collection_name=self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=_make_vector_params(self._vector_size, self._distance),
            )

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Embed and upsert document chunks with tenant metadata in payload."""

        if not chunks:
            return
        points = []
        for chunk in chunks:
            vector = self._embedding_provider.embed(chunk.text)
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id))
            self._chunk_to_uuid[chunk.id] = point_uuid
            self._uuid_to_chunk[point_uuid] = chunk.id
            payload = {
                "customer_id": chunk.customer_id,
                "code": chunk.certificate_code,
                "certificate_id": chunk.certificate_id,
                "source_type": chunk.source_type,
                "path": chunk.path,
                "text": chunk.text,
            }
            points.append(_make_point(point_uuid, vector, payload))
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query: str, allowed_ids: set[str], top_k: int) -> list[tuple[str, float]]:
        """Search for chunks matching *query*, restricted to *allowed_ids*.

        Returns ``(chunk_id, score)`` tuples sorted by descending score.
        Chunks not in *allowed_ids* are never returned.
        """

        if not allowed_ids:
            return []
        uuid_allowed = {self._chunk_to_uuid[cid] for cid in allowed_ids if cid in self._chunk_to_uuid}
        if not uuid_allowed:
            return []
        query_vector = self._embedding_provider.embed(query)
        query_filter = _make_filter(uuid_allowed)
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        return [
            (self._uuid_to_chunk.get(str(point.id), str(point.id)), float(point.score))
            for point in response.points
        ]

    @classmethod
    def from_settings(cls, settings: Any, embedding_provider: EmbeddingProviderProtocol) -> QdrantVectorIndex:
        """Build a QdrantVectorIndex from application settings (real mode only).

        Imports qdrant-client lazily so mock mode never requires the package.
        """

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "Real Qdrant mode requires the optional 'qdrant-client' package."
            ) from exc

        client_kwargs: dict[str, Any] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            client_kwargs["api_key"] = settings.qdrant_api_key
        client = QdrantClient(**client_kwargs)
        return cls(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding_provider=embedding_provider,
            vector_size=embedding_provider.dimension,
        )
