from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from app.domain.models import DocumentChunk
from app.logging import get_logger
from app.observability import trace_span

logger = get_logger("retrieval.qdrant")


class EmbeddingProviderProtocol(Protocol):
    """Minimal interface QdrantVectorIndex needs from an embedding provider."""

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

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
    payload. Retrieval is restricted by ``allowed_customer_ids`` using Qdrant
    payload filtering so tenant isolation is enforced at the index level without
    in-memory lookup dictionaries.

    Collection initialization is idempotent: the collection is created only if
    it does not already exist, preserving any previously ingested data.

    Security notes:
    - PII columns never reach this index. The MySQLLoader maps only allowlisted
      text to DocumentChunk.text; the embedding provider only sees that
      sanitized text. PII (password, ruc, email, phone) is structurally
      excluded at the connector level.
    - Tenant isolation: every search call receives an ``allowed_customer_ids``
      set computed from the caller's AccessScope. Points outside this set are
      filtered server-side by Qdrant payload and never returned, preventing
      cross-tenant data leakage.
    - A search with an empty ``allowed_customer_ids`` set returns no results,
      ensuring fail-closed behavior when no tenant scope is available.
    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embedding_provider: EmbeddingProviderProtocol,
        vector_size: int,
        distance: str = "Cosine",
        upsert_batch_size: int = 200,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._vector_size = vector_size
        self._distance = distance
        self._upsert_batch_size = upsert_batch_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist (idempotent)."""

        if not self._client.collection_exists(collection_name=self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=_make_vector_params(self._vector_size, self._distance),
            )

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Embed and upsert document chunks with tenant metadata in payload.

        Idempotent: chunks whose deterministic UUID already exists in Qdrant
        are skipped (no re-embedding, no re-write).
        """

        started_at = time.perf_counter()
        logger.info("qdrant.upsert.start", total_chunks=len(chunks))
        with trace_span("rag.ingest.upsert") as span:
            if not chunks:
                span.set_attribute("rag.ingest.new_chunks", 0)
                span.set_attribute("rag.ingest.skipped_chunks", 0)
                span.set_attribute("rag.duration_ms", int((time.perf_counter() - started_at) * 1000))
                logger.info("qdrant.upsert.complete", new_chunks=0, skipped=0, duration_ms=int((time.perf_counter() - started_at) * 1000))
                return

            # Compute deterministic UUIDs for all chunks
            candidate_uuids: dict[str, DocumentChunk] = {}
            for chunk in chunks:
                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id))
                candidate_uuids[point_uuid] = chunk

            # Check which UUIDs already exist in Qdrant
            existing = self._client.retrieve(
                collection_name=self._collection_name,
                ids=list(candidate_uuids.keys()),
                with_payload=False,
                with_vectors=False,
            )
            existing_ids = {str(point.id) for point in existing}

            # Only embed + upsert what's new
            new_chunks = [chunk for uuid_str, chunk in candidate_uuids.items() if uuid_str not in existing_ids]
            skipped_count = len(chunks) - len(new_chunks)
            if not new_chunks:
                span.set_attribute("rag.ingest.new_chunks", 0)
                span.set_attribute("rag.ingest.skipped_chunks", skipped_count)
                span.set_attribute("rag.duration_ms", int((time.perf_counter() - started_at) * 1000))
                logger.info("qdrant.upsert.complete", new_chunks=0, skipped=skipped_count, duration_ms=int((time.perf_counter() - started_at) * 1000))
                return

            # Batch embed all new chunks at once
            texts = [chunk.text for chunk in new_chunks]
            vectors = self._embedding_provider.embed_batch(texts)

            points = []
            for chunk, vector in zip(new_chunks, vectors, strict=True):
                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id))
                payload = {
                    "chunk_id": chunk.id,
                    "customer_id": chunk.customer_id,
                    "code": chunk.certificate_code,
                    "certificate_id": chunk.certificate_id,
                    "source_type": chunk.source_type,
                    "path": chunk.path,
                    "text": chunk.text,
                }
                points.append(_make_point(point_uuid, vector, payload))

            # Upsert in batches to stay under Qdrant's 32MB payload limit
            batch_size = self._upsert_batch_size
            for i in range(0, len(points), batch_size):
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points[i : i + batch_size],
                )

            span.set_attribute("rag.ingest.new_chunks", len(new_chunks))
            span.set_attribute("rag.ingest.skipped_chunks", skipped_count)
            span.set_attribute("rag.duration_ms", int((time.perf_counter() - started_at) * 1000))
            logger.info("qdrant.upsert.complete", new_chunks=len(new_chunks), skipped=skipped_count, duration_ms=int((time.perf_counter() - started_at) * 1000))

    def search(
        self, query: str, allowed_customer_ids: set[int] | None, top_k: int
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for chunks matching *query*, filtered by *allowed_customer_ids*.

        ``allowed_customer_ids``:
        - ``None`` — global scope (ADMIN/TECHNICIAN): return all chunks.
        - ``set()`` — empty set: return no results (fail-closed).
        - ``{1, 2}`` — return only chunks whose payload ``customer_id`` is in the set.

        Returns ``(chunk_id, score, payload)`` triples.
        """

        started_at = time.perf_counter()
        allowed_count = len(allowed_customer_ids) if allowed_customer_ids is not None else 0
        logger.debug("qdrant.search.start", top_k=top_k, allowed_count=allowed_count)

        if allowed_customer_ids is not None and not allowed_customer_ids:
            logger.debug("qdrant.search.complete", results=0, duration_ms=int((time.perf_counter() - started_at) * 1000))
            return []  # fail-closed for empty scope

        query_vector = self._embedding_provider.embed(query)

        # Build filter: None = no filter (global scope), set = filter by customer_id
        query_filter = None
        if allowed_customer_ids is not None:
            from qdrant_client.models import FieldCondition, Filter, MatchAny

            query_filter = Filter(
                must=[
                    FieldCondition(key="customer_id", match=MatchAny(any=list(allowed_customer_ids)))
                ]
            )

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        result = [
            (point.payload.get("chunk_id", str(point.id)), float(point.score), point.payload)
            for point in response.points
        ]
        logger.debug("qdrant.search.complete", results=len(result), duration_ms=int((time.perf_counter() - started_at) * 1000))
        return result

    def get_stats(self) -> dict[str, Any]:
        """Return collection statistics for ingestion tracking."""

        try:
            info = self._client.get_collection(self._collection_name)
            return {
                "total_points": info.points_count or 0,
                "indexed_vectors": info.indexed_vectors_count or 0,
            }
        except Exception:
            return {"total_points": 0, "indexed_vectors": 0}

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
            upsert_batch_size=settings.qdrant_upsert_batch_size,
        )
