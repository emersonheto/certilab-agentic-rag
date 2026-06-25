from __future__ import annotations

from app.domain.models import DocumentChunk, RetrievedSource
from app.ingestion.indexer import InMemoryVectorIndex
from app.retrieval.constants import DEFAULT_RETRIEVAL_TOP_K
from app.retrieval.protocols import VectorIndex
from app.retrieval.query import extract_certificate_code
from app.security.access_control import AccessScope


class SemanticRetriever:
    """Tenant-aware semantic-ish retrieval over PDF text chunks.

    Accepts an optional ``index`` implementing the VectorIndex protocol.
    When no index is provided, an InMemoryVectorIndex is built internally
    (mock/hermetic mode). In real mode, a QdrantVectorIndex is injected.

    Tenant isolation is enforced at the index level via ``allowed_customer_ids``
    payload filtering — no in-memory chunk lookups are needed for Qdrant mode.
    """

    def __init__(self, chunks: list[DocumentChunk] | None = None, index: VectorIndex | None = None) -> None:
        if index is None:
            # Mock mode: need chunks to build InMemoryVectorIndex
            self._chunks = chunks or []
            index = InMemoryVectorIndex()
            index.upsert(self._chunks)
        else:
            # Real mode: Qdrant has everything, no need to keep chunks in memory
            self._chunks = []
        self._index = index

    def retrieve(
        self, question: str, scope: AccessScope, top_k: int = DEFAULT_RETRIEVAL_TOP_K
    ) -> tuple[str, list[RetrievedSource]]:
        # Determine customer filter from scope
        allowed_customer_ids: set[int] | None
        if scope.is_global:
            allowed_customer_ids = None  # ADMIN/TECHNICIAN: see all
        else:
            assert scope.customer_id is not None, "CLIENT scope must have customer_id"
            allowed_customer_ids = {scope.customer_id}  # CLIENT: only own data

        # If we have chunks (mock mode) and a specific code is requested, filter pre-search
        requested_code = extract_certificate_code(question)
        if requested_code is not None and self._chunks:
            # In mock mode, filter chunks by code and build customer set from them
            matching_customer_ids = {
                chunk.customer_id
                for chunk in self._chunks
                if chunk.certificate_code == requested_code
            }
            if allowed_customer_ids is None:
                allowed_customer_ids = matching_customer_ids
            else:
                allowed_customer_ids &= matching_customer_ids
            if not allowed_customer_ids:
                return "No se encontraron fragmentos técnicos relevantes dentro del alcance autorizado.", []

        results = self._index.search(question, allowed_customer_ids, top_k=top_k)
        if not results:
            return "No se encontraron fragmentos técnicos relevantes dentro del alcance autorizado.", []

        sources = []
        for chunk_id, score, payload in results:
            sources.append(
                RetrievedSource(
                    certificate_id=payload.get("certificate_id"),
                    code=payload.get("code", ""),
                    customer_id=payload.get("customer_id"),
                    source_type=payload.get("source_type", "semantic"),
                    path=payload.get("path", ""),
                    snippet=payload.get("text", "")[:260],
                    score=score,
                )
            )
        answer = "Resumen basado en documentos autorizados: " + " ".join(source.snippet for source in sources[:2])
        return answer, sources
