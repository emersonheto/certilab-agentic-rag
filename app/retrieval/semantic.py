from __future__ import annotations

from app.domain.models import DocumentChunk, RetrievedSource
from app.ingestion.indexer import InMemoryVectorIndex
from app.retrieval.constants import DEFAULT_RETRIEVAL_TOP_K
from app.retrieval.protocols import VectorIndex
from app.retrieval.query import extract_certificate_code
from app.security.access_control import AccessScope, filter_chunks


class SemanticRetriever:
    """Tenant-aware semantic-ish retrieval over PDF text chunks.

    Accepts an optional ``index`` implementing the VectorIndex protocol.
    When no index is provided, an InMemoryVectorIndex is built internally
    (mock/hermetic mode). In real mode, a QdrantVectorIndex is injected.
    """

    def __init__(self, chunks: list[DocumentChunk], index: VectorIndex | None = None) -> None:
        self._chunks = chunks
        self._chunk_lookup: dict[str, DocumentChunk] = {chunk.id: chunk for chunk in chunks}
        if index is None:
            index = InMemoryVectorIndex()
            index.upsert(chunks)
        self._index = index

    def retrieve(
        self, question: str, scope: AccessScope, top_k: int = DEFAULT_RETRIEVAL_TOP_K
    ) -> tuple[str, list[RetrievedSource]]:
        visible_chunks = filter_chunks(scope, self._chunks)
        requested_code = extract_certificate_code(question)
        if requested_code is not None:
            visible_chunks = [chunk for chunk in visible_chunks if chunk.certificate_code == requested_code]
        allowed_ids = {chunk.id for chunk in visible_chunks}
        results = self._index.search(question, allowed_ids, top_k=top_k)
        if not results:
            return "No se encontraron fragmentos técnicos relevantes dentro del alcance autorizado.", []
        sources = []
        for chunk_id, score in results:
            chunk = self._chunk_lookup.get(chunk_id)
            if chunk is None:
                continue
            sources.append(
                RetrievedSource(
                    certificate_id=chunk.certificate_id,
                    code=chunk.certificate_code,
                    customer_id=chunk.customer_id,
                    source_type=chunk.source_type,
                    path=chunk.path,
                    snippet=chunk.text[:260],
                    score=score,
                )
            )
        answer = "Resumen basado en documentos autorizados: " + " ".join(source.snippet for source in sources[:2])
        return answer, sources
