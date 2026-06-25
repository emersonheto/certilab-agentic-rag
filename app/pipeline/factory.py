"""Pipeline factory — builds the configured RAG pipeline.

Moved from ``app/graph.py`` (PR 4 structural refactor). Contains
``build_rag_pipeline`` and all helper functions used during construction and
re-indexing.
"""

from __future__ import annotations

from app.config import Settings
from app.pipeline.deterministic import CertilabRagPipeline
from app.pipeline.protocols import RagPipeline
from app.retrieval.protocols import VectorIndex


def build_rag_pipeline(settings: Settings) -> RagPipeline:
    """Build the configured graph engine with graceful fallback for optional extras.

    Mock mode (default) loads from JSON fixtures and uses an in-memory index.
    Real mode loads from MySQL and uses Qdrant; both paths use the same
    pipeline class for consistent routing behavior.
    """

    if settings.app_mode == "real":
        return _build_real_pipeline(settings)
    return _build_mock_pipeline(settings)


def _build_mock_pipeline(settings: Settings) -> RagPipeline:
    """Build the hermetic mock pipeline from JSON fixtures."""

    if settings.graph_engine == "langgraph":
        try:
            from app.pipeline.langgraph import CertilabLangGraphPipeline

            return CertilabLangGraphPipeline(
                data_dir=settings.data_dir,
                top_k=settings.default_top_k,
                settings=settings,
            )
        except ImportError:
            pass
    return CertilabRagPipeline(
        data_dir=settings.data_dir,
        top_k=settings.default_top_k,
        settings=settings,
    )


def _reindex_pipeline(index: VectorIndex, settings: Settings) -> dict:
    """Reload data from MySQL, rebuild chunks, and upsert only what's new.

    Incremental: checks which certificates already have PDF chunks in Qdrant
    before downloading from S3. Only downloads and embeds new/changed PDFs.
    """

    from app.ingestion.mysql_loader import MySQLLoader
    from app.ingestion.splitter import build_metadata_chunks, build_s3_pdf_chunks
    from app.retrieval.qdrant_index import QdrantVectorIndex
    from app.tools.mysql_connector import MySQLCertificateConnector, MySQLConnectorConfig

    connector = MySQLCertificateConnector(MySQLConnectorConfig.from_settings(settings))
    loader = MySQLLoader(connector)
    _, certificates, _ = loader.load()

    # 1. Metadata chunks (always cheap, no S3 needed)
    metadata_chunks = build_metadata_chunks(certificates)

    # 2. Find which certificates already have PDF chunks in Qdrant
    already_indexed_certs: set[int] = set()
    if isinstance(index, QdrantVectorIndex):
        already_indexed_certs = _find_indexed_cert_ids(index)

    # 3. Only download PDFs for certificates NOT yet indexed
    pdf_texts: dict[int, str] = {}
    certs_needing_pdf = [
        c for c in certificates
        if c.pdf_path and c.id not in already_indexed_certs
    ]
    if certs_needing_pdf:
        from app.ingestion.pdf_extractor import S3PdfTextExtractor

        extractor = S3PdfTextExtractor(settings)
        for cert in certs_needing_pdf:
            try:
                pdf_texts[cert.id] = extractor.extract_text(cert.pdf_path)
            except Exception:
                pass

    # 4. Build chunks only for new PDFs
    pdf_chunks = build_s3_pdf_chunks(certificates, pdf_texts)
    chunks = [*metadata_chunks, *pdf_chunks]

    # 5. Upsert (idempotent — skips existing metadata chunks)
    index.upsert(chunks)

    return {
        "indexed": len(pdf_chunks),
        "skipped": len(already_indexed_certs),
        "total_certificates": len(certificates),
        "certificates_with_pdf": sum(1 for c in certificates if c.pdf_path),
        "certificates_embedded": len(already_indexed_certs) + len(pdf_texts),
        "orphaned_pdfs": sum(1 for c in certificates if c.pdf_path) - len(already_indexed_certs) - len(pdf_texts),
    }


def _find_indexed_cert_ids(index: VectorIndex) -> set[int]:
    """Find certificate IDs that already have pdf_text chunks in Qdrant."""

    from app.retrieval.qdrant_index import QdrantVectorIndex
    from qdrant_client.models import FieldCondition, MatchValue, Filter

    if not isinstance(index, QdrantVectorIndex):
        return set()
    try:
        # Scroll all pdf_text points, extract unique certificate_ids
        cert_ids: set[int] = set()
        offset = None
        while True:
            results, offset = index._client.scroll(
                collection_name=index._collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="source_type", match=MatchValue(value="pdf_text"))
                ]),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                cid = point.payload.get("certificate_id")
                if cid is not None:
                    cert_ids.add(int(cid))
            if offset is None:
                break
        return cert_ids
    except Exception:
        return set()


def _build_real_pipeline(settings: Settings) -> RagPipeline:
    """Build the real-stack pipeline from MySQL + Qdrant.

    Loads certificate metadata from MySQL for structured retrieval and
    connects to the existing Qdrant collection for semantic retrieval.
    Does NOT re-ingest PDFs — that is the job of ``reindex()``.
    """

    from app.ingestion.mysql_loader import MySQLLoader
    from app.ingestion.splitter import build_metadata_chunks
    from app.retrieval.qdrant_index import QdrantVectorIndex
    from app.tools.embeddings import EmbeddingProviderConfig, EmbeddingsProvider
    from app.tools.mysql_connector import MySQLCertificateConnector, MySQLConnectorConfig

    connector = MySQLCertificateConnector(MySQLConnectorConfig.from_settings(settings))
    loader = MySQLLoader(connector)
    customers, certificates, histories = loader.load()

    # Minimal metadata chunks for SemanticRetriever fallback only
    metadata_chunks = build_metadata_chunks(certificates)

    embedding_provider = EmbeddingsProvider(EmbeddingProviderConfig.from_settings(settings))
    index = QdrantVectorIndex.from_settings(settings, embedding_provider)

    if settings.graph_engine == "langgraph":
        try:
            from app.pipeline.langgraph import CertilabLangGraphPipeline

            return CertilabLangGraphPipeline(
                top_k=settings.default_top_k,
                settings=settings,
                customers=customers,
                certificates=certificates,
                histories=histories,
                chunks=metadata_chunks,
                index=index,
            )
        except ImportError:
            pass
    return CertilabRagPipeline(
        top_k=settings.default_top_k,
        settings=settings,
        customers=customers,
        certificates=certificates,
        histories=histories,
        chunks=metadata_chunks,
        index=index,
    )
