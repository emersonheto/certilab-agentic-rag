from __future__ import annotations

import os

import pytest


@pytest.mark.requires_mysql
@pytest.mark.requires_qdrant
def test_mysql_to_qdrant_full_round_trip(qdrant_available: str | None) -> None:
    """Full integration: MySQL → MySQLLoader → chunks → Qdrant → search.

    Requires both a reachable MySQL instance (DB_HOST set) and Qdrant
    (QDRANT_URL set). Skips automatically otherwise.
    """

    if not qdrant_available:
        pytest.skip("Qdrant not reachable")
    if not os.environ.get("DB_HOST"):
        pytest.skip("DB_HOST not set")

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        pytest.skip("qdrant-client not installed")

    import uuid

    from app.config import Settings
    from app.ingestion.mysql_loader import MySQLLoader
    from app.ingestion.splitter import build_metadata_chunks
    from app.retrieval.qdrant_index import QdrantVectorIndex
    from app.tools.embeddings import EmbeddingProviderConfig, EmbeddingsProvider
    from app.tools.mysql_connector import MySQLCertificateConnector, MySQLConnectorConfig

    # Build settings from current environment (real DB credentials)
    settings = Settings()
    assert settings.app_mode == "real" or os.environ.get("APP_MODE") == "real", "Set APP_MODE=real for integration tests"

    connector = MySQLCertificateConnector(MySQLConnectorConfig.from_settings(settings))
    loader = MySQLLoader(connector)
    customers, certificates, histories = loader.load()

    assert len(certificates) > 0, "MySQL returned no certificates — check test DB seed data"

    # Verify no PII leaked into domain models
    for cert in certificates:
        assert not hasattr(cert, "ruc")
        assert not hasattr(cert, "email")
        assert not hasattr(cert, "phone")

    chunks = build_metadata_chunks(certificates)
    assert len(chunks) > 0

    collection = f"itest-mysql-{uuid.uuid4().hex[:8]}"
    embeddings = EmbeddingsProvider(EmbeddingProviderConfig.from_settings(settings))
    client = QdrantClient(url=qdrant_available)
    index = QdrantVectorIndex(
        client=client,
        collection_name=collection,
        embedding_provider=embeddings,
        vector_size=embeddings.dimension,
    )
    index.upsert(chunks)

    # Search with all chunk IDs (admin scope)
    all_ids = {chunk.id for chunk in chunks}
    results = index.search(certificates[0].code, allowed_ids=all_ids, top_k=5)

    assert len(results) > 0
    for chunk_id, score in results:
        assert chunk_id in all_ids

    try:
        client.delete_collection(collection_name=collection)
    except Exception:
        pass


@pytest.mark.requires_mysql
def test_mysql_loader_returns_canonical_models_from_real_db() -> None:
    """Verify MySQLLoader maps real rows correctly (requires real MySQL)."""

    if not os.environ.get("DB_HOST"):
        pytest.skip("DB_HOST not set")

    from app.config import Settings
    from app.ingestion.mysql_loader import MySQLLoader
    from app.tools.mysql_connector import MySQLCertificateConnector, MySQLConnectorConfig

    settings = Settings()
    connector = MySQLCertificateConnector(MySQLConnectorConfig.from_settings(settings))
    loader = MySQLLoader(connector)

    customers, certificates, histories = loader.load()

    if certificates:
        cert = certificates[0]
        # Canonical field names must be populated
        assert cert.code
        assert cert.customer_id
        assert cert.status
        assert cert.emitted_at
        # PII must be absent
        assert not hasattr(cert, "ruc")
        assert not hasattr(cert, "email")
        assert not hasattr(cert, "phone")
        assert not hasattr(cert, "password")
