import builtins
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.domain.models import DocumentChunk, Role
from app.graph import CertilabRagPipeline
from app.observability import configure_observability, trace_span
from app.observability import phoenix
from app.security.access_control import Principal


class FailingSpan:
    def set_attribute(self, key: str, value: object) -> None:
        raise RuntimeError("span attribute failure")


class FailingSpanContext:
    def __enter__(self) -> FailingSpan:
        return FailingSpan()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        raise RuntimeError("span end failure")


class FailingStartSpanContext:
    def __enter__(self) -> FailingSpan:
        raise RuntimeError("span start failure")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        return None


class FailingTracer:
    def __init__(self, *, fail_on_start: bool = False) -> None:
        self._fail_on_start = fail_on_start

    def start_as_current_span(self, name: str) -> FailingSpanContext | FailingStartSpanContext:
        if self._fail_on_start:
            return FailingStartSpanContext()
        return FailingSpanContext()


def test_observability_configuration_is_noop_when_disabled() -> None:
    settings = Settings(phoenix_enabled=False)

    configure_observability(settings)

    with trace_span("test.disabled", {"test.enabled": False}) as span:
        span.set_attribute("test.source_count", 0)


def test_trace_span_does_not_raise_without_runtime_configuration() -> None:
    with trace_span("test.no_runtime", {"test.question_length": 12}) as span:
        span.set_attribute("test.route", "structured")


def test_enabled_observability_degrades_to_noop_when_optional_dependencies_are_missing() -> None:
    phoenix._configured = False
    phoenix._tracer = None
    original_import = builtins.__import__

    def raise_for_opentelemetry(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name.startswith("opentelemetry"):
            raise ImportError(name)
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = raise_for_opentelemetry
    try:
        configure_observability(Settings(phoenix_enabled=True))
    finally:
        builtins.__import__ = original_import

    with trace_span("test.missing_dependencies") as span:
        span.set_attribute("test.source_count", 0)


def test_trace_span_suppresses_tracer_lifecycle_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phoenix, "_tracer", FailingTracer(fail_on_start=True))

    with trace_span("test.failing_start", {"test.source_count": 1}) as span:
        span.set_attribute("test.route", "structured")


def test_graph_ask_still_works_with_failing_enabled_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phoenix, "_configured", True)
    monkeypatch.setattr(phoenix, "_tracer", FailingTracer())
    pipeline = CertilabRagPipeline(Path("data"))
    principal = Principal(role=Role.CLIENT, customer_id=101, user_id=1010)

    response = pipeline.ask("¿Cuántos certificados hay?", principal)

    assert response.route == "structured"
    assert response.sources
    assert {source.customer_id for source in response.sources} == {101}


def test_ask_endpoint_still_works_with_failing_enabled_observability(
    monkeypatch: pytest.MonkeyPatch, demo_client_101_token: str
) -> None:
    from app.main import create_app

    monkeypatch.setattr(phoenix, "_configured", True)
    monkeypatch.setattr(phoenix, "_tracer", FailingTracer())
    client = TestClient(create_app())

    response = client.post(
        "/ask",
        headers={"X-Demo-Token": demo_client_101_token},
        json={"question": "¿Cuántos certificados hay?"},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "structured"


def test_api_startup_still_works_when_observability_setup_fails(
    monkeypatch: pytest.MonkeyPatch, demo_client_101_token: str
) -> None:
    from app.main import create_app

    def fail_observability_setup(settings: Settings) -> None:
        raise RuntimeError("Phoenix setup failed")

    monkeypatch.setattr("app.main.configure_observability", fail_observability_setup)
    client = TestClient(create_app())

    health_response = client.get("/health")
    ask_response = client.post(
        "/ask",
        headers={"X-Demo-Token": demo_client_101_token},
        json={"question": "certificados"},
    )

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert ask_response.status_code == 200
    assert ask_response.json()["route"] == "structured"


def test_graph_ask_still_works_when_phoenix_is_disabled() -> None:
    settings = Settings(phoenix_enabled=False)
    configure_observability(settings)
    pipeline = CertilabRagPipeline(Path("data"))
    principal = Principal(role=Role.CLIENT, customer_id=101, user_id=1010)

    response = pipeline.ask("¿Cuántos certificados hay?", principal)

    assert response.route == "structured"
    assert response.sources
    assert {source.customer_id for source in response.sources} == {101}


# ---------------------------------------------------------------------------
# Recording tracer helpers — spy on span attributes for payload safety tests
# ---------------------------------------------------------------------------


class _RecordingSpan:
    """Records every set_attribute call made during a trace."""

    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _RecordingSpanContext:
    def __init__(self, span: _RecordingSpan) -> None:
        self.span = span

    def __enter__(self) -> _RecordingSpan:
        return self.span

    def __exit__(self, *args: object) -> None:
        return None


class _RecordingTracer:
    """Inject into phoenix._tracer to record span attributes without
    exporting them.

    Records every span created in ``self.spans`` (in creation order), and
    keeps ``self.last_span`` as the most recently created span.
    """

    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []
        self.last_span = _RecordingSpan()

    def start_as_current_span(self, name: str) -> _RecordingSpanContext:
        span = _RecordingSpan()
        self.spans.append(span)
        self.last_span = span
        return _RecordingSpanContext(span)


# ---------------------------------------------------------------------------
# Embedding observability tests
# ---------------------------------------------------------------------------


def test_embed_works_with_failing_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the tracer fails, embed() must still return a correct vector."""
    from app.observability import phoenix
    from app.tools.embeddings import EmbeddingProviderConfig, EmbeddingsProvider

    monkeypatch.setattr(phoenix, "_configured", True)
    monkeypatch.setattr(phoenix, "_tracer", FailingTracer())

    config = EmbeddingProviderConfig(
        provider="local",
        openai_api_key=None,
        openai_embedding_model="text-embedding-3-small",
        sentence_transformers_model="all-MiniLM-L6-v2",
    )
    provider = EmbeddingsProvider(config)
    provider._local_engine = None
    provider._local_error = RuntimeError("model not available")

    vector = provider.embed("test text")

    assert isinstance(vector, list)
    assert all(v == 0.0 for v in vector)
    assert len(vector) == 384


def test_embed_batch_works_with_failing_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the tracer fails, embed_batch() must still return correct vectors."""
    from app.observability import phoenix
    from app.tools.embeddings import EmbeddingProviderConfig, EmbeddingsProvider

    monkeypatch.setattr(phoenix, "_configured", True)
    monkeypatch.setattr(phoenix, "_tracer", FailingTracer())

    config = EmbeddingProviderConfig(
        provider="local",
        openai_api_key=None,
        openai_embedding_model="text-embedding-3-small",
        sentence_transformers_model="all-MiniLM-L6-v2",
    )
    provider = EmbeddingsProvider(config)
    provider._local_engine = None
    provider._local_error = RuntimeError("model not available")

    result = provider.embed_batch(["hello", "world"])

    assert len(result) == 2
    assert all(len(v) == 384 for v in result)


def test_embed_does_not_export_text_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed() span attributes must NOT contain the input text (PII safety)."""
    from app.observability import phoenix
    from app.tools.embeddings import EmbeddingProviderConfig, EmbeddingsProvider

    tracer = _RecordingTracer()
    monkeypatch.setattr(phoenix, "_tracer", tracer)

    config = EmbeddingProviderConfig(
        provider="local",
        openai_api_key=None,
        openai_embedding_model="text-embedding-3-small",
        sentence_transformers_model="all-MiniLM-L6-v2",
    )
    provider = EmbeddingsProvider(config)
    provider._local_engine = None
    provider._local_error = RuntimeError("model not available")

    provider.embed("super secret PII text")

    # Verify no PII leaked into attribute values
    for value in tracer.last_span.attributes.values():
        assert "super secret PII text" not in str(value)
        assert "PII" not in str(value)

    # Verify only expected safe attributes are present
    expected_keys = {"rag.embed.provider", "rag.embed.vector_dim", "rag.duration_ms"}
    assert set(tracer.last_span.attributes.keys()) == expected_keys
    assert tracer.last_span.attributes["rag.embed.provider"] == "zero_fallback"
    assert tracer.last_span.attributes["rag.embed.vector_dim"] == 384


def test_embed_batch_does_not_export_text_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_batch() span attributes must NOT contain any input text (PII safety)."""
    from app.observability import phoenix
    from app.tools.embeddings import EmbeddingProviderConfig, EmbeddingsProvider

    tracer = _RecordingTracer()
    monkeypatch.setattr(phoenix, "_tracer", tracer)

    config = EmbeddingProviderConfig(
        provider="local",
        openai_api_key=None,
        openai_embedding_model="text-embedding-3-small",
        sentence_transformers_model="all-MiniLM-L6-v2",
    )
    provider = EmbeddingsProvider(config)
    provider._local_engine = None
    provider._local_error = RuntimeError("model not available")

    provider.embed_batch(["super secret PII text", "another PII chunk"])

    # Verify no PII leaked into any recorded span
    for span in tracer.spans:
        for value in span.attributes.values():
            assert "super secret PII text" not in str(value)
            assert "another PII chunk" not in str(value)
            assert "PII" not in str(value)

    # Verify only expected safe attributes are present on the embed_batch span
    batch_span = tracer.last_span  # innermost span
    expected_keys = {"rag.embed.batch_size", "rag.embed.provider", "rag.duration_ms"}
    assert set(batch_span.attributes.keys()) == expected_keys
    assert batch_span.attributes["rag.embed.batch_size"] == 2
    assert batch_span.attributes["rag.embed.provider"] == "zero_fallback"


# ---------------------------------------------------------------------------
# Qdrant upsert observability tests
# ---------------------------------------------------------------------------


class _FakeQdrantClient:
    """Minimal in-memory stand-in for testing upsert with failing tracing."""

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, object]] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def create_collection(self, collection_name: str, **kwargs: object) -> None:
        self._collections[collection_name] = {}

    def retrieve(
        self, collection_name: str, ids: list[str], **kwargs: object
    ) -> list[object]:
        return []

    def upsert(self, collection_name: str, points: list[object]) -> None:
        for point in points:
            self._collections[collection_name][point.id] = point  # type: ignore[attr-defined]


class _FakeEmbeddingsProvider:
    @property
    def dimension(self) -> int:
        return 8

    def embed(self, text: str) -> list[float]:
        return [0.0] * 8

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]


def test_upsert_works_with_failing_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the tracer fails, upsert() must still insert chunks."""
    from app.observability import phoenix
    from app.retrieval.qdrant_index import QdrantVectorIndex

    monkeypatch.setattr(phoenix, "_configured", True)
    monkeypatch.setattr(phoenix, "_tracer", FailingTracer())

    client = _FakeQdrantClient()
    embeddings = _FakeEmbeddingsProvider()
    index = QdrantVectorIndex(
        client=client,
        collection_name="test",
        embedding_provider=embeddings,
        vector_size=8,
    )

    chunk = DocumentChunk(
        id="test-chunk",
        certificate_id=1,
        certificate_code="CERT-001",
        customer_id=101,
        source_type="pdf_text",
        path="test.txt",
        text="test content",
    )

    index.upsert([chunk])

    # Verify upsert actually stored the chunk
    assert len(client._collections.get("test", {})) == 1


def test_upsert_does_not_export_text_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """upsert() span attributes must NOT contain chunk text, code, or
    customer_id (PII safety)."""
    from app.observability import phoenix
    from app.retrieval.qdrant_index import QdrantVectorIndex

    tracer = _RecordingTracer()
    monkeypatch.setattr(phoenix, "_tracer", tracer)

    client = _FakeQdrantClient()
    embeddings = _FakeEmbeddingsProvider()
    index = QdrantVectorIndex(
        client=client,
        collection_name="test",
        embedding_provider=embeddings,
        vector_size=8,
    )

    chunk = DocumentChunk(
        id="pii-chunk",
        certificate_id=1,
        certificate_code="SECRET-CODE-999",
        customer_id=999,
        source_type="pdf_text",
        path="pii-test.txt",
        text="super secret certificate content",
    )

    index.upsert([chunk])

    # Verify NO PII leaked into any recorded span attribute value
    for span in tracer.spans:
        for value in span.attributes.values():
            assert "super secret certificate content" not in str(value)
            assert "SECRET-CODE-999" not in str(value)
            assert "secret" not in str(value).lower()
            assert "999" not in str(value)

    # Find the upsert span (rag.ingest.upsert) among all recorded spans.
    # The upsert span is first recorded (outermost), embed_batch is second.
    upsert_span = tracer.spans[0] if len(tracer.spans) >= 1 else tracer.last_span
    expected_keys = {"rag.ingest.new_chunks", "rag.ingest.skipped_chunks", "rag.duration_ms"}
    assert set(upsert_span.attributes.keys()) == expected_keys
    assert upsert_span.attributes["rag.ingest.new_chunks"] == 1
    assert upsert_span.attributes["rag.ingest.skipped_chunks"] == 0
