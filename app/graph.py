from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

from app.config import Settings
from app.domain.models import Certificate, CertificateHistory, Customer, DocumentChunk, RetrievedSource
from app.ingestion.loader import load_certificates, load_customers, load_histories, load_pdf_texts
from app.ingestion.splitter import build_pdf_chunks
from app.observability import trace_span
from app.retrieval.constants import DEFAULT_RETRIEVAL_TOP_K
from app.retrieval.grader import route_question
from app.retrieval.protocols import VectorIndex
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.structured import StructuredRetriever
from app.schemas import AskResponse, Source
from app.security.access_control import AccessScope, Principal, scope_from_principal
from app.tools.openai_client import AnswerGenerator
from app.tools.web_search import TavilyWebSearch, WebSearchConfig, WebSearchResult

RouteName = Literal["structured", "semantic", "combined", "web_search"]


class RagPipeline(Protocol):
    def ask(self, question: str, principal: Principal) -> AskResponse:
        """Answer a tenant-scoped RAG question."""


class CertilabRagPipeline:
    """Agentic routing pipeline with deterministic local behavior by default."""

    def __init__(
        self,
        data_dir: Path | None = None,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        settings: Settings | None = None,
        *,
        customers: list[Customer] | None = None,
        certificates: list[Certificate] | None = None,
        histories: list[CertificateHistory] | None = None,
        chunks: list[DocumentChunk] | None = None,
        index: VectorIndex | None = None,
    ) -> None:
        if customers is None or certificates is None or histories is None:
            assert data_dir is not None, "data_dir is required when customers/certificates/histories are not provided"
            customers = load_customers(data_dir)
            certificates = load_certificates(data_dir)
            histories = load_histories(data_dir)
        if chunks is None:
            assert data_dir is not None
            chunks = build_pdf_chunks(certificates, load_pdf_texts(data_dir))
        self._top_k = top_k
        self._structured = StructuredRetriever(customers, certificates, histories)
        self._semantic = SemanticRetriever(chunks, index=index)
        self._web_search = TavilyWebSearch(WebSearchConfig(tavily_api_key=settings.tavily_api_key if settings else None))
        self._answer_generator = AnswerGenerator(settings) if settings is not None else None

    def ask(self, question: str, principal: Principal) -> AskResponse:
        scope = scope_from_principal(principal)
        pipeline_started_at = perf_counter()
        with trace_span(
            "rag.ask",
            {
                "rag.role": principal.role.value,
                "rag.has_customer_scope": principal.customer_id is not None,
                "rag.question_length": len(question),
            },
        ) as span:
            route = self._route_question(question)
            if route == "structured":
                answer, sources = self._retrieve_structured(question, scope)
            elif route == "semantic":
                answer, sources = self._retrieve_semantic(question, scope)
            elif route == "web_search":
                answer, sources = self._retrieve_web(question)
            else:
                answer, sources = self._combined(question, scope)

            unique_sources = _dedupe_sources(sources)
            answer = self._generate_answer(question, unique_sources, answer)
            span.set_attribute("rag.route", route)
            span.set_attribute("rag.source_count", len(unique_sources))
            span.set_attribute("rag.certificate_count", _certificate_count(unique_sources))
            span.set_attribute("rag.duration_ms", _elapsed_ms(pipeline_started_at))
            return AskResponse(answer=answer, route=route, sources=[_to_schema(source) for source in unique_sources])

    def _generate_answer(self, question: str, sources: list[RetrievedSource], fallback_answer: str) -> str:
        started_at = perf_counter()
        with trace_span("rag.generate_answer", {"rag.question_length": len(question), "rag.source_count": len(sources)}) as span:
            answer = (
                self._answer_generator.generate(question, sources, fallback_answer)
                if self._answer_generator is not None
                else fallback_answer
            )
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            span.set_attribute("rag.used_fallback", answer == fallback_answer)
            return answer

    def _combined(self, question: str, scope: AccessScope) -> tuple[str, list[RetrievedSource]]:
        started_at = perf_counter()
        with trace_span("rag.retrieve.combined", _scope_attributes(scope)) as span:
            structured_answer, structured_sources = self._retrieve_structured(question, scope)
            semantic_answer, semantic_sources = self._retrieve_semantic(question, scope)
            sources = [*structured_sources, *semantic_sources]
            span.set_attribute("rag.structured_source_count", len(structured_sources))
            span.set_attribute("rag.semantic_source_count", len(semantic_sources))
            span.set_attribute("rag.source_count", len(sources))
            span.set_attribute("rag.certificate_count", _certificate_count(sources))
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            return f"{structured_answer} {semantic_answer}", sources

    def _route_question(self, question: str) -> RouteName:
        started_at = perf_counter()
        with trace_span("rag.route_decision", {"rag.question_length": len(question)}) as span:
            route = route_question(question)
            span.set_attribute("rag.route", route)
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            return route

    def _retrieve_structured(self, question: str, scope: AccessScope) -> tuple[str, list[RetrievedSource]]:
        started_at = perf_counter()
        with trace_span("rag.retrieve.structured", _scope_attributes(scope)) as span:
            answer, sources = self._structured.retrieve(question, scope)
            span.set_attribute("rag.source_count", len(sources))
            span.set_attribute("rag.certificate_count", _certificate_count(sources))
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            return answer, sources

    def _retrieve_semantic(self, question: str, scope: AccessScope) -> tuple[str, list[RetrievedSource]]:
        started_at = perf_counter()
        with trace_span("rag.retrieve.semantic", _scope_attributes(scope)) as span:
            answer, sources = self._semantic.retrieve(question, scope, self._top_k)
            span.set_attribute("rag.source_count", len(sources))
            span.set_attribute("rag.certificate_count", _certificate_count(sources))
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            return answer, sources

    def _retrieve_web(self, question: str) -> tuple[str, list[RetrievedSource]]:
        started_at = perf_counter()
        with trace_span("rag.retrieve.web_search", {"rag.question_length": len(question)}) as span:
            results = self._web_search.search(question)
            sources = _web_sources(results)
            span.set_attribute("rag.web_result_count", len(results))
            span.set_attribute("rag.source_count", len(sources))
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            answer = "Contexto web opcional: " + " ".join(result.snippet for result in results[:3])
            return answer, sources


def _to_schema(source: RetrievedSource) -> Source:
    return Source(
        certificate_id=source.certificate_id,
        code=source.code,
        customer_id=source.customer_id,
        source_type=source.source_type,
        source_id=_source_id(source),
        snippet=source.snippet,
    )


def _source_id(source: RetrievedSource) -> str:
    if source.code:
        return f"{source.source_type}:{source.code}"
    return source.source_type


def _dedupe_sources(sources: list[RetrievedSource]) -> list[RetrievedSource]:
    seen: set[tuple[int | None, str, str]] = set()
    unique: list[RetrievedSource] = []
    for source in sources:
        key = (source.certificate_id, source.source_type, source.snippet)
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


def _scope_attributes(scope: AccessScope) -> dict[str, str | bool]:
    return {"rag.role": scope.role.value, "rag.has_customer_scope": scope.customer_id is not None}


def _certificate_count(sources: list[RetrievedSource]) -> int:
    return len({source.certificate_id for source in sources if source.certificate_id is not None})


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _web_sources(results: list[WebSearchResult]) -> list[RetrievedSource]:
    return [
        RetrievedSource(
            certificate_id=None,
            code=None,
            customer_id=None,
            source_type="web_search",
            path=result.url,
            snippet=f"{result.title}: {result.snippet}",
            score=0.0 if result.source_note == "fallback" else 1.0,
        )
        for result in results
    ]


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
            from app.graph_langgraph import CertilabLangGraphPipeline

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


def _build_real_pipeline(settings: Settings) -> RagPipeline:
    """Build the real-stack pipeline from MySQL + Qdrant.

    All optional dependencies (pymysql, qdrant-client, sentence-transformers,
    openai) are imported lazily inside this function so mock mode never
    requires them.
    """

    from app.ingestion.mysql_loader import MySQLLoader
    from app.ingestion.splitter import build_metadata_chunks
    from app.retrieval.qdrant_index import QdrantVectorIndex
    from app.tools.embeddings import EmbeddingProviderConfig, EmbeddingsProvider
    from app.tools.mysql_connector import MySQLCertificateConnector, MySQLConnectorConfig

    connector = MySQLCertificateConnector(MySQLConnectorConfig.from_settings(settings))
    loader = MySQLLoader(connector)
    customers, certificates, histories = loader.load()

    chunks = build_metadata_chunks(certificates)

    embedding_provider = EmbeddingsProvider(EmbeddingProviderConfig.from_settings(settings))
    index = QdrantVectorIndex.from_settings(settings, embedding_provider)
    index.upsert(chunks)

    if settings.graph_engine == "langgraph":
        try:
            from app.graph_langgraph import CertilabLangGraphPipeline

            return CertilabLangGraphPipeline(
                top_k=settings.default_top_k,
                settings=settings,
                customers=customers,
                certificates=certificates,
                histories=histories,
                chunks=chunks,
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
        chunks=chunks,
        index=index,
    )
