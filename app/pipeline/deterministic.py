"""Deterministic RAG pipeline implementation.

Moved from ``app/graph.py`` (PR 4 structural refactor). This module contains the
``CertilabRagPipeline`` class with deterministic local behavior.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from app.config import Settings
from app.domain.models import (
    Certificate,
    CertificateHistory,
    Customer,
    DocumentChunk,
    RetrievedSource,
)
from app.ingestion.loader import (
    load_certificates,
    load_customers,
    load_histories,
    load_pdf_texts,
)
from app.ingestion.splitter import build_pdf_chunks
from app.logging import get_logger
from app.observability import trace_span
from app.pipeline.guards import OFF_TOPIC_ANSWER, _check_off_topic
from app.pipeline.utils import (
    _certificate_count,
    _dedupe_sources,
    _elapsed_ms,
    _scope_attributes,
    _to_schema,
    _web_sources,
)
from app.retrieval.constants import DEFAULT_RETRIEVAL_TOP_K
from app.retrieval.grader import route_question
from app.retrieval.protocols import VectorIndex
from app.retrieval.query import extract_customer_id
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.structured import StructuredRetriever
from app.schemas import AskResponse, RouteName
from app.security.access_control import (
    AccessScope,
    Principal,
    scope_from_principal,
)
from app.tools.openai_client import AnswerGenerator
from app.tools.web_search import TavilyWebSearch, WebSearchConfig

logger = get_logger("pipeline.deterministic")


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
        self._settings = settings
        self._top_k = top_k
        self._structured = StructuredRetriever(customers, certificates, histories)
        self._semantic = SemanticRetriever(chunks, index=index)
        self._web_search = TavilyWebSearch(
            WebSearchConfig(
                tavily_api_key=settings.tavily_api_key if settings else None,
                max_results=settings.web_search_max_results if settings else 3,
            )
        )
        self._answer_generator = AnswerGenerator(settings) if settings is not None else None

    def reindex(self) -> dict:
        """Reload data from MySQL, rebuild chunks, and re-upsert into the vector index.

        Uses the idempotent upsert so only new/changed chunks are embedded.
        Mock mode returns 0/0 without reloading.
        """

        if not self._settings or self._settings.app_mode != "real":
            return {"indexed": 0, "skipped": 0}
        from app.pipeline.factory import _reindex_pipeline

        return _reindex_pipeline(self._semantic._index, self._settings)

    def ask(self, question: str, principal: Principal) -> AskResponse:
        scope = scope_from_principal(principal)

        # Early exit for off-topic questions (greetings, math, off-domain)
        if _check_off_topic(question):
            logger.info("pipeline.off_topic")
            return AskResponse(route="off_topic", answer=OFF_TOPIC_ANSWER, sources=[])

        # Scope narrowing: if question mentions a specific customer, narrow the
        # access scope so all retrievers only return data for that customer.
        # RBAC guard: a CLIENT user cannot escalate to a different customer.
        mentioned_customer = extract_customer_id(question)
        if mentioned_customer is not None:
            if scope.is_global:
                scope = AccessScope(
                    role=scope.role,
                    customer_id=mentioned_customer,
                    user_id=scope.user_id,
                )
            # If not global (CLIENT role), ignore extracted customer_id unless
            # it matches the client's own scope (effectively a no-op).

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
            logger.info("pipeline.route", route=route)
            logger.info("pipeline.retrieve.start", route=route)
            retrieve_started = perf_counter()
            if route == "structured":
                answer, sources = self._retrieve_structured(question, scope)
            elif route == "semantic":
                answer, sources = self._retrieve_semantic(question, scope)
            elif route == "web_search":
                answer, sources = self._retrieve_web(question)
            else:
                answer, sources = self._combined(question, scope)
            logger.info("pipeline.retrieve.complete", source_count=len(sources), duration_ms=int((perf_counter() - retrieve_started) * 1000))

            unique_sources = _dedupe_sources(sources)
            logger.info("pipeline.generate.start")
            generate_started = perf_counter()
            answer = self._generate_answer(question, unique_sources, answer, scope)
            logger.info("pipeline.generate.complete", used_openai=self._answer_generator is not None, duration_ms=int((perf_counter() - generate_started) * 1000))
            span.set_attribute("rag.route", route)
            span.set_attribute("rag.source_count", len(unique_sources))
            span.set_attribute("rag.certificate_count", _certificate_count(unique_sources))
            span.set_attribute("rag.duration_ms", _elapsed_ms(pipeline_started_at))
            return AskResponse(answer=answer, route=route, sources=[_to_schema(source) for source in unique_sources])

    def _generate_answer(
        self,
        question: str,
        sources: list[RetrievedSource],
        fallback_answer: str,
        scope: AccessScope | None = None,
    ) -> str:
        started_at = perf_counter()
        with trace_span("rag.generate_answer", {"rag.question_length": len(question), "rag.source_count": len(sources)}) as span:
            scope_facts: str | None = None
            if scope is not None:
                customer_part = f"customer_id={scope.customer_id}" if scope.customer_id is not None else "all customers"
                scope_facts = f"{customer_part}, {len(sources)} sources retrieved"
            answer = (
                self._answer_generator.generate(question, sources, fallback_answer, scope_facts=scope_facts)
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
