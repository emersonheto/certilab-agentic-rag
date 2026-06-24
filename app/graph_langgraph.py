from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.domain.models import Certificate, CertificateHistory, Customer, DocumentChunk, RetrievedSource
from app.graph import _certificate_count, _dedupe_sources, _elapsed_ms, _scope_attributes, _to_schema, _web_sources
from app.ingestion.loader import load_certificates, load_customers, load_histories, load_pdf_texts
from app.ingestion.splitter import build_pdf_chunks
from app.observability import trace_span
from app.retrieval.constants import DEFAULT_RETRIEVAL_TOP_K
from app.retrieval.grader import route_question
from app.retrieval.protocols import VectorIndex
from app.retrieval.structured import StructuredRetriever
from app.retrieval.semantic import SemanticRetriever
from app.schemas import AskResponse
from app.security.access_control import AccessScope, Principal, scope_from_principal
from app.tools.openai_client import AnswerGenerator
from app.tools.web_search import TavilyWebSearch, WebSearchConfig, WebSearchResult

RouteName = Literal["structured", "semantic", "combined", "web_search"]

class CertilabGraphState(TypedDict):
    question: str
    principal: Principal
    scope: AccessScope
    route: RouteName
    structured_answer: str
    semantic_answer: str
    web_answer: str
    structured_sources: list[RetrievedSource]
    semantic_sources: list[RetrievedSource]
    web_results: list[WebSearchResult]
    final_answer: str
    sources: list[RetrievedSource]


class CompiledGraph(Protocol):
    def invoke(self, input: CertilabGraphState) -> CertilabGraphState:
        """Run the graph with a Certilab state payload."""


class CertilabLangGraphPipeline:
    """Course-style LangGraph StateGraph pipeline for Certilab RAG."""

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
        self._settings = settings or Settings()
        if customers is None or certificates is None or histories is None:
            assert data_dir is not None
            certificates = load_certificates(data_dir)
            customers = load_customers(data_dir)
            histories = load_histories(data_dir)
        if chunks is None:
            assert data_dir is not None
            chunks = build_pdf_chunks(certificates, load_pdf_texts(data_dir))
        self._top_k = top_k
        self._structured = StructuredRetriever(customers, certificates, histories)
        self._semantic = SemanticRetriever(chunks, index=index)
        self._web_search = TavilyWebSearch(WebSearchConfig(tavily_api_key=self._settings.tavily_api_key))
        self._answer_generator = AnswerGenerator(self._settings)
        self._graph = self._build_graph()

    def ask(self, question: str, principal: Principal) -> AskResponse:
        started_at = perf_counter()
        with trace_span(
            "rag.langgraph.ask",
            {
                "rag.role": principal.role.value,
                "rag.has_customer_scope": principal.customer_id is not None,
                "rag.question_length": len(question),
            },
        ) as span:
            final_state = self._graph.invoke(_initial_state(question, principal))
            sources = _dedupe_sources(final_state["sources"])
            route = final_state["route"]
            span.set_attribute("rag.engine", "langgraph")
            span.set_attribute("rag.route", route)
            span.set_attribute("rag.source_count", len(sources))
            span.set_attribute("rag.certificate_count", _certificate_count(sources))
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            return AskResponse(answer=final_state["final_answer"], route=route, sources=[_to_schema(source) for source in sources])

    def _build_graph(self) -> CompiledGraph:
        graph = StateGraph(CertilabGraphState)
        graph.add_node("route_question", self._route_question)
        graph.add_node("retrieve_structured", self._retrieve_structured)
        graph.add_node("retrieve_semantic", self._retrieve_semantic)
        graph.add_node("web_search", self._web_search_node)
        graph.add_node("generate_answer", self._generate_answer)

        graph.add_edge(START, "route_question")
        graph.add_conditional_edges(
            "route_question",
            _route_after_classification,
            {
                "structured": "retrieve_structured",
                "semantic": "retrieve_semantic",
                "combined": "retrieve_structured",
                "web_search": "web_search",
            },
        )
        graph.add_conditional_edges(
            "retrieve_structured",
            _route_after_structured,
            {"combined": "retrieve_semantic", "generate_answer": "generate_answer"},
        )
        graph.add_edge("retrieve_semantic", "generate_answer")
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", END)
        return cast(CompiledGraph, graph.compile())

    def _route_question(self, state: CertilabGraphState) -> CertilabGraphState:
        started_at = perf_counter()
        with trace_span("rag.langgraph.route_question", {"rag.question_length": len(state["question"])}) as span:
            route = route_question(state["question"])
            span.set_attribute("rag.route", route)
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            return {**state, "route": route}

    def _retrieve_structured(self, state: CertilabGraphState) -> CertilabGraphState:
        started_at = perf_counter()
        with trace_span("rag.langgraph.retrieve_structured", _scope_attributes(state["scope"])) as span:
            answer, sources = self._structured.retrieve(state["question"], state["scope"])
            span.set_attribute("rag.source_count", len(sources))
            span.set_attribute("rag.certificate_count", _certificate_count(sources))
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            return {**state, "structured_answer": answer, "structured_sources": sources}

    def _retrieve_semantic(self, state: CertilabGraphState) -> CertilabGraphState:
        started_at = perf_counter()
        with trace_span("rag.langgraph.retrieve_semantic", _scope_attributes(state["scope"])) as span:
            answer, sources = self._semantic.retrieve(state["question"], state["scope"], self._top_k)
            span.set_attribute("rag.source_count", len(sources))
            span.set_attribute("rag.certificate_count", _certificate_count(sources))
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            return {**state, "semantic_answer": answer, "semantic_sources": sources}

    def _web_search_node(self, state: CertilabGraphState) -> CertilabGraphState:
        started_at = perf_counter()
        with trace_span("rag.langgraph.web_search", {"rag.question_length": len(state["question"])}) as span:
            results = self._web_search.search(state["question"])
            span.set_attribute("rag.web_result_count", len(results))
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            answer = "Contexto web opcional: " + " ".join(result.snippet for result in results[:3])
            return {**state, "web_answer": answer, "web_results": results}

    def _generate_answer(self, state: CertilabGraphState) -> CertilabGraphState:
        started_at = perf_counter()
        sources = [*state["structured_sources"], *state["semantic_sources"], *_web_sources(state["web_results"])]
        fallback_answer = _fallback_answer(state)
        with trace_span(
            "rag.langgraph.generate_answer",
            {"rag.question_length": len(state["question"]), "rag.source_count": len(sources)},
        ) as span:
            answer = self._answer_generator.generate(state["question"], sources, fallback_answer)
            span.set_attribute("rag.used_fallback", answer == fallback_answer)
            span.set_attribute("rag.duration_ms", _elapsed_ms(started_at))
            return {**state, "sources": sources, "final_answer": answer}


def _initial_state(question: str, principal: Principal) -> CertilabGraphState:
    return {
        "question": question,
        "principal": principal,
        "scope": scope_from_principal(principal),
        "route": "structured",
        "structured_answer": "",
        "semantic_answer": "",
        "web_answer": "",
        "structured_sources": [],
        "semantic_sources": [],
        "web_results": [],
        "final_answer": "",
        "sources": [],
    }


def _route_after_classification(state: CertilabGraphState) -> RouteName:
    return state["route"]


def _route_after_structured(state: CertilabGraphState) -> Literal["combined", "generate_answer"]:
    return "combined" if state["route"] == "combined" else "generate_answer"


def _fallback_answer(state: CertilabGraphState) -> str:
    if state["route"] == "combined":
        return f"{state['structured_answer']} {state['semantic_answer']}"
    if state["route"] == "semantic":
        return state["semantic_answer"]
    if state["route"] == "web_search":
        return state["web_answer"]
    return state["structured_answer"]
