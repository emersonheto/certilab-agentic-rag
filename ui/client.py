"""Async wrapper over the synchronous RAG pipeline, exposing phases for progressive UI.

The ``PipelineClient`` wraps each phase method of a ``RagPipeline`` in
``asyncio.to_thread()`` so the UI event loop is not blocked during retrieval
or generation.  This enables the Chainlit handler to open and close
progressive ``cl.Step`` elements with real intermediate data.

The client transparently supports both ``CertilabRagPipeline`` (deterministic
— methods take individual params) and ``CertilabLangGraphPipeline`` (state
graph — methods take a ``CertilabGraphState`` dict).  Detection is automatic
via the presence of a ``_graph`` attribute on the pipeline object.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from app.domain.models import RetrievedSource
from app.pipeline.guards import OFF_TOPIC_ANSWER, _check_off_topic
from app.pipeline.utils import _dedupe_sources, _to_schema, _web_sources
from app.retrieval.query import extract_customer_id
from app.schemas import AskResponse, RouteName, Source
from app.security.access_control import AccessScope, Principal, scope_from_principal


class PipelineClient:
    """Async wrapper over a synchronous RAG pipeline.

    Each phase method calls the corresponding synchronous pipeline method
    via ``asyncio.to_thread()`` to keep the event loop responsive.

    Supports both deterministic (direct-param) and LangGraph (state-dict)
    pipeline variants, auto-detected at construction time.
    """

    def __init__(self, pipeline: Any, principal: Principal) -> None:
        self._pipeline = pipeline
        self._principal = principal
        self._scope = scope_from_principal(principal)
        # LangGraph pipelines expose a compiled ``_graph`` attribute;
        # deterministic pipelines do not.
        self._is_langgraph = hasattr(pipeline, "_graph")

    def _build_state(self, question: str, **extra: Any) -> dict[str, Any]:
        """Build a minimal ``CertilabGraphState`` for LangGraph pipeline calls."""
        return {
            "question": question,
            "principal": self._principal,
            "scope": self._scope,
            "route": "structured",
            "structured_answer": "",
            "semantic_answer": "",
            "web_answer": "",
            "structured_sources": [],
            "semantic_sources": [],
            "web_results": [],
            "final_answer": "",
            "sources": [],
            **extra,
        }

    # ── Phase 1 ────────────────────────────────────────────────────────────────

    async def route(self, question: str) -> RouteName:
        """Phase 1: route the question.

        Args:
            question: The user's question.

        Returns:
            The route name (e.g. ``"structured"``, ``"semantic"``, etc.).
        """
        if _check_off_topic(question):
            return "off_topic"  # type: ignore[return-value]
        if self._is_langgraph:
            state = self._build_state(question)
            result = await asyncio.to_thread(self._pipeline._route_question, state)
            return cast("RouteName", result["route"])
        result = await asyncio.to_thread(self._pipeline._route_question, question)
        return cast("RouteName", result)

    # ── Phase 2 ────────────────────────────────────────────────────────────────

    async def retrieve(
        self, question: str, route: str,
    ) -> tuple[str, list[Source]]:
        """Phase 2: retrieve sources based on the route.

        Dispatches to the appropriate pipeline retrieval method based on the
        route determined in phase 1.

        Args:
            question: The user's question.
            route: The route name from ``route()``.

        Returns:
            A tuple ``(answer_text, sources)`` where ``answer_text`` is the
            fallback answer from the retrieval phase and ``sources`` are the
            retrieved :class:`Source` objects.
        """
        if route == "off_topic":
            return "", []

        # Scope narrowing: if question mentions a specific customer and the
        # current scope is global (admin/technician), narrow temporarily so
        # all retrievers only return data for that customer.
        extracted_id = extract_customer_id(question)
        if extracted_id is not None and self._scope.is_global:
            saved_scope = self._scope
            self._scope = AccessScope(
                role=saved_scope.role,
                customer_id=extracted_id,
                user_id=saved_scope.user_id,
            )
            try:
                return await self._do_retrieve(question, route)
            finally:
                self._scope = saved_scope
        return await self._do_retrieve(question, route)

    async def _do_retrieve(
        self, question: str, route: str,
    ) -> tuple[str, list[Source]]:
        """Dispatch retrieval to the configured pipeline based on route."""
        if self._is_langgraph:
            return await self._langgraph_retrieve(question, route)
        return await self._deterministic_retrieve(question, route)

    async def _deterministic_retrieve(
        self, question: str, route: str,
    ) -> tuple[str, list[Source]]:
        """Retrieve phase for deterministic (direct-param) pipelines."""
        if route == "structured":
            answer, retrieved = await asyncio.to_thread(
                self._pipeline._retrieve_structured, question, self._scope,
            )
        elif route == "semantic":
            answer, retrieved = await asyncio.to_thread(
                self._pipeline._retrieve_semantic, question, self._scope,
            )
        elif route == "web_search":
            answer, retrieved = await asyncio.to_thread(
                self._pipeline._retrieve_web, question,
            )
        else:
            # combined (or unknown routes — fall back to combined)
            answer, retrieved = await asyncio.to_thread(
                self._pipeline._combined, question, self._scope,
            )
        return answer, [_to_schema(s) for s in retrieved]

    async def _langgraph_retrieve(
        self, question: str, route: str,
    ) -> tuple[str, list[Source]]:
        """Retrieve phase for LangGraph (state-dict) pipelines."""
        state = self._build_state(question)

        if route == "structured":
            result = await asyncio.to_thread(
                self._pipeline._retrieve_structured, state,
            )
            return result["structured_answer"], [
                _to_schema(s) for s in result["structured_sources"]
            ]

        if route == "semantic":
            result = await asyncio.to_thread(
                self._pipeline._retrieve_semantic, state,
            )
            return result["semantic_answer"], [
                _to_schema(s) for s in result["semantic_sources"]
            ]

        if route == "web_search":
            result = await asyncio.to_thread(
                self._pipeline._web_search_node, state,
            )
            sources = _web_sources(result["web_results"])
            return result["web_answer"], [_to_schema(s) for s in sources]

        # combined — run both structured and semantic, merge results
        s1 = await asyncio.to_thread(
            self._pipeline._retrieve_structured, {**state, "route": "combined"},
        )
        s2 = await asyncio.to_thread(
            self._pipeline._retrieve_semantic, {
                **state,
                "route": "combined",
                "structured_answer": s1["structured_answer"],
                "structured_sources": s1["structured_sources"],
            },
        )
        combined_answer = f"{s1['structured_answer']} {s2['semantic_answer']}"
        combined_sources = [*s1["structured_sources"], *s2["semantic_sources"]]
        return combined_answer, [_to_schema(s) for s in combined_sources]

    # ── Phase 3 ────────────────────────────────────────────────────────────────

    async def generate(
        self, question: str, sources: list[Source], fallback_answer: str,
    ) -> str:
        """Phase 3: generate the final answer.

        Deduplicates sources internally before calling the pipeline's
        ``_generate_answer`` method.

        Args:
            question: The user's question.
            sources: Sources from ``retrieve()`` (will be deduplicated).
            fallback_answer: The fallback text to use if no answer generator
                is configured.

        Returns:
            The generated answer string.
        """
        # Convert Source → RetrievedSource for the pipeline method
        retrieved = [
            RetrievedSource(
                certificate_id=s.certificate_id,
                code=s.code,
                customer_id=s.customer_id,
                source_type=s.source_type,
                path=s.source_id,
                snippet=s.snippet,
            )
            for s in sources
        ]
        unique = _dedupe_sources(retrieved)

        if self._is_langgraph:
            state = self._build_state(
                question,
                structured_sources=unique,
                structured_answer=fallback_answer,
            )
            result = await asyncio.to_thread(
                self._pipeline._generate_answer, state,
            )
            return cast("str", result["final_answer"])

        return await asyncio.to_thread(
            self._pipeline._generate_answer, question, unique, fallback_answer,
        )

    # ── Full convenience ────────────────────────────────────────────────────────

    async def ask(self, question: str) -> AskResponse:
        """Convenience: run all three phases and return a complete response.

        This is the main entry point for non-progressive UIs.  For progressive
        ``cl.Step`` integration, call ``route()``, ``retrieve()``, and
        ``generate()`` individually.
        """
        route_name = await self.route(question)
        if route_name == "off_topic":
            return AskResponse(route="off_topic", answer=OFF_TOPIC_ANSWER, sources=[])
        fallback_answer, sources = await self.retrieve(question, route_name)
        answer = await self.generate(question, sources, fallback_answer)
        return AskResponse(answer=answer, route=route_name, sources=sources)
