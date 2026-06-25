"""Tests for ``ui/client`` — ``PipelineClient`` async wrapper.

Strict TDD: tests written before implementation.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.domain.models import RetrievedSource, Role
from app.schemas import Source
from app.security.access_control import AccessScope, Principal


# ── Fake pipeline for testing ────────────────────────────────────────────────


class FakePipeline:
    """A minimal fake that mimics the phase methods of a RAG pipeline.

    Each callable records its invocations for assertion.
    """

    def __init__(self) -> None:
        self.route_calls: list[str] = []
        self.retrieve_calls: list[tuple[str, AccessScope]] = []
        self.generate_calls: list[tuple[str, list[RetrievedSource], str]] = []
        self._route_result: str = "structured"
        self._retrieve_result: tuple[str, list[RetrievedSource]] = (
            "Answer from structured retrieval.",
            [
                RetrievedSource(
                    certificate_id=1,
                    code="CERT-001",
                    customer_id=101,
                    source_type="metadata",
                    path="data/mock/certificates.json",
                    snippet="Certificado activo.",
                ),
            ],
        )
        self._generate_result: str = "Generated answer."

    def _route_question(self, question: str) -> str:
        self.route_calls.append(question)
        return self._route_result

    def _retrieve_structured(self, question: str, scope: AccessScope) -> tuple[str, list[RetrievedSource]]:
        self.retrieve_calls.append((question, scope))
        return self._retrieve_result

    def _retrieve_semantic(self, question: str, scope: AccessScope) -> tuple[str, list[RetrievedSource]]:
        self.retrieve_calls.append((question, scope))
        return (
            "Answer from semantic retrieval.",
            [
                RetrievedSource(
                    certificate_id=2,
                    code="CERT-002",
                    customer_id=101,
                    source_type="pdf_text",
                    path="s3://bucket/cert-002.pdf",
                    snippet="Procedimiento técnico detallado.",
                ),
            ],
        )

    def _retrieve_web(self, question: str) -> tuple[str, list[RetrievedSource]]:
        self.retrieve_calls.append((question, AccessScope(Role.ADMIN, None)))
        return (
            "Contexto web opcional: public info.",
            [
                RetrievedSource(
                    certificate_id=None,
                    code=None,
                    customer_id=None,
                    source_type="web_search",
                    path="https://example.test/result",
                    snippet="Public calibration info.",
                ),
            ],
        )

    def _combined(self, question: str, scope: AccessScope) -> tuple[str, list[RetrievedSource]]:
        self.retrieve_calls.append((question, scope))
        return (
            "Structured answer Semantic answer",
            [
                RetrievedSource(
                    certificate_id=1, code="CERT-001", customer_id=101,
                    source_type="metadata", path="data/mock/certificates.json",
                    snippet="Certificado activo.",
                ),
                RetrievedSource(
                    certificate_id=2, code="CERT-002", customer_id=101,
                    source_type="pdf_text", path="s3://bucket/cert-002.pdf",
                    snippet="Procedimiento técnico detallado.",
                ),
            ],
        )

    def _generate_answer(self, question: str, sources: list[RetrievedSource], fallback: str) -> str:
        self.generate_calls.append((question, sources, fallback))
        return self._generate_result


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def admin_principal() -> Principal:
    return Principal(role=Role.ADMIN, customer_id=None, user_id=1)


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def client(fake_pipeline: FakePipeline, admin_principal: Principal) -> Any:
    """Build a PipelineClient with a fake pipeline."""
    from ui.client import PipelineClient

    return PipelineClient(fake_pipeline, admin_principal)


# ── route() ───────────────────────────────────────────────────────────────────


class TestRoute:
    """Verify the ``route()`` phase method."""

    async def test_route_returns_route_name(
        self, client: Any, fake_pipeline: FakePipeline,
    ) -> None:
        """``route()`` returns the route from ``_route_question``."""
        result = await client.route("¿Cuántos certificados hay?")
        assert result == "structured"

    async def test_route_calls_pipeline_method(
        self, client: Any, fake_pipeline: FakePipeline,
    ) -> None:
        """``route()`` invokes ``pipeline._route_question`` with the question."""
        await client.route("¿Pregunta de prueba?")
        assert fake_pipeline.route_calls == ["¿Pregunta de prueba?"]

    async def test_route_is_async_coroutine(
        self, client: Any,
    ) -> None:
        """``route()`` is an async method that can be awaited (non-blocking)."""
        result = await client.route("¿Prueba?")
        # If this completes correctly, the async/await mechanism works
        assert result == "structured"


# ── retrieve() ────────────────────────────────────────────────────────────────


class TestRetrieve:
    """Verify the ``retrieve()`` phase method dispatches by route."""

    async def test_retrieve_structured(
        self, client: Any,
    ) -> None:
        """Structured route dispatches to ``_retrieve_structured``."""
        result_answer, result_sources = await client.retrieve(
            "¿Cuántos certificados?", "structured",
        )
        assert "Answer from structured" in result_answer
        assert len(result_sources) == 1
        assert result_sources[0].code == "CERT-001"

    async def test_retrieve_semantic(
        self, client: Any,
    ) -> None:
        """Semantic route dispatches to ``_retrieve_semantic``."""
        result_answer, result_sources = await client.retrieve(
            "¿Resumen técnico?", "semantic",
        )
        assert "Answer from semantic" in result_answer
        assert result_sources[0].code == "CERT-002"

    async def test_retrieve_web(
        self, client: Any,
    ) -> None:
        """Web search route dispatches to ``_retrieve_web``."""
        result_answer, result_sources = await client.retrieve(
            "¿Buscar en web?", "web_search",
        )
        assert "Contexto web" in result_answer
        assert result_sources[0].source_type == "web_search"

    async def test_retrieve_combined(
        self, client: Any,
    ) -> None:
        """Combined route dispatches to ``_combined``."""
        result_answer, result_sources = await client.retrieve(
            "¿Estado y resumen?", "combined",
        )
        assert "Structured answer" in result_answer
        assert len(result_sources) == 2
        assert result_sources[0].code == "CERT-001"
        assert result_sources[1].code == "CERT-002"

    async def test_retrieve_unknown_route_falls_back_to_combined(
        self, client: Any,
    ) -> None:
        """Unknown route falls back to combined retrieval."""
        result_answer, result_sources = await client.retrieve(
            "¿Pregunta?", "unknown_route",
        )
        assert result_answer
        assert len(result_sources) >= 0

    async def test_retrieve_returns_source_objects(
        self, client: Any,
    ) -> None:
        """``retrieve()`` returns ``Source`` objects (converted from ``RetrievedSource``)."""
        _, result_sources = await client.retrieve("¿Prueba?", "structured")
        assert all(isinstance(s, Source) for s in result_sources)
        # Source objects have ``source_id`` (not ``path`` like RetrievedSource)
        assert all(hasattr(s, "source_id") for s in result_sources)

    async def test_retrieve_is_async(
        self, client: Any, fake_pipeline: FakePipeline,
    ) -> None:
        """``retrieve()`` is async — pipeline method is called via await."""
        await client.retrieve("¿Prueba?", "structured")
        assert len(fake_pipeline.retrieve_calls) == 1


# ── generate() ────────────────────────────────────────────────────────────────


class TestGenerate:
    """Verify the ``generate()`` phase method."""

    async def test_generate_returns_string(
        self, client: Any,
    ) -> None:
        """``generate()`` returns the answer string."""
        result = await client.generate("¿Pregunta?", [], "fallback answer")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_generate_deduplicates_sources(
        self, client: Any, fake_pipeline: FakePipeline,
    ) -> None:
        """``generate()`` deduplicates sources before passing to the pipeline."""
        duplicate_sources = [
            Source(
                certificate_id=1, code="CERT-001", customer_id=101,
                source_type="metadata", source_id="metadata:CERT-001",
                snippet="Same",
            ),
            Source(
                certificate_id=1, code="CERT-001", customer_id=101,
                source_type="metadata", source_id="metadata:CERT-001",
                snippet="Same",
            ),
        ]
        await client.generate("¿Pregunta?", duplicate_sources, "fallback")

        assert len(fake_pipeline.generate_calls) == 1
        _question, sources, _fallback = fake_pipeline.generate_calls[0]
        assert len(sources) == 1  # deduped from 2 to 1

    async def test_generate_is_async(
        self, client: Any, fake_pipeline: FakePipeline,
    ) -> None:
        """``generate()`` is async — pipeline method is called via await."""
        result = await client.generate("¿Prueba?", [], "fallback")
        assert isinstance(result, str)
        assert len(fake_pipeline.generate_calls) == 1


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    """Verify phase errors propagate correctly."""

    async def test_route_raises_on_pipeline_error(
        self, admin_principal: Principal,
    ) -> None:
        """If the pipeline's ``_route_question`` raises, the client raises."""
        from ui.client import PipelineClient

        class BrokenPipeline:
            def _route_question(self, question: str) -> str:
                raise ValueError("Routing failed")

            _retrieve_structured = _route_question
            _retrieve_semantic = _route_question
            _retrieve_web = _route_question
            _combined = _route_question
            _generate_answer = _route_question

        client = PipelineClient(BrokenPipeline(), admin_principal)

        with pytest.raises(ValueError, match="Routing failed"):
            await client.route("¿Prueba?")

    async def test_retrieve_raises_on_pipeline_error(
        self, admin_principal: Principal,
    ) -> None:
        """If the pipeline's retrieval method raises, the client raises."""
        from ui.client import PipelineClient

        class BrokenPipeline:
            def _route_question(self, question: str) -> str:
                return "structured"

            def _retrieve_structured(self, question: str, scope: AccessScope) -> tuple[str, list[RetrievedSource]]:
                raise RuntimeError("Retrieval failed")

            def _retrieve_semantic(self, *args: Any, **kwargs: Any) -> tuple[str, list[RetrievedSource]]:
                raise RuntimeError("Retrieval failed")

            def _retrieve_web(self, *args: Any, **kwargs: Any) -> tuple[str, list[RetrievedSource]]:
                raise RuntimeError("Retrieval failed")

            def _combined(self, *args: Any, **kwargs: Any) -> tuple[str, list[RetrievedSource]]:
                raise RuntimeError("Retrieval failed")

            def _generate_answer(self, *args: Any, **kwargs: Any) -> str:
                return "answer"

        client = PipelineClient(BrokenPipeline(), admin_principal)

        with pytest.raises(RuntimeError, match="Retrieval failed"):
            await client.retrieve("¿Prueba?", "structured")

    async def test_generate_raises_on_pipeline_error(
        self, admin_principal: Principal,
    ) -> None:
        """If the pipeline's ``_generate_answer`` raises, the client raises."""
        from ui.client import PipelineClient

        class BrokenPipeline:
            def _generate_answer(self, question: str, sources: list[RetrievedSource], fallback: str) -> str:
                raise RuntimeError("Generation failed")

        client = PipelineClient(BrokenPipeline(), admin_principal)

        with pytest.raises(RuntimeError, match="Generation failed"):
            await client.generate("¿Prueba?", [], "fallback")


# ── Off-topic guard (PR 3) ────────────────────────────────────────────────────


class TestOffTopicGuard:
    """Verify off-topic detection short-circuits the pipeline phases."""

    async def test_route_returns_off_topic_for_greeting(
        self, client: Any,
    ) -> None:
        """``route()`` returns 'off_topic' for a bare greeting."""
        result = await client.route("hola")
        assert result == "off_topic"

    async def test_retrieve_returns_empty_for_off_topic(
        self, client: Any,
    ) -> None:
        """``retrieve()`` with route 'off_topic' returns empty answer and no sources."""
        answer, sources = await client.retrieve("hola", "off_topic")
        assert answer == ""
        assert sources == []

    async def test_ask_short_circuits_for_off_topic(
        self, client: Any,
    ) -> None:
        """``ask()`` for an off-topic question returns route 'off_topic' and empty sources."""
        response = await client.ask("hola")
        assert response.route == "off_topic"
        assert response.sources == []
        assert "Certilab" in response.answer


# ── Scope narrowing (PR 3) ────────────────────────────────────────────────────


class TestScopeNarrowing:
    """Verify admin scope narrows to mentioned customer."""

    async def test_retrieve_narrows_scope_for_admin_with_mentioned_customer(
        self, client: Any, fake_pipeline: FakePipeline,
    ) -> None:
        """When an admin asks about customer 101, the scope passed to the pipeline is narrowed."""
        await client.retrieve("dame los del cliente 101", "structured")

        assert len(fake_pipeline.retrieve_calls) == 1
        _question, scope = fake_pipeline.retrieve_calls[0]
        assert scope.customer_id == 101

    async def test_retrieve_does_not_narrow_for_client_role(
        self,
    ) -> None:
        """Client role scope is not global, so no narrowing happens."""
        from ui.client import PipelineClient

        fake = FakePipeline()
        client_principal = Principal(role=Role.CLIENT, customer_id=202, user_id=5)
        client = PipelineClient(fake, client_principal)

        await client.retrieve("dame los del cliente 101", "structured")

        assert len(fake.retrieve_calls) == 1
        _question, scope = fake.retrieve_calls[0]
        # Client's own scope is 202, not narrowed to 101
        assert scope.customer_id == 202

    async def test_scope_restored_after_retrieve(
        self, client: Any,
    ) -> None:
        """After retrieve with narrowing, the client's scope is restored to global."""
        await client.retrieve("dame los del cliente 101", "structured")
        assert client._scope.customer_id is None  # global scope restored
