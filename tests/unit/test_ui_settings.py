"""Tests for ``ui/settings`` — profile resolution, format_answer, and module integrity.

Strict TDD: tests written before implementation.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.domain.models import Role
from app.security.access_control import Principal


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        # Provide all demo tokens so _principal_mock does not raise
        demo_admin_token="admin-token",
        demo_technician_token="technician-token",
        demo_client_101_token="client-101-token",
        demo_client_202_token="client-202-token",
        **overrides,
    )


# ── _PROFILES integrity ──────────────────────────────────────────────────────


class TestProfiles:
    """Verify ``_PROFILES`` constant after importing ``ui.settings``."""

    # ── RED step: reference module that does not exist yet ────────────────

    def test_module_imports_without_chainlit(self) -> None:
        """Import ``ui.settings`` succeeds when ``chainlit`` is not installed.

        The settings module must not depend on chainlit at import time.
        """
        # Purge any cached ui.settings to force a fresh import
        sys.modules.pop("ui.settings", None)

        # Ensure chainlit is NOT available (simulate missing optional dep)
        monkeypatch: Any = pytest.MonkeyPatch()
        monkeypatch.setitem(sys.modules, "chainlit", None)  # type: ignore[arg-type]
        try:
            mod = importlib.import_module("ui.settings")
        finally:
            monkeypatch.undo()
        # Clean up so other tests reimport cleanly
        sys.modules.pop("ui.settings", None)

        assert hasattr(mod, "_PROFILES")

    def test_profile_ids_are_unique(self) -> None:
        """Every profile has a unique ``id``."""
        from ui.settings import _PROFILES

        ids = [p["id"] for p in _PROFILES]
        assert len(ids) == len(set(ids)), f"Duplicate profile ids: {ids}"

    def test_expected_admin_profile(self) -> None:
        """Admin profile has expected fields."""
        from ui.settings import _PROFILES

        admin = next(p for p in _PROFILES if p["id"] == "admin")
        assert admin["name"] == "Admin"
        assert admin["icon"] == "🛡️"

    def test_expected_technician_profile(self) -> None:
        """Technician profile has expected fields."""
        from ui.settings import _PROFILES

        tech = next(p for p in _PROFILES if p["id"] == "technician")
        assert tech["name"] == "Técnico"
        assert tech["icon"] == "🔧"

    def test_expected_client_101_profile(self) -> None:
        """Client 101 profile has expected fields."""
        from ui.settings import _PROFILES

        c = next(p for p in _PROFILES if p["id"] == "client_101")
        assert c["name"] == "Cliente 101"
        assert c["icon"] == "🏢"

    def test_expected_client_202_profile(self) -> None:
        """Client 202 profile has expected fields."""
        from ui.settings import _PROFILES

        c = next(p for p in _PROFILES if p["id"] == "client_202")
        assert c["name"] == "Cliente 202"
        assert c["icon"] == "🏭"

    def test_all_profiles_have_required_keys(self) -> None:
        """Every profile dict contains ``id``, ``name``, ``markdown_description``, ``icon``."""
        from ui.settings import _PROFILES

        required = {"id", "name", "markdown_description", "icon"}
        for profile in _PROFILES:
            assert required.issubset(profile.keys()), f"Profile {profile['id']} missing keys"


# ── _build_principal / _principal_mock ────────────────────────────────────────


class TestPrincipalBuilders:
    """Verify ``_build_principal`` and ``_principal_mock`` resolve tokens correctly."""

    def test_principal_mock_returns_admin(self) -> None:
        """``_principal_mock('admin', settings)`` returns ADMIN Principal."""
        from ui.settings import _principal_mock

        settings = _make_settings()
        principal = _principal_mock("admin", settings)

        assert principal.role == Role.ADMIN
        assert principal.customer_id is None
        assert principal.user_id == 1

    def test_principal_mock_returns_technician(self) -> None:
        """``_principal_mock('technician', settings)`` returns TECHNICIAN Principal."""
        from ui.settings import _principal_mock

        settings = _make_settings()
        principal = _principal_mock("technician", settings)

        assert principal.role == Role.TECHNICIAN
        assert principal.customer_id is None

    def test_principal_mock_returns_client_101(self) -> None:
        """``_principal_mock('client_101', settings)`` returns CLIENT Principal scoped to 101."""
        from ui.settings import _principal_mock

        settings = _make_settings()
        principal = _principal_mock("client_101", settings)

        assert principal.role == Role.CLIENT
        assert principal.customer_id == 101
        assert principal.user_id == 1010

    def test_principal_mock_returns_client_202(self) -> None:
        """``_principal_mock('client_202', settings)`` returns CLIENT Principal scoped to 202."""
        from ui.settings import _principal_mock

        settings = _make_settings()
        principal = _principal_mock("client_202", settings)

        assert principal.role == Role.CLIENT
        assert principal.customer_id == 202
        assert principal.user_id == 2020

    def test_principal_mock_raises_on_missing_token(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``_principal_mock`` raises ``RuntimeError`` when demo token is not configured."""
        from ui.settings import _principal_mock

        # The conftest sets env vars for all demo tokens; clear them so
        # pydantic-settings does not inject them as kwargs to Settings().
        monkeypatch.delenv("DEMO_ADMIN_TOKEN", raising=False)
        monkeypatch.delenv("DEMO_TECHNICIAN_TOKEN", raising=False)
        monkeypatch.delenv("DEMO_CLIENT_101_TOKEN", raising=False)
        monkeypatch.delenv("DEMO_CLIENT_202_TOKEN", raising=False)

        # Build settings without any demo token env — all default to None.
        settings = Settings(_env_file=None)

        with pytest.raises(RuntimeError, match="Demo token for profile"):
            _principal_mock("admin", settings)

    def test_build_principal_mock_mode(self) -> None:
        """``_build_principal`` delegates to ``_principal_mock`` when ``app_mode != 'real'``."""
        from ui.settings import _build_principal

        settings = _make_settings(app_mode="mock")
        principal = _build_principal("admin", settings)

        assert principal.role == Role.ADMIN


# ── NODE_DISPLAY_NAMES ────────────────────────────────────────────────────────


class TestNodeDisplayNames:
    """Verify the step-display-name constants."""

    def test_has_route_key(self) -> None:
        """``NODE_DISPLAY_NAMES`` contains a ``route`` key."""
        from ui.settings import NODE_DISPLAY_NAMES

        assert "route" in NODE_DISPLAY_NAMES

    def test_has_retrieve_key(self) -> None:
        """``NODE_DISPLAY_NAMES`` contains a ``retrieve`` key."""
        from ui.settings import NODE_DISPLAY_NAMES

        assert "retrieve" in NODE_DISPLAY_NAMES

    def test_has_generate_key(self) -> None:
        """``NODE_DISPLAY_NAMES`` contains a ``generate`` key."""
        from ui.settings import NODE_DISPLAY_NAMES

        assert "generate" in NODE_DISPLAY_NAMES

    def test_entries_are_non_empty_strings(self) -> None:
        """Every value in ``NODE_DISPLAY_NAMES`` is a non-empty string."""
        from ui.settings import NODE_DISPLAY_NAMES

        for key, value in NODE_DISPLAY_NAMES.items():
            assert isinstance(value, str) and len(value) > 0, f"Entry {key!r} is empty"


# ── format_answer ─────────────────────────────────────────────────────────────


class TestFormatAnswer:
    """Verify ``format_answer()`` builds the final markdown message."""

    def _make_response(
        self,
        answer: str = "Respuesta de prueba.",
        route: str = "structured",
        sources: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Build a minimal AskResponse-like object."""
        from app.schemas import AskResponse, Source

        return AskResponse(
            answer=answer,
            route=route,
            sources=[Source(**s) for s in (sources or [])],
        )

    def test_format_answer_includes_answer(self) -> None:
        """The answer text appears in the formatted output."""
        from ui.settings import format_answer

        result = format_answer(self._make_response(answer="Hola mundo"))
        assert "Hola mundo" in result

    def test_format_answer_structured_route_has_indicator(self) -> None:
        """Structured route output includes route label."""
        from ui.settings import format_answer

        result = format_answer(self._make_response(route="structured"))
        assert "estructurada" in result

    def test_format_answer_semantic_route_has_indicator(self) -> None:
        """Semantic route output includes route label."""
        from ui.settings import format_answer

        result = format_answer(self._make_response(route="semantic"))
        assert "semántica" in result.lower() or "semantica" in result.lower()

    def test_format_answer_web_search_route_has_indicator(self) -> None:
        """Web search route output includes route label."""
        from ui.settings import format_answer

        result = format_answer(self._make_response(route="web_search"))
        assert "web" in result.lower()

    def test_format_answer_combined_route_has_indicator(self) -> None:
        """Combined route output includes route label."""
        from ui.settings import format_answer

        result = format_answer(self._make_response(route="combined"))
        assert "combinada" in result.lower()

    def test_format_answer_with_sources_includes_source_list(self) -> None:
        """When sources are present, a sources heading and items appear."""
        from ui.settings import format_answer

        result = format_answer(
            self._make_response(
                sources=[
                    {"certificate_id": 1, "code": "CERT-001", "customer_id": 101,
                     "source_type": "metadata", "source_id": "metadata:CERT-001",
                     "snippet": "Detalle"},
                ]
            )
        )
        assert "CERT-001" in result
        assert "cliente 101" in result
        assert "Fuentes" in result

    def test_format_answer_without_sources_omits_source_section(self) -> None:
        """When no sources, no sources section appears."""
        from ui.settings import format_answer

        result = format_answer(self._make_response(sources=[]))
        assert "Fuentes" not in result

    # ── Source-type grouping (WARNING-1) ──────────────────────────────────

    def test_format_answer_mixed_types_shows_group_headers(self) -> None:
        """Mixed source types show per-type group headers."""
        from ui.settings import format_answer

        result = format_answer(
            self._make_response(
                route="combined",
                sources=[
                    {
                        "certificate_id": 1, "code": "CERT-001", "customer_id": 101,
                        "source_type": "structured", "source_id": "str:1",
                        "snippet": "Dato estructurado.",
                    },
                    {
                        "certificate_id": 2, "code": "CERT-002", "customer_id": 102,
                        "source_type": "pdf_text", "source_id": "s3://bucket/cert-002.pdf",
                        "snippet": "Texto extraído.",
                    },
                ],
            ),
        )
        assert "📋 **Datos estructurados**" in result
        assert "📄 **Documentos PDF**" in result
        assert "CERT-001" in result
        assert "CERT-002" in result

    def test_format_answer_only_structured_no_pdf_header(self) -> None:
        """Single type (structured) shows flat list, no pdf header."""
        from ui.settings import format_answer

        result = format_answer(
            self._make_response(
                sources=[
                    {
                        "certificate_id": 1, "code": "CERT-001", "customer_id": 101,
                        "source_type": "structured", "source_id": "str:1",
                        "snippet": "Dato estructurado.",
                    },
                ],
            ),
        )
        assert "📄 **Documentos PDF**" not in result
        assert "📋 **Datos estructurados**" not in result
        assert "CERT-001" in result
        assert "Fuentes consultadas" in result

    def test_format_answer_only_pdf_no_structured_header(self) -> None:
        """Single type (pdf_text) shows flat list, no structured header."""
        from ui.settings import format_answer

        result = format_answer(
            self._make_response(
                sources=[
                    {
                        "certificate_id": 2, "code": "CERT-002", "customer_id": 102,
                        "source_type": "pdf_text", "source_id": "s3://bucket/cert-002.pdf",
                        "snippet": "Texto extraído.",
                    },
                ],
            ),
        )
        assert "📋 **Datos estructurados**" not in result
        assert "CERT-002" in result
        assert "Fuentes consultadas" in result

    def test_format_answer_only_web_no_other_headers(self) -> None:
        """Single type (web) shows flat list."""
        from ui.settings import format_answer

        result = format_answer(
            self._make_response(
                route="web_search",
                sources=[
                    {
                        "certificate_id": None, "code": "WEB-RES-1", "customer_id": None,
                        "source_type": "web", "source_id": "web:https://example.com",
                        "snippet": "Resultado web.",
                    },
                ],
            ),
        )
        assert "🌐 **Resultados web**" not in result
        assert "Fuentes consultadas" in result

    def test_format_answer_ordering_within_group_preserved(self) -> None:
        """Sources within the same type keep original order."""
        from ui.settings import format_answer

        result = format_answer(
            self._make_response(
                route="combined",
                sources=[
                    {
                        "certificate_id": 1, "code": "CERT-A", "customer_id": 101,
                        "source_type": "structured", "source_id": "str:1",
                        "snippet": "Primero.",
                    },
                    {
                        "certificate_id": 2, "code": "CERT-B", "customer_id": 102,
                        "source_type": "structured", "source_id": "str:2",
                        "snippet": "Segundo.",
                    },
                    {
                        "certificate_id": 3, "code": "CERT-C", "customer_id": 103,
                        "source_type": "pdf_text", "source_id": "s3://bucket/cert-c.pdf",
                        "snippet": "PDF.",
                    },
                ],
            ),
        )
        # Within structured group, CERT-A should come before CERT-B
        assert result.index("CERT-A") < result.index("CERT-B")

    def test_format_answer_unknown_source_type(self) -> None:
        """Unknown source_type gets 'Otras fuentes' header."""
        from ui.settings import format_answer

        result = format_answer(
            self._make_response(
                route="combined",
                sources=[
                    {
                        "certificate_id": 1, "code": "CERT-001", "customer_id": 101,
                        "source_type": "structured", "source_id": "str:1",
                        "snippet": "Estructurado.",
                    },
                    {
                        "certificate_id": 2, "code": "CERT-002", "customer_id": 102,
                        "source_type": "api_result", "source_id": "api:2",
                        "snippet": "API.",
                    },
                ],
            ),
        )
        assert "📎 **Otras fuentes**" in result


# ── _get_index_stats / _embedding_label ───────────────────────────────────────


class TestIndexHelpers:
    """Helper functions for admin index stats."""

    def test_get_index_stats_mock_mode(self) -> None:
        """In mock mode, stats return a safe fallback."""
        from ui.settings import _get_index_stats

        settings = _make_settings(app_mode="mock")
        stats = _get_index_stats(settings)
        assert stats["status"] == "mock"

    def test_embedding_label_returns_local_by_default(self) -> None:
        """Without OpenAI config, label is 'local'."""
        from ui.settings import _embedding_label

        settings = _make_settings(embedding_provider="local", openai_api_key=None)
        assert _embedding_label(settings) == "local"

    def test_embedding_label_returns_openai_with_key(self) -> None:
        """With API key, label resolves to 'openai'."""
        from ui.settings import _embedding_label

        settings = _make_settings(embedding_provider="auto", openai_api_key="sk-xxx")
        assert _embedding_label(settings) == "openai"
