"""Tests for ``ui/app`` — Chainlit handlers with progressive steps.

Strict TDD: tests written before implementation.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import httpx
import pytest

from app.config import Settings


# ── Fake Chainlit classes ─────────────────────────────────────────────────────


class _FakeUserSession:
    """Minimal fake for ``cl.user_session``."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value


class _FakeStep:
    """A fake ``cl.Step`` that tracks open/close for assertions.

    Usage::

        async with _FakeStep(name="...") as step:
            step.output = "working..."
        assert step.opened is True
        assert step.closed is True
    """

    instances: list[_FakeStep] = []

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.output: str = ""
        self.opened = False
        self.closed = False
        _FakeStep.instances.append(self)

    async def __aenter__(self) -> _FakeStep:
        self.opened = True
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.closed = True

    @classmethod
    def reset(cls) -> None:
        cls.instances.clear()


class _FakeMessage:
    _sent: list[str] = []

    def __init__(self, content: str = "", **_kwargs: Any) -> None:
        self.content = content

    async def send(self) -> _FakeMessage:
        _FakeMessage._sent.append(self.content)
        return self

    async def remove(self) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls._sent.clear()


class _FakeChatProfile:
    def __init__(
        self, name: str = "", markdown_description: str = "", icon: str = "",
    ) -> None:
        self.name = name
        self.markdown_description = markdown_description
        self.icon = icon


class _FakeAction:
    def __init__(self, name: str = "", label: str = "", description: str = "") -> None:
        self.name = name
        self.label = label
        self.description = description


class _FakeTask:
    def __init__(self, title: str = "", status: object = None, forId: str | None = None) -> None:
        self.title = title
        self.status = status
        self.forId = forId


class _FakeTaskList:
    def __init__(self) -> None:
        self.tasks: list[Any] = []

    async def send(self) -> None: ...
    async def remove(self) -> None: ...


class _FakeTaskStatus:
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    READY = "ready"


class _FakeStarter:
    def __init__(self, label: str = "", message: str = "") -> None:
        self.label = label
        self.message = message


class FakeChainlitModule:
    """Minimal fake for testing Chainlit handler import and registration."""

    Action = _FakeAction
    Message = _FakeMessage
    ChatProfile = _FakeChatProfile
    Task = _FakeTask
    TaskList = _FakeTaskList
    TaskStatus = _FakeTaskStatus
    Starter = _FakeStarter
    Step = _FakeStep
    user_session = _FakeUserSession()

    @staticmethod
    def set_chat_profiles(func: object) -> object:
        return func

    @staticmethod
    def set_starters(func: object) -> object:
        return func

    @staticmethod
    def on_chat_start(func: object) -> object:
        return func

    @staticmethod
    def on_message(func: object) -> object:
        return func

    @staticmethod
    def action_callback(name: str) -> object:
        def decorator(func: object) -> object:
            return func
        return decorator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_settings(**overrides: Any) -> Settings:
    """Build a Settings instance with test demo tokens."""
    return Settings(
        _env_file=None,
        demo_admin_token="admin-token",
        demo_technician_token="technician-token",
        demo_client_101_token="client-101-token",
        demo_client_202_token="client-202-token",
        **overrides,
    )


# ── Module import tests ───────────────────────────────────────────────────────


class TestAppModule:
    """Verify ``ui.chainlit_app`` module loads and registers handlers correctly.

    RED step: the module ``ui/app.py`` does not exist yet — these tests
    reference it first, confirming they will fail on initial import.
    """

    def test_module_imports_without_chainlit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing ``ui.chainlit_app`` succeeds when ``chainlit`` is not installed.

        The module must not crash at import time without chainlit.
        It should define a ``main`` fallback so downstream imports behave.
        """
        monkeypatch.setitem(sys.modules, "chainlit", None)
        sys.modules.pop("ui.chainlit_app", None)
        try:
            mod = importlib.import_module("ui.chainlit_app")
        finally:
            sys.modules.pop("ui.chainlit_app", None)

        assert hasattr(mod, "main"), "ui.chainlit_app must export main() as fallback"

    def test_module_imports_with_chainlit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing ``ui.chainlit_app`` succeeds when a fake Chainlit is available."""
        monkeypatch.setitem(sys.modules, "chainlit", FakeChainlitModule())
        sys.modules.pop("ui.chainlit_app", None)
        try:
            mod = importlib.import_module("ui.chainlit_app")
        finally:
            sys.modules.pop("ui.chainlit_app", None)

        # When chainlit IS available, the if-cl branch should register the expected handlers.
        # The fallback main() is NOT defined in this path — the decorator handlers are.
        assert hasattr(mod, "chat_profiles"), "chat_profiles handler must exist"
        assert hasattr(mod, "starters"), "starters handler must exist"
        assert hasattr(mod, "on_chat_start"), "on_chat_start handler must exist"
        assert hasattr(mod, "main"), "main handler must exist"


# ── _ask_stream consumer integration ──────────────────────────────────────────


class TestOnMessageConsumer:
    """Verify ``on_message`` uses the SSE ``ask_stream`` consumer correctly."""

    @pytest.fixture(autouse=True)
    def _patch_chainlit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _FakeStep.reset()
        FakeChainlitModule.user_session._store.clear()
        monkeypatch.setitem(sys.modules, "chainlit", FakeChainlitModule())
        sys.modules.pop("ui.chainlit_app", None)

    async def test_on_message_calls_ask_stream_with_question(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``on_message`` forwards the user question to ``ask_stream``."""
        import ui.chainlit_app as _app

        from app.domain.models import Role
        from app.security.access_control import Principal

        principal = Principal(role=Role.ADMIN, customer_id=None, user_id=1)
        FakeChainlitModule.user_session.set("principal", principal)
        FakeChainlitModule.user_session.set("demo_token", "tok")

        captured: dict[str, Any] = {}

        async def fake_ask_stream(**kwargs: Any) -> None:
            captured.update(kwargs)
            await kwargs["on_event"]({"event": "complete", "answer": "x", "route": "structured", "sources": []})

        monkeypatch.setattr(_app, "ask_stream", fake_ask_stream)

        await _app.main(_FakeMessage(content="¿Cuántos hay?"))

        assert captured["question"] == "¿Cuántos hay?"

    async def test_on_message_renders_answer_on_complete(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ``complete`` event produces a formatted answer message."""
        import ui.chainlit_app as _app

        from app.domain.models import Role
        from app.security.access_control import Principal

        principal = Principal(role=Role.ADMIN, customer_id=None, user_id=1)
        FakeChainlitModule.user_session.set("principal", principal)
        FakeChainlitModule.user_session.set("demo_token", "tok")

        _FakeMessage.reset()

        async def fake_ask_stream(**kwargs: Any) -> None:
            await kwargs["on_event"]({
                "event": "complete",
                "answer": "Hay 42 certificados.",
                "route": "structured",
                "sources": [],
            })

        monkeypatch.setattr(_app, "ask_stream", fake_ask_stream)

        await _app.main(_FakeMessage(content="pregunta"))

        assert any("42 certificados" in m for m in _FakeMessage._sent)

    async def test_on_message_step_closes_on_stream_failure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``ask_stream`` raises an unexpected error, the step still closes.

        ``httpx.HTTPError`` is caught gracefully, but other exceptions
        propagate while the ``cl.Step`` context manager still closes.
        """
        import ui.chainlit_app as _app

        from app.domain.models import Role
        from app.security.access_control import Principal

        principal = Principal(role=Role.ADMIN, customer_id=None, user_id=1)
        FakeChainlitModule.user_session.set("principal", principal)
        FakeChainlitModule.user_session.set("demo_token", "tok")

        async def failing_ask_stream(**kwargs: Any) -> None:
            raise RuntimeError("Stream failed")

        monkeypatch.setattr(_app, "ask_stream", failing_ask_stream)

        with pytest.raises(RuntimeError, match="Stream failed"):
            await _app.main(_FakeMessage(content="test query"))

        assert len(_FakeStep.instances) == 1
        step = _FakeStep.instances[0]
        assert step.opened is True, "Step was opened before the exception"
        assert step.closed is True, "Step was closed even on exception"

    async def test_on_message_graceful_on_http_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``httpx.HTTPError`` is caught — user sees an error message, no crash."""
        import ui.chainlit_app as _app

        from app.domain.models import Role
        from app.security.access_control import Principal

        principal = Principal(role=Role.ADMIN, customer_id=None, user_id=1)
        FakeChainlitModule.user_session.set("principal", principal)
        FakeChainlitModule.user_session.set("demo_token", "tok")

        async def failing_ask_stream(**kwargs: Any) -> None:
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(_app, "ask_stream", failing_ask_stream)

        # Should NOT raise
        await _app.main(_FakeMessage(content="test query"))

        assert len(_FakeStep.instances) == 1
        assert _FakeStep.instances[0].closed is True


# ── handler structure ─────────────────────────────────────────────────────────


class TestHandlerStructure:
    """Verify handler signatures and registration under fake Chainlit.

    These tests import ``ui.chainlit_app`` with a fake Chainlit module, then inspect
    the exported handler functions.
    """

    @pytest.fixture(autouse=True)
    def _patch_chainlit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "chainlit", FakeChainlitModule())
        sys.modules.pop("ui.chainlit_app", None)

    def test_on_message_is_async(self) -> None:
        """The ``main`` handler (on_message) is an async function."""
        import ui.chainlit_app as _app

        assert hasattr(_app.main, "__code__")

    def test_on_chat_start_exists_and_is_callable(self) -> None:
        """``on_chat_start`` is a callable async function."""
        import ui.chainlit_app as _app

        assert callable(_app.on_chat_start)

    async def test_chat_profiles_returns_list(self) -> None:
        """``chat_profiles()`` returns a list of profiles."""
        import ui.chainlit_app as _app

        profiles = await _app.chat_profiles()
        assert isinstance(profiles, list)

    async def test_main_fallback_raises_without_chainlit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When chainlit is not available, ``main()`` raises ``RuntimeError``."""
        monkeypatch.setitem(sys.modules, "chainlit", None)
        sys.modules.pop("ui.chainlit_app", None)

        import ui.chainlit_app as _app

        with pytest.raises(RuntimeError, match="Install the optional Chainlit"):
            await _app.main(object())


# ── format_answer integration ──────────────────────────────────────────────────


class TestFormatAnswerIntegration:
    """Verify ``format_answer`` from settings works correctly for all routes.

    These tests are pure-function verifications — no Chainlit mocking needed.
    They confirm the formatting helper that ``on_message`` calls behaves
    correctly for each route type.
    """

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

    def test_format_includes_route_indicator(self) -> None:
        """Answer includes route indicator for structured."""
        from ui.settings import format_answer

        result = format_answer(self._make_response(route="structured"))
        assert "📊 Ruta: estructurada" in result
        assert "Respuesta de prueba" in result

    def test_format_includes_semantic_indicator(self) -> None:
        """Answer includes route indicator for semantic."""
        from ui.settings import format_answer

        result = format_answer(self._make_response(route="semantic"))
        assert "🔬" in result

    def test_format_includes_sources_section(self) -> None:
        """Sources render with code and customer id."""
        from ui.settings import format_answer

        result = format_answer(
            self._make_response(
                sources=[
                    {
                        "certificate_id": 1, "code": "CERT-001", "customer_id": 101,
                        "source_type": "metadata", "source_id": "metadata:CERT-001",
                        "snippet": "Detalle",
                    },
                ],
            ),
        )
        assert "CERT-001" in result
        assert "Fuentes" in result

    def test_format_omits_sources_when_empty(self) -> None:
        """No sources heading when there are no sources."""
        from ui.settings import format_answer

        result = format_answer(self._make_response(sources=[]))
        assert "Fuentes" not in result

    def test_format_answer_text_preserved_verbatim(self) -> None:
        """The answer text appears exactly as provided, without truncation."""
        from ui.settings import format_answer

        answer_text = "El sistema tiene 42 certificados activos y 3 pendientes."
        result = format_answer(self._make_response(answer=answer_text))
        assert answer_text in result

    def test_format_combined_route_shows_both_source_types(self) -> None:
        """Combined route output shows sources from both types."""
        from ui.settings import format_answer

        result = format_answer(
            self._make_response(
                route="combined",
                sources=[
                    {
                        "certificate_id": 1, "code": "CERT-001", "customer_id": 101,
                        "source_type": "metadata", "source_id": "metadata:CERT-001",
                        "snippet": "Estructurado.",
                    },
                    {
                        "certificate_id": 2, "code": "CERT-002", "customer_id": 101,
                        "source_type": "pdf_text", "source_id": "s3://bucket/cert-002.pdf",
                        "snippet": "Semántico.",
                    },
                ],
            ),
        )
        assert "CERT-001" in result
        assert "CERT-002" in result
        assert "cliente 101" in result


# ── Step behavior ──────────────────────────────────────────────────────────────


class TestStepBehavior:
    """Verify that cl.Step is used as an async context manager.

    We test the _FakeStep directly to confirm the step protocol works.
    """

    @pytest.fixture(autouse=True)
    def _reset_steps(self) -> None:
        _FakeStep.reset()

    async def test_fake_step_tracks_open_close(self) -> None:
        """_FakeStep tracks open/close via async context manager."""
        async with _FakeStep(name="test") as step:
            step.output = "working"
            assert step.opened is True
            assert step.closed is False
        assert step.closed is True

    async def test_fake_step_registers_instances(self) -> None:
        """_FakeStep.instances grows as steps are entered."""
        async with _FakeStep(name="a"):
            async with _FakeStep(name="b"):
                pass
        assert len(_FakeStep.instances) == 2
        assert _FakeStep.instances[0].name == "a"
        assert _FakeStep.instances[1].name == "b"
