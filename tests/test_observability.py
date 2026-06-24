import builtins
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.domain.models import Role
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
