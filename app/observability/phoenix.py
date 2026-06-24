from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import wraps
from types import TracebackType
from typing import Any, ParamSpec, Protocol, TypeVar

from app.config import Settings


SpanAttributeValue = str | int | float | bool | list[str] | list[int] | list[float] | list[bool]
SpanAttributes = Mapping[str, SpanAttributeValue]

P = ParamSpec("P")
R = TypeVar("R")


class SpanHandle(Protocol):
    """Small span surface used by application code without importing OpenTelemetry."""

    def set_attribute(self, key: str, value: SpanAttributeValue) -> None:
        """Attach a safe attribute to the current span."""


class _NoOpSpan:
    def set_attribute(self, key: str, value: SpanAttributeValue) -> None:
        return None


class _OpenTelemetrySpan:
    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: SpanAttributeValue) -> None:
        try:
            self._span.set_attribute(key, value)
        except Exception:
            return None


_tracer: Any | None = None
_configured = False
_noop_span = _NoOpSpan()


def configure_observability(settings: Settings) -> None:
    """Configure optional Phoenix/OpenTelemetry tracing.

    The application must remain runnable in mock mode without Phoenix or
    OpenTelemetry installed, so configuration failures intentionally degrade to
    no-op tracing.
    """

    global _configured, _tracer

    if _configured:
        return

    _configured = True
    if not settings.phoenix_enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except ImportError:
        return

    try:
        project_attribute = _project_name_attribute()
        resource = Resource.create({project_attribute: settings.phoenix_project_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint)
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(settings.phoenix_project_name)
    except Exception:
        _tracer = None


@contextmanager
def trace_span(name: str, attributes: SpanAttributes | None = None) -> Iterator[SpanHandle]:
    """Create a trace span or safely no-op when observability is unavailable."""

    if _tracer is None:
        yield _noop_span
        return

    span_context: Any | None = None
    try:
        span_context = _tracer.start_as_current_span(name)
        span = span_context.__enter__()
    except Exception:
        yield _noop_span
        return

    handle = _OpenTelemetrySpan(span)
    for key, value in (attributes or {}).items():
        handle.set_attribute(key, value)

    exc_type: type[BaseException] | None = None
    exc_value: BaseException | None = None
    traceback: TracebackType | None = None
    try:
        yield handle
    except BaseException as exc:
        exc_type = type(exc)
        exc_value = exc
        traceback = exc.__traceback__
        raise
    finally:
        try:
            span_context.__exit__(exc_type, exc_value, traceback)
        except Exception:
            pass


def traced(
    name: str, attributes: SpanAttributes | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a sync function with a trace span."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with trace_span(name, attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _project_name_attribute() -> str:
    try:
        from openinference.semconv.resource import ResourceAttributes
    except ImportError:
        return "openinference.project.name"
    return str(ResourceAttributes.PROJECT_NAME)
