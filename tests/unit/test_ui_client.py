"""Tests for ``ui/client`` — HTTP SSE consumer for ``/ask/stream``.

Strict TDD: tests written before implementation.

The old ``PipelineClient`` (260 lines of duplicated pipeline logic) is
replaced by a ~50-line ``ask_stream()`` that POSTs to the backend and
consumes SSE events. All routing/off-topic/scope/RBAC logic now lives
exclusively in the backend ``/ask`` and ``/ask/stream`` endpoints.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_sse_body(*events: dict[str, Any]) -> str:
    """Build an SSE body string from a sequence of event dicts."""
    parts = [f"data: {json.dumps(e)}\n\n" for e in events]
    return "".join(parts)


def make_transport(
    body: str = "",
    status_code: int = 200,
    received: dict[str, Any] | None = None,
) -> httpx.MockTransport:
    """Build a MockTransport that returns ``body`` and records the request.

    Args:
        body: The SSE body to return.
        status_code: HTTP status code for the response.
        received: Optional dict populated with request details for assertions.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if received is not None:
            received["url"] = str(request.url)
            received["method"] = request.method
            received["headers"] = dict(request.headers)
            received["content"] = request.content
        return httpx.Response(
            status_code,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    return httpx.MockTransport(handler)


# ── ask_stream — event consumption ────────────────────────────────────────────


class TestEventConsumption:
    """Verify SSE events are parsed and passed to the callback."""

    async def test_invokes_callback_for_each_event(self) -> None:
        from ui.client import ask_stream

        events: list[dict[str, Any]] = []

        async def on_event(event: dict[str, Any]) -> None:
            events.append(event)

        body = make_sse_body(
            {"event": "start", "question_length": 5},
            {"event": "complete", "route": "structured", "answer": "x", "sources": []},
        )
        transport = make_transport(body)
        await ask_stream("hola?", "tok", "http://api", on_event, transport=transport)

        assert len(events) == 2
        assert events[0]["event"] == "start"
        assert events[1]["event"] == "complete"

    async def test_skips_non_data_lines(self) -> None:
        """Comment lines and blank separators are not passed to the callback."""
        from ui.client import ask_stream

        events: list[dict[str, Any]] = []

        async def on_event(event: dict[str, Any]) -> None:
            events.append(event)

        # Mix in comments and event-id lines that must be ignored
        body = (
            ": keepalive\n\n"
            "id: 42\n\n"
            'data: {"event": "start"}\n\n'
            "\n"
            'data: {"event": "complete"}\n\n'
        )
        transport = make_transport(body)
        await ask_stream("q", "tok", "http://api", on_event, transport=transport)

        assert len(events) == 2

    async def test_empty_data_payload_is_skipped(self) -> None:
        """A bare ``data:`` with no payload is skipped."""
        from ui.client import ask_stream

        events: list[dict[str, Any]] = []

        async def on_event(event: dict[str, Any]) -> None:
            events.append(event)

        body = "data:\n\ndata: {\"event\": \"complete\"}\n\n"
        transport = make_transport(body)
        await ask_stream("q", "tok", "http://api", on_event, transport=transport)

        assert len(events) == 1
        assert events[0]["event"] == "complete"

    async def test_strips_data_prefix_with_or_without_space(self) -> None:
        """Both ``data: {...}`` and ``data:{...}`` are parsed."""
        from ui.client import ask_stream

        events: list[dict[str, Any]] = []

        async def on_event(event: dict[str, Any]) -> None:
            events.append(event)

        body = 'data:{"event": "start"}\n\ndata: {"event": "complete"}\n\n'
        transport = make_transport(body)
        await ask_stream("q", "tok", "http://api", on_event, transport=transport)

        assert events[0]["event"] == "start"
        assert events[1]["event"] == "complete"


# ── ask_stream — request construction ─────────────────────────────────────────


class TestRequestConstruction:
    """Verify the HTTP request is built correctly."""

    async def test_posts_to_ask_stream_url(self) -> None:
        from ui.client import ask_stream

        received: dict[str, Any] = {}
        transport = make_transport(make_sse_body({"event": "start"}), received=received)

        async def on_event(event: dict[str, Any]) -> None:
            pass

        await ask_stream("q", "tok", "http://localhost:8000", on_event, transport=transport)

        assert received["url"] == "http://localhost:8000/ask/stream"
        assert received["method"] == "POST"

    async def test_strips_trailing_slash_from_api_url(self) -> None:
        from ui.client import ask_stream

        received: dict[str, Any] = {}
        transport = make_transport(make_sse_body({"event": "start"}), received=received)

        async def on_event(event: dict[str, Any]) -> None:
            pass

        await ask_stream("q", "tok", "http://localhost:8000/", on_event, transport=transport)

        assert received["url"] == "http://localhost:8000/ask/stream"

    async def test_sends_question_in_json_body(self) -> None:
        from ui.client import ask_stream

        received: dict[str, Any] = {}
        transport = make_transport(make_sse_body({"event": "start"}), received=received)

        async def on_event(event: dict[str, Any]) -> None:
            pass

        await ask_stream("¿Cuántos certificados?", "tok", "http://api", on_event, transport=transport)

        body = json.loads(received["content"])
        assert body == {"question": "¿Cuántos certificados?"}

    async def test_default_token_header_is_demo(self) -> None:
        from ui.client import ask_stream

        received: dict[str, Any] = {}
        transport = make_transport(make_sse_body({"event": "start"}), received=received)

        async def on_event(event: dict[str, Any]) -> None:
            pass

        await ask_stream("q", "my-token", "http://api", on_event, transport=transport)

        assert received["headers"]["x-demo-token"] == "my-token"

    async def test_custom_token_header(self) -> None:
        from ui.client import ask_stream

        received: dict[str, Any] = {}
        transport = make_transport(make_sse_body({"event": "start"}), received=received)

        async def on_event(event: dict[str, Any]) -> None:
            pass

        await ask_stream(
            "q", "key-123", "http://api", on_event,
            token_header="X-API-Key", transport=transport,
        )

        assert received["headers"]["x-api-key"] == "key-123"

    async def test_sets_accept_event_stream_header(self) -> None:
        from ui.client import ask_stream

        received: dict[str, Any] = {}
        transport = make_transport(make_sse_body({"event": "start"}), received=received)

        async def on_event(event: dict[str, Any]) -> None:
            pass

        await ask_stream("q", "tok", "http://api", on_event, transport=transport)

        assert received["headers"]["accept"] == "text/event-stream"


# ── ask_stream — error handling ───────────────────────────────────────────────


class TestErrorHandling:
    """Verify HTTP errors propagate."""

    async def test_raises_on_http_error_status(self) -> None:
        from ui.client import ask_stream

        transport = make_transport("Not Found", status_code=404)

        async def on_event(event: dict[str, Any]) -> None:
            pass

        with pytest.raises(httpx.HTTPStatusError):
            await ask_stream("q", "tok", "http://api", on_event, transport=transport)

    async def test_does_not_invoke_callback_on_error(self) -> None:
        from ui.client import ask_stream

        transport = make_transport("error", status_code=500)
        called: list[dict[str, Any]] = []

        async def on_event(event: dict[str, Any]) -> None:
            called.append(event)

        with pytest.raises(httpx.HTTPStatusError):
            await ask_stream("q", "tok", "http://api", on_event, transport=transport)

        assert called == []
