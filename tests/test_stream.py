"""Tests for the ``POST /ask/stream`` SSE endpoint.

Strict TDD: tests written before implementation.

The endpoint wraps the same atomic ``pipeline.ask()`` used by ``/ask``,
emitting SSE events (``start`` → ``complete`` / ``error``) so the UI can
consume them over HTTP instead of duplicating pipeline logic in-process.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> TestClient:
    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    return TestClient(create_app())


def _parse_sse(text: str) -> list[dict[str, object]]:
    """Parse an SSE stream body into a list of event dicts.

    Each SSE record is ``data: {json}\\n\\n``. Non-data lines (comments,
    blank separators) are skipped.
    """
    events: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload:
            continue
        events.append(json.loads(payload))
    return events


# ── Happy path ────────────────────────────────────────────────────────────────


def test_ask_stream_returns_sse_events(client: TestClient, demo_admin_token: str) -> None:
    """A valid request yields a 200 with start and complete SSE events."""
    response = client.post(
        "/ask/stream",
        headers={"X-Demo-Token": demo_admin_token},
        json={"question": "¿Cuántos certificados hay?"},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_types = [e["event"] for e in events]
    assert "start" in event_types
    assert "complete" in event_types


def test_ask_stream_start_event_has_question_length(
    client: TestClient, demo_admin_token: str,
) -> None:
    """The start event reports the incoming question length."""
    question = "¿Cuántos certificados hay?"
    response = client.post(
        "/ask/stream",
        headers={"X-Demo-Token": demo_admin_token},
        json={"question": question},
    )
    events = _parse_sse(response.text)
    start = next(e for e in events if e["event"] == "start")
    assert start["question_length"] == len(question)


def test_ask_stream_complete_event_has_answer_and_route(
    client: TestClient, demo_admin_token: str,
) -> None:
    """The complete event carries a non-empty answer and a valid route."""
    response = client.post(
        "/ask/stream",
        headers={"X-Demo-Token": demo_admin_token},
        json={"question": "¿Cuántos certificados hay?"},
    )
    events = _parse_sse(response.text)
    complete = next(e for e in events if e["event"] == "complete")
    assert isinstance(complete["answer"], str) and complete["answer"]
    assert complete["route"] in {"structured", "semantic", "combined", "web_search", "off_topic"}
    assert isinstance(complete["sources"], list)


def test_ask_stream_complete_event_serializes_sources(
    client: TestClient, demo_client_101_token: str,
) -> None:
    """Sources in the complete event are JSON-serializable Source dicts."""
    response = client.post(
        "/ask/stream",
        headers={"X-Demo-Token": demo_client_101_token},
        json={"question": "¿Cuántos certificados hay?"},
    )
    events = _parse_sse(response.text)
    complete = next(e for e in events if e["event"] == "complete")
    for source in complete["sources"]:
        assert "source_type" in source
        assert "source_id" in source
        assert "snippet" in source


# ── Off-topic ─────────────────────────────────────────────────────────────────


def test_ask_stream_off_topic_returns_greeting(
    client: TestClient, demo_admin_token: str,
) -> None:
    """An off-topic greeting yields route=off_topic with a non-empty answer."""
    response = client.post(
        "/ask/stream",
        headers={"X-Demo-Token": demo_admin_token},
        json={"question": "hola"},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    complete = next(e for e in events if e["event"] == "complete")
    assert complete["route"] == "off_topic"
    assert complete["answer"]
    assert complete["sources"] == []


# ── Scope isolation ───────────────────────────────────────────────────────────


def test_ask_stream_respects_client_scope(
    client: TestClient, demo_client_101_token: str,
) -> None:
    """A client-scoped token only returns sources for that client."""
    response = client.post(
        "/ask/stream",
        headers={"X-Demo-Token": demo_client_101_token},
        json={"question": "¿Cuántos certificados hay?"},
    )
    events = _parse_sse(response.text)
    complete = next(e for e in events if e["event"] == "complete")
    assert complete["route"] == "structured"
    customer_ids = {s["customer_id"] for s in complete["sources"]}
    assert customer_ids == {101}


# ── Auth ─────────────────────────────────────────────────────────────────────


def test_ask_stream_requires_auth(client: TestClient) -> None:
    """Missing demo token returns 401."""
    response = client.post("/ask/stream", json={"question": "¿Cuántos certificados hay?"})
    assert response.status_code == 401


def test_ask_stream_rejects_invalid_token(client: TestClient) -> None:
    """Invalid demo token returns 401."""
    response = client.post(
        "/ask/stream",
        headers={"X-Demo-Token": "not-a-demo-token"},
        json={"question": "¿Cuántos certificados hay?"},
    )
    assert response.status_code == 401


def test_ask_stream_rejects_caller_controlled_scope_fields(
    client: TestClient, demo_client_101_token: str,
) -> None:
    """Extra body fields (scope escalation attempts) are rejected (422)."""
    response = client.post(
        "/ask/stream",
        headers={"X-Demo-Token": demo_client_101_token},
        json={"question": "¿Cuántos certificados hay?", "role": "admin", "customer_id": 202},
    )
    assert response.status_code == 422
