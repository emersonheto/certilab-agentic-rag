"""HTTP SSE client for the Certilab RAG API.

Replaces the in-process ``PipelineClient``. All pipeline logic (routing,
off-topic detection, scope narrowing, generation, RBAC) lives in the
backend ``/ask`` and ``/ask/stream`` endpoints — the UI just consumes
Server-Sent Events over HTTP.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


async def ask_stream(
    question: str,
    token: str,
    api_url: str,
    on_event: EventHandler,
    *,
    token_header: str = "X-Demo-Token",
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """POST to ``/ask/stream`` and consume SSE events.

    Args:
        question: The user's question.
        token: Auth token (demo token or API key).
        api_url: Base URL of the API (e.g. ``"http://localhost:8000"``).
        on_event: Async callback invoked once per parsed SSE event.
        token_header: Header name carrying the auth token.
        transport: Optional httpx transport for testing (inject MockTransport).
    """
    url = f"{api_url.rstrip('/')}/ask/stream"
    async with httpx.AsyncClient(timeout=300.0, transport=transport) as client:
        async with client.stream(
            "POST",
            url,
            json={"question": question},
            headers={token_header: token, "Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data:
                    continue
                await on_event(json.loads(data))
