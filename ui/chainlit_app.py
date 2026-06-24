from __future__ import annotations

from app.config import Settings, get_settings
from app.graph import build_rag_pipeline
from app.security.demo_auth import principal_from_demo_token

try:
    import chainlit as cl
except ImportError:  # pragma: no cover - exercised by import-only environments
    cl = None  # type: ignore[assignment]


def _answer(question: str) -> str:
    settings = get_settings()
    token = _default_demo_token(settings)
    if token is None:
        return "Chainlit demo is not configured. Set CHAINLIT_DEMO_TOKEN, or configure a demo client token."
    principal = principal_from_demo_token(token, settings)
    response = build_rag_pipeline(settings).ask(question, principal)
    source_count = len(response.sources)
    return f"{response.answer}\n\nRuta: {response.route} · Fuentes: {source_count}"


def _default_demo_token(settings: Settings) -> str | None:
    return settings.chainlit_demo_token or settings.demo_client_101_token or settings.demo_client_202_token


if cl is not None:

    @cl.on_message
    async def main(message: cl.Message) -> None:
        await cl.Message(content=_answer(message.content)).send()

else:

    async def main(message: object) -> None:
        raise RuntimeError("Install the optional Chainlit dependency to run the UI.")
