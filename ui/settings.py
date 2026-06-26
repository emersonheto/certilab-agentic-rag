"""Application settings and helpers for the Chainlit UI.

Extracted from ``ui/chainlit_app.py`` to separate UI configuration, profile
resolution, and formatting logic from Chainlit handler registration.

This module has zero runtime dependency on ``chainlit`` — it can be imported
safely even when chainlit is not installed.
"""

from __future__ import annotations

import os
from typing import Any

from app.config import Settings
from app.schemas import AskResponse
from app.security.access_control import Principal


# ── Profile definitions ──────────────────────────────────────────────────────
# Each profile maps to a Principal at session start.
# In real mode: resolved from API keys.
# In mock mode: resolved from demo tokens.

_PROFILES = [
    {
        "id": "admin",
        "name": "Admin",
        "markdown_description": "Acceso total — ve todos los certificados de todos los clientes.",
        "icon": "🛡️",
    },
    {
        "id": "technician",
        "name": "Técnico",
        "markdown_description": "Acceso global — puede consultar cualquier registro pero no administrar.",
        "icon": "🔧",
    },
    {
        "id": "client_101",
        "name": "Cliente 101",
        "markdown_description": "Acceso restringido — solo ve los certificados del cliente 101.",
        "icon": "🏢",
    },
    {
        "id": "client_202",
        "name": "Cliente 202",
        "markdown_description": "Acceso restringido — solo ve los certificados del cliente 202.",
        "icon": "🏭",
    },
]


def _build_principal(profile_id: str, settings: Settings) -> Principal:
    """Resolve a Principal for the given profile, adapting to APP_MODE."""

    if settings.app_mode == "real":
        return _principal_real(profile_id, settings)
    return _principal_mock(profile_id, settings)


def _principal_real(profile_id: str, settings: Settings) -> Principal:
    from app.security.api_key_auth import principal_from_api_key

    if profile_id == "admin":
        key = os.environ.get("CHAINLIT_API_KEY_ADMIN") or settings.api_key_admin or ""
        return principal_from_api_key(key, settings)

    if profile_id == "technician":
        key = os.environ.get("CHAINLIT_API_KEY_TECHNICIAN") or settings.api_key_technician or ""
        return principal_from_api_key(key, settings)

    if profile_id == "client_101":
        key = os.environ.get("CHAINLIT_API_KEY_CLIENT") or settings.api_key_client or ""
        return principal_from_api_key(key, settings, customer_id=101)

    if profile_id == "client_202":
        key = os.environ.get("CHAINLIT_API_KEY_CLIENT") or settings.api_key_client or ""
        return principal_from_api_key(key, settings, customer_id=202)

    raise ValueError(f"Unknown profile: {profile_id}")


def _principal_mock(profile_id: str, settings: Settings) -> Principal:
    from app.security.demo_auth import principal_from_demo_token

    token_map = {
        "admin": settings.demo_admin_token,
        "technician": settings.demo_technician_token,
        "client_101": settings.demo_client_101_token,
        "client_202": settings.demo_client_202_token,
    }
    token = token_map.get(profile_id)
    if not token:
        raise RuntimeError(
            f"Demo token for profile '{profile_id}' is not configured. "
            "Set DEMO_ADMIN_TOKEN / DEMO_TECHNICIAN_TOKEN / DEMO_CLIENT_101_TOKEN / "
            "DEMO_CLIENT_202_TOKEN in your .env."
        )
    return principal_from_demo_token(token, settings)


# ── Index / embedding helpers ───────────────────────────────────────────────


def _get_index_stats(settings: Settings) -> dict[str, str | int | None]:
    """Get current Qdrant collection stats for the admin view.

    Only imports ``qdrant_client`` in real mode.  In mock mode or on error
    returns a safe fallback dict so the UI never crashes on an index stat
    refresh.
    """
    if settings.app_mode != "real":
        return {
            "points": "N/A (modo mock)",
            "collection": settings.qdrant_collection,
            "status": "mock",
        }
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url)
        info = client.get_collection(settings.qdrant_collection)
        return {
            "points": info.points_count,
            "indexed_vectors": info.indexed_vectors_count,
            "collection": settings.qdrant_collection,
            "status": info.status,
        }
    except Exception as exc:
        return {
            "points": "?",
            "collection": settings.qdrant_collection,
            "status": str(exc),
        }


def _embedding_label(settings: Settings) -> str:
    """Return a human-readable embedding provider label.

    OpenAI is the only supported embedding provider.
    """
    return "openai"


# ── Step / display constants ────────────────────────────────────────────────

NODE_DISPLAY_NAMES: dict[str, str] = {
    "route": "🧭 Enrutando consulta",
    "retrieve": "🔍 Recuperando fuentes",
    "generate": "✍️ Generando respuesta",
}

ROUTE_LABELS: dict[str, str] = {
    "structured": "📊 Ruta: estructurada",
    "semantic": "🔬 Ruta: semántica",
    "combined": "🔄 Ruta: combinada",
    "web_search": "🌐 Ruta: búsqueda web",
}


# ── Answer formatting ────────────────────────────────────────────────────────


_MAX_DISPLAYED_SOURCES = 8


def format_answer(response: AskResponse) -> str:
    """Build the final markdown answer with a route indicator and source list.

    Displays up to ``_MAX_DISPLAYED_SOURCES`` sources. When the pipeline
    returns more, a summary line indicates how many are omitted.

    Args:
        response: The pipeline response containing answer, route, and sources.

    Returns:
        A markdown string ready to send as a ``cl.Message``.
    """
    parts: list[str] = []

    # Route indicator
    route_label = ROUTE_LABELS.get(response.route, f"📌 Ruta: {response.route}")
    parts.append(f"*{route_label}*\n")

    # Answer body
    parts.append(response.answer)

    # Source list — capped at _MAX_DISPLAYED_SOURCES, grouped by type when mixed
    if response.sources:
        total = len(response.sources)
        displayed = response.sources[:_MAX_DISPLAYED_SOURCES]
        omitted = total - len(displayed)

        source_types = {s.source_type for s in displayed}
        type_headers: dict[str, str] = {
            "structured": "📋 **Datos estructurados**",
            "metadata": "📋 **Datos estructurados**",
            "pdf_text": "📄 **Documentos PDF**",
            "web": "🌐 **Resultados web**",
        }

        if len(source_types) == 1:
            # Single type — flat list
            lines = [
                f"- **{s.code or s.certificate_id}** · cliente {s.customer_id}"
                for s in displayed
            ]
            if omitted:
                lines.append(f"- *… y {omitted} fuente(s) más*")
            parts.append(f"\n---\n📄 **Fuentes consultadas**\n" + "\n".join(lines))
        else:
            # Multiple types — group by source_type
            groups: dict[str, list[Any]] = {}
            for s in displayed:
                groups.setdefault(s.source_type, []).append(s)

            group_lines: list[str] = []
            for source_type, sources in groups.items():
                header = type_headers.get(source_type, "📎 **Otras fuentes**")
                group_lines.append(header)
                for s in sources:
                    group_lines.append(
                        f"- **{s.code or s.certificate_id}** · cliente {s.customer_id}",
                    )
            if omitted:
                group_lines.append(f"*… y {omitted} fuente(s) más*")

            parts.append("\n---\n" + "\n".join(group_lines))

    return "\n\n".join(parts)
