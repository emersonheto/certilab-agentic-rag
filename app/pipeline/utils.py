"""Utility functions extracted from the main pipeline.

These functions were originally defined in ``app.graph`` and are extracted here
for reuse across pipeline variants. Both files keep the same definitions
temporarily (PR 4 will remove them from ``app.graph``).
"""

from __future__ import annotations

from time import perf_counter

from app.domain.models import RetrievedSource
from app.schemas import Source
from app.security.access_control import AccessScope
from app.tools.web_search import WebSearchResult


def _source_id(source: RetrievedSource) -> str:
    if source.code:
        return f"{source.source_type}:{source.code}"
    return source.source_type


def _to_schema(source: RetrievedSource) -> Source:
    return Source(
        certificate_id=source.certificate_id,
        code=source.code,
        customer_id=source.customer_id,
        source_type=source.source_type,
        source_id=_source_id(source),
        snippet=source.snippet,
    )


def _dedupe_sources(sources: list[RetrievedSource]) -> list[RetrievedSource]:
    """Remove duplicate sources by (certificate_id, source_type, snippet)."""
    seen: set[tuple[int | None, str, str]] = set()
    unique: list[RetrievedSource] = []
    for source in sources:
        key = (source.certificate_id, source.source_type, source.snippet)
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


def _web_sources(results: list[WebSearchResult]) -> list[RetrievedSource]:
    """Convert web search results to RetrievedSource list."""
    return [
        RetrievedSource(
            certificate_id=None,
            code=None,
            customer_id=None,
            source_type="web_search",
            path=result.url,
            snippet=f"{result.title}: {result.snippet}",
            score=0.0 if result.source_note == "fallback" else 1.0,
        )
        for result in results
    ]


def _elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds since *started_at*."""
    return round((perf_counter() - started_at) * 1000, 3)


def _scope_attributes(scope: AccessScope) -> dict[str, str | bool]:
    """Build OTEL span attributes from an access scope."""
    return {"rag.role": scope.role.value, "rag.has_customer_scope": scope.customer_id is not None}


def _certificate_count(sources: list[RetrievedSource]) -> int:
    """Count unique certificate IDs across sources."""
    return len({source.certificate_id for source in sources if source.certificate_id is not None})
