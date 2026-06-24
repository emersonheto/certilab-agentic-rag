from dataclasses import dataclass
from typing import Any

from app.security.payload_sanitizer import sanitize_query_for_external_search, should_block_external_search


@dataclass(frozen=True)
class WebSearchConfig:
    tavily_api_key: str | None


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source_note: str = "tavily"


class TavilyWebSearch:
    """Future optional fallback for public web context, never for private certificates."""

    def __init__(self, config: WebSearchConfig) -> None:
        self._config = config
        self._client: Any | None = None

    def search(self, query: str) -> list[WebSearchResult]:
        """Search Tavily when configured, otherwise return a safe fallback note."""

        if should_block_external_search(query):
            return [_fallback_result("Web search skipped because the query may contain private or sensitive data.")]
        if not self._config.tavily_api_key:
            return [_fallback_result("Tavily API key is not configured.")]
        try:
            safe_query = sanitize_query_for_external_search(query)
            if not safe_query:
                return [_fallback_result("Web search skipped because the sanitized query is empty.")]
            response = self._get_client().search(query=safe_query, max_results=3)
        except Exception as exc:
            return [_fallback_result(f"Tavily search unavailable: {type(exc).__name__}.")]
        raw_results = response.get("results", []) if isinstance(response, dict) else []
        results: list[WebSearchResult] = []
        for item in raw_results[:3]:
            if not isinstance(item, dict):
                continue
            results.append(
                WebSearchResult(
                    title=str(item.get("title") or "External result"),
                    url=str(item.get("url") or ""),
                    snippet=str(item.get("content") or item.get("snippet") or "")[:500],
                )
            )
        return results or [_fallback_result("Tavily returned no public results.")]

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from tavily import TavilyClient
            except ImportError as exc:
                raise RuntimeError("Tavily search requires the optional 'tavily-python' package.") from exc
            self._client = TavilyClient(api_key=self._config.tavily_api_key)
        return self._client


def _fallback_result(note: str) -> WebSearchResult:
    return WebSearchResult(title="Web search fallback", url="", snippet=note, source_note="fallback")
