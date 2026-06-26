from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.logging import get_logger
from app.observability import trace_span

logger = get_logger("tools.embeddings")

# OpenAI text-embedding-3-small output dimension.
_OPENAI_EMBEDDING_DIM = 1536


@dataclass(frozen=True)
class EmbeddingProviderConfig:
    """Configuration for the OpenAI embedding provider."""

    openai_api_key: str | None
    openai_embedding_model: str
    embedding_batch_size: int = 100

    @classmethod
    def from_settings(cls, settings: Settings) -> EmbeddingProviderConfig:
        """Build provider config from application settings."""

        return cls(
            openai_api_key=settings.openai_api_key,
            openai_embedding_model=settings.openai_embedding_model,
            embedding_batch_size=settings.embedding_batch_size,
        )


class EmbeddingsProvider:
    """OpenAI-only embedding provider.

    Raises on failure — callers (pipeline, Qdrant) handle the exception and
    fall back to structured-only retrieval or error responses.

    Security notes:
    - The provider never receives PII columns. Only allowlisted text
      constructed by the loader is passed to ``embed``. PII (password, ruc,
      email, phone) is excluded at the connector/loader level.
    """

    def __init__(self, config: EmbeddingProviderConfig) -> None:
        self._config = config
        self._client: Any = None

    @property
    def active_provider(self) -> str:
        """Return the resolved provider name (always 'openai')."""

        return "openai"

    @property
    def dimension(self) -> int:
        """Return the expected vector dimension for the active provider."""

        return _OPENAI_EMBEDDING_DIM

    def embed(self, text: str) -> list[float]:
        """Embed a single text via OpenAI."""

        started_at = time.perf_counter()
        with trace_span("rag.embed") as span:
            logger.info("embeddings.embed.start")
            result = self._call_openai([text])
            span.set_attribute("rag.embed.provider", "openai")
            span.set_attribute("rag.embed.vector_dim", len(result[0]))
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            span.set_attribute("rag.duration_ms", duration_ms)
            logger.info("embeddings.embed.complete", duration_ms=duration_ms)
            return result[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, returning one vector per input."""

        started_at = time.perf_counter()
        with trace_span("rag.embed_batch") as span:
            logger.info("embeddings.batch.start", batch_size=len(texts))
            span.set_attribute("rag.embed.batch_size", len(texts))
            if not texts:
                span.set_attribute("rag.embed.provider", "openai")
                span.set_attribute("rag.duration_ms", int((time.perf_counter() - started_at) * 1000))
                logger.info("embeddings.batch.complete", batch_size=0, duration_ms=int((time.perf_counter() - started_at) * 1000))
                return []
            result = self._call_openai(texts)
            span.set_attribute("rag.embed.provider", "openai")
            span.set_attribute("rag.duration_ms", int((time.perf_counter() - started_at) * 1000))
            logger.info("embeddings.batch.complete", batch_size=len(texts), duration_ms=int((time.perf_counter() - started_at) * 1000))
            return result

    def _call_openai(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API, batching if needed."""

        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._config.openai_api_key)

        batch_size = self._config.embedding_batch_size
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self._config.openai_embedding_model,
                input=batch,
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            results.extend(list(d.embedding) for d in sorted_data)
        return results
