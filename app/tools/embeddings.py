from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.config import Settings

ProviderName = Literal["openai", "local"]

# OpenAI text-embedding-3-small output dimension.
_OPENAI_EMBEDDING_DIM = 1536
# sentence-transformers all-MiniLM-L6-v2 output dimension.
_LOCAL_EMBEDDING_DIM = 384


@dataclass(frozen=True)
class EmbeddingProviderConfig:
    """Configuration for embedding provider selection and model names."""

    provider: Literal["auto", "openai", "local"]
    openai_api_key: str | None
    openai_embedding_model: str
    sentence_transformers_model: str

    @classmethod
    def from_settings(cls, settings: Settings) -> EmbeddingProviderConfig:
        """Build provider config from application settings."""

        return cls(
            provider=settings.embedding_provider,
            openai_api_key=settings.openai_api_key,
            openai_embedding_model=settings.openai_embedding_model,
            sentence_transformers_model=settings.sentence_transformers_model,
        )


class EmbeddingsProvider:
    """Embedding provider with OpenAI default and deterministic local fallback.

    Provider selection is driven by config:
    - ``openai``: use OpenAI if API key is present; otherwise fall back to local.
    - ``local``: always use sentence-transformers (offline).
    - ``auto`` (default): OpenAI when key present, local otherwise.

    If the local provider also fails (package not installed, model unavailable),
    a deterministic zero-vector is returned so the pipeline degrades gracefully
    rather than crashing.

    Security notes:
    - The provider never receives PII columns. Only allowlisted text
      constructed by the loader is passed to ``embed``. PII (password, ruc,
      email, phone) is excluded at the connector/loader level.
    - Provider selection is deterministic: the same config always resolves to
      the same provider, with no randomness in the fallback path.
    - When all providers fail, a deterministic zero vector is returned so the
      pipeline degrades gracefully rather than crashing or leaking partial data.
    """

    def __init__(self, config: EmbeddingProviderConfig) -> None:
        self._config = config
        self._active_provider: ProviderName = self._resolve_provider()
        self._local_engine: Any = None
        self._local_error: Exception | None = None
        self._openai_client: Any = None
        if self._active_provider == "local":
            self._init_local()

    @property
    def active_provider(self) -> str:
        """Return the resolved provider name ('openai' or 'local')."""

        return self._active_provider

    @property
    def dimension(self) -> int:
        """Return the expected vector dimension for the active provider."""

        if self._active_provider == "openai":
            return _OPENAI_EMBEDDING_DIM
        if self._local_engine is not None:
            try:
                return int(self._local_engine.get_sentence_embedding_dimension())
            except Exception:
                pass
        return _LOCAL_EMBEDDING_DIM

    def embed(self, text: str) -> list[float]:
        """Embed a single text, falling back deterministically on any failure."""

        if self._active_provider == "openai":
            try:
                return self._embed_openai(text)
            except Exception:
                # Fall back to local on any OpenAI error
                pass
        return self._embed_local_or_zero(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, returning one vector per input."""

        return [self.embed(text) for text in texts]

    def _resolve_provider(self) -> ProviderName:
        if self._config.provider == "local":
            return "local"
        # openai or auto: use OpenAI only if key is present
        if self._config.provider == "openai" and not self._config.openai_api_key:
            return "local"
        if self._config.openai_api_key:
            return "openai"
        return "local"

    def _init_local(self) -> None:
        """Lazily load the sentence-transformers model."""

        try:
            from sentence_transformers import SentenceTransformer

            self._local_engine = SentenceTransformer(self._config.sentence_transformers_model)
        except ImportError:
            self._local_error = ImportError(
                "sentence-transformers is not installed. Install with: uv pip install '.[local-embeddings]'"
            )
        except Exception as exc:
            self._local_error = exc

    def _embed_openai(self, text: str) -> list[float]:
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self._config.openai_api_key)
        response = self._openai_client.embeddings.create(
            model=self._config.openai_embedding_model,
            input=text,
        )
        return list(response.data[0].embedding)

    def _embed_local_or_zero(self, text: str) -> list[float]:
        if self._local_engine is not None:
            try:
                vector = self._local_engine.encode(text)
                return list(vector)
            except Exception:
                pass
        return self._zero_vector()

    @staticmethod
    def _zero_vector() -> list[float]:
        """Return a deterministic zero vector (used when all providers fail)."""

        return [0.0] * _LOCAL_EMBEDDING_DIM
