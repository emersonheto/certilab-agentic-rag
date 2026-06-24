from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.domain.models import RetrievedSource
from app.security.payload_sanitizer import summarize_sources_for_external_payload


@dataclass(frozen=True)
class OpenAIClientConfig:
    """OpenAI settings for future real embedding and generation paths."""

    api_key: str | None
    embedding_model: str
    chat_model: str

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAIClientConfig:
        """Build OpenAI config without requiring credentials in mock mode."""

        return cls(
            api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            chat_model=settings.openai_chat_model,
        )

    def require_api_key(self) -> str:
        """Return the API key only when a real LLM path explicitly needs it."""

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required only when using real OpenAI embeddings or chat generation.")
        return self.api_key


class OpenAIClientAdapter:
    """Lazy OpenAI SDK adapter for future real LLM integrations."""

    def __init__(self, config: OpenAIClientConfig) -> None:
        self._config = config
        self._client: Any | None = None

    @property
    def embedding_model(self) -> str:
        return self._config.embedding_model

    @property
    def chat_model(self) -> str:
        return self._config.chat_model

    def client(self) -> Any:
        """Build the OpenAI SDK client lazily and only for real LLM paths."""

        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Real OpenAI mode requires the optional 'openai' package.") from exc
        return OpenAI(api_key=self._config.require_api_key())


class AnswerGenerator:
    """Generate final answers with OpenAI when available, otherwise preserve deterministic output."""

    def __init__(self, settings: Settings, adapter: OpenAIClientAdapter | None = None) -> None:
        self._settings = settings
        config = OpenAIClientConfig.from_settings(settings)
        self._adapter = adapter or OpenAIClientAdapter(config)

    def generate(self, question: str, sources: list[RetrievedSource], fallback_answer: str) -> str:
        """Return an OpenAI answer only for explicitly configured real mode.

        The prompt uses sanitized snippets and source metadata only. If the SDK,
        credentials, or model call are unavailable, the deterministic answer is
        returned so mock mode and tests stay offline.
        """

        if self._settings.app_mode != "real" or not self._settings.openai_api_key:
            return fallback_answer
        try:
            response = self._adapter.client().chat.completions.create(
                model=self._adapter.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the provided Certilab context. "
                            "If context is insufficient, say that authorized context was not found."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _build_generation_prompt(question, sources, fallback_answer),
                    },
                ],
                temperature=0.0,
            )
        except Exception:
            return fallback_answer
        content = response.choices[0].message.content if response.choices else None
        return content.strip() if isinstance(content, str) and content.strip() else fallback_answer


def _build_generation_prompt(question: str, sources: list[RetrievedSource], fallback_answer: str) -> str:
    safe_context = summarize_sources_for_external_payload(sources)
    return (
        f"Question length: {len(question)} characters\n"
        f"Deterministic draft length: {len(fallback_answer)} characters\n"
        f"Authorized context:\n{safe_context or 'No authorized sources.'}"
    )
