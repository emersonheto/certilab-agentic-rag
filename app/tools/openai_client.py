from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.domain.models import RetrievedSource
from app.logging import get_logger
from app.security.payload_sanitizer import summarize_sources_for_external_payload

logger = get_logger("tools.openai")

SYSTEM_PROMPT = (
    "You are Certilab Assistant. A deterministic pipeline has retrieved verified "
    "data and produced a draft answer. Reformulate it into a clear, natural "
    "response in the user's language.\n\n"
    "Rules:\n"
    "- Trust the draft and context as ground truth. Do not second-guess.\n"
    "- [REDACTED_CUSTOMER] and [REDACTED_CERTIFICATE_CODE] are sanitized placeholders — treat as normal.\n"
    '- The "Scope" line in context tells you the customer and result count — use for a confident answer.\n'
    "- Never invent data not present in the draft or context.\n"
    'If the draft is empty: reply "No encontré información disponible para esa consulta."'
)


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

    def generate(
        self,
        question: str,
        sources: list[RetrievedSource],
        fallback_answer: str,
        scope_facts: str | None = None,
    ) -> str:
        """Return an OpenAI answer only for explicitly configured real mode.

        The prompt uses sanitized snippets and source metadata only. If the SDK,
        credentials, or model call are unavailable, the deterministic answer is
        returned so mock mode and tests stay offline.
        """

        if self._settings.app_mode != "real" or not self._settings.openai_api_key:
            return fallback_answer
        model = self._adapter.chat_model
        try:
            logger.info("openai.generate.start", model=model)
            started = time.perf_counter()
            response = self._adapter.client().chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": _build_generation_prompt(question, sources, fallback_answer, scope_facts),
                    },
                ],
                temperature=self._settings.openai_temperature,
            )
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.info("openai.generate.complete", model=model, duration_ms=elapsed)
        except Exception:
            logger.warning("openai.generate.fallback")
            return fallback_answer
        content = response.choices[0].message.content if response.choices else None
        return content.strip() if isinstance(content, str) and content.strip() else fallback_answer


def _build_generation_prompt(
    question: str,
    sources: list[RetrievedSource],
    fallback_answer: str,
    scope_facts: str | None = None,
) -> str:
    safe_context = summarize_sources_for_external_payload(sources)
    scope_section = f"Scope: {scope_facts}\n" if scope_facts else ""
    return (
        f"Question: {question[:500]}\n"
        f"Draft answer: {fallback_answer[:500]}\n"
        f"{scope_section}"
        f"Authorized context:\n{safe_context or 'No authorized sources.'}"
    )
