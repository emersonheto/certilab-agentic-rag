"""Tests verifying the four new Settings knobs are actually wired into consumers.

Strict TDD: tests written before the wiring code changes. Each test verifies
that the settings value flows through to where the hardcoded magic number used
to live.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.config import Settings


# ── openai_client: temperature wiring ────────────────────────────────────────


def test_answer_generator_passes_settings_temperature() -> None:
    """AnswerGenerator uses settings.openai_temperature instead of hardcoded 0.0."""
    from app.tools.openai_client import AnswerGenerator

    settings = Settings(
        _env_file=None,
        app_mode="real",
        openai_api_key="test-key",
        openai_temperature=0.7,
    )

    captured: dict[str, object] = {}
    fake_message = MagicMock()
    fake_message.content = "Generated answer."
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_message)]

    fake_completions = MagicMock()
    fake_completions.create = lambda **kwargs: (captured.update(kwargs), fake_response)[1]
    fake_client = MagicMock()
    fake_client.chat.completions = fake_completions

    fake_adapter = MagicMock()
    fake_adapter.chat_model = "gpt-4o-mini"
    fake_adapter.client.return_value = fake_client

    generator = AnswerGenerator(settings, adapter=fake_adapter)
    generator.generate("question?", [], "fallback")

    assert captured["temperature"] == 0.7


# ── web_search: max_results wiring ───────────────────────────────────────────


def test_web_search_config_has_max_results_default() -> None:
    """WebSearchConfig exposes max_results with default 3."""
    from app.tools.web_search import WebSearchConfig

    config = WebSearchConfig(tavily_api_key=None)
    assert config.max_results == 3


def test_web_search_config_accepts_custom_max_results() -> None:
    """WebSearchConfig accepts a custom max_results from settings."""
    from app.tools.web_search import WebSearchConfig

    config = WebSearchConfig(tavily_api_key=None, max_results=5)
    assert config.max_results == 5


def test_web_search_uses_config_max_results_in_search() -> None:
    """TavilyWebSearch.search passes config.max_results to the Tavily client."""
    from app.tools.web_search import TavilyWebSearch, WebSearchConfig

    config = WebSearchConfig(tavily_api_key="fake-key", max_results=7)
    search = TavilyWebSearch(config)

    captured: dict[str, object] = {}
    fake_response = {"results": [{"title": "T", "url": "u", "content": "c"}]}
    fake_client = MagicMock()
    fake_client.search = lambda **kwargs: (captured.update(kwargs), fake_response)[1]
    search._client = fake_client

    search.search("certificado")
    assert captured["max_results"] == 7


# ── qdrant_index: upsert_batch_size wiring ───────────────────────────────────


def test_qdrant_index_constructor_accepts_batch_size() -> None:
    """QdrantVectorIndex stores a configurable upsert_batch_size."""
    from app.retrieval.qdrant_index import QdrantVectorIndex

    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    fake_embedding = MagicMock()
    fake_embedding.dimension = 384

    index = QdrantVectorIndex(
        client=fake_client,
        collection_name="test",
        embedding_provider=fake_embedding,
        vector_size=384,
        upsert_batch_size=50,
    )
    assert index._upsert_batch_size == 50


def test_qdrant_index_default_batch_size_is_200() -> None:
    """QdrantVectorIndex defaults upsert_batch_size to 200."""
    from app.retrieval.qdrant_index import QdrantVectorIndex

    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    fake_embedding = MagicMock()
    fake_embedding.dimension = 384

    index = QdrantVectorIndex(
        client=fake_client,
        collection_name="test",
        embedding_provider=fake_embedding,
        vector_size=384,
    )
    assert index._upsert_batch_size == 200


# ── embeddings: embedding_batch_size wiring ──────────────────────────────────


def test_embedding_provider_config_has_batch_size_default() -> None:
    """EmbeddingProviderConfig exposes embedding_batch_size with default 100."""
    from app.tools.embeddings import EmbeddingProviderConfig

    config = EmbeddingProviderConfig(
        openai_api_key=None,
        openai_embedding_model="text-embedding-3-small",
    )
    assert config.embedding_batch_size == 100


def test_embedding_provider_config_from_settings_reads_batch_size() -> None:
    """EmbeddingProviderConfig.from_settings picks up embedding_batch_size."""
    from app.tools.embeddings import EmbeddingProviderConfig

    settings = Settings(_env_file=None, embedding_batch_size=64)
    config = EmbeddingProviderConfig.from_settings(settings)
    assert config.embedding_batch_size == 64
