from __future__ import annotations

from app.tools.embeddings import EmbeddingsProvider, EmbeddingProviderConfig


def test_missing_openai_api_key_triggers_local_fallback() -> None:
    """When EMBEDDING_PROVIDER=auto/openai but no API key, fall back to local."""
    config = EmbeddingProviderConfig(
        provider="auto",
        openai_api_key=None,
        openai_embedding_model="text-embedding-3-small",
        sentence_transformers_model="all-MiniLM-L6-v2",
    )

    provider = EmbeddingsProvider(config)

    assert provider.active_provider == "local"


def test_openai_provider_selected_when_key_present() -> None:
    """When API key is present and provider is auto/openai, use OpenAI."""
    config = EmbeddingProviderConfig(
        provider="auto",
        openai_api_key="sk-test-key-123",
        openai_embedding_model="text-embedding-3-small",
        sentence_transformers_model="all-MiniLM-L6-v2",
    )

    provider = EmbeddingsProvider(config)

    assert provider.active_provider == "openai"


def test_explicit_local_provider_ignores_openai_key() -> None:
    """When EMBEDDING_PROVIDER=local, always use local even if OpenAI key exists."""
    config = EmbeddingProviderConfig(
        provider="local",
        openai_api_key="sk-test-key-123",
        openai_embedding_model="text-embedding-3-small",
        sentence_transformers_model="all-MiniLM-L6-v2",
    )

    provider = EmbeddingsProvider(config)

    assert provider.active_provider == "local"


def test_embed_returns_zero_vector_when_all_providers_fail() -> None:
    """Deterministic zero-vector fallback when both OpenAI and local fail."""
    config = EmbeddingProviderConfig(
        provider="local",
        openai_api_key=None,
        openai_embedding_model="text-embedding-3-small",
        sentence_transformers_model="all-MiniLM-L6-v2",
    )

    provider = EmbeddingsProvider(config)
    # Simulate local model failure by patching _embed_local to raise
    provider._local_engine = None
    provider._local_error = RuntimeError("model not available")

    vector = provider.embed("test text")

    assert isinstance(vector, list)
    assert all(v == 0.0 for v in vector)
    assert len(vector) > 0


def test_embed_deterministic_for_same_input() -> None:
    """Same input must produce the same vector (no randomness in fallback)."""
    config = EmbeddingProviderConfig(
        provider="local",
        openai_api_key=None,
        openai_embedding_model="text-embedding-3-small",
        sentence_transformers_model="all-MiniLM-L6-v2",
    )

    provider = EmbeddingsProvider(config)
    provider._local_engine = None
    provider._local_error = RuntimeError("no model")

    v1 = provider.embed("hello world")
    v2 = provider.embed("hello world")

    assert v1 == v2


def test_from_settings_builds_config() -> None:
    from app.config import Settings

    settings = Settings(_env_file=None)
    config = EmbeddingProviderConfig.from_settings(settings)

    assert config.provider == "auto"
    assert config.openai_embedding_model == "text-embedding-3-small"
    assert config.sentence_transformers_model == "all-MiniLM-L6-v2"
