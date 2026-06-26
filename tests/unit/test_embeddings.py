"""Tests for the OpenAI-only embedding provider.

STRICT TDD: written before the implementation rewrite. The provider is now
OpenAI-only — there is no local/sentence-transformers fallback and no
zero-vector path. The OpenAI client call is mocked so no network/model
loading occurs and the suite stays fast (<3s).
"""

from __future__ import annotations

from typing import Any

import pytest

from app.tools.embeddings import EmbeddingProviderConfig, EmbeddingsProvider

_OPENAI_DIM = 1536


def _make_provider(monkeypatch: pytest.MonkeyPatch, *, stub: Any | None = None) -> EmbeddingsProvider:
    """Build an OpenAI-only provider with ``_call_openai`` stubbed out."""

    config = EmbeddingProviderConfig(
        openai_api_key="sk-test-key",
        openai_embedding_model="text-embedding-3-small",
    )
    provider = EmbeddingsProvider(config)
    if stub is None:
        stub = lambda texts: [[0.0] * _OPENAI_DIM for _ in texts]  # noqa: E731
    monkeypatch.setattr(provider, "_call_openai", stub)
    return provider


# ── provider identity ─────────────────────────────────────────────────────────


def test_active_provider_is_openai() -> None:
    """The only supported provider is OpenAI."""

    provider = _make_provider(pytest.MonkeyPatch())

    assert provider.active_provider == "openai"


def test_dimension_is_1536() -> None:
    """OpenAI text-embedding-3-small always emits 1536-dim vectors."""

    provider = _make_provider(pytest.MonkeyPatch())

    assert provider.dimension == _OPENAI_DIM


# ── embed / embed_batch ───────────────────────────────────────────────────────


def test_embed_calls_openai_and_returns_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed() delegates to the OpenAI call and returns its single vector."""

    seen: list[list[str]] = []
    expected = [0.1] * _OPENAI_DIM

    def fake_call(texts: list[str]) -> list[list[float]]:
        seen.append(texts)
        return [expected]

    provider = _make_provider(monkeypatch, stub=fake_call)

    result = provider.embed("hello world")

    assert result == expected
    assert seen == [["hello world"]]


def test_embed_batch_returns_correct_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_batch returns exactly one vector per input text."""

    texts = ["hello world", "test text", "third one"]

    def fake_call(batch: list[str]) -> list[list[float]]:
        return [[float(i)] * _OPENAI_DIM for i, _ in enumerate(batch)]

    provider = _make_provider(monkeypatch, stub=fake_call)

    result = provider.embed_batch(texts)

    assert len(result) == 3
    assert all(len(v) == _OPENAI_DIM for v in result)
    # order is preserved
    assert result[0][0] == 0.0
    assert result[1][0] == 1.0
    assert result[2][0] == 2.0


def test_embed_batch_empty_list_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_batch([]) short-circuits without calling OpenAI."""

    called: list[bool] = []

    def fake_call(batch: list[str]) -> list[list[float]]:
        called.append(True)
        return []

    provider = _make_provider(monkeypatch, stub=fake_call)

    result = provider.embed_batch([])

    assert result == []
    assert called == []  # _call_openai must NOT be invoked


class _FakeOpenAIClient:
    """Stand-in for the OpenAI client that records create() call shapes."""

    def __init__(self) -> None:
        self.create_inputs: list[list[str]] = []

        class _Embeddings:
            def __init__(self, outer: _FakeOpenAIClient) -> None:
                self._outer = outer

            def create(self, model: str, input: list[str]) -> Any:
                self._outer.create_inputs.append(input)

                class _Item:
                    def __init__(self, index: int) -> None:
                        self.index = index
                        self.embedding = [0.0] * _OPENAI_DIM

                class _Resp:
                    def __init__(self, items: list[_Item]) -> None:
                        self.data = items

                return _Resp([_Item(i) for i in range(len(input))])

        self.embeddings = _Embeddings(self)


def test_call_openai_chunks_by_batch_size() -> None:
    """_call_openai splits inputs into embedding_batch_size-sized requests."""

    config = EmbeddingProviderConfig(
        openai_api_key="sk-test-key",
        openai_embedding_model="text-embedding-3-small",
        embedding_batch_size=2,
    )
    provider = EmbeddingsProvider(config)
    provider._client = _FakeOpenAIClient()

    texts = ["a", "b", "c", "d", "e"]
    result = provider._call_openai(texts)

    # one vector per input, order preserved
    assert len(result) == 5
    assert all(len(v) == _OPENAI_DIM for v in result)
    # batch_size=2 → 3 client calls of sizes 2, 2, 1
    assert [len(c) for c in provider._client.create_inputs] == [2, 2, 1]


# ── config wiring ─────────────────────────────────────────────────────────────


def test_from_settings_builds_config() -> None:
    from app.config import Settings

    settings = Settings(_env_file=None, embedding_batch_size=42)
    config = EmbeddingProviderConfig.from_settings(settings)

    assert config.openai_embedding_model == "text-embedding-3-small"
    assert config.embedding_batch_size == 42


def test_config_default_batch_size() -> None:
    config = EmbeddingProviderConfig(
        openai_api_key="sk-test",
        openai_embedding_model="text-embedding-3-small",
    )

    assert config.embedding_batch_size == 100
