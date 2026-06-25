"""Tests for ``app.pipeline.protocols``.

Strict TDD: tests written before implementation.
"""

from __future__ import annotations

from app.pipeline.protocols import RagPipeline


def test_rag_pipeline_protocol_importable() -> None:
    """RagPipeline protocol can be imported from app.pipeline.protocols."""
    assert callable(RagPipeline.ask)


def test_rag_pipeline_is_protocol() -> None:
    """RagPipeline is a Protocol (not a concrete class)."""
    from typing import Protocol
    assert issubclass(RagPipeline, Protocol)


def test_rag_pipeline_has_ask_and_reindex() -> None:
    """RagPipeline defines ask() and reindex() methods."""
    import inspect
    sig_ask = inspect.signature(RagPipeline.ask)
    sig_reindex = inspect.signature(RagPipeline.reindex)
    assert "question" in sig_ask.parameters
    assert "principal" in sig_ask.parameters
    assert list(sig_reindex.parameters.keys()) == ["self"]  # only self


def test_rag_pipeline_old_import_still_works() -> None:
    """The old import from app.graph still works."""
    from app.graph import RagPipeline as OldRagPipeline  # noqa: F811
    assert callable(OldRagPipeline.ask)
