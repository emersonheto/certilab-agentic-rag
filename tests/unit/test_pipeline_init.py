"""Tests for ``app.pipeline`` package init.

Strict TDD: test written before implementation.
"""

from __future__ import annotations


def test_pipeline_package_has_docstring() -> None:
    """The ``app.pipeline`` package has a proper docstring."""
    import app.pipeline  # noqa: F811

    doc = app.pipeline.__doc__
    assert doc is not None
    assert "Pipeline package" in doc
