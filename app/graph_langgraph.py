"""Backward-compatibility shim. Use ``app.pipeline.langgraph`` for new code."""

from app.pipeline.langgraph import CertilabLangGraphPipeline

__all__ = ["CertilabLangGraphPipeline"]
