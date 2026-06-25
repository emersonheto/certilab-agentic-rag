"""Backward-compatibility shim. Use ``app.pipeline.*`` for new code."""

from app.pipeline.deterministic import CertilabRagPipeline
from app.pipeline.factory import build_rag_pipeline
from app.pipeline.guards import (
    OFF_TOPIC_ANSWER,
    _DOMAIN_KEYWORDS,
    _OFF_TOPIC_GREETINGS,
    _OFF_TOPIC_KEYWORDS,
    _OFF_TOPIC_MATH_RE,
    _check_off_topic,
)
from app.pipeline.protocols import RagPipeline
from app.pipeline.utils import (
    _certificate_count,
    _dedupe_sources,
    _elapsed_ms,
    _scope_attributes,
    _source_id,
    _to_schema,
    _web_sources,
)

__all__ = [
    "RagPipeline",
    "CertilabRagPipeline",
    "build_rag_pipeline",
    "_check_off_topic",
    "OFF_TOPIC_ANSWER",
    "_OFF_TOPIC_GREETINGS",
    "_OFF_TOPIC_KEYWORDS",
    "_OFF_TOPIC_MATH_RE",
    "_DOMAIN_KEYWORDS",
    "_dedupe_sources",
    "_to_schema",
    "_source_id",
    "_web_sources",
    "_elapsed_ms",
    "_scope_attributes",
    "_certificate_count",
]
