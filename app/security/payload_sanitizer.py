from __future__ import annotations

import re
from collections import Counter

from app.domain.models import RetrievedSource


MAX_SAFE_SNIPPET_CHARS = 160
MAX_OPENAI_SOURCES = 6
MAX_TAVILY_QUERY_CHARS = 160

REDACTED_CERTIFICATE = "[REDACTED_CERTIFICATE_CODE]"
REDACTED_SECRET = "[REDACTED_SECRET]"
REDACTED_LOCATION = "[REDACTED_LOCATION]"
REDACTED_CUSTOMER = "[REDACTED_CUSTOMER]"

CERTIFICATE_CODE_PATTERN = re.compile(r"\bCERT-\d{4}-\d{3,}\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"\b(?:https?|s3|mysql|postgres(?:ql)?|redis|mongodb)://\S+", re.IGNORECASE)
DSN_PATTERN = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s]+", re.IGNORECASE)
SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"\b(?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?key|secret[_-]?key)\s*[:=]\s*[^\s,;]+",
    re.IGNORECASE,
)
OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
BEARER_TOKEN_PATTERN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE)
STORAGE_PATH_PATTERN = re.compile(r"(?:\bdata/|\bs3://|/[^\s]+|[A-Za-z]:\\)[^\s,;]*", re.IGNORECASE)
CUSTOMER_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:customer|cliente|client|tenant|empresa)\s*[:#-]?\s*[A-Za-z0-9_-]+",
    re.IGNORECASE,
)


def sanitize_text_for_external_payload(value: str) -> str:
    """Redact identifiers and secrets before text leaves the local trust boundary."""

    sanitized = CERTIFICATE_CODE_PATTERN.sub(REDACTED_CERTIFICATE, value)
    sanitized = SECRET_ASSIGNMENT_PATTERN.sub(REDACTED_SECRET, sanitized)
    sanitized = OPENAI_KEY_PATTERN.sub(REDACTED_SECRET, sanitized)
    sanitized = BEARER_TOKEN_PATTERN.sub(REDACTED_SECRET, sanitized)
    sanitized = URL_PATTERN.sub(REDACTED_LOCATION, sanitized)
    sanitized = DSN_PATTERN.sub(REDACTED_LOCATION, sanitized)
    sanitized = STORAGE_PATH_PATTERN.sub(REDACTED_LOCATION, sanitized)
    sanitized = CUSTOMER_IDENTIFIER_PATTERN.sub(REDACTED_CUSTOMER, sanitized)
    return " ".join(sanitized.split())


def minimize_snippet_for_external_payload(snippet: str) -> str:
    """Return the smallest useful sanitized snippet for LLM context."""

    return sanitize_text_for_external_payload(snippet)[:MAX_SAFE_SNIPPET_CHARS]


def summarize_sources_for_external_payload(sources: list[RetrievedSource]) -> str:
    """Build count-first source context without raw customer IDs or certificate codes."""

    type_counts = Counter(source.source_type for source in sources)
    type_summary = ", ".join(f"{source_type}={count}" for source_type, count in sorted(type_counts.items()))
    lines = [
        f"Source count: {len(sources)}",
        f"Certificate-bearing source count: {_certificate_bearing_source_count(sources)}",
        f"Source types: {type_summary or 'none'}",
    ]
    for index, source in enumerate(sources[:MAX_OPENAI_SOURCES], start=1):
        snippet = minimize_snippet_for_external_payload(source.snippet)
        lines.append(f"- source_{index}: type={source.source_type}; snippet={snippet or '[empty]'}")
    return "\n".join(lines)


def should_block_external_search(query: str) -> bool:
    """Prevent private or customer-specific material from reaching public web search."""

    return any(
        pattern.search(query)
        for pattern in (
            CERTIFICATE_CODE_PATTERN,
            URL_PATTERN,
            DSN_PATTERN,
            SECRET_ASSIGNMENT_PATTERN,
            OPENAI_KEY_PATTERN,
            BEARER_TOKEN_PATTERN,
            STORAGE_PATH_PATTERN,
            CUSTOMER_IDENTIFIER_PATTERN,
        )
    )


def sanitize_query_for_external_search(query: str) -> str:
    """Sanitize and cap public web-search queries."""

    return sanitize_text_for_external_payload(query)[:MAX_TAVILY_QUERY_CHARS]


def _certificate_bearing_source_count(sources: list[RetrievedSource]) -> int:
    return sum(1 for source in sources if source.certificate_id is not None or source.code is not None)
