"""Tests for app/tools/openai_client.py — system prompt and generation prompt."""

from __future__ import annotations

from app.domain.models import RetrievedSource
from app.tools.openai_client import SYSTEM_PROMPT, _build_generation_prompt

# ─── Task 3.1: System prompt guardrails ──────────────────────────────────


def test_system_prompt_contains_reformulate() -> None:
    """The system prompt must frame the LLM as a reformulator."""
    assert "reformulate" in SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_redacted_customer() -> None:
    """The system prompt must explain [REDACTED_CUSTOMER] placeholders."""
    assert "[REDACTED_CUSTOMER]" in SYSTEM_PROMPT


def test_system_prompt_mentions_redacted_certificate_code() -> None:
    """The system prompt must explain [REDACTED_CERTIFICATE_CODE] placeholders."""
    assert "[REDACTED_CERTIFICATE_CODE]" in SYSTEM_PROMPT


def test_system_prompt_does_not_contain_reject_guardrail() -> None:
    """The system prompt must NOT contain guardrail-trigger phrases like 'reject'."""
    assert "reject" not in SYSTEM_PROMPT.lower()


def test_system_prompt_does_not_contain_lo_siento_pero() -> None:
    """The system prompt must NOT contain 'lo siento, pero' which triggers guardrails."""
    assert "lo siento" not in SYSTEM_PROMPT.lower()


def test_system_prompt_under_150_tokens() -> None:
    """The system prompt must be under 150 tokens (roughly < 600 chars for Spanish)."""
    # Rough heuristic: 1 token ≈ 4 chars for Spanish text
    assert len(SYSTEM_PROMPT) < 600, (
        f"System prompt too long: {len(SYSTEM_PROMPT)} chars "
        f"(~{len(SYSTEM_PROMPT) // 4} estimated tokens, max 150)"
    )


# ─── Task 3.1: System prompt mentions scope fact awareness ────────────────


def test_system_prompt_mentions_scope() -> None:
    """The system prompt must reference the scope fact from context."""
    assert "Scope" in SYSTEM_PROMPT or "scope" in SYSTEM_PROMPT.lower()


def test_system_prompt_contains_no_second_guess() -> None:
    """The system prompt must instruct to trust the draft, not second-guess."""
    assert "second-guess" in SYSTEM_PROMPT or "trust" in SYSTEM_PROMPT.lower()


# ─── Helpers for _build_generation_prompt tests ──────────────────────────


def _make_source(customer_id: int | None = 1) -> RetrievedSource:
    return RetrievedSource(
        certificate_id=1,
        code="CERT-2025-001",
        customer_id=customer_id,
        source_type="structured",
        path="data/certificates.json",
        snippet="Certificate for customer 1",
        score=1.0,
    )


# ─── Task 3.2: _build_generation_prompt with scope_facts ──────────────────


def test_build_generation_prompt_includes_scope_facts_when_provided() -> None:
    """The prompt must include a Scope: section when scope_facts is provided."""
    sources = [_make_source()]
    prompt = _build_generation_prompt("test question", sources, "test answer", scope_facts="customer_id=101, 1 sources retrieved")
    assert "Scope:" in prompt
    assert "customer_id=101" in prompt


def test_build_generation_prompt_omits_scope_section_when_none() -> None:
    """The prompt must NOT include a Scope: section when scope_facts is None (default)."""
    sources = [_make_source()]
    prompt = _build_generation_prompt("test question", sources, "test answer")
    assert "Scope:" not in prompt
    # Verify the rest of the structure is still correct
    assert prompt.startswith("Question: test question")
    assert "Draft answer: test answer" in prompt
    assert "Authorized context:" in prompt


def test_build_generation_prompt_scope_facts_contains_customer_id() -> None:
    """The scope_facts content must include the customer_id when provided."""
    sources = [_make_source()]
    prompt = _build_generation_prompt(
        "test question", sources, "test answer",
        scope_facts="customer_id=202, 3 sources retrieved",
    )
    assert "customer_id=202" in prompt
    assert "3 sources" in prompt


def test_build_generation_prompt_with_empty_sources_no_scope() -> None:
    """Without scope_facts and with empty sources, the prompt must still render cleanly."""
    prompt = _build_generation_prompt("test question", [], "test answer")
    assert "Scope:" not in prompt
    assert "Authorized context:" in prompt
    assert "Source count: 0" in prompt


def test_build_generation_prompt_scope_precedes_context() -> None:
    """The Scope: line must appear between Draft answer and Authorized context."""
    sources = [_make_source()]
    prompt = _build_generation_prompt("test question", sources, "test answer", scope_facts="customer_id=101, 1 sources")
    draft_idx = prompt.index("Draft answer:")
    scope_idx = prompt.index("Scope:")
    ctx_idx = prompt.index("Authorized context:")
    assert draft_idx < scope_idx < ctx_idx, (
        f"Expected order: Draft answer ({draft_idx}) < Scope ({scope_idx}) "
        f"< Authorized context ({ctx_idx})"
    )
