"""Off-topic detection guards extracted from the main pipeline.

These constants and the ``_check_off_topic`` function were originally defined
in ``app.graph`` and are extracted here for reuse across pipeline variants.
Both files keep the same definitions temporarily (PR 4 will remove them from
``app.graph``).
"""

from __future__ import annotations

import re


# ── Off-topic detection ─────────────────────────────────────────────────

_OFF_TOPIC_GREETINGS: set[str] = {
    "hola", "hi", "hello", "buenos días", "buenas tardes",
    "buenas noches", "buen día", "saludos", "hey",
}

_OFF_TOPIC_KEYWORDS: set[str] = {
    "clima", "tiempo atmosférico", "deporte",
    "fútbol", "noticias", "política",
}

_OFF_TOPIC_MATH_RE = re.compile(r"^\s*[\d\s\+\-\*\/\(\)]+\s*$")

_DOMAIN_KEYWORDS: set[str] = {
    "certificado", "cliente", "estado", "emitido", "pendiente",
    "código", "laboratorio", "certificate", "customer",
    "muestra", "análisis", "analisis", "resultado",
}

OFF_TOPIC_ANSWER = (
    "Hola, soy el asistente de Certilab. Solo puedo ayudarte con consultas "
    "sobre certificados, estados, clientes y documentos del sistema. "
    "¿En qué te puedo ayudar?"
)


def _check_off_topic(question: str) -> bool:
    """Return True if the question is off-topic for Certilab assistant.

    Detects greetings, off-domain topics, and pure math expressions.
    If the question ALSO contains any domain keyword, it is let through
    (mixed-message guard).
    """
    normalized = question.lower().strip()

    # Mixed-message guard: if any domain keyword is present, let through
    for keyword in _DOMAIN_KEYWORDS:
        if keyword in normalized:
            return False

    # Pure math (no domain keywords — guard already passed above)
    if _OFF_TOPIC_MATH_RE.match(question):
        return True

    # Greetings (exact match — full question is just a greeting)
    if normalized in _OFF_TOPIC_GREETINGS:
        return True

    # Off-domain keywords (substring — "clima" in "qué es el clima hoy")
    for keyword in _OFF_TOPIC_KEYWORDS:
        if keyword in normalized:
            return True

    return False
