import re
from typing import Literal

Route = Literal["structured", "semantic", "combined", "web_search"]

STRUCTURED_TERMS = {
    "certificado",
    "certificados",
    "estado",
    "fecha",
    "emitido",
    "emitidos",
    "cliente",
    "clientes",
    "cuántos",
    "cuantos",
    "cantidad",
    "vigente",
    "pendiente",
}
SEMANTIC_TERMS = {
    "resumen",
    "contenido",
    "procedimiento",
    "procedimientos",
    "técnico",
    "tecnico",
    "calibración",
    "calibracion",
    "observación",
    "observacion",
    "pdf",
    "informe",
}
WEB_SEARCH_TERMS = {
    "actualidad",
    "externa",
    "externas",
    "externo",
    "internet",
    "publico",
    "público",
    "tavily",
    "web",
}


def route_question(question: str) -> Route:
    """Classify the question into the simplest deterministic route."""

    tokens = set(re.findall(r"[a-záéíóúñ]+", question.lower()))
    if tokens & WEB_SEARCH_TERMS:
        return "web_search"
    has_structured = bool(tokens & STRUCTURED_TERMS)
    has_semantic = bool(tokens & SEMANTIC_TERMS)
    if has_structured and has_semantic:
        return "combined"
    if has_semantic:
        return "semantic"
    return "structured"
