from pathlib import Path

from app.domain.models import Role
from app.graph import CertilabRagPipeline
from app.security.access_control import Principal


ADMIN = Principal(role=Role.ADMIN, customer_id=None, user_id=1)
TECHNICIAN = Principal(role=Role.TECHNICIAN, customer_id=None, user_id=2)
CLIENT_101 = Principal(role=Role.CLIENT, customer_id=101, user_id=1010)
CLIENT_202 = Principal(role=Role.CLIENT, customer_id=202, user_id=2020)


def test_structured_retrieval_respects_client_tenant() -> None:
    pipeline = CertilabRagPipeline(Path("data"))
    response = pipeline.ask("¿Cuántos certificados hay?", CLIENT_101)
    assert response.route == "structured"
    assert response.sources
    assert {source.customer_id for source in response.sources} == {101}
    assert "2 certificados" in response.answer


def test_semantic_retrieval_returns_pdf_source() -> None:
    pipeline = CertilabRagPipeline(Path("data"))
    response = pipeline.ask("Resumen del procedimiento técnico de temperatura", ADMIN)
    assert response.route == "semantic"
    assert response.sources
    assert response.sources[0].source_type == "pdf_text"


def test_combined_route_includes_metadata_and_pdf_sources() -> None:
    pipeline = CertilabRagPipeline(Path("data"))
    response = pipeline.ask("Estado del certificado y resumen del procedimiento", TECHNICIAN)
    assert response.route == "combined"
    assert {source.source_type for source in response.sources} >= {"metadata", "pdf_text"}


def test_semantic_retrieval_respects_client_tenant() -> None:
    pipeline = CertilabRagPipeline(Path("data"))
    response = pipeline.ask("Resumen del procedimiento técnico y observaciones internas", CLIENT_101)
    assert response.route == "semantic"
    assert response.sources
    assert {source.customer_id for source in response.sources} == {101}


def test_semantic_retrieval_does_not_fallback_for_out_of_scope_code() -> None:
    pipeline = CertilabRagPipeline(Path("data"))
    response = pipeline.ask("Resumen del procedimiento técnico de temperatura CERT-2025-002", CLIENT_101)
    assert response.route == "semantic"
    assert response.sources == []


def test_combined_retrieval_respects_client_tenant() -> None:
    pipeline = CertilabRagPipeline(Path("data"))
    response = pipeline.ask("Estado del certificado y resumen del procedimiento de temperatura", CLIENT_101)
    assert response.route == "combined"
    assert response.sources
    assert {source.customer_id for source in response.sources} == {101}


def test_structured_code_lookup_returns_visible_certificate() -> None:
    pipeline = CertilabRagPipeline(Path("data"))
    response = pipeline.ask("Estado del certificado CERT-2025-002", CLIENT_202)
    assert response.route == "structured"
    assert [source.code for source in response.sources] == ["CERT-2025-002"]
    assert "CERT-2025-002" in response.answer


def test_structured_code_lookup_does_not_return_out_of_scope_certificate() -> None:
    pipeline = CertilabRagPipeline(Path("data"))
    response = pipeline.ask("Estado del certificado CERT-2025-002", CLIENT_101)
    assert response.route == "structured"
    assert response.sources == []
    assert "autorizado" in response.answer


def test_structured_code_lookup_does_not_fallback_for_unknown_certificate() -> None:
    pipeline = CertilabRagPipeline(Path("data"))
    response = pipeline.ask("Estado del certificado CERT-2025-999", ADMIN)
    assert response.route == "structured"
    assert response.sources == []
    assert "No se encontró" in response.answer


def test_response_sources_use_sanitized_source_ids() -> None:
    pipeline = CertilabRagPipeline(Path("data"))
    response = pipeline.ask("Resumen del procedimiento técnico de temperatura", ADMIN)
    assert response.sources
    assert all("data/" not in source.source_id for source in response.sources)
