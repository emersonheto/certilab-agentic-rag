import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> TestClient:
    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    return TestClient(create_app())


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_client_scope(client: TestClient, demo_client_101_token: str) -> None:
    response = client.post(
        "/ask",
        headers={"X-Demo-Token": demo_client_101_token},
        json={"question": "¿Cuántos certificados hay?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "structured"
    assert {source["customer_id"] for source in body["sources"]} == {101}
    assert all("data/" not in source["source_id"] for source in body["sources"])


def test_ask_endpoint_rejects_missing_demo_token(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "¿Cuántos certificados hay?"})
    assert response.status_code == 401


def test_ask_endpoint_rejects_invalid_demo_token(client: TestClient) -> None:
    response = client.post(
        "/ask",
        headers={"X-Demo-Token": "not-a-demo-token"},
        json={"question": "¿Cuántos certificados hay?"},
    )
    assert response.status_code == 401


def test_ask_endpoint_rejects_caller_controlled_scope_fields(
    client: TestClient, demo_client_101_token: str
) -> None:
    response = client.post(
        "/ask",
        headers={"X-Demo-Token": demo_client_101_token},
        json={"question": "¿Cuántos certificados hay?", "role": "admin", "customer_id": 202, "user_id": 999},
    )
    assert response.status_code == 422


def test_reindex_endpoint_admin_allowed(client: TestClient, demo_admin_token: str) -> None:
    """Admin role can call /admin/reindex and gets mock-mode response."""
    response = client.post("/admin/reindex", headers={"X-Demo-Token": demo_admin_token})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed"] == 0
    assert body["skipped"] == 0
    assert isinstance(body["duration_ms"], (int, float))
    assert "note" in body


def test_reindex_endpoint_technician_denied(client: TestClient, demo_technician_token: str) -> None:
    """Technician role gets 403 on /admin/reindex."""
    response = client.post("/admin/reindex", headers={"X-Demo-Token": demo_technician_token})
    assert response.status_code == 403


def test_reindex_endpoint_client_denied(client: TestClient, demo_client_101_token: str) -> None:
    """Client role gets 403 on /admin/reindex."""
    response = client.post("/admin/reindex", headers={"X-Demo-Token": demo_client_101_token})
    assert response.status_code == 403


def test_reindex_endpoint_requires_auth(client: TestClient) -> None:
    """Unauthenticated call to /admin/reindex gets 401."""
    response = client.post("/admin/reindex")
    assert response.status_code == 401
