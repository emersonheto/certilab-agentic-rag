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
