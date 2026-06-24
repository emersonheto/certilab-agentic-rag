"""Tests for the real-mode X-API-Key authentication adapter.

The valid-key derivation is exercised at the adapter level (pure, deterministic
and independent of MySQL/Qdrant, which the real pipeline requires). The HTTP
401 behaviour and the mock-mode regression are exercised through the `/ask`
endpoint, where the auth dependency runs before the pipeline is built.
"""

from collections.abc import Iterator
from typing import cast

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.domain.models import Role
from app.security.access_control import Principal
from app.security.api_key_auth import principal_from_api_key

API_KEY_ADMIN = "admin-secret"
API_KEY_TECHNICIAN = "tech-secret"
API_KEY_CLIENT_101 = "client-101-secret"
API_KEY_CLIENT_202 = "client-202-secret"


def _real_settings() -> Settings:
    return Settings(
        app_mode="real",
        api_key_admin=API_KEY_ADMIN,
        api_key_technician=API_KEY_TECHNICIAN,
        api_key_client_101=API_KEY_CLIENT_101,
        api_key_client_202=API_KEY_CLIENT_202,
    )


@pytest.fixture()
def real_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("APP_MODE", "real")
    monkeypatch.setenv("API_KEY_ADMIN", API_KEY_ADMIN)
    monkeypatch.setenv("API_KEY_TECHNICIAN", API_KEY_TECHNICIAN)
    monkeypatch.setenv("API_KEY_CLIENT_101", API_KEY_CLIENT_101)
    monkeypatch.setenv("API_KEY_CLIENT_202", API_KEY_CLIENT_202)

    from app.main import create_app

    get_settings.cache_clear()
    client = TestClient(create_app())
    try:
        yield client
    finally:
        get_settings.cache_clear()


@pytest.fixture()
def demo_client_101_token() -> str:
    import os

    return cast(str, os.environ["DEMO_CLIENT_101_TOKEN"])


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [
        (API_KEY_ADMIN, Principal(role=Role.ADMIN, customer_id=None, user_id=1)),
        (API_KEY_TECHNICIAN, Principal(role=Role.TECHNICIAN, customer_id=None, user_id=2)),
        (API_KEY_CLIENT_101, Principal(role=Role.CLIENT, customer_id=101, user_id=1010)),
        (API_KEY_CLIENT_202, Principal(role=Role.CLIENT, customer_id=202, user_id=2020)),
    ],
)
def test_valid_api_key_derives_principal(api_key: str, expected: Principal) -> None:
    assert principal_from_api_key(api_key, _real_settings()) == expected


def test_invalid_api_key_rejected_by_adapter() -> None:
    with pytest.raises(ValueError):
        principal_from_api_key("definitely-not-a-real-key", _real_settings())


def test_real_mode_rejects_missing_api_key(real_client: TestClient) -> None:
    response = real_client.post("/ask", json={"question": "¿Cuántos certificados hay?"})
    assert response.status_code == 401


def test_real_mode_rejects_invalid_api_key(real_client: TestClient) -> None:
    response = real_client.post(
        "/ask",
        headers={"X-API-Key": "definitely-not-a-real-key"},
        json={"question": "¿Cuántos certificados hay?"},
    )
    assert response.status_code == 401


def test_real_mode_ignores_demo_token(real_client: TestClient, demo_client_101_token: str) -> None:
    """In real mode the demo adapter is not wired: a demo token must not authenticate."""

    response = real_client.post(
        "/ask",
        headers={"X-Demo-Token": demo_client_101_token},
        json={"question": "¿Cuántos certificados hay?"},
    )
    assert response.status_code == 401


def test_mock_mode_still_uses_demo_adapter(demo_client_101_token: str) -> None:
    """Regression: APP_MODE=mock keeps the demo adapter and its X-Demo-Token header."""

    from app.main import create_app

    get_settings.cache_clear()
    client = TestClient(create_app())
    response = client.post(
        "/ask",
        headers={"X-Demo-Token": demo_client_101_token},
        json={"question": "¿Cuántos certificados hay?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "structured"
    assert {source["customer_id"] for source in body["sources"]} == {101}
