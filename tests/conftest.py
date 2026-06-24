import os
import socket
from collections.abc import Callable
from pathlib import Path

import pytest


EXTERNAL_SERVICE_ENV_KEYS = {
    "APP_MODE",
    "AWS_ACCESS_KEY_ID",
    "AWS_BUCKET",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_STORAGE_PREFIX",
    "CERTIFICATES_STORAGE_DISK",
    "DB_DATABASE",
    "DB_HOST",
    "DB_PASSWORD",
    "DB_PORT",
    "DB_USERNAME",
    "EMBEDDING_PROVIDER",
    "MYSQL_READONLY_DSN",
    "OPENAI_API_KEY",
    "PHOENIX_COLLECTOR_ENDPOINT",
    "PHOENIX_ENABLED",
    "PHOENIX_PROJECT_NAME",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_URL",
    "S3_BUCKET_NAME",
    "SENTENCE_TRANSFORMERS_MODEL",
    "TAVILY_API_KEY",
}

_INTEGRATION_MARKERS = ("requires_qdrant", "requires_mysql")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_qdrant: requires a running Qdrant instance (QDRANT_URL)")
    config.addinivalue_line("markers", "requires_mysql: requires a reachable MySQL instance (DB_HOST)")
    _force_test_env(os.environ.__setitem__)
    _clear_settings_cache()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip integration tests when the required service env var is absent."""

    skip_qdrant = pytest.mark.skip(reason="QDRANT_URL not set; skipping Qdrant integration test")
    skip_mysql = pytest.mark.skip(reason="DB_HOST not set; skipping MySQL integration test")

    qdrant_available = bool(os.environ.get("QDRANT_URL"))
    mysql_available = bool(os.environ.get("DB_HOST"))

    for item in items:
        if "requires_qdrant" in item.keywords and not qdrant_available:
            item.add_marker(skip_qdrant)
        if "requires_mysql" in item.keywords and not mysql_available:
            item.add_marker(skip_mysql)


@pytest.fixture(autouse=True)
def isolate_external_service_env(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    # Integration tests with service markers keep real env vars
    is_integration = any(request.node.get_closest_marker(m) for m in _INTEGRATION_MARKERS)
    if not is_integration:
        for key in EXTERNAL_SERVICE_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        _force_test_env(monkeypatch.setenv)
    _clear_settings_cache()
    yield
    _clear_settings_cache()


@pytest.fixture()
def demo_client_101_token() -> str:
    return os.environ["DEMO_CLIENT_101_TOKEN"]


@pytest.fixture()
def qdrant_available() -> str | None:
    """Return the Qdrant URL if a Qdrant instance is reachable, else None."""

    url = os.environ.get("QDRANT_URL")
    if not url:
        return None
    # Quick TCP reachability check
    try:
        host = url.split("//")[-1].split("/")[0].split(":")[0]
        port_str = url.split("//")[-1].split("/")[0].split(":")[1] if ":" in url.split("//")[-1].split("/")[0] else "6333"
        with socket.create_connection((host, int(port_str)), timeout=2):
            pass
    except (OSError, ValueError):
        return None
    return url


def _clear_settings_cache() -> None:
    from app.config import get_settings

    get_settings.cache_clear()


def _force_test_env(set_env: Callable[[str, str], object]) -> None:
    set_env("CERTILAB_RAG_DISABLE_DOTENV", "true")
    set_env("PHOENIX_ENABLED", "false")
    for key, value in _demo_tokens_from_example().items():
        set_env(key, value)


def _demo_tokens_from_example() -> dict[str, str]:
    tokens: dict[str, str] = {}
    for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key.startswith("DEMO_"):
            tokens[key] = value.strip('"')
    return tokens
