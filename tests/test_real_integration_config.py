from pathlib import Path

import pytest

from app.config import Settings, resolve_env_file
from app.tools.mysql_connector import (
    ALLOWED_SELECT_QUERIES,
    SENSITIVE_COLUMN_NAMES,
    MySQLCertificateConnector,
    MySQLConnectorConfig,
)
from app.tools.openai_client import OpenAIClientConfig
from app.tools.s3_loader import S3LoaderConfig


def test_settings_default_to_mock_mode_without_real_credentials() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_mode == "mock"
    assert settings.openai_api_key is None


def test_settings_load_configured_external_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "operator.env"
    env_file.write_text("APP_MODE=real\nOPENAI_API_KEY=test-openai-key\n", encoding="utf-8")
    monkeypatch.delenv("CERTILAB_RAG_DISABLE_DOTENV", raising=False)
    monkeypatch.setenv("CERTILAB_RAG_ENV_FILE", str(env_file))

    settings = Settings()

    assert resolve_env_file() == env_file
    assert settings.app_mode == "real"
    assert settings.openai_api_key == "test-openai-key"


def test_settings_skip_dotenv_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "operator.env"
    env_file.write_text("APP_MODE=real\nOPENAI_API_KEY=test-openai-key\n", encoding="utf-8")
    monkeypatch.setenv("CERTILAB_RAG_DISABLE_DOTENV", "true")
    monkeypatch.setenv("CERTILAB_RAG_ENV_FILE", str(env_file))

    settings = Settings()

    assert resolve_env_file() is None
    assert settings.app_mode == "mock"
    assert settings.openai_api_key is None


def test_settings_support_laravel_mysql_and_aws_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    monkeypatch.setenv("DB_PORT", "3307")
    monkeypatch.setenv("DB_DATABASE", "certilab")
    monkeypatch.setenv("DB_USERNAME", "readonly")
    monkeypatch.setenv("DB_PASSWORD", "placeholder-password")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    monkeypatch.setenv("AWS_BUCKET", "certificates-bucket")
    monkeypatch.setenv("AWS_STORAGE_PREFIX", "certificates/pdfs")
    monkeypatch.setenv("CERTIFICATES_STORAGE_DISK", "s3")

    settings = Settings(_env_file=None)

    assert MySQLConnectorConfig.from_settings(settings).connection_kwargs() == {
        "host": "127.0.0.1",
        "port": 3307,
        "database": "certilab",
        "user": "readonly",
        "password": "placeholder-password",
        "charset": "utf8mb4",
        "read_timeout": 10,
        "write_timeout": 10,
    }
    s3_config = S3LoaderConfig.from_settings(settings)
    assert s3_config.region == "us-east-2"
    assert s3_config.bucket_name == "certificates-bucket"
    assert s3_config.prefix == "certificates/pdfs"
    assert s3_config.storage_disk == "s3"


def test_openai_config_does_not_require_key_until_real_llm_path() -> None:
    settings = Settings(_env_file=None)
    config = OpenAIClientConfig.from_settings(settings)

    assert config.embedding_model == "text-embedding-3-small"
    assert config.chat_model == "gpt-4o-mini"
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        config.require_api_key()


def test_mysql_allowlisted_queries_are_select_only_and_exclude_sensitive_columns() -> None:
    connector = MySQLCertificateConnector(MySQLConnectorConfig(host="localhost", database="db", username="u", password="p"))

    assert connector.allowed_queries == ALLOWED_SELECT_QUERIES
    for query in ALLOWED_SELECT_QUERIES.values():
        normalized = " ".join(query.lower().split())
        selected_columns = normalized.split(" from ", maxsplit=1)[0].replace("select ", "").replace(" ", "").split(",")
        assert normalized.startswith("select ")
        assert ";" not in normalized
        assert SENSITIVE_COLUMN_NAMES.isdisjoint(selected_columns)


def test_mysql_connection_config_requires_complete_real_db_settings() -> None:
    config = MySQLConnectorConfig(host="localhost", database="certilab", username="readonly")

    with pytest.raises(ValueError, match="DB_PASSWORD"):
        config.connection_kwargs()
