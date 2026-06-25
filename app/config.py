from functools import lru_cache
import os
from pathlib import Path
from typing import Any
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.retrieval.constants import DEFAULT_RETRIEVAL_TOP_K


DEFAULT_OPERATOR_ENV_FILE = Path.home() / ".config" / "certilab-agentic-rag" / ".env"
DOTENV_DISABLE_VALUES = {"1", "true", "yes", "on"}


def resolve_env_file() -> Path | None:
    """Return the external operator env file to load, if dotenv loading is enabled."""

    if os.getenv("CERTILAB_RAG_DISABLE_DOTENV", "").casefold() in DOTENV_DISABLE_VALUES:
        return None

    configured_env_file = os.getenv("CERTILAB_RAG_ENV_FILE")
    if configured_env_file:
        return Path(configured_env_file).expanduser()

    if DEFAULT_OPERATOR_ENV_FILE.exists():
        return DEFAULT_OPERATOR_ENV_FILE

    return None


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=None, env_file_encoding="utf-8", extra="ignore")

    def __init__(self, **values: Any) -> None:
        if "_env_file" not in values:
            values["_env_file"] = resolve_env_file()
        super().__init__(**values)

    app_name: str = "Certilab Agentic RAG"
    environment: str = "local"
    app_mode: Literal["mock", "real"] = "mock"
    graph_engine: Literal["deterministic", "langgraph"] = "langgraph"
    data_dir: Path = Field(default=Path("data"))
    default_top_k: int = DEFAULT_RETRIEVAL_TOP_K

    demo_admin_token: str | None = None
    demo_technician_token: str | None = None
    demo_client_101_token: str | None = None
    demo_client_202_token: str | None = None
    chainlit_demo_token: str | None = None

    # Real-mode API key authentication. Required only when APP_MODE=real;
    # the X-API-Key header is validated against these operator-issued secrets.
    api_key_admin: str | None = None
    api_key_technician: str | None = None
    api_key_client: str | None = None

    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    tavily_api_key: str | None = None
    qdrant_url: str | None = None
    qdrant_collection: str = "certilab-rag"
    qdrant_api_key: str | None = None
    embedding_provider: Literal["auto", "openai", "local"] = "auto"
    sentence_transformers_model: str = "all-MiniLM-L6-v2"
    mysql_readonly_dsn: str | None = None
    db_host: str | None = None
    db_port: int = 3306
    db_database: str | None = None
    db_username: str | None = None
    db_password: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str | None = Field(default=None, validation_alias=AliasChoices("AWS_REGION", "AWS_DEFAULT_REGION"))
    s3_bucket_name: str | None = Field(default=None, validation_alias=AliasChoices("S3_BUCKET_NAME", "AWS_BUCKET"))
    aws_storage_prefix: str | None = None
    certificates_storage_disk: str | None = None

    phoenix_enabled: bool = False
    phoenix_project_name: str = "certilab-agentic-rag"
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings for dependency injection."""

    return Settings()
