from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from app.config import Settings


class DatabaseCursor(Protocol):
    def execute(self, query: str, params: tuple[object, ...]) -> object: ...

    def fetchall(self) -> list[dict[str, object]]: ...


class DatabaseConnection(Protocol):
    def cursor(self) -> Any: ...

    def close(self) -> None: ...


ALLOWED_CERTIFICATE_COLUMNS = (
    "id",
    "code",
    "case_number",
    "request_number",
    "document_type",
    "issue_date",
    "service_date",
    "customer_id",
    "user_id",
    "pdf_document_path",
    "qr_code",
    "status",
)
ALLOWED_CUSTOMER_COLUMNS = ("id", "company_name")
ALLOWED_HISTORY_COLUMNS = ("id", "certification_id", "user_id", "action", "ip", "user_agent", "created_at")
SENSITIVE_COLUMN_NAMES = {
    "password",
    "plain_password",
    "remember_token",
    "api_token",
    "ruc",
    "email",
    "phone",
}

ALLOWED_SELECT_QUERIES: dict[str, str] = {
    "certificates": """
        SELECT id, code, case_number, request_number, document_type, issue_date,
               service_date, customer_id, user_id, pdf_document_path, qr_code, status
        FROM certificates
        WHERE (%s IS NULL OR customer_id = %s)
        ORDER BY issue_date DESC, id DESC
    """,
    "customers": """
        SELECT id, company_name
        FROM customers
        ORDER BY id ASC
    """,
    "histories": """
        SELECT id, certification_id, user_id, action, ip, user_agent, created_at
        FROM certification_histories
        WHERE (%s IS NULL OR certification_id = %s)
        ORDER BY created_at DESC, id DESC
    """,
}


@dataclass(frozen=True)
class MySQLConnectorConfig:
    """Read-only MySQL connection settings for certificate metadata."""

    readonly_dsn: str | None = None
    host: str | None = None
    port: int = 3306
    database: str | None = None
    username: str | None = None
    password: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> MySQLConnectorConfig:
        """Build connector config from safe or Laravel-compatible settings."""

        return cls(
            readonly_dsn=settings.mysql_readonly_dsn,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_database,
            username=settings.db_username,
            password=settings.db_password,
        )

    def connection_kwargs(self) -> dict[str, object]:
        """Return PyMySQL keyword arguments without opening a connection."""

        if self.readonly_dsn:
            return {"dsn": self.readonly_dsn}
        missing = [
            name
            for name, value in {
                "DB_HOST": self.host,
                "DB_DATABASE": self.database,
                "DB_USERNAME": self.username,
                "DB_PASSWORD": self.password,
            }.items()
            if value in (None, "")
        ]
        if missing:
            raise ValueError(f"Missing MySQL settings for real mode: {', '.join(missing)}")
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.username,
            "password": self.password,
            "charset": "utf8mb4",
            "read_timeout": 10,
            "write_timeout": 10,
        }


class MySQLCertificateConnector:
    """Read-only connector for Laravel/MySQL certificate metadata exports."""

    def __init__(self, config: MySQLConnectorConfig) -> None:
        self._config = config

    @property
    def allowed_queries(self) -> dict[str, str]:
        """Expose query templates for tests and review without accepting arbitrary SQL."""

        return ALLOWED_SELECT_QUERIES.copy()

    def fetch_certificates(self, customer_id: int | None = None) -> list[dict[str, object]]:
        """Fetch allowlisted certificate metadata with an optional tenant filter."""

        return self._fetch_allowed("certificates", (customer_id, customer_id))

    def fetch_customers(self) -> list[dict[str, object]]:
        """Fetch allowlisted customer metadata."""

        return self._fetch_allowed("customers", ())

    def fetch_histories(self, certificate_id: int | None = None) -> list[dict[str, object]]:
        """Fetch allowlisted certificate history entries."""

        return self._fetch_allowed("histories", (certificate_id, certificate_id))

    def _fetch_allowed(self, query_name: str, params: tuple[object, ...]) -> list[dict[str, object]]:
        query = ALLOWED_SELECT_QUERIES[query_name]
        self._validate_query(query)
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
        finally:
            connection.close()
        return list(rows)

    def _connect(self) -> DatabaseConnection:
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError("Real MySQL mode requires the optional 'pymysql' package.") from exc

        kwargs = self._config.connection_kwargs()
        if "dsn" in kwargs:
            raise RuntimeError("MYSQL_READONLY_DSN is documented for operators; PyMySQL real mode uses DB_* settings.")
        return cast(DatabaseConnection, pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **kwargs))

    @staticmethod
    def _validate_query(query: str) -> None:
        normalized = " ".join(query.lower().split())
        if not normalized.startswith("select "):
            raise ValueError("Only allowlisted SELECT queries are supported.")
        if ";" in normalized:
            raise ValueError("Multiple SQL statements are not supported.")
        selected_columns = normalized.split(" from ", maxsplit=1)[0].replace("select ", "").replace(" ", "")
        if any(column in selected_columns.split(",") for column in SENSITIVE_COLUMN_NAMES):
            raise ValueError("Sensitive columns are not allowed in connector queries.")
