from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from app.config import Settings


@dataclass(frozen=True)
class S3LoaderConfig:
    """S3 settings for certificate PDFs."""

    bucket_name: str | None
    region: str | None
    prefix: str = ""
    access_key_id: str | None = None
    secret_access_key: str | None = None
    storage_disk: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> S3LoaderConfig:
        """Build S3 config from safe or Laravel-compatible settings."""

        return cls(
            bucket_name=settings.s3_bucket_name,
            region=settings.aws_region,
            prefix=settings.aws_storage_prefix or "",
            access_key_id=settings.aws_access_key_id,
            secret_access_key=settings.aws_secret_access_key,
            storage_disk=settings.certificates_storage_disk,
        )


class S3PdfTextLoader:
    """Lazy S3 PDF loader with prefix-bound key resolution."""

    def __init__(self, config: S3LoaderConfig) -> None:
        self._config = config
        self._client: Any | None = None

    @property
    def normalized_prefix(self) -> str:
        """Return the configured prefix without leading/trailing separators."""

        return self._normalize_prefix(self._config.prefix)

    def resolve_certificate_pdf_key(self, certificate_pdf_path: str) -> str:
        """Resolve a certificate PDF key under the configured storage prefix."""

        candidate = certificate_pdf_path.strip()
        if not candidate:
            raise ValueError("Certificate PDF key cannot be empty.")
        if candidate.startswith("/"):
            raise ValueError("Certificate PDF key must be relative to the bucket.")

        path = PurePosixPath(candidate)
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("Certificate PDF key contains unsafe path segments.")

        key = path.as_posix()
        prefix = self.normalized_prefix
        if not prefix:
            return key
        if key == prefix or key.startswith(f"{prefix}/"):
            return key
        return f"{prefix}/{key}"

    def generate_presigned_url(self, certificate_pdf_path: str, expires_in: int = 900) -> str:
        """Generate a presigned GET URL after enforcing the configured prefix."""

        key = self.resolve_certificate_pdf_key(certificate_pdf_path)
        client = self._get_client()
        return str(
            client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._require_bucket(), "Key": key},
                ExpiresIn=expires_in,
            )
        )

    def fetch_pdf_bytes(self, certificate_pdf_path: str) -> bytes:
        """Fetch PDF bytes after enforcing the configured prefix."""

        key = self.resolve_certificate_pdf_key(certificate_pdf_path)
        response = self._get_client().get_object(Bucket=self._require_bucket(), Key=key)
        body = response["Body"].read()
        if not isinstance(body, bytes):
            raise RuntimeError("S3 object body did not return bytes.")
        return body

    def load_certificate_pdf_text(self, certificate_code: str, customer_id: int) -> str:
        """Placeholder for future PDF text extraction after authorization."""

        raise NotImplementedError(
            f"S3 PDF text extraction is not implemented yet for certificate {certificate_code} and customer {customer_id}."
        )

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Real S3 mode requires the optional 'boto3' package.") from exc

        kwargs: dict[str, object] = {}
        if self._config.region:
            kwargs["region_name"] = self._config.region
        if self._config.access_key_id and self._config.secret_access_key:
            kwargs["aws_access_key_id"] = self._config.access_key_id
            kwargs["aws_secret_access_key"] = self._config.secret_access_key
        return boto3.client("s3", **kwargs)

    def _require_bucket(self) -> str:
        if not self._config.bucket_name:
            raise ValueError("Missing S3 bucket setting for real mode: S3_BUCKET_NAME or AWS_BUCKET.")
        return self._config.bucket_name

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        normalized = prefix.strip().strip("/")
        if not normalized:
            return ""
        path = PurePosixPath(normalized)
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("S3 prefix contains unsafe path segments.")
        return path.as_posix()
