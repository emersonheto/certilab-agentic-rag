import pytest

from app.tools.s3_loader import S3LoaderConfig, S3PdfTextLoader


def test_s3_loader_resolves_relative_pdf_key_under_prefix() -> None:
    loader = S3PdfTextLoader(S3LoaderConfig(bucket_name="bucket", region="us-east-1", prefix="certificates/pdfs"))

    assert loader.resolve_certificate_pdf_key("2025/CERT-001.pdf") == "certificates/pdfs/2025/CERT-001.pdf"


def test_s3_loader_keeps_key_already_under_prefix() -> None:
    loader = S3PdfTextLoader(S3LoaderConfig(bucket_name="bucket", region="us-east-1", prefix="/certificates/pdfs/"))

    assert loader.resolve_certificate_pdf_key("certificates/pdfs/2025/CERT-001.pdf") == (
        "certificates/pdfs/2025/CERT-001.pdf"
    )


@pytest.mark.parametrize(
    "unsafe_key",
    ["../CERT-001.pdf", "2025/../CERT-001.pdf", "/certificates/pdfs/CERT-001.pdf", ""],
)
def test_s3_loader_rejects_unsafe_keys(unsafe_key: str) -> None:
    loader = S3PdfTextLoader(S3LoaderConfig(bucket_name="bucket", region="us-east-1", prefix="certificates/pdfs"))

    with pytest.raises(ValueError):
        loader.resolve_certificate_pdf_key(unsafe_key)


def test_s3_loader_rejects_unsafe_prefix() -> None:
    loader = S3PdfTextLoader(S3LoaderConfig(bucket_name="bucket", region="us-east-1", prefix="certificates/../pdfs"))

    with pytest.raises(ValueError, match="prefix"):
        _ = loader.normalized_prefix


def test_s3_loader_requires_boto3_only_when_real_client_is_used() -> None:
    loader = S3PdfTextLoader(S3LoaderConfig(bucket_name="bucket", region="us-east-1", prefix="certificates"))

    assert loader.resolve_certificate_pdf_key("CERT-001.pdf") == "certificates/CERT-001.pdf"
