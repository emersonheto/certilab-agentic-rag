from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.ingestion.pdf_extractor import S3PdfTextExtractor


class FakeS3PdfTextLoader:
    """Stand-in for S3PdfTextLoader that returns fake PDF bytes."""

    def __init__(self, config: object) -> None:
        self.config = config
        self.fetch_count = 0

    def fetch_pdf_bytes(self, pdf_path: str) -> bytes:
        self.fetch_count += 1
        return b"%PDF-1.4 fake pdf content for " + pdf_path.encode()


@pytest.fixture()
def settings() -> Settings:
    return Settings(_env_file=None)


@patch("app.ingestion.pdf_extractor._extract_text_from_bytes")
def test_extract_text_returns_text(mock_extract: MagicMock, settings: Settings) -> None:
    mock_extract.return_value = "Extracted PDF text content"
    extractor = S3PdfTextExtractor(settings)
    extractor._loader = FakeS3PdfTextLoader(None)  # type: ignore[assignment]

    text = extractor.extract_text("certificates/test.pdf")

    assert text == "Extracted PDF text content"


@patch("app.ingestion.pdf_extractor._extract_text_from_bytes")
def test_extract_text_caching_skips_fetch(mock_extract: MagicMock, settings: Settings) -> None:
    mock_extract.return_value = "Cached text"
    extractor = S3PdfTextExtractor(settings)
    fake = FakeS3PdfTextLoader(None)
    extractor._loader = fake  # type: ignore[assignment]

    text1 = extractor.extract_text("certificates/dup.pdf")
    text2 = extractor.extract_text("certificates/dup.pdf")

    assert text1 == text2
    assert fake.fetch_count == 1  # Only fetched once
    mock_extract.assert_called_once()


@patch("app.ingestion.pdf_extractor._extract_text_from_bytes")
def test_extract_batch_returns_dict(mock_extract: MagicMock, settings: Settings) -> None:
    mock_extract.side_effect = lambda b: f"Text from {b.decode()[:40]}..."
    extractor = S3PdfTextExtractor(settings)
    fake = FakeS3PdfTextLoader(None)
    extractor._loader = fake  # type: ignore[assignment]

    paths = ["certificates/a.pdf", "certificates/b.pdf"]
    result = extractor.extract_batch(paths)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert "certificates/a.pdf" in result
    assert "certificates/b.pdf" in result


@patch("app.ingestion.pdf_extractor._extract_text_from_bytes")
def test_extract_text_empty_pdf(mock_extract: MagicMock, settings: Settings) -> None:
    mock_extract.return_value = ""
    extractor = S3PdfTextExtractor(settings)
    extractor._loader = FakeS3PdfTextLoader(None)  # type: ignore[assignment]

    text = extractor.extract_text("certificates/empty.pdf")
    assert text == ""
