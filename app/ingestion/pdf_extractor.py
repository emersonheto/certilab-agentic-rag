from __future__ import annotations

import io

from app.config import Settings
from app.tools.s3_loader import S3LoaderConfig, S3PdfTextLoader


class S3PdfTextExtractor:
    """Extract text from certificate PDFs stored in S3.

    Caches extracted text by S3 path so repeated requests for the same
    certificate do not re-download or re-parse the PDF.
    """

    def __init__(self, settings: Settings) -> None:
        self._loader = S3PdfTextLoader(S3LoaderConfig.from_settings(settings))
        self._cache: dict[str, str] = {}

    def extract_text(self, pdf_path: str) -> str:
        """Download PDF from S3 and extract text using pdfplumber."""
        if pdf_path in self._cache:
            return self._cache[pdf_path]
        pdf_bytes = self._loader.fetch_pdf_bytes(pdf_path)
        text = _extract_text_from_bytes(pdf_bytes)
        self._cache[pdf_path] = text
        return text

    def extract_batch(self, paths: list[str]) -> dict[str, str]:
        """Extract text from multiple PDFs. Returns ``{path: text}``."""
        return {path: self.extract_text(path) for path in paths}


def _extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Parse PDF bytes with pdfplumber and return all page text joined."""
    import pdfplumber

    texts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            texts.append(text)
    return "\n".join(texts).strip()
