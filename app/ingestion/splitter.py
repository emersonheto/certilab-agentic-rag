import hashlib

from app.domain.models import Certificate, DocumentChunk


def split_text(text: str, *, chunk_size: int = 420, overlap: int = 60) -> list[str]:
    """Split text into deterministic overlapping chunks."""

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_pdf_chunks(certificates: list[Certificate], pdf_texts: dict[str, str]) -> list[DocumentChunk]:
    """Create metadata-enriched chunks from mock PDF text."""

    chunks: list[DocumentChunk] = []
    certificates_by_file = {certificate.pdf_path.split("/")[-1]: certificate for certificate in certificates}
    for filename, text in pdf_texts.items():
        certificate = certificates_by_file.get(filename)
        if certificate is None:
            continue
        for index, chunk_text in enumerate(split_text(text)):
            digest = hashlib.sha256(f"{certificate.code}:{index}:{chunk_text}".encode()).hexdigest()[:16]
            chunks.append(
                DocumentChunk(
                    id=digest,
                    certificate_id=certificate.id,
                    certificate_code=certificate.code,
                    customer_id=certificate.customer_id,
                    source_type="pdf_text",
                    path=f"data/pdf_text/{filename}",
                    text=chunk_text,
                )
            )
    return chunks


def build_metadata_chunks(certificates: list[Certificate]) -> list[DocumentChunk]:
    """Create metadata-enriched chunks from certificate fields (real mode).

    Used when PDF text is not yet available. Constructs a safe text
    representation from non-PII certificate metadata fields only.
    """

    chunks: list[DocumentChunk] = []
    for certificate in certificates:
        parts = [
            certificate.code,
            certificate.status,
            certificate.equipment,
            certificate.emitted_at.isoformat(),
        ]
        if certificate.document_type:
            parts.append(certificate.document_type)
        if certificate.case_number:
            parts.append(certificate.case_number)
        metadata_text = " ".join(part for part in parts if part)
        for index, chunk_text in enumerate(split_text(metadata_text)):
            digest = hashlib.sha256(f"{certificate.code}:{index}:{chunk_text}".encode()).hexdigest()[:16]
            chunks.append(
                DocumentChunk(
                    id=digest,
                    certificate_id=certificate.id,
                    certificate_code=certificate.code,
                    customer_id=certificate.customer_id,
                    source_type="metadata",
                    path=f"certificates/{certificate.code}",
                    text=chunk_text,
                )
            )
    return chunks
