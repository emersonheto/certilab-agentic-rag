from __future__ import annotations

import re

from app.domain.models import Certificate, CertificateHistory, Customer, RetrievedSource
from app.retrieval.query import extract_certificate_code
from app.security.access_control import AccessScope, filter_certificates

# Maximum items returned in listing and count responses.
# Prevents overwhelming the LLM context and the UI with hundreds of items.
_MAX_LISTING_RESULTS = 20
_MAX_COUNT_SOURCES = 5  # count answers only need a few representative sources

# Keywords that indicate the question is about customers, not certificates.
_CUSTOMER_KEYWORDS = {"cliente", "clientes", "customer", "customers", "empresa", "empresas"}
_COUNT_KEYWORDS = {"cuánt", "cuant", "cantidad", "cuantos", "cuántos", "total", "count", "how many"}


class StructuredRetriever:
    """Tenant-aware lookup over anonymized certificate metadata."""

    def __init__(self, customers: list[Customer], certificates: list[Certificate], histories: list[CertificateHistory]) -> None:
        self._customers = {customer.id: customer for customer in customers}
        self._certificates = certificates
        self._histories_by_certificate: dict[int, list[CertificateHistory]] = {}
        for history in histories:
            self._histories_by_certificate.setdefault(history.certificate_id, []).append(history)

    def retrieve(self, question: str, scope: AccessScope) -> tuple[str, list[RetrievedSource]]:
        visible = filter_certificates(scope, self._certificates)
        requested_code = extract_certificate_code(question)
        if requested_code is not None:
            return self._answer_code_lookup(requested_code, visible)

        # Strip punctuation from tokens so "¿cuántos" matches "cuántos"
        tokens = set(re.sub(r"[¿?¡!,\.]", "", t) for t in question.lower().split())
        is_count_question = bool(tokens & _COUNT_KEYWORDS)
        is_customer_question = bool(tokens & _CUSTOMER_KEYWORDS)

        if is_count_question and is_customer_question:
            return self._answer_customer_count(visible)
        if is_count_question:
            return self._answer_count(visible)
        return self._answer_listing(visible)

    def _answer_code_lookup(
        self, requested_code: str, visible: list[Certificate]
    ) -> tuple[str, list[RetrievedSource]]:
        certificate = next((item for item in self._certificates if item.code == requested_code), None)
        if certificate is None:
            return f"No se encontró el certificado {requested_code}.", []

        if certificate not in visible:
            return f"No se encontró un certificado autorizado para el código {requested_code}.", []

        return self._answer_listing([certificate])

    def _answer_customer_count(self, certificates: list[Certificate]) -> tuple[str, list[RetrievedSource]]:
        """Count distinct customers visible in scope."""
        distinct_ids = sorted({c.customer_id for c in certificates})
        total = len(distinct_ids)
        names = [
            self._customers[cid].name if cid in self._customers else f"Cliente {cid}"
            for cid in distinct_ids[:5]
        ]
        sample = ", ".join(names)
        suffix = f" y {total - 5} más" if total > 5 else ""
        answer = f"Hay {total} cliente(s) con certificados visibles en el sistema ({sample}{suffix})."
        # Return a small representative set of sources (one cert per customer, max 5)
        rep_certs = [next(c for c in certificates if c.customer_id == cid) for cid in distinct_ids[:_MAX_COUNT_SOURCES]]
        return answer, [self._source_from_certificate(c) for c in rep_certs]

    def _answer_count(self, certificates: list[Certificate]) -> tuple[str, list[RetrievedSource]]:
        by_status: dict[str, int] = {}
        for certificate in certificates:
            by_status[certificate.status] = by_status.get(certificate.status, 0) + 1
        status_text = ", ".join(f"{status}: {count}" for status, count in sorted(by_status.items())) or "sin registros"
        answer = f"Se encontraron {len(certificates)} certificados visibles para el alcance consultado ({status_text})."
        # Return only a small representative sample as sources (not all certs)
        sample = certificates[:_MAX_COUNT_SOURCES]
        return answer, [self._source_from_certificate(c) for c in sample]

    def _answer_listing(self, certificates: list[Certificate]) -> tuple[str, list[RetrievedSource]]:
        if not certificates:
            return "No hay certificados visibles para el alcance consultado.", []
        total = len(certificates)
        capped = certificates[:_MAX_LISTING_RESULTS]
        lines = []
        for certificate in capped:
            customer = self._customers.get(certificate.customer_id)
            customer_name = customer.name if customer else f"Cliente {certificate.customer_id}"
            lines.append(f"{certificate.code}: {certificate.status}, emitido el {certificate.emitted_at.isoformat()}, cliente {customer_name}.")
        if total > _MAX_LISTING_RESULTS:
            omitted = total - _MAX_LISTING_RESULTS
            lines.append(f"... y {omitted} certificado(s) más no mostrados.")
        return " ".join(lines), [self._source_from_certificate(c) for c in capped]

    def _source_from_certificate(self, certificate: Certificate) -> RetrievedSource:
        histories = self._histories_by_certificate.get(certificate.id, [])
        history_note = histories[-1].note if histories else "Sin historial asociado."
        return RetrievedSource(
            certificate_id=certificate.id,
            code=certificate.code,
            customer_id=certificate.customer_id,
            source_type="metadata",
            path="data/mock/certificates.json",
            snippet=f"{certificate.code} | {certificate.status} | {certificate.emitted_at.isoformat()} | {history_note}",
        )
