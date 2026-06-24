from app.domain.models import Certificate, CertificateHistory, Customer, RetrievedSource
from app.retrieval.query import extract_certificate_code
from app.security.access_control import AccessScope, filter_certificates


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

        question_lc = question.lower()
        if "cuánt" in question_lc or "cuant" in question_lc or "cantidad" in question_lc or "count" in question_lc:
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

    def _answer_count(self, certificates: list[Certificate]) -> tuple[str, list[RetrievedSource]]:
        by_status: dict[str, int] = {}
        for certificate in certificates:
            by_status[certificate.status] = by_status.get(certificate.status, 0) + 1
        status_text = ", ".join(f"{status}: {count}" for status, count in sorted(by_status.items())) or "sin registros"
        answer = f"Se encontraron {len(certificates)} certificados visibles para el alcance consultado ({status_text})."
        return answer, [self._source_from_certificate(certificate) for certificate in certificates]

    def _answer_listing(self, certificates: list[Certificate]) -> tuple[str, list[RetrievedSource]]:
        if not certificates:
            return "No hay certificados visibles para el alcance consultado.", []
        lines = []
        for certificate in certificates:
            customer = self._customers.get(certificate.customer_id)
            customer_name = customer.name if customer else f"Cliente {certificate.customer_id}"
            lines.append(f"{certificate.code}: {certificate.status}, emitido el {certificate.emitted_at.isoformat()}, cliente {customer_name}.")
        return " ".join(lines), [self._source_from_certificate(certificate) for certificate in certificates]

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
