from datetime import date

import pytest

from app.domain.models import Certificate
from app.security.access_control import AccessDeniedError, build_scope, filter_certificates


def test_client_requires_customer_id() -> None:
    with pytest.raises(AccessDeniedError):
        build_scope("client", None, user_id=10)


def test_client_scope_filters_other_customers() -> None:
    certificates = [
        Certificate(1, "CERT-A", 101, "emitido", date(2025, 1, 1), 1, "Equipo A", "a.txt"),
        Certificate(2, "CERT-B", 202, "emitido", date(2025, 1, 2), 1, "Equipo B", "b.txt"),
    ]
    visible = filter_certificates(build_scope("client", 101), certificates)
    assert [certificate.customer_id for certificate in visible] == [101]


def test_admin_scope_can_see_all_customers() -> None:
    certificates = [
        Certificate(1, "CERT-A", 101, "emitido", date(2025, 1, 1), 1, "Equipo A", "a.txt"),
        Certificate(2, "CERT-B", 202, "emitido", date(2025, 1, 2), 1, "Equipo B", "b.txt"),
    ]
    visible = filter_certificates(build_scope("admin", None), certificates)
    assert len(visible) == 2
