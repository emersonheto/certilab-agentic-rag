from dataclasses import dataclass

from app.domain.models import Certificate, DocumentChunk, Role


class AccessDeniedError(ValueError):
    """Raised when a caller tries to access data outside its scope."""


@dataclass(frozen=True)
class Principal:
    """Authenticated caller identity used to enforce tenant isolation."""

    role: Role
    customer_id: int | None
    user_id: int


@dataclass(frozen=True)
class AccessScope:
    role: Role
    customer_id: int | None
    user_id: int | None = None

    @property
    def is_global(self) -> bool:
        return self.role in {Role.ADMIN, Role.TECHNICIAN}


def build_scope(role: str, customer_id: int | None, user_id: int | None = None) -> AccessScope:
    """Build and validate the caller access scope."""

    parsed_role = Role(role)
    if parsed_role is Role.CLIENT and customer_id is None:
        raise AccessDeniedError("Client role requires customer_id.")
    return AccessScope(role=parsed_role, customer_id=customer_id, user_id=user_id)


def scope_from_principal(principal: Principal) -> AccessScope:
    """Build an access scope from a trusted authenticated principal."""

    return build_scope(principal.role.value, principal.customer_id, principal.user_id)


def can_access_customer(scope: AccessScope, customer_id: int) -> bool:
    """Return whether a scope can access a customer-owned record."""

    if scope.is_global:
        return True
    return scope.customer_id == customer_id


def filter_certificates(scope: AccessScope, certificates: list[Certificate]) -> list[Certificate]:
    """Apply tenant/customer isolation to certificate records."""

    return [certificate for certificate in certificates if can_access_customer(scope, certificate.customer_id)]


def filter_chunks(scope: AccessScope, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Apply tenant/customer isolation to indexed text chunks."""

    return [chunk for chunk in chunks if can_access_customer(scope, chunk.customer_id)]
