from typing import Annotated

from fastapi import Header, HTTPException, status

from app.config import Settings, get_settings
from app.domain.models import Role
from app.security.access_control import Principal


async def get_demo_principal(x_demo_token: Annotated[str | None, Header()] = None) -> Principal:
    """Resolve a local/demo principal from an allowlisted header token.

    This adapter is intentionally suitable only for local demos. Production must
    replace it with identity verified by Laravel, JWT, an authenticated session,
    or an internal gateway before building a Principal.
    """

    if x_demo_token is None:
        raise _unauthorized()

    principal = _demo_principals().get(x_demo_token)
    if principal is None:
        raise _unauthorized()
    return principal


def principal_from_demo_token(token: str, settings: Settings) -> Principal:
    """Resolve a demo principal for non-HTTP adapters such as Chainlit."""

    principal = _demo_principals_from_settings(settings).get(token)
    if principal is None:
        raise ValueError("Invalid demo credentials.")
    return principal


def _demo_principals() -> dict[str, Principal]:
    settings = get_settings()
    return _demo_principals_from_settings(settings)


def _demo_principals_from_settings(settings: Settings) -> dict[str, Principal]:
    configured: list[tuple[str | None, Principal]] = [
        (settings.demo_admin_token, Principal(role=Role.ADMIN, customer_id=None, user_id=1)),
        (settings.demo_technician_token, Principal(role=Role.TECHNICIAN, customer_id=None, user_id=2)),
        (settings.demo_client_101_token, Principal(role=Role.CLIENT, customer_id=101, user_id=1010)),
        (settings.demo_client_202_token, Principal(role=Role.CLIENT, customer_id=202, user_id=2020)),
    ]
    return {token: principal for token, principal in configured if token}


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid demo credentials.",
    )
