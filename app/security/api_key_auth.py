from typing import Annotated

from fastapi import Header, HTTPException, status

from app.config import Settings, get_settings
from app.domain.models import Role
from app.security.access_control import Principal


async def get_api_key_principal(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    """Resolve a principal from an operator-issued API key (real mode).

    Keys are provisioned as operator environment variables (never in the
    database) so the service stays standable without MySQL. This adapter is the
    real-mode counterpart of the demo adapter and can be swapped to a DB- or
    Laravel-backed store later without changing call sites.
    """

    if x_api_key is None:
        raise _unauthorized()
    principal = _api_key_principals().get(x_api_key)
    if principal is None:
        raise _unauthorized()
    return principal


def principal_from_api_key(api_key: str, settings: Settings) -> Principal:
    """Resolve an API-key principal for non-HTTP adapters such as Chainlit."""

    principal = _api_key_principals_from_settings(settings).get(api_key)
    if principal is None:
        raise ValueError("Invalid API key.")
    return principal


def _api_key_principals() -> dict[str, Principal]:
    return _api_key_principals_from_settings(get_settings())


def _api_key_principals_from_settings(settings: Settings) -> dict[str, Principal]:
    configured: list[tuple[str | None, Principal]] = [
        (settings.api_key_admin, Principal(role=Role.ADMIN, customer_id=None, user_id=1)),
        (settings.api_key_technician, Principal(role=Role.TECHNICIAN, customer_id=None, user_id=2)),
        (settings.api_key_client_101, Principal(role=Role.CLIENT, customer_id=101, user_id=1010)),
        (settings.api_key_client_202, Principal(role=Role.CLIENT, customer_id=202, user_id=2020)),
    ]
    return {api_key: principal for api_key, principal in configured if api_key}


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
    )
