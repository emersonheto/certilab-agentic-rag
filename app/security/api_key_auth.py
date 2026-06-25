from typing import Annotated

from fastapi import Header, HTTPException, status

from app.config import Settings, get_settings
from app.domain.models import Role
from app.security.access_control import Principal


async def get_api_key_principal(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    x_customer_id: Annotated[int | None, Header(alias="X-Customer-Id")] = None,
    settings: Settings = None,
) -> Principal:
    """Resolve a principal from a role-based API key (real mode).

    One key per role (admin, technician, client). Clients must also send
    X-Customer-Id to declare which tenant they belong to.
    """

    if settings is None:
        settings = get_settings()

    if x_api_key is None:
        raise _unauthorized()

    if x_api_key == settings.api_key_admin:
        return Principal(role=Role.ADMIN, customer_id=None, user_id=1)

    if x_api_key == settings.api_key_technician:
        return Principal(role=Role.TECHNICIAN, customer_id=None, user_id=2)

    if x_api_key == settings.api_key_client:
        if x_customer_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-Customer-Id header is required for client role.",
            )
        return Principal(role=Role.CLIENT, customer_id=x_customer_id, user_id=x_customer_id)

    raise _unauthorized()


def principal_from_api_key(
    api_key: str, settings: Settings, customer_id: int | None = None
) -> Principal:
    """Resolve an API-key principal for non-HTTP adapters such as Chainlit."""

    if api_key == settings.api_key_admin:
        return Principal(role=Role.ADMIN, customer_id=None, user_id=1)
    if api_key == settings.api_key_technician:
        return Principal(role=Role.TECHNICIAN, customer_id=None, user_id=2)
    if api_key == settings.api_key_client:
        if customer_id is None:
            raise ValueError("customer_id is required for client API key.")
        return Principal(role=Role.CLIENT, customer_id=customer_id, user_id=customer_id)
    raise ValueError("Invalid API key.")


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
    )
