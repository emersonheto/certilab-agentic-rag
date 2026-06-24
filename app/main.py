from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.config import Settings
from app.config import get_settings
from app.graph import RagPipeline, build_rag_pipeline
from app.observability import configure_observability
from app.schemas import AskRequest, AskResponse, HealthResponse
from app.security.access_control import AccessDeniedError, Principal
from app.security.api_key_auth import get_api_key_principal
from app.security.demo_auth import get_demo_principal


async def get_principal(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    x_demo_token: Annotated[str | None, Header(alias="X-Demo-Token")] = None,
) -> Principal:
    """Select the auth adapter from APP_MODE.

    Mock mode keeps the demo adapter (and the 84 existing token-based tests),
    while real mode authenticates callers with the operator-issued X-API-Key.
    """

    if settings.app_mode == "real":
        return await get_api_key_principal(x_api_key)
    return await get_demo_principal(x_demo_token)


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_optional_observability(settings)
    app = FastAPI(title=settings.app_name, version="0.1.0")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service=settings.app_name, environment=settings.environment)

    @app.post("/ask", response_model=AskResponse, tags=["rag"])
    async def ask(payload: AskRequest, principal: Annotated[Principal, Depends(get_principal)]) -> AskResponse:
        try:
            return get_pipeline(settings).ask(payload.question, principal)
        except AccessDeniedError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    return app


def get_pipeline(settings: Settings) -> RagPipeline:
    return build_rag_pipeline(settings)


def _configure_optional_observability(settings: Settings) -> None:
    """Configure tracing without letting optional telemetry block API startup."""

    try:
        configure_observability(settings)
    except Exception:
        return None


app = create_app()
