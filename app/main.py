import time
import uuid
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.config import Settings
from app.config import get_settings
from app.domain.models import Role
from app.graph import RagPipeline, build_rag_pipeline
from app.logging import configure_logging, get_logger
from app.observability import configure_observability
from app.schemas import AskRequest, AskResponse, EvaluateResponse, HealthResponse, ReindexResponse
from app.security.access_control import AccessDeniedError, Principal
from app.security.api_key_auth import get_api_key_principal
from app.security.demo_auth import get_demo_principal

logger = get_logger("api")


async def get_principal(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    x_customer_id: Annotated[int | None, Header(alias="X-Customer-Id")] = None,
    x_demo_token: Annotated[str | None, Header(alias="X-Demo-Token")] = None,
) -> Principal:
    """Select the auth adapter from APP_MODE.

    Mock mode keeps the demo adapter (and the 84 existing token-based tests),
    while real mode authenticates callers with the operator-issued X-API-Key.
    Clients in real mode also send X-Customer-Id for tenant scope.
    """

    if settings.app_mode == "real":
        return await get_api_key_principal(x_api_key, x_customer_id, settings)
    return await get_demo_principal(x_demo_token)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, json=settings.log_json)
    _configure_optional_observability(settings)
    app = FastAPI(title=settings.app_name, version="0.1.0")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service=settings.app_name, environment=settings.environment)

    @app.post("/ask", response_model=AskResponse, tags=["rag"])
    async def ask(payload: AskRequest, principal: Annotated[Principal, Depends(get_principal)]) -> AskResponse:
        request_id = str(uuid.uuid4())[:8]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            role=principal.role.value,
            question_length=len(payload.question),
        )
        logger.info("api.ask.start")
        try:
            started = time.perf_counter()
            response = get_pipeline(settings).ask(payload.question, principal)
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.info("api.ask.complete", route=response.route, source_count=len(response.sources), duration_ms=elapsed)
            return response
        except AccessDeniedError as exc:
            logger.warning("api.ask.access_denied")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except Exception:
            logger.exception("api.ask.error")
            raise

    @app.post("/admin/reindex", response_model=ReindexResponse, tags=["admin"])
    async def admin_reindex(principal: Annotated[Principal, Depends(get_principal)]) -> ReindexResponse:
        if principal.role is not Role.ADMIN:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required.")
        if settings.app_mode != "real":
            return ReindexResponse(
                status="ok",
                indexed=0,
                skipped=0,
                duration_ms=0,
                note="reindex not available in mock mode",
            )
        from time import perf_counter

        started_at = perf_counter()
        result = get_pipeline(settings).reindex()
        elapsed = (perf_counter() - started_at) * 1000
        return ReindexResponse(
            status="ok",
            indexed=result["indexed"],
            skipped=result["skipped"],
            duration_ms=round(elapsed, 3),
        )

    @app.post("/admin/evaluate", response_model=EvaluateResponse, tags=["admin"])
    async def admin_evaluate(
        principal: Annotated[Principal, Depends(get_principal)],
    ) -> EvaluateResponse:
        if principal.role is not Role.ADMIN:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required.")
        if settings.app_mode != "real":
            return EvaluateResponse(status="ok", evaluated=0, experiment_id="")
        if not settings.phoenix_enabled:
            return EvaluateResponse(status="ok", evaluated=0, experiment_id="")

        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        from time import perf_counter

        started_at = perf_counter()
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            summary = await loop.run_in_executor(pool, _run_eval_sync, settings)
        elapsed = (perf_counter() - started_at) * 1000
        logger.info("admin.evaluate.complete", duration_ms=round(elapsed), test_cases=summary["evaluated"])
        return EvaluateResponse(
            status="ok",
            evaluated=summary["evaluated"],
            experiment_id=summary.get("experiment_id", ""),
        )

    return app


def _run_eval_sync(settings: Settings) -> dict:
    """Synchronous evaluation wrapper for the thread-pool."""
    from app.eval import run_evaluation

    return run_evaluation(settings)


_pipeline_cache: RagPipeline | None = None


def get_pipeline(settings: Settings) -> RagPipeline:
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = build_rag_pipeline(settings)
    return _pipeline_cache


def _configure_optional_observability(settings: Settings) -> None:
    """Configure tracing without letting optional telemetry block API startup."""

    try:
        configure_observability(settings)
    except Exception:
        return None


app = create_app()
