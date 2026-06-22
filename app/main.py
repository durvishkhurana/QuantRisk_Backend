import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from app.config import get_settings
from app.database import Base, engine
from prometheus_client import make_asgi_app
from app.middleware.metrics_middleware import MetricsMiddleware
from app.routers import alerts, auth, portfolios, risk, websocket

logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title=settings.app_name, docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json")

# Only echo raw exception text to clients outside production, where it aids local
# debugging. In production we log the detail server-side and return a generic
# message so internal errors / SQL aren't leaked to callers.
_expose_errors = settings.environment.lower() != "production"

_allowed_origins = settings.cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)


@app.middleware("http")
async def ensure_cors_on_error_responses(request: Request, call_next):
    """Some proxy/error paths omit CORS; mirror allowed Origin on every response."""
    response = await call_next(request)
    origin = request.headers.get("origin")
    if origin and origin in _allowed_origins and "access-control-allow-origin" not in response.headers:
        response.headers["access-control-allow-origin"] = origin
        response.headers["access-control-allow-credentials"] = "true"
    return response


@app.on_event("startup")
async def startup() -> None:
    # In production, Alembic migrations are the single source of truth for the
    # schema (`alembic upgrade head` on deploy). Running create_all there would
    # risk ORM/migration drift, so it's limited to dev/test convenience.
    if settings.environment.lower() == "production":
        logger.info("startup: skipping create_all (Alembic owns the schema in production)")
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.exception_handler(SQLAlchemyError)
async def db_exception_handler(_: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("database_error")
    message = str(exc) if _expose_errors else "A database error occurred."
    return JSONResponse(
        status_code=500,
        content={"error": "DATABASE_ERROR", "message": message, "status": 500},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    error_code = str(exc.detail).upper().replace(" ", "_")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error_code, "message": str(exc.detail), "status": exc.status_code},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("internal_error")
    message = str(exc) if _expose_errors else "An internal error occurred."
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": message, "status": 500},
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": settings.app_name}


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(auth.router)
app.include_router(portfolios.router)
app.include_router(risk.router)
app.include_router(risk.task_router)
app.include_router(alerts.router)
app.include_router(alerts.portfolio_alert_router)
app.include_router(websocket.router)
