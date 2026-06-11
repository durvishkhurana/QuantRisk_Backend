from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from app.config import get_settings
from app.database import Base, engine
from prometheus_client import make_asgi_app
from app.middleware.metrics_middleware import MetricsMiddleware
from app.routers import alerts, auth, portfolios, risk, websocket

settings = get_settings()
app = FastAPI(title=settings.app_name, docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)


@app.on_event("startup")
async def startup() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.exception_handler(SQLAlchemyError)
async def db_exception_handler(_: Request, exc: SQLAlchemyError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "DATABASE_ERROR", "message": str(exc), "status": 500},
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
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": str(exc), "status": 500},
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
