import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent


def _env_files() -> tuple[str, ...]:
    paths: list[Path] = [_REPO_ROOT / ".env", _BACKEND_DIR / ".env", Path(".env")]
    return tuple(str(p) for p in paths if p.is_file()) or (".env",)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "QuantRisk Engine"
    environment: str = "development"
    frontend_url: str = "http://localhost:5173"
    frontend_urls: str | None = Field(default=None, alias="FRONTEND_URLS")

    database_url: str = Field(
        default="postgresql+asyncpg://quantrisk:password@localhost:5432/quantrisk_db",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    jwt_secret_key: str = Field(default="change-me", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_hours: int = Field(default=24, alias="JWT_EXPIRE_HOURS")

    alpha_vantage_key: str | None = Field(default=None, alias="ALPHA_VANTAGE_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    risk_compute_interval_seconds: int = Field(default=60, alias="RISK_COMPUTE_INTERVAL_SECONDS")
    lookback_days: int = Field(default=252, alias="LOOKBACK_DAYS")
    var_confidence_95: float = Field(default=0.05, alias="VAR_CONFIDENCE_95")
    var_confidence_99: float = Field(default=0.01, alias="VAR_CONFIDENCE_99")
    margin_warning_threshold: float = Field(default=0.85, alias="MARGIN_WARNING_THRESHOLD")

    celery_broker_url: str = Field(default="redis://localhost:6379/0", alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/1", alias="CELERY_RESULT_BACKEND")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")

    supabase_url: str | None = Field(default=None, alias="NEXT_PUBLIC_SUPABASE_URL")
    supabase_publishable_key: str | None = Field(
        default=None, alias="NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"
    )
    supabase_secret_key: str | None = Field(default=None, alias="SUPABASE_SECRET_KEY")

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: object) -> object:
        supabase_db = os.getenv("SUPABASE_DATABASE_URL")
        if supabase_db:
            value = supabase_db
        if isinstance(value, str) and value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value

    def cors_origins(self) -> list[str]:
        if self.frontend_urls:
            values = [v.strip() for v in self.frontend_urls.split(",") if v.strip()]
            if values:
                return values
        return [self.frontend_url]


@lru_cache
def get_settings() -> Settings:
    return Settings()
