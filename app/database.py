from collections.abc import AsyncGenerator
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import get_settings

class Base(DeclarativeBase):
    pass


settings = get_settings()


def _asyncpg_connect_args(database_url: str) -> dict:
    # Supabase Supavisor (transaction pooler) does not support asyncpg prepared statements.
    if "pooler.supabase.com" in database_url or ":6543/" in database_url:
        return {
            "statement_cache_size": 0,
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
        }
    return {}


def _engine_kwargs(database_url: str) -> dict:
    kwargs: dict = {
        "echo": False,
        "future": True,
        "connect_args": _asyncpg_connect_args(database_url),
    }
    if "pooler.supabase.com" in database_url or ":6543/" in database_url:
        kwargs["poolclass"] = NullPool
    return kwargs


engine = create_async_engine(settings.database_url, **_engine_kwargs(settings.database_url))
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
