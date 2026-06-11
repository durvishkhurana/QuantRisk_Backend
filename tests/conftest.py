"""Shared pytest configuration — set test env before any app imports."""
from __future__ import annotations

import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://quantrisk:password@localhost:5432/quantrisk_test",
)
os.environ.setdefault("JWT_SECRET_KEY", "pytest-jwt-secret-key-change-in-ci")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/15")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/15")

from collections.abc import AsyncGenerator
from decimal import Decimal
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.config import get_settings

get_settings.cache_clear()


@pytest.fixture
async def _clean_database() -> AsyncGenerator[None, None]:
    from app.database import engine

    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE users RESTART IDENTITY CASCADE"))
    yield


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    import asyncio

    from starlette.background import BackgroundTasks

    from app.main import app

    _orig_add_task = BackgroundTasks.add_task

    def _add_task_run_after_response(self, func, *args, **kwargs):
        async def _runner() -> None:
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)

        _orig_add_task(self, _runner)

    BackgroundTasks.add_task = _add_task_run_after_response  # type: ignore[method-assign]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    BackgroundTasks.add_task = _orig_add_task  # type: ignore[method-assign]


@pytest.fixture
def market_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    import pandas as pd
    from datetime import date, timedelta

    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def fast_info(self) -> dict:
            return {"lastPrice": 150.0 if self.symbol != "SPY" else 450.0}

        def history(self, period: str = "1d") -> pd.DataFrame:
            return pd.DataFrame({"Close": [150.0]})

        @property
        def info(self) -> dict:
            return {"sector": "Technology"}

    def _fake_download(ticker: str, period: str = "1y", auto_adjust: bool = False, progress: bool = False) -> pd.DataFrame:
        days = 280
        end = date.today()
        dates = pd.bdate_range(end=end, periods=days)
        base = 100.0 if str(ticker).upper() != "SPY" else 400.0
        closes = base + np.linspace(-2, 2, len(dates))
        return pd.DataFrame({"Close": closes, "Volume": np.full(len(dates), 1_000_000)}, index=dates)

    async def _noop_publish(*_args, **_kwargs) -> None:
        return None

    async def _noop_broadcast(*_args, **_kwargs) -> None:
        return None

    def _noop_send_task(*_args, **_kwargs) -> None:
        return None

    async def _cache_miss(*_args, **_kwargs):
        return None

    async def _cache_set(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.services.market_data.cache_get_json", _cache_miss)
    monkeypatch.setattr("app.services.market_data.cache_set_json", _cache_set)
    monkeypatch.setattr("app.services.market_data.yf.Ticker", _FakeTicker)
    monkeypatch.setattr("app.services.market_data.yf.download", _fake_download)
    monkeypatch.setattr("app.services.redis_client.publish_json", _noop_publish)
    monkeypatch.setattr("app.services.alerts.socket_manager.broadcast", _noop_broadcast)
    monkeypatch.setattr("app.workers.celery_app.celery_app.send_task", _noop_send_task)
    monkeypatch.setattr("app.routers.portfolios.celery_app.send_task", _noop_send_task)
    monkeypatch.setattr("app.routers.risk.celery_app.send_task", _noop_send_task)


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    email = f"user-{uuid4()}@example.com"
    password = "password123"
    resp = await client.post("/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def portfolio_with_position(client: AsyncClient, auth_headers: dict[str, str]) -> str:
    create = await client.post(
        "/portfolios/",
        json={"name": "Test Portfolio", "margin_limit": 0.05},
        headers=auth_headers,
    )
    assert create.status_code == 200, create.text
    portfolio_id = create.json()["portfolio_id"]

    add = await client.post(
        f"/portfolios/{portfolio_id}/positions",
        json={"ticker": "AAPL", "quantity": 10, "purchase_price": 100},
        headers=auth_headers,
    )
    assert add.status_code == 200, add.text
    return portfolio_id
