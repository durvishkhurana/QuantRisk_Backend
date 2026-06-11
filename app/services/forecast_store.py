from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import VolatilityForecast


async def save_forecast(
    session: AsyncSession,
    portfolio_id: uuid.UUID,
    ticker: str,
    predicted_vol: float,
    garch_vol: float,
    lstm_mae: float,
    garch_mae: float,
    lstm_rmse: float,
    garch_rmse: float,
    direction_accuracy: float,
    regime: str,
    adjusted_var_95: float | None,
    computed_at: datetime | None = None,
) -> VolatilityForecast:
    improvement_pct = None
    if garch_mae and garch_mae > 0:
        improvement_pct = (garch_mae - lstm_mae) / garch_mae * 100.0
    row = VolatilityForecast(
        portfolio_id=portfolio_id,
        ticker=ticker.upper(),
        predicted_vol=predicted_vol,
        garch_vol=garch_vol,
        lstm_mae=lstm_mae,
        garch_mae=garch_mae,
        lstm_rmse=lstm_rmse,
        garch_rmse=garch_rmse,
        direction_accuracy=direction_accuracy,
        vol_regime=regime,
        adjusted_var_95=adjusted_var_95,
        improvement_pct=improvement_pct,
        computed_at=computed_at or datetime.now(timezone.utc),
    )
    session.add(row)
    await session.flush()
    return row


async def get_latest_forecast(
    session: AsyncSession,
    portfolio_id: uuid.UUID,
    ticker: str,
) -> VolatilityForecast | None:
    result = await session.execute(
        select(VolatilityForecast)
        .where(
            VolatilityForecast.portfolio_id == portfolio_id,
            VolatilityForecast.ticker == ticker.upper(),
        )
        .order_by(VolatilityForecast.computed_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_forecast_history(
    session: AsyncSession,
    portfolio_id: uuid.UUID,
    ticker: str,
    days: int = 30,
) -> list[VolatilityForecast]:
    result = await session.execute(
        select(VolatilityForecast)
        .where(
            VolatilityForecast.portfolio_id == portfolio_id,
            VolatilityForecast.ticker == ticker.upper(),
        )
        .order_by(VolatilityForecast.computed_at.desc())
        .limit(days)
    )
    return list(result.scalars().all())


async def get_latest_forecasts_for_portfolio(
    session: AsyncSession,
    portfolio_id: uuid.UUID,
) -> list[VolatilityForecast]:
    """Most recent forecast row per ticker for a portfolio."""
    result = await session.execute(
        select(VolatilityForecast)
        .where(VolatilityForecast.portfolio_id == portfolio_id)
        .order_by(VolatilityForecast.ticker, VolatilityForecast.computed_at.desc())
    )
    rows = result.scalars().all()
    seen: set[str] = set()
    latest: list[VolatilityForecast] = []
    for row in rows:
        if row.ticker in seen:
            continue
        seen.add(row.ticker)
        latest.append(row)
    return latest
