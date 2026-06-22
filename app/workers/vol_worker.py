"""Asynchronous LSTM/GARCH volatility forecasting.

Training is CPU-bound and slow (Torch + arch), so it must never run inside the
risk-compute request path or the 60s Celery beat. This worker trains/refreshes
per-ticker volatility models and persists them; the risk pipeline only reads the
latest stored forecast (see app.services.risk.compute_portfolio_risk).
"""
from __future__ import annotations

import asyncio
import logging
from decimal import Decimal

from sqlalchemy import select

from app.database import SessionLocal
from app.models import Position, RiskComputation
from app.services.forecast_store import get_latest_forecast, save_forecast
from app.services.market_data import get_last_snapshot_price
from app.services.portfolio_service import compute_portfolio_value, compute_position_values, compute_weights
from app.services.return_matrix import build_returns_matrix
from app.services.volatility_forecaster import VolatilityForecaster, forecast_is_fresh
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.vol_worker.compute_volatility_forecasts")
def compute_volatility_forecasts(risk_computation_id: str) -> dict:
    return asyncio.run(_compute(risk_computation_id))


async def _compute(risk_computation_id: str) -> dict:
    async with SessionLocal() as db:
        row = (
            await db.execute(select(RiskComputation).where(RiskComputation.id == risk_computation_id))
        ).scalar_one_or_none()
        if not row:
            return {"status": "not_found"}

        positions = (
            await db.execute(select(Position).where(Position.portfolio_id == row.portfolio_id))
        ).scalars().all()
        if not positions:
            return {"status": "no_positions"}

        tickers = [p.ticker.upper() for p in positions]
        returns = await build_returns_matrix(db, tickers)
        if returns.empty:
            return {"status": "no_returns"}

        # Snapshot-based weights keep the worker self-contained (no live quotes).
        prices: dict[str, Decimal] = {}
        for pos in positions:
            snap = await get_last_snapshot_price(db, pos.ticker)
            if snap is not None:
                prices[pos.ticker] = snap
        weighted_positions = [p for p in positions if p.ticker in prices]
        weight_by_ticker: dict[str, float] = {}
        if weighted_positions:
            position_values = compute_position_values(weighted_positions, prices)
            portfolio_value = compute_portfolio_value(position_values)
            weights = compute_weights(weighted_positions, position_values, portfolio_value)
            weight_by_ticker = {
                p.ticker: abs(float(weights[i])) if i < len(weights) else 0.0
                for i, p in enumerate(weighted_positions)
            }

        var_95 = float(row.var_95)
        trained = 0
        skipped = 0
        failed = 0
        for pos in positions:
            ticker = pos.ticker
            if ticker not in returns.columns:
                continue
            cached = await get_latest_forecast(db, row.portfolio_id, ticker)
            if cached and forecast_is_fresh(cached.computed_at):
                skipped += 1
                continue
            try:
                series = returns[ticker].dropna()
                ticker_hist_var = var_95 * weight_by_ticker.get(ticker, 0.0)
                forecaster = VolatilityForecaster(series, ticker=ticker)
                result = forecaster.evaluate_and_forecast(historical_var_95=ticker_hist_var)
                m = result.metrics
                await save_forecast(
                    db,
                    row.portfolio_id,
                    ticker,
                    m.predicted_vol,
                    m.garch_vol,
                    m.lstm_mae,
                    m.garch_mae,
                    m.lstm_rmse,
                    m.garch_rmse,
                    m.direction_accuracy,
                    m.vol_regime,
                    result.adjusted_var_95,
                )
                trained += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("vol_forecast_failed ticker=%s error=%s", ticker, exc)
                failed += 1
        await db.commit()
        return {"status": "ok", "trained": trained, "skipped": skipped, "failed": failed}
