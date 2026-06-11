from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import get_settings
from app.models import Portfolio, Position
from app.services.market_data import get_last_snapshot_price, get_latest_price
from app.services.margin_evaluator import evaluate_margin
from app.services.portfolio_service import compute_portfolio_value, compute_position_values, compute_weights
from app.services.return_matrix import build_returns_matrix
from app.services.shap_service import compute_shap_like_attribution
from app.services.stress_engine import calculate_beta, run_stress_tests
from app.services.correlation_service import compute_correlation_regime
from app.services.var_engine import compute_monte_carlo_var, compute_var_cvar
from app.services.volatility_forecaster import (
    VolatilityForecaster,
    forecast_is_fresh,
)
from app.services.forecast_store import get_latest_forecast, save_forecast

logger = logging.getLogger(__name__)


settings = get_settings()


@dataclass
class RiskComputationResult:
    portfolio_value: Decimal
    var_95: Decimal
    var_99: Decimal
    cvar_95: Decimal
    margin_utilization: Decimal
    margin_status: str
    stress_mild: Decimal
    stress_moderate: Decimal
    stress_severe: Decimal
    shap_json: dict[str, float]
    computation_ms: int
    mc_var_95: Decimal | None = None
    mc_var_99: Decimal | None = None
    mc_cvar_95: Decimal | None = None
    mc_skewness: float | None = None
    mc_kurtosis: float | None = None
    mc_histogram: list[dict] | None = None
    correlation_json: dict | None = None
    vol_forecasts: list[dict] | None = None
    adjusted_var_95_portfolio: float | None = None


async def _load_positions(session: AsyncSession, portfolio_id: str) -> list[Position]:
    result = await session.execute(select(Position).where(Position.portfolio_id == portfolio_id))
    return list(result.scalars().all())


async def _load_returns_matrix(session: AsyncSession, tickers: list[str]):
    # Backward-compatible helper to keep tests/imports stable.
    return await build_returns_matrix(session, tickers)


def _calculate_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    return calculate_beta(asset_returns, market_returns)


async def compute_portfolio_risk(session: AsyncSession, portfolio: Portfolio) -> RiskComputationResult:
    start = time.perf_counter()
    positions = await _load_positions(session, str(portfolio.id))
    if not positions:
        raise ValueError("Portfolio has no positions")

    tickers = [p.ticker.upper() for p in positions]
    prices: dict[str, Decimal] = {}
    for pos in positions:
        try:
            latest = await get_latest_price(pos.ticker)
        except Exception:  # noqa: BLE001
            fallback = await get_last_snapshot_price(session, pos.ticker)
            if fallback is None:
                raise
            latest = fallback
        prices[pos.ticker] = latest
    position_values = compute_position_values(positions, prices)

    portfolio_value = compute_portfolio_value(position_values)
    weights = compute_weights(positions, position_values, portfolio_value)

    returns = await build_returns_matrix(session, tickers + ["SPY"])
    if returns.empty:
        raise ValueError("Not enough historical price data. Backfill prices first.")

    asset_returns = returns[[t for t in tickers if t in returns.columns]]
    port_series = asset_returns.to_numpy() @ weights[: asset_returns.shape[1]]
    var_95, var_99, cvar_95 = compute_var_cvar(
        port_series,
        portfolio_value,
        settings.var_confidence_95,
        settings.var_confidence_99,
    )

    market_returns = returns["SPY"].dropna().to_numpy() if "SPY" in returns.columns else np.array([])
    returns_by_ticker = {pos.ticker: returns[pos.ticker].dropna().to_numpy() for pos in positions if pos.ticker in returns.columns}
    stress_losses = run_stress_tests(positions, returns_by_ticker, market_returns, position_values)

    scenarios = asset_returns.to_numpy()
    shap_json = compute_shap_like_attribution(scenarios, weights, portfolio_value, positions)

    utilization, status = evaluate_margin(var_95, Decimal(portfolio.margin_limit), portfolio_value)

    mc_result = None
    if asset_returns.shape[0] >= 30 and asset_returns.shape[1] == len(weights):
        w = weights[: asset_returns.shape[1]]
        mc_result = compute_monte_carlo_var(asset_returns.to_numpy(), w, portfolio_value)

    correlation_json = compute_correlation_regime(asset_returns)

    vol_forecasts: list[dict] = []
    adjusted_var_95_portfolio: float | None = None
    vol_start = time.perf_counter()
    weight_by_ticker = {
        pos.ticker: float(weights[i]) if i < len(weights) else 0.0
        for i, pos in enumerate(positions)
        if pos.ticker in asset_returns.columns
    }
    for pos in positions:
        ticker = pos.ticker
        if ticker not in returns.columns:
            continue
        try:
            cached = await get_latest_forecast(session, portfolio.id, ticker)
            if cached and forecast_is_fresh(cached.computed_at):
                vol_forecasts.append(
                    {
                        "ticker": ticker,
                        "predicted_vol": float(cached.predicted_vol),
                        "garch_vol": float(cached.garch_vol),
                        "lstm_mae": float(cached.lstm_mae),
                        "garch_mae": float(cached.garch_mae),
                        "improvement_pct": float(cached.improvement_pct) if cached.improvement_pct is not None else None,
                        "vol_regime": cached.vol_regime,
                        "adjusted_var_95": float(cached.adjusted_var_95) if cached.adjusted_var_95 is not None else None,
                    }
                )
                continue
            if time.perf_counter() - vol_start > 10:
                logger.warning("vol_forecast_time_budget_exceeded", extra={"portfolio_id": str(portfolio.id)})
                break
            series = returns[ticker].dropna()
            w = abs(weight_by_ticker.get(ticker, 0.0))
            ticker_hist_var = float(var_95) * w
            forecaster = VolatilityForecaster(series, ticker=ticker)
            result = forecaster.evaluate_and_forecast(historical_var_95=ticker_hist_var)
            m = result.metrics
            await save_forecast(
                session,
                portfolio.id,
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
            improvement = (m.garch_mae - m.lstm_mae) / m.garch_mae * 100.0 if m.garch_mae else None
            vol_forecasts.append(
                {
                    "ticker": ticker,
                    "predicted_vol": m.predicted_vol,
                    "garch_vol": m.garch_vol,
                    "lstm_mae": m.lstm_mae,
                    "garch_mae": m.garch_mae,
                    "improvement_pct": improvement,
                    "vol_regime": m.vol_regime,
                    "adjusted_var_95": result.adjusted_var_95,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("vol_forecast_failed", extra={"ticker": ticker, "error": str(exc)})
            continue

    if vol_forecasts:
        total_w = sum(abs(weight_by_ticker.get(v["ticker"], 0.0)) for v in vol_forecasts)
        if total_w > 0:
            adjusted_var_95_portfolio = sum(
                (v.get("adjusted_var_95") or 0.0) * abs(weight_by_ticker.get(v["ticker"], 0.0)) for v in vol_forecasts
            ) / total_w

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return RiskComputationResult(
        portfolio_value=portfolio_value.quantize(Decimal("0.01")),
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        margin_utilization=utilization,
        margin_status=status,
        stress_mild=stress_losses["mild"],
        stress_moderate=stress_losses["moderate"],
        stress_severe=stress_losses["severe"],
        shap_json=shap_json,
        computation_ms=elapsed_ms,
        mc_var_95=mc_result.mc_var_95 if mc_result else None,
        mc_var_99=mc_result.mc_var_99 if mc_result else None,
        mc_cvar_95=mc_result.mc_cvar_95 if mc_result else None,
        mc_skewness=mc_result.skewness if mc_result else None,
        mc_kurtosis=mc_result.kurtosis if mc_result else None,
        mc_histogram=mc_result.histogram if mc_result else None,
        correlation_json=correlation_json,
        vol_forecasts=vol_forecasts or None,
        adjusted_var_95_portfolio=adjusted_var_95_portfolio,
    )
