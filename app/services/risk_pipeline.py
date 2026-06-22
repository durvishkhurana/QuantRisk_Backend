"""Risk persistence + serialization pipeline.

This lives in the service layer (not the router) so that Celery workers and the
HTTP routers both depend *downward* on it — previously `risk_worker` imported
`_persist_computation` from `routers.risk`, inverting the dependency direction.
"""
from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MarginEvent, Portfolio, Position, RiskComputation, ShapAttribution
from app.schemas import (
    MonteCarloOut,
    RiskOut,
    ShapContributionOut,
    StressScenarioOut,
    VolForecastOut,
)
from app.services.alerts import socket_manager
from app.services.forecast_store import get_latest_forecasts_for_portfolio
from app.services.nlp_report_service import generate_risk_narrative
from app.services.redis_client import publish_json
from app.services.risk import compute_portfolio_risk
from app.workers.celery_app import celery_app

# Async tasks queued after every computation. Narrative is generated exactly once
# (here via Celery, or lazily on first GET if no broker is available). Volatility
# model training (LSTM/GARCH) and kernel SHAP are offloaded so they never block
# the compute hot path or the 60s beat.
_ASYNC_TASKS = (
    "app.workers.shap_worker.compute_kernel_shap_for_computation",
    "app.workers.risk_worker.generate_risk_narrative_task",
    "app.workers.vol_worker.compute_volatility_forecasts",
)


async def get_shap_map_for_row(db: AsyncSession, row: RiskComputation) -> dict[str, float]:
    kernel_rows = (
        await db.execute(
            select(ShapAttribution).where(
                ShapAttribution.risk_computation_id == row.id,
                ShapAttribution.method == "kernel",
            )
        )
    ).scalars().all()
    if kernel_rows:
        return {r.ticker: float(r.shap_value) for r in kernel_rows}
    return {k: float(v) for k, v in (row.shap_json or {}).items()}


def monte_carlo_from_row(row: RiskComputation) -> MonteCarloOut | None:
    if row.mc_var_95 is None:
        return None
    return MonteCarloOut(
        var_95=Decimal(row.mc_var_95),
        var_99=Decimal(row.mc_var_99 or 0),
        cvar_95=Decimal(row.mc_cvar_95 or 0),
        skewness=float(row.mc_skewness or 0),
        kurtosis=float(row.mc_kurtosis or 0),
        n_simulations=10000,
        histogram=row.mc_histogram,
    )


def forecast_rows_to_out(rows) -> list[VolForecastOut]:
    return [
        VolForecastOut(
            ticker=r.ticker,
            predicted_vol=float(r.predicted_vol),
            garch_vol=float(r.garch_vol),
            lstm_mae=float(r.lstm_mae),
            garch_mae=float(r.garch_mae),
            improvement_pct=float(r.improvement_pct) if r.improvement_pct is not None else None,
            vol_regime=r.vol_regime,
            adjusted_var_95=float(r.adjusted_var_95) if r.adjusted_var_95 is not None else None,
        )
        for r in rows
    ]


def portfolio_adjusted_var(forecasts: list[VolForecastOut]) -> float | None:
    vals = [f.adjusted_var_95 for f in forecasts if f.adjusted_var_95 is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def map_risk_row_from_shap(
    row: RiskComputation,
    shap_map: dict[str, float],
    vol_forecasts: list[VolForecastOut] | None = None,
) -> RiskOut:
    shap_items = sorted(shap_map.items(), key=lambda x: x[1], reverse=True)
    var = Decimal(row.var_95) if Decimal(row.var_95) != 0 else Decimal("1")
    shap = [
        ShapContributionOut(
            ticker=t,
            contribution=Decimal(str(v)),
            pct_of_var=(Decimal(str(v)) / var * Decimal("100")).quantize(Decimal("0.01")),
        )
        for t, v in shap_items
    ]
    pv = Decimal(row.portfolio_value) if Decimal(row.portfolio_value) != 0 else Decimal("1")
    return RiskOut(
        computed_at=row.computed_at,
        portfolio_value=Decimal(row.portfolio_value),
        var_95=Decimal(row.var_95),
        var_99=Decimal(row.var_99),
        cvar_95=Decimal(row.cvar_95),
        margin_utilization=Decimal(row.margin_utilization),
        margin_status=row.margin_status,
        stress_tests={
            "mild": StressScenarioOut(
                loss=Decimal(row.stress_mild),
                pct=(Decimal(row.stress_mild) / pv * Decimal("100")).quantize(Decimal("0.01")),
            ),
            "moderate": StressScenarioOut(
                loss=Decimal(row.stress_moderate),
                pct=(Decimal(row.stress_moderate) / pv * Decimal("100")).quantize(Decimal("0.01")),
            ),
            "severe": StressScenarioOut(
                loss=Decimal(row.stress_severe),
                pct=(Decimal(row.stress_severe) / pv * Decimal("100")).quantize(Decimal("0.01")),
            ),
        },
        shap_attribution=shap,
        computation_ms=row.computation_ms,
        monte_carlo=monte_carlo_from_row(row),
        risk_narrative=row.risk_narrative,
        vol_forecasts=vol_forecasts,
        adjusted_var_95_portfolio=portfolio_adjusted_var(vol_forecasts) if vol_forecasts else None,
    )


async def refresh_risk_narrative(db: AsyncSession, row: RiskComputation):
    shap_map = await get_shap_map_for_row(db, row)
    forecast_rows = await get_latest_forecasts_for_portfolio(db, row.portfolio_id)
    vol_out = forecast_rows_to_out(forecast_rows) if forecast_rows else None
    mapped = map_risk_row_from_shap(row, shap_map, vol_out)
    result = await generate_risk_narrative(mapped.model_dump(mode="json"))
    row.risk_narrative = result.text
    await db.commit()
    await db.refresh(row)
    return result


async def persist_computation(db: AsyncSession, portfolio: Portfolio, triggered_by: str = "manual") -> RiskComputation:
    res = await compute_portfolio_risk(db, portfolio)
    pos_result = await db.execute(select(Position.id).where(Position.portfolio_id == portfolio.id))
    n_positions = len(pos_result.scalars().all())

    row = RiskComputation(
        portfolio_id=portfolio.id,
        portfolio_value=res.portfolio_value,
        var_95=res.var_95,
        var_99=res.var_99,
        cvar_95=res.cvar_95,
        margin_utilization=res.margin_utilization,
        margin_status=res.margin_status,
        stress_mild=res.stress_mild,
        stress_moderate=res.stress_moderate,
        stress_severe=res.stress_severe,
        shap_json=res.shap_json,
        n_positions=n_positions,
        lookback_days=252,
        computation_ms=res.computation_ms,
        triggered_by=triggered_by,
        mc_var_95=res.mc_var_95,
        mc_var_99=res.mc_var_99,
        mc_cvar_95=res.mc_cvar_95,
        mc_skewness=Decimal(str(res.mc_skewness)) if res.mc_skewness is not None else None,
        mc_kurtosis=Decimal(str(res.mc_kurtosis)) if res.mc_kurtosis is not None else None,
        mc_histogram=res.mc_histogram,
        correlation_json=res.correlation_json,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)

    for ticker, contribution in (row.shap_json or {}).items():
        pct = (Decimal(str(contribution)) / Decimal(row.var_95) * Decimal("100")) if Decimal(row.var_95) else Decimal("0")
        db.add(
            ShapAttribution(
                risk_computation_id=row.id,
                ticker=ticker,
                shap_value=Decimal(str(contribution)).quantize(Decimal("0.01")),
                method="linear",
                pct_of_var=pct.quantize(Decimal("0.01")),
                position_weight=None,
            )
        )
    await db.commit()

    for task_name in _ASYNC_TASKS:
        try:
            celery_app.send_task(task_name, args=[str(row.id)])
        except Exception:
            # Keep the API resilient when the broker is unavailable in local dev.
            pass

    shap_map = await get_shap_map_for_row(db, row)
    forecast_rows = await get_latest_forecasts_for_portfolio(db, portfolio.id)
    vol_out = forecast_rows_to_out(forecast_rows) if forecast_rows else None
    mapped = map_risk_row_from_shap(row, shap_map, vol_out)

    if row.margin_status in {"BREACH", "WARNING"}:
        event = MarginEvent(
            portfolio_id=portfolio.id,
            event_type=row.margin_status,
            var_95=row.var_95,
            margin_limit=portfolio.margin_limit,
            margin_utilization=row.margin_utilization,
            risk_computation_id=row.id,
        )
        db.add(event)
        await db.commit()
        margin_payload = {"type": f"margin_{row.margin_status.lower()}", "payload": mapped.model_dump(mode="json")}
        await socket_manager.broadcast(str(portfolio.id), margin_payload)
        await publish_json(f"alerts:{portfolio.id}", margin_payload)

    corr = res.correlation_json or {}
    if corr.get("regime") == "STRESS" and row.margin_status != "BREACH":
        corr_event = MarginEvent(
            portfolio_id=portfolio.id,
            event_type="CORRELATION_ALERT",
            var_95=row.var_95,
            margin_limit=portfolio.margin_limit,
            margin_utilization=row.margin_utilization,
            risk_computation_id=row.id,
        )
        db.add(corr_event)
        await db.commit()
        corr_payload = {"type": "CORRELATION_ALERT", "payload": corr}
        await socket_manager.broadcast(str(portfolio.id), corr_payload)
        await publish_json(f"alerts:{portfolio.id}", corr_payload)

    risk_payload = {"type": "risk_update", "payload": mapped.model_dump(mode="json")}
    await socket_manager.broadcast(str(portfolio.id), risk_payload)
    await publish_json(f"alerts:{portfolio.id}", risk_payload)
    return row
