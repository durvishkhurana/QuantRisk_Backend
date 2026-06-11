from decimal import Decimal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from sqlalchemy import select

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user

from app.database import get_db

from app.models import MarginEvent, Portfolio, Position, RiskComputation, ShapAttribution, User

from app.schemas import (

    CorrelationRegimeOut,

    KupiecResult,

    MonteCarloOut,

    OptimizationResult,

    RebalancingAction,

    EfficientFrontierPoint,

    RiskOut,

    ShapContributionOut,

    StressScenarioOut,

    VolForecastOut,

    VolForecastDetailOut,

    VolForecastHistoryPoint,

)

from app.services.alerts import socket_manager

from app.services.backtest_service import run_kupiec_test

from app.services.market_data import get_last_snapshot_price, get_latest_price

from app.services.optimizer_service import run_portfolio_optimizer

from app.services.portfolio_service import compute_position_values, compute_portfolio_value, compute_weights

from app.services.redis_client import publish_json

from app.services.return_matrix import build_returns_matrix

from app.services.risk import compute_portfolio_risk
from app.services.forecast_store import get_forecast_history, get_latest_forecasts_for_portfolio

from app.task_state import get_task_status, new_task_id, set_task_status

from app.workers.celery_app import celery_app



router = APIRouter(prefix="/portfolios/{portfolio_id}/risk", tags=["risk"])

task_router = APIRouter(prefix="/tasks", tags=["tasks"])





async def _get_shap_map_for_row(db: AsyncSession, row: RiskComputation) -> dict[str, float]:

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





def _monte_carlo_from_row(row: RiskComputation) -> MonteCarloOut | None:

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





def _forecast_rows_to_out(rows) -> list[VolForecastOut]:
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


def _portfolio_adjusted_var(forecasts: list[VolForecastOut]) -> float | None:
    vals = [f.adjusted_var_95 for f in forecasts if f.adjusted_var_95 is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _map_risk_row_from_shap(
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

                pct=(Decimal(row.stress_mild) / Decimal(row.portfolio_value) * Decimal("100")).quantize(Decimal("0.01")),

            ),

            "moderate": StressScenarioOut(

                loss=Decimal(row.stress_moderate),

                pct=(Decimal(row.stress_moderate) / Decimal(row.portfolio_value) * Decimal("100")).quantize(Decimal("0.01")),

            ),

            "severe": StressScenarioOut(

                loss=Decimal(row.stress_severe),

                pct=(Decimal(row.stress_severe) / Decimal(row.portfolio_value) * Decimal("100")).quantize(Decimal("0.01")),

            ),

        },

        shap_attribution=shap,

        computation_ms=row.computation_ms,

        monte_carlo=_monte_carlo_from_row(row),

        risk_narrative=row.risk_narrative,

        vol_forecasts=vol_forecasts,

        adjusted_var_95_portfolio=_portfolio_adjusted_var(vol_forecasts) if vol_forecasts else None,

    )





async def _get_owned_portfolio(db: AsyncSession, portfolio_id: str, user_id: str) -> Portfolio:

    result = await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id, Portfolio.user_id == user_id))

    portfolio = result.scalar_one_or_none()

    if not portfolio:

        raise HTTPException(status_code=404, detail="Portfolio not found")

    return portfolio





async def _persist_computation(db: AsyncSession, portfolio: Portfolio, triggered_by: str = "manual") -> RiskComputation:

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

    try:

        celery_app.send_task("app.workers.shap_worker.compute_kernel_shap_for_computation", args=[str(row.id)])

    except Exception:

        pass

    try:

        celery_app.send_task("app.workers.risk_worker.generate_risk_narrative_task", args=[str(row.id)])

    except Exception:

        pass



    shap_map = await _get_shap_map_for_row(db, row)

    forecast_rows = await get_latest_forecasts_for_portfolio(db, portfolio.id)

    vol_out = _forecast_rows_to_out(forecast_rows) if forecast_rows else None

    mapped = _map_risk_row_from_shap(row, shap_map, vol_out)



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





async def _run_task(task_id: str, portfolio_id: str, user_id: str) -> None:
    await set_task_status(task_id, "RUNNING")
    db_gen = get_db()
    db = await anext(db_gen)
    try:
        portfolio = await _get_owned_portfolio(db, portfolio_id, user_id)
        row = await _persist_computation(db, portfolio, triggered_by="manual")
        shap_map = await _get_shap_map_for_row(db, row)
        result = _map_risk_row_from_shap(row, shap_map).model_dump(mode="json")
        await set_task_status(task_id, "SUCCESS", result=result)
    except Exception as exc:  # noqa: BLE001
        await set_task_status(
            task_id,
            "FAILED",
            result={"error": str(exc)},
            error=str(exc),
        )
    finally:
        await db.close()
        await db_gen.aclose()





@router.get("", response_model=RiskOut)

async def get_latest_risk(

    portfolio_id: str,

    db: AsyncSession = Depends(get_db),

    user: User = Depends(get_current_user),

) -> RiskOut:

    await _get_owned_portfolio(db, portfolio_id, str(user.id))

    result = await db.execute(

        select(RiskComputation).where(RiskComputation.portfolio_id == portfolio_id).order_by(RiskComputation.computed_at.desc()).limit(1)

    )

    row = result.scalar_one_or_none()

    if not row:

        raise HTTPException(status_code=404, detail="No computation found for portfolio")

    shap_map = await _get_shap_map_for_row(db, row)

    forecast_rows = await get_latest_forecasts_for_portfolio(db, row.portfolio_id)

    vol_out = _forecast_rows_to_out(forecast_rows) if forecast_rows else None

    return _map_risk_row_from_shap(row, shap_map, vol_out)





@router.get("/volatility-forecast", response_model=list[VolForecastDetailOut])
async def get_volatility_forecast(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[VolForecastDetailOut]:
    await _get_owned_portfolio(db, portfolio_id, str(user.id))
    latest = await get_latest_forecasts_for_portfolio(db, portfolio_id)
    if not latest:
        return []
    out: list[VolForecastDetailOut] = []
    for row in latest:
        history = await get_forecast_history(db, portfolio_id, row.ticker, days=30)
        out.append(
            VolForecastDetailOut(
                ticker=row.ticker,
                predicted_vol=float(row.predicted_vol),
                garch_vol=float(row.garch_vol),
                lstm_mae=float(row.lstm_mae),
                garch_mae=float(row.garch_mae),
                improvement_pct=float(row.improvement_pct) if row.improvement_pct is not None else None,
                vol_regime=row.vol_regime,
                adjusted_var_95=float(row.adjusted_var_95) if row.adjusted_var_95 is not None else None,
                lstm_rmse=float(row.lstm_rmse),
                garch_rmse=float(row.garch_rmse),
                direction_accuracy=float(row.direction_accuracy),
                history=[
                    VolForecastHistoryPoint(
                        computed_at=h.computed_at,
                        predicted_vol=float(h.predicted_vol),
                        garch_vol=float(h.garch_vol),
                    )
                    for h in reversed(history)
                ],
            )
        )
    return out


@router.get("/history")

async def get_risk_history(

    portfolio_id: str,

    days: int = 30,

    db: AsyncSession = Depends(get_db),

    user: User = Depends(get_current_user),

) -> list[RiskOut]:

    await _get_owned_portfolio(db, portfolio_id, str(user.id))

    result = await db.execute(

        select(RiskComputation).where(RiskComputation.portfolio_id == portfolio_id).order_by(RiskComputation.computed_at.desc()).limit(days)

    )

    rows = result.scalars().all()

    out: list[RiskOut] = []

    for row in rows:

        shap_map = await _get_shap_map_for_row(db, row)

        forecast_rows = await get_latest_forecasts_for_portfolio(db, row.portfolio_id)

        vol_out = _forecast_rows_to_out(forecast_rows) if forecast_rows else None

        out.append(_map_risk_row_from_shap(row, shap_map, vol_out))

    return out





@router.get("/optimize", response_model=OptimizationResult)

async def optimize_portfolio(

    portfolio_id: str,

    db: AsyncSession = Depends(get_db),

    user: User = Depends(get_current_user),

) -> OptimizationResult:

    portfolio = await _get_owned_portfolio(db, portfolio_id, str(user.id))

    positions = (await db.execute(select(Position).where(Position.portfolio_id == portfolio.id))).scalars().all()

    if not positions:

        raise HTTPException(status_code=400, detail="Portfolio has no positions")



    tickers = [p.ticker.upper() for p in positions]

    prices: dict[str, Decimal] = {}

    for pos in positions:

        try:

            prices[pos.ticker] = await get_latest_price(pos.ticker)

        except Exception:  # noqa: BLE001

            fallback = await get_last_snapshot_price(db, pos.ticker)

            if fallback is None:

                raise HTTPException(status_code=400, detail=f"No price for {pos.ticker}") from None

            prices[pos.ticker] = fallback



    position_values = compute_position_values(positions, prices)

    portfolio_value = compute_portfolio_value(position_values)

    weights = compute_weights(positions, position_values, portfolio_value)



    returns = await build_returns_matrix(db, tickers)

    available = [t for t in tickers if t in returns.columns]

    if len(available) < 2:

        raise HTTPException(status_code=400, detail="Insufficient history for optimization")



    opt = run_portfolio_optimizer(

        returns[available],

        available,

        weights[: len(available)],

        portfolio_value,

        prices,

        {p.ticker: Decimal(p.quantity) for p in positions},

    )

    return OptimizationResult(

        current_var_95=opt["current_var_95"],

        optimized_var_95=opt["optimized_var_95"],

        var_reduction_pct=opt["var_reduction_pct"],

        rebalancing_actions=[RebalancingAction(**a) for a in opt["rebalancing_actions"]],

        efficient_frontier=[EfficientFrontierPoint(**p) for p in opt["efficient_frontier"]],

    )





@router.get("/correlation", response_model=CorrelationRegimeOut)

async def get_correlation(

    portfolio_id: str,

    db: AsyncSession = Depends(get_db),

    user: User = Depends(get_current_user),

) -> CorrelationRegimeOut:

    await _get_owned_portfolio(db, portfolio_id, str(user.id))

    result = await db.execute(

        select(RiskComputation).where(RiskComputation.portfolio_id == portfolio_id).order_by(RiskComputation.computed_at.desc()).limit(1)

    )

    row = result.scalar_one_or_none()

    if not row or not row.correlation_json:

        raise HTTPException(status_code=404, detail="No correlation data found")

    data = row.correlation_json

    return CorrelationRegimeOut(**data)





@router.get("/backtest", response_model=KupiecResult)

async def get_backtest(

    portfolio_id: str,

    days: int = 252,

    db: AsyncSession = Depends(get_db),

    user: User = Depends(get_current_user),

) -> KupiecResult:

    await _get_owned_portfolio(db, portfolio_id, str(user.id))

    result = await run_kupiec_test(db, portfolio_id, confidence_level=0.95, lookback_days=days)

    return KupiecResult(**result)





@router.post("/compute")

async def compute_now(

    portfolio_id: str,

    background_tasks: BackgroundTasks,

    db: AsyncSession = Depends(get_db),

    user: User = Depends(get_current_user),

) -> dict:

    await _get_owned_portfolio(db, portfolio_id, str(user.id))

    task_id = new_task_id()
    await set_task_status(task_id, "PENDING")
    background_tasks.add_task(_run_task, task_id, portfolio_id, str(user.id))
    return {"task_id": task_id}





@task_router.get("/{task_id}")

async def get_task(task_id: str) -> dict:
    record = await get_task_status(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": record["status"], "result": record.get("result")}

