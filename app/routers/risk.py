from decimal import Decimal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import Portfolio, Position, RiskComputation, User
from app.schemas import (
    CorrelationRegimeOut,
    KupiecResult,
    OptimizationResult,
    RebalancingAction,
    EfficientFrontierPoint,
    RiskOut,
    VolForecastDetailOut,
    VolForecastHistoryPoint,
)
from app.services.backtest_service import run_kupiec_test
from app.services.forecast_store import get_forecast_history, get_latest_forecasts_for_portfolio
from app.services.market_data import get_last_snapshot_price, get_latest_price
from app.services.optimizer_service import run_portfolio_optimizer
from app.services.portfolio_service import compute_position_values, compute_portfolio_value, compute_weights
from app.services.return_matrix import build_returns_matrix
from app.services.risk_pipeline import (
    forecast_rows_to_out as _forecast_rows_to_out,
    get_shap_map_for_row as _get_shap_map_for_row,
    map_risk_row_from_shap as _map_risk_row_from_shap,
    persist_computation as _persist_computation,
    refresh_risk_narrative as _refresh_risk_narrative,
)
from app.task_state import get_task_status, new_task_id, set_task_status

router = APIRouter(prefix="/portfolios/{portfolio_id}/risk", tags=["risk"])
task_router = APIRouter(prefix="/tasks", tags=["tasks"])


async def _get_owned_portfolio(db: AsyncSession, portfolio_id: str, user_id: str) -> Portfolio:
    result = await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id, Portfolio.user_id == user_id))
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio


async def _run_task(task_id: str, portfolio_id: str, user_id: str) -> None:
    await set_task_status(task_id, "RUNNING", user_id=user_id)
    db_gen = get_db()
    db = await anext(db_gen)
    try:
        portfolio = await _get_owned_portfolio(db, portfolio_id, user_id)
        row = await _persist_computation(db, portfolio, triggered_by="manual")
        shap_map = await _get_shap_map_for_row(db, row)
        result = _map_risk_row_from_shap(row, shap_map).model_dump(mode="json")
        await set_task_status(task_id, "SUCCESS", result=result, user_id=user_id)
    except Exception as exc:  # noqa: BLE001
        await set_task_status(
            task_id,
            "FAILED",
            result={"error": str(exc)},
            error=str(exc),
            user_id=user_id,
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
    if not row.risk_narrative:
        await _refresh_risk_narrative(db, row)
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
    # Volatility forecasts are a point-in-time view; attach them only to the most
    # recent row rather than pretending today's forecast applied to every past
    # computation. (Also avoids re-querying forecasts once per history row.)
    forecast_rows = await get_latest_forecasts_for_portfolio(db, portfolio_id)
    vol_out = _forecast_rows_to_out(forecast_rows) if forecast_rows else None
    out: list[RiskOut] = []
    for idx, row in enumerate(rows):
        shap_map = await _get_shap_map_for_row(db, row)
        out.append(_map_risk_row_from_shap(row, shap_map, vol_out if idx == 0 else None))
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
    return CorrelationRegimeOut(**row.correlation_json)


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


@router.post("/narrative")
async def regenerate_narrative(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    await _get_owned_portfolio(db, portfolio_id, str(user.id))
    result = await db.execute(
        select(RiskComputation)
        .where(RiskComputation.portfolio_id == portfolio_id)
        .order_by(RiskComputation.computed_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="No computation found for portfolio")
    narrative = await _refresh_risk_narrative(db, row)
    return {"risk_narrative": row.risk_narrative, "source": narrative.source}


@router.post("/compute")
async def compute_now(
    portfolio_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    await _get_owned_portfolio(db, portfolio_id, str(user.id))
    task_id = new_task_id()
    await set_task_status(task_id, "PENDING", user_id=str(user.id))
    background_tasks.add_task(_run_task, task_id, portfolio_id, str(user.id))
    return {"task_id": task_id}


@task_router.get("/{task_id}")
async def get_task(task_id: str, user: User = Depends(get_current_user)) -> dict:
    record = await get_task_status(task_id)
    # Treat "not yours" the same as "not found" so task IDs can't be probed.
    if not record or record.get("user_id") != str(user.id):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": record["status"], "result": record.get("result")}
