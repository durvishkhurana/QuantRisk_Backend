import asyncio
from decimal import Decimal
from sqlalchemy import select
from app.database import SessionLocal
from app.models import Position, RiskComputation, ShapAttribution
from app.services.market_data import get_last_snapshot_price
from app.services.portfolio_service import compute_portfolio_value, compute_position_values, compute_weights
from app.services.return_matrix import build_returns_matrix
from app.services.shap_kernel import compute_kernel_shap
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.shap_worker.compute_kernel_shap_for_computation")
def compute_kernel_shap_for_computation(risk_computation_id: str) -> dict:
    return asyncio.run(_compute(risk_computation_id))


async def _compute(risk_computation_id: str) -> dict:
    async with SessionLocal() as db:
        row = (await db.execute(select(RiskComputation).where(RiskComputation.id == risk_computation_id))).scalar_one_or_none()
        if not row:
            return {"status": "not_found"}

        positions = (
            await db.execute(select(Position).where(Position.portfolio_id == row.portfolio_id).order_by(Position.ticker.asc()))
        ).scalars().all()
        if not positions:
            return {"status": "no_positions"}

        tickers = [p.ticker for p in positions]
        returns = await build_returns_matrix(db, tickers)
        if returns.empty:
            return {"status": "no_returns"}

        # Use the same **market-value** weights as the inline linear attribution
        # (value / portfolio_value), not quantity-normalized weights — otherwise
        # the kernel and linear SHAP numbers disagree for the same computation.
        ordered_positions = [p for p in positions if p.ticker in returns.columns]
        if not ordered_positions:
            return {"status": "no_ordered_cols"}
        prices: dict[str, Decimal] = {}
        for pos in ordered_positions:
            snap = await get_last_snapshot_price(db, pos.ticker)
            if snap is not None:
                prices[pos.ticker] = snap
        priced = [p for p in ordered_positions if p.ticker in prices]
        if not priced:
            return {"status": "no_prices"}
        position_values = compute_position_values(priced, prices)
        portfolio_value = compute_portfolio_value(position_values)
        weights = compute_weights(priced, position_values, portfolio_value)
        if weights.size == 0:
            return {"status": "zero_value"}
        scenarios = returns[[p.ticker for p in priced]].to_numpy()
        contributions = compute_kernel_shap(scenarios, weights, Decimal(row.portfolio_value), priced)

        for ticker, val in contributions.items():
            pct = (Decimal(str(val)) / Decimal(row.var_95) * Decimal("100")) if Decimal(row.var_95) else Decimal("0")
            db.add(
                ShapAttribution(
                    risk_computation_id=row.id,
                    ticker=ticker,
                    shap_value=Decimal(str(val)).quantize(Decimal("0.01")),
                    method="kernel",
                    pct_of_var=pct.quantize(Decimal("0.01")),
                    position_weight=None,
                )
            )
        await db.commit()
        return {"status": "ok", "n": len(contributions)}
