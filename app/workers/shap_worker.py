import asyncio
from decimal import Decimal
import numpy as np
from sqlalchemy import select
from app.database import SessionLocal
from app.models import Position, RiskComputation, ShapAttribution
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

        ordered_cols = [t for t in tickers if t in returns.columns]
        if not ordered_cols:
            return {"status": "no_ordered_cols"}
        scenarios = returns[ordered_cols].to_numpy()

        qty = np.array([float(p.quantity) for p in positions[: len(ordered_cols)]], dtype=float)
        qty = np.abs(qty)
        if float(np.sum(qty)) == 0:
            return {"status": "zero_qty"}
        weights = qty / np.sum(qty)
        contributions = compute_kernel_shap(scenarios, weights, Decimal(row.portfolio_value), positions[: len(ordered_cols)])

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
