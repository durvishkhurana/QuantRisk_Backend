from decimal import Decimal
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth import get_current_user
from app.database import get_db
from app.models import Portfolio, Position, RiskComputation, User
from app.schemas import (
    AggregateRiskResponse,
    PortfolioCreateIn,
    PortfolioPatchIn,
    PortfolioOut,
    PortfolioRiskBreakdown,
    PositionCreateIn,
    PositionOut,
    PositionPatchIn,
)
import numpy as np
from app.services.market_data import backfill_history, get_latest_price, get_latest_price_with_fallback, get_sector
from app.workers.celery_app import celery_app

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


async def _get_owned_portfolio(db: AsyncSession, portfolio_id: str, user_id: str) -> Portfolio:
    result = await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id, Portfolio.user_id == user_id))
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio


@router.post("", response_model=PortfolioOut)
async def create_portfolio(
    payload: PortfolioCreateIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PortfolioOut:
    portfolio = Portfolio(user_id=user.id, name=payload.name, margin_limit=payload.margin_limit)
    db.add(portfolio)
    await db.commit()
    await db.refresh(portfolio)
    return PortfolioOut(
        portfolio_id=portfolio.id,
        name=portfolio.name,
        margin_limit=Decimal(portfolio.margin_limit),
        positions_count=0,
        total_value=Decimal("0"),
        latest_risk=None,
    )


@router.get("/aggregate", response_model=AggregateRiskResponse)
async def aggregate_risk(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AggregateRiskResponse:
    portfolios = (await db.execute(select(Portfolio).where(Portfolio.user_id == user.id, Portfolio.is_active.is_(True)))).scalars().all()
    breakdown: list[PortfolioRiskBreakdown] = []
    total_value = Decimal("0")
    individual_vars: list[float] = []
    value_weights: list[float] = []
    var_ratios: list[tuple] = []

    for p in portfolios:
        latest = (
            await db.execute(
                select(RiskComputation).where(RiskComputation.portfolio_id == p.id).order_by(RiskComputation.computed_at.desc()).limit(1)
            )
        ).scalar_one_or_none()
        value = Decimal(latest.portfolio_value) if latest else Decimal("0")
        var_95 = Decimal(latest.var_95) if latest else Decimal("0")
        status = latest.margin_status if latest else "OK"
        total_value += value
        breakdown.append(
            PortfolioRiskBreakdown(
                portfolio_id=p.id,
                name=p.name,
                value=value,
                var_95=var_95,
                var_pct_of_total=0.0,
                margin_status=status,
            )
        )
        individual_vars.append(float(var_95))
        value_weights.append(float(value))
        if value > 0:
            var_ratios.append((p.id, float(var_95 / value)))

    if total_value > 0:
        for item in breakdown:
            item.var_pct_of_total = round(float(item.var_95 / total_value * Decimal("100")), 2)

    aggregate_var = Decimal("0")
    if individual_vars and sum(value_weights) > 0:
        w = np.array(value_weights, dtype=float)
        w = w / w.sum()
        corr = np.eye(len(w))
        if len(w) > 1:
            corr = np.full((len(w), len(w)), 0.35)
            np.fill_diagonal(corr, 1.0)
        diversification = float(np.sqrt(w.T @ corr @ w))
        aggregate_var = Decimal(str(round(sum(individual_vars) * diversification, 2)))

    most_exposed = None
    most_diversifying = None
    if var_ratios:
        most_exposed = max(var_ratios, key=lambda x: x[1])[0]
        most_diversifying = min(var_ratios, key=lambda x: x[1])[0]

    return AggregateRiskResponse(
        total_portfolio_value=total_value,
        aggregate_var_95=aggregate_var,
        portfolio_count=len(portfolios),
        breakdown=breakdown,
        most_exposed_portfolio_id=most_exposed,
        most_diversifying_portfolio_id=most_diversifying,
    )


@router.get("/{portfolio_id}")
async def get_portfolio(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    portfolio = await _get_owned_portfolio(db, portfolio_id, str(user.id))
    positions = (await db.execute(select(Position).where(Position.portfolio_id == portfolio.id))).scalars().all()
    position_rows = []
    total = Decimal("0")
    for pos in positions:
        purchase = Decimal(pos.purchase_price)
        price = await get_latest_price_with_fallback(pos.ticker, purchase)
        market = price * Decimal(pos.quantity)
        total += market
        position_rows.append(
            PositionOut(
                position_id=pos.id,
                ticker=pos.ticker,
                quantity=Decimal(pos.quantity),
                purchase_price=Decimal(pos.purchase_price),
                current_price=price,
                market_value=market.quantize(Decimal("0.01")),
                sector=pos.sector,
            ).model_dump()
        )
    return {
        "portfolio_id": str(portfolio.id),
        "name": portfolio.name,
        "margin_limit": float(portfolio.margin_limit),
        "total_value": float(total),
        "positions": position_rows,
    }


@router.delete("/{portfolio_id}", status_code=204)
async def delete_portfolio(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    portfolio = await _get_owned_portfolio(db, portfolio_id, str(user.id))
    await db.delete(portfolio)
    await db.commit()


@router.patch("/{portfolio_id}", response_model=PortfolioOut)
async def patch_portfolio(
    portfolio_id: str,
    payload: PortfolioPatchIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PortfolioOut:
    portfolio = await _get_owned_portfolio(db, portfolio_id, str(user.id))
    if payload.name is not None:
        portfolio.name = payload.name
    if payload.margin_limit is not None:
        portfolio.margin_limit = payload.margin_limit
    await db.commit()
    await db.refresh(portfolio)

    pos_count_q = await db.execute(select(func.count(Position.id)).where(Position.portfolio_id == portfolio.id))
    pos_count = int(pos_count_q.scalar_one())
    
    latest_risk_q = await db.execute(
        select(RiskComputation).where(RiskComputation.portfolio_id == portfolio.id).order_by(RiskComputation.computed_at.desc()).limit(1)
    )
    latest = latest_risk_q.scalar_one_or_none()
    
    return PortfolioOut(
        portfolio_id=portfolio.id,
        name=portfolio.name,
        margin_limit=Decimal(portfolio.margin_limit),
        positions_count=pos_count,
        total_value=Decimal(latest.portfolio_value) if latest else Decimal("0"),
        latest_risk={
            "margin_status": latest.margin_status,
            "var_95": float(latest.var_95),
            "margin_utilization": float(latest.margin_utilization),
        }
        if latest
        else None,
    )



@router.post("/{portfolio_id}/positions", response_model=PositionOut)
async def add_position(
    portfolio_id: str,
    payload: PositionCreateIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PositionOut:
    portfolio = await _get_owned_portfolio(db, portfolio_id, str(user.id))
    ticker = payload.ticker.upper()
    existing = await db.execute(select(Position).where(Position.portfolio_id == portfolio.id, Position.ticker == ticker))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Ticker already exists in portfolio")

    sector = await get_sector(ticker)
    position = Position(
        portfolio_id=portfolio.id,
        ticker=ticker,
        quantity=payload.quantity,
        purchase_price=payload.purchase_price,
        sector=sector,
    )
    db.add(position)
    await db.commit()
    await db.refresh(position)

    await backfill_history(db, ticker)
    await backfill_history(db, "SPY")
    try:
        celery_app.send_task("app.workers.history_loader.backfill_ticker_history", args=[ticker])
    except Exception:
        # Keep API resilient if broker is not available in local dev.
        pass
    price = await get_latest_price(ticker)
    market_value = (price * payload.quantity).quantize(Decimal("0.01"))

    return PositionOut(
        position_id=position.id,
        ticker=ticker,
        quantity=Decimal(position.quantity),
        purchase_price=Decimal(position.purchase_price),
        current_price=price,
        market_value=market_value,
        sector=sector,
    )


@router.patch("/{portfolio_id}/positions/{position_id}", response_model=PositionOut)
async def patch_position(
    portfolio_id: str,
    position_id: str,
    payload: PositionPatchIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PositionOut:
    portfolio = await _get_owned_portfolio(db, portfolio_id, str(user.id))
    result = await db.execute(select(Position).where(Position.id == position_id, Position.portfolio_id == portfolio.id))
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")

    if payload.quantity is not None:
        position.quantity = payload.quantity
    if payload.purchase_price is not None:
        position.purchase_price = payload.purchase_price
    await db.commit()
    await db.refresh(position)

    price = await get_latest_price(position.ticker)
    market_value = (price * Decimal(position.quantity)).quantize(Decimal("0.01"))
    return PositionOut(
        position_id=position.id,
        ticker=position.ticker,
        quantity=Decimal(position.quantity),
        purchase_price=Decimal(position.purchase_price),
        current_price=price,
        market_value=market_value,
        sector=position.sector,
    )


@router.delete("/{portfolio_id}/positions/{position_id}", status_code=204)
async def delete_position(
    portfolio_id: str,
    position_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    portfolio = await _get_owned_portfolio(db, portfolio_id, str(user.id))
    result = await db.execute(select(Position).where(Position.id == position_id, Position.portfolio_id == portfolio.id))
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")
    await db.delete(position)
    await db.commit()


@router.get("", response_model=list[PortfolioOut])
async def list_portfolios(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)) -> list[PortfolioOut]:
    result = await db.execute(select(Portfolio).where(Portfolio.user_id == user.id).order_by(Portfolio.created_at.desc()))
    portfolios = result.scalars().all()
    out: list[PortfolioOut] = []
    for p in portfolios:
        pos_count_q = await db.execute(select(func.count(Position.id)).where(Position.portfolio_id == p.id))
        pos_count = int(pos_count_q.scalar_one())
        latest_risk_q = await db.execute(
            select(RiskComputation).where(RiskComputation.portfolio_id == p.id).order_by(RiskComputation.computed_at.desc()).limit(1)
        )
        latest = latest_risk_q.scalar_one_or_none()
        out.append(
            PortfolioOut(
                portfolio_id=p.id,
                name=p.name,
                margin_limit=Decimal(p.margin_limit),
                positions_count=pos_count,
                total_value=Decimal(latest.portfolio_value) if latest else Decimal("0"),
                latest_risk={
                    "margin_status": latest.margin_status,
                    "var_95": float(latest.var_95),
                    "margin_utilization": float(latest.margin_utilization),
                }
                if latest
                else None,
            )
        )
    return out
