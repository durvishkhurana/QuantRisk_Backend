from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Position, PriceSnapshot, RiskComputation


def _insufficient_result(
    rows_count: int,
    calibration: str,
    message: str,
    *,
    actual_violations: int = 0,
    violation_rate: float = 0.0,
    violation_dates: list[date] | None = None,
    series: list[dict] | None = None,
) -> dict:
    return {
        "total_days": rows_count,
        "expected_violations": round(rows_count * 0.05, 2),
        "actual_violations": actual_violations,
        "violation_rate": violation_rate,
        "kupiec_lr_statistic": None,
        "model_valid": None,
        "calibration": calibration,
        "violation_dates": violation_dates or [],
        "series": series or [],
        "message": message,
    }


async def run_kupiec_test(
    session: AsyncSession,
    portfolio_id: str,
    confidence_level: float = 0.95,
    lookback_days: int = 252,
) -> dict:
    rows = (
        await session.execute(
            select(RiskComputation)
            .where(RiskComputation.portfolio_id == portfolio_id)
            .order_by(RiskComputation.computed_at.desc())
            .limit(lookback_days)
        )
    ).scalars().all()

    if len(rows) < 30:
        return _insufficient_result(
            len(rows),
            "INSUFFICIENT_DATA",
            f"Need at least 30 days of risk history for backtesting. Currently have {len(rows)} days.",
        )

    if not rows:
        return _insufficient_result(
            0,
            "INSUFFICIENT_DATA",
            "Need at least 30 days of risk history for backtesting. Currently have 0 days.",
        )

    rows = sorted(rows, key=lambda r: r.computed_at)
    positions = (await session.execute(select(Position).where(Position.portfolio_id == portfolio_id))).scalars().all()
    tickers = [p.ticker for p in positions]
    qty_map = {p.ticker: float(p.quantity) for p in positions}

    snaps = (await session.execute(select(PriceSnapshot).where(PriceSnapshot.ticker.in_(tickers)))).scalars().all()
    price_df = pd.DataFrame([{"ticker": s.ticker, "date": s.date, "close": float(s.close)} for s in snaps])
    if price_df.empty:
        daily_pnl: dict[date, float] = {}
    else:
        prices = price_df.pivot_table(index="date", columns="ticker", values="close").sort_index()
        values = prices.apply(lambda row: sum(float(row.get(t, 0)) * qty_map.get(t, 0) for t in tickers), axis=1)
        daily_pnl = {}
        for idx in range(1, len(values)):
            d = values.index[idx]
            if hasattr(d, "date"):
                d = d.date()
            prev_val = float(values.iloc[idx - 1])
            cur_val = float(values.iloc[idx])
            daily_pnl[d] = prev_val - cur_val

    p = 1.0 - confidence_level
    violations = 0
    violation_dates: list[date] = []
    series: list[dict] = []

    prev_var: Decimal | None = None
    prev_day: date | None = None
    for row in rows:
        day = row.computed_at.date()
        var_95 = Decimal(row.var_95)
        violated = False
        if prev_var is not None and prev_day is not None:
            loss = daily_pnl.get(day, 0.0)
            if loss > float(prev_var):
                violations += 1
                violation_dates.append(day)
                violated = True
        series.append({"date": day, "var_95": var_95, "violated": violated})
        prev_var = var_95
        prev_day = day

    total_days = len(rows) - 1 if len(rows) > 1 else 0
    expected = total_days * p if total_days else 0.0
    n = violations
    t = max(total_days, 1)
    violation_rate = n / t if t else 0.0

    if n == 0 or n == t:
        lr_stat = None
        model_valid = None
        calibration = "INSUFFICIENT_VIOLATIONS" if n == 0 else "ALL_DAYS_VIOLATED"
        message = (
            "No VaR violations observed; Kupiec LR is undefined."
            if n == 0
            else "Violations on every day; Kupiec LR is undefined."
        )
    else:
        lr_stat = -2 * (
            (t - n) * math.log(max(1 - p, 1e-12))
            + n * math.log(max(p, 1e-12))
            - (t - n) * math.log(max(1 - n / t, 1e-12))
            - n * math.log(max(n / t, 1e-12))
        )
        model_valid = lr_stat < 3.841
        message = None
        if expected > 0 and n < expected * 0.5:
            calibration = "OVERESTIMATES_RISK"
        elif expected > 0 and n > expected * 2:
            calibration = "UNDERESTIMATES_RISK"
        else:
            calibration = "WELL_CALIBRATED"

    return {
        "total_days": total_days,
        "expected_violations": round(expected, 2),
        "actual_violations": n,
        "violation_rate": round(violation_rate, 4),
        "kupiec_lr_statistic": round(lr_stat, 4) if lr_stat is not None else None,
        "model_valid": model_valid,
        "calibration": calibration,
        "violation_dates": violation_dates,
        "series": series,
        "message": message,
    }
