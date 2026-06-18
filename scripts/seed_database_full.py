"""
Seed PostgreSQL with users, portfolios, stock positions, price history, and risk rows.

Use this when the app shows logins but empty portfolios or $0 / no risk charts.
Reads DATABASE_URL or SUPABASE_DATABASE_URL from the repo root .env.

Run from backend/ (after `alembic upgrade head`):

  python scripts/seed_database_full.py
  python scripts/seed_database_full.py --force-prices
  python scripts/seed_database_full.py --skip-risk

Log in at /auth with demo@quantrisk.com / QuantRisk2025! or tester@quantrisk.com / Tester2025!
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import delete, func, select

_BACKEND = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_BACKEND / ".env")

sys.path.insert(0, str(_BACKEND))

from app.auth import hash_password  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import Portfolio, Position, PriceSnapshot, RiskComputation, ShapAttribution, User  # noqa: E402

get_settings.cache_clear()

TEST_USERS: list[tuple[str, str, str]] = [
    ("demo@quantrisk.com", "QuantRisk2025!", "Primary demo — full portfolio + risk"),
    ("analyst@quantrisk.com", "Analyst2025!", "Secondary — smaller tech-heavy book"),
    ("tester@quantrisk.com", "Tester2025!", "Showcase — five themed portfolios with live-style data"),
]

# ticker, quantity, purchase_price
DEMO_PORTFOLIO_SPECS: list[tuple[str, list[tuple[str, float, float]]]] = [
    (
        "Demo Portfolio",
        [
            ("AAPL", 120, 185.00),
            ("MSFT", 80, 390.00),
            ("GOOGL", 45, 175.00),
            ("NVDA", 100, 120.00),
            ("TSLA", 50, 250.00),
            ("JPM", 150, 195.00),
            ("GS", 60, 480.00),
            ("SPY", 200, 450.00),
        ],
    ),
    (
        "Analyst Growth Book",
        [
            ("AAPL", 40, 190.00),
            ("MSFT", 30, 400.00),
            ("NVDA", 75, 115.00),
            ("AMD", 90, 140.00),
            ("META", 35, 520.00),
        ],
    ),
]

# Five portfolios for tester@quantrisk.com (ticker, quantity, purchase_price)
TESTER_PORTFOLIO_SPECS: list[tuple[str, list[tuple[str, float, float]]]] = [
    (
        "Mega Cap Technology",
        [
            ("AAPL", 150, 178.00),
            ("MSFT", 90, 380.00),
            ("GOOGL", 60, 165.00),
            ("META", 45, 480.00),
            ("NVDA", 120, 95.00),
        ],
    ),
    (
        "Global Financials",
        [
            ("JPM", 200, 185.00),
            ("BAC", 400, 32.00),
            ("GS", 75, 450.00),
            ("MS", 110, 88.00),
            ("V", 85, 270.00),
        ],
    ),
    (
        "Consumer & Retail",
        [
            ("AMZN", 80, 165.00),
            ("WMT", 120, 68.00),
            ("COST", 35, 720.00),
            ("HD", 55, 340.00),
            ("MCD", 90, 285.00),
        ],
    ),
    (
        "Healthcare Defensive",
        [
            ("JNJ", 140, 155.00),
            ("UNH", 50, 520.00),
            ("PFE", 300, 28.00),
            ("ABBV", 70, 165.00),
            ("MRK", 95, 115.00),
        ],
    ),
    (
        "ETF Core Allocation",
        [
            ("SPY", 180, 480.00),
            ("QQQ", 100, 420.00),
            ("IWM", 150, 195.00),
            ("VTI", 130, 240.00),
            ("AGG", 220, 98.00),
        ],
    ),
]

SECTOR_BY_TICKER: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "NVDA": "Technology",
    "TSLA": "Consumer Cyclical",
    "AMD": "Technology",
    "META": "Technology",
    "JPM": "Financials",
    "GS": "Financials",
    "SPY": "ETF",
    "BAC": "Financials",
    "MS": "Financials",
    "V": "Financials",
    "AMZN": "Consumer Cyclical",
    "WMT": "Consumer Defensive",
    "COST": "Consumer Defensive",
    "HD": "Consumer Cyclical",
    "MCD": "Consumer Cyclical",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "QQQ": "ETF",
    "IWM": "ETF",
    "VTI": "ETF",
    "AGG": "ETF",
}

MIN_HISTORY_ROWS = 200
LOOKBACK_DAYS = 252

# Approximate recent closes for synthetic history when yfinance is unavailable.
REFERENCE_CLOSE: dict[str, float] = {
    "AAPL": 220.0,
    "MSFT": 420.0,
    "GOOGL": 175.0,
    "NVDA": 135.0,
    "TSLA": 250.0,
    "AMD": 160.0,
    "META": 580.0,
    "JPM": 210.0,
    "GS": 520.0,
    "SPY": 550.0,
    "BAC": 38.0,
    "MS": 105.0,
    "V": 310.0,
    "AMZN": 195.0,
    "WMT": 75.0,
    "COST": 920.0,
    "HD": 390.0,
    "MCD": 295.0,
    "JNJ": 158.0,
    "UNH": 580.0,
    "PFE": 26.0,
    "ABBV": 178.0,
    "MRK": 125.0,
    "QQQ": 510.0,
    "IWM": 220.0,
    "VTI": 280.0,
    "AGG": 98.0,
}


def _sector(ticker: str) -> str | None:
    return SECTOR_BY_TICKER.get(ticker.upper())


async def _get_or_create_user(db, email: str, password: str) -> User:
    result = await db.execute(select(User).where(User.email == email.lower()))
    user = result.scalar_one_or_none()
    if user:
        user.password_hash = hash_password(password)
        print(f"  User exists (password reset): {email}")
        return user
    user = User(email=email.lower(), password_hash=hash_password(password))
    db.add(user)
    await db.flush()
    print(f"  Created user: {email}")
    return user


async def _get_or_create_portfolio(db, user: User, name: str) -> Portfolio:
    result = await db.execute(select(Portfolio).where(Portfolio.user_id == user.id, Portfolio.name == name))
    portfolio = result.scalar_one_or_none()
    if portfolio:
        print(f"  Portfolio exists: {name} ({portfolio.id})")
        return portfolio
    portfolio = Portfolio(
        user_id=user.id,
        name=name,
        margin_limit=Decimal("0.05"),
        is_active=True,
    )
    db.add(portfolio)
    await db.flush()
    print(f"  Created portfolio: {name} ({portfolio.id})")
    return portfolio


async def _upsert_positions(
    db,
    portfolio: Portfolio,
    positions: list[tuple[str, float, float]],
) -> list[str]:
    tickers: list[str] = []
    for raw_ticker, qty, px in positions:
        ticker = raw_ticker.upper()
        tickers.append(ticker)
        existing = await db.execute(
            select(Position).where(Position.portfolio_id == portfolio.id, Position.ticker == ticker)
        )
        pos = existing.scalar_one_or_none()
        if pos:
            pos.quantity = Decimal(str(qty))
            pos.purchase_price = Decimal(str(px))
            pos.sector = _sector(ticker)
            print(f"    Updated {ticker} × {qty}")
        else:
            db.add(
                Position(
                    portfolio_id=portfolio.id,
                    ticker=ticker,
                    quantity=Decimal(str(qty)),
                    purchase_price=Decimal(str(px)),
                    sector=_sector(ticker),
                )
            )
            print(f"    Added {ticker} × {qty}")
    await db.flush()
    return tickers


def _synthetic_price_rows(ticker: str, lookback_days: int) -> list[PriceSnapshot]:
    ticker = ticker.upper()
    base = REFERENCE_CLOSE.get(ticker, 100.0)
    end = date.today()
    dates = pd.bdate_range(end=end, periods=lookback_days)
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    daily_returns = rng.normal(0.0002, 0.011, len(dates))
    closes = base * np.exp(np.cumsum(daily_returns - daily_returns.mean()))
    rows: list[PriceSnapshot] = []
    for dt, close in zip(dates, closes, strict=False):
        rows.append(
            PriceSnapshot(
                ticker=ticker,
                date=date.fromisoformat(str(dt.date())),
                close=Decimal(str(round(float(close), 4))),
                volume=1_000_000,
                source="synthetic",
            )
        )
    return rows


async def _insert_price_rows(db, ticker: str, rows: list[PriceSnapshot]) -> int:
    if not rows:
        return 0
    db.add_all(rows)
    await db.flush()
    return len(rows)


async def ensure_price_history(
    db,
    ticker: str,
    *,
    lookback_days: int = LOOKBACK_DAYS,
    force: bool = False,
    synthetic_only: bool = False,
) -> int:
    ticker = ticker.upper()
    count_q = await db.execute(
        select(func.count()).select_from(PriceSnapshot).where(PriceSnapshot.ticker == ticker)
    )
    count = int(count_q.scalar_one())
    if count >= MIN_HISTORY_ROWS and not force:
        print(f"    Prices {ticker}: {count} rows (skip)")
        return count

    if count > 0 and force:
        await db.execute(delete(PriceSnapshot).where(PriceSnapshot.ticker == ticker))
        await db.flush()
        print(f"    Prices {ticker}: cleared {count} rows (force)")

    rows: list[PriceSnapshot] = []
    if not synthetic_only:
        try:
            history = yf.download(ticker, period="1y", auto_adjust=False, progress=False)
            if not history.empty:
                history = history.tail(lookback_days)
                for dt, row in history.iterrows():
                    close = row.get("Close")
                    if pd.isna(close):
                        continue
                    rows.append(
                        PriceSnapshot(
                            ticker=ticker,
                            date=date.fromisoformat(str(dt.date())),
                            close=Decimal(str(round(float(close), 4))),
                            volume=int(row.get("Volume")) if not pd.isna(row.get("Volume")) else None,
                            source="yfinance",
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"    Prices {ticker}: yfinance error ({exc})", file=sys.stderr)

    if not rows:
        rows = _synthetic_price_rows(ticker, lookback_days)
        print(f"    Prices {ticker}: using {len(rows)} synthetic rows (yfinance unavailable)")

    inserted = await _insert_price_rows(db, ticker, rows)
    print(f"    Prices {ticker}: stored {inserted} rows")
    return inserted


async def _persist_risk_seed(db, portfolio: Portfolio) -> RiskComputation | None:
    from app.services.risk import compute_portfolio_risk

    try:
        res = await compute_portfolio_risk(db, portfolio)
    except Exception as exc:  # noqa: BLE001
        print(f"    Risk failed for {portfolio.name}: {exc}", file=sys.stderr)
        return None

    pos_count = int(
        (
            await db.execute(select(func.count()).select_from(Position).where(Position.portfolio_id == portfolio.id))
        ).scalar_one()
    )
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
        n_positions=pos_count,
        lookback_days=LOOKBACK_DAYS,
        computation_ms=res.computation_ms,
        triggered_by="seed",
        mc_var_95=res.mc_var_95,
        mc_var_99=res.mc_var_99,
        mc_cvar_95=res.mc_cvar_95,
        mc_skewness=Decimal(str(res.mc_skewness)) if res.mc_skewness is not None else None,
        mc_kurtosis=Decimal(str(res.mc_kurtosis)) if res.mc_kurtosis is not None else None,
        mc_histogram=res.mc_histogram,
        correlation_json=res.correlation_json,
    )
    db.add(row)
    await db.flush()

    for ticker, contribution in (row.shap_json or {}).items():
        var_d = Decimal(row.var_95)
        if var_d:
            pct = Decimal(str(contribution)) / var_d * Decimal("100")
            pct = max(Decimal("-9999.99"), min(Decimal("9999.99"), pct))
        else:
            pct = Decimal("0")
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
    await db.flush()
    print(
        f"    Risk: value=${float(row.portfolio_value):,.0f} "
        f"VaR95=${float(row.var_95):,.0f} status={row.margin_status}"
    )
    return row


async def run_seed(*, force_prices: bool, skip_risk: bool, synthetic_only: bool) -> None:
    settings = get_settings()
    db_label = settings.database_url.split("@")[-1] if "@" in settings.database_url else "(local)"
    print(f"Database target: …@{db_label}")
    print("Seeding full demo data…\n")

    # Avoid Redis dependency during offline seeding; snapshots supply prices.
    import app.services.market_data as market_data

    async def _cache_miss(*_args, **_kwargs):
        return None

    async def _cache_set(*_args, **_kwargs):
        return None

    market_data.cache_get_json = _cache_miss
    market_data.cache_set_json = _cache_set

    async def _price_from_db_first(ticker: str) -> Decimal:
        async with SessionLocal() as snap_db:
            from app.services.market_data import get_last_snapshot_price

            snap = await get_last_snapshot_price(snap_db, ticker)
            if snap is not None:
                return snap
        return await market_data.get_latest_price(ticker)

    market_data.get_latest_price = _price_from_db_first

    import app.services.risk as risk_module

    risk_module.get_latest_price = _price_from_db_first

    class _FastVolForecaster:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def evaluate_and_forecast(self, historical_var_95: float = 0.0):
            from types import SimpleNamespace

            metrics = SimpleNamespace(
                predicted_vol=0.2,
                garch_vol=0.21,
                lstm_mae=0.01,
                garch_mae=0.02,
                lstm_rmse=0.01,
                garch_rmse=0.02,
                direction_accuracy=0.55,
                vol_regime="NORMAL",
            )
            return SimpleNamespace(metrics=metrics, adjusted_var_95=historical_var_95)

    risk_module.VolatilityForecaster = _FastVolForecaster

    users_by_email: dict[str, User] = {}
    all_tickers: set[str] = {"SPY"}

    async with SessionLocal() as db:
        for email, password, note in TEST_USERS:
            user = await _get_or_create_user(db, email, password)
            users_by_email[email] = user
            print(f"    ({note})\n")

        demo_user = users_by_email["demo@quantrisk.com"]
        analyst_user = users_by_email["analyst@quantrisk.com"]
        tester_user = users_by_email["tester@quantrisk.com"]

        portfolio_jobs: list[tuple[User, str, list[tuple[str, float, float]]]] = [
            (demo_user, DEMO_PORTFOLIO_SPECS[0][0], DEMO_PORTFOLIO_SPECS[0][1]),
            (analyst_user, DEMO_PORTFOLIO_SPECS[1][0], DEMO_PORTFOLIO_SPECS[1][1]),
        ]
        for name, positions in TESTER_PORTFOLIO_SPECS:
            portfolio_jobs.append((tester_user, name, positions))

        portfolios: list[Portfolio] = []
        for user, name, positions in portfolio_jobs:
            print(f"Portfolio for {user.email}: {name}")
            portfolio = await _get_or_create_portfolio(db, user, name)
            tickers = await _upsert_positions(db, portfolio, positions)
            all_tickers.update(tickers)
            portfolios.append(portfolio)
            print()

        await db.commit()

        print("Backfilling price history (yfinance)…")
        async with SessionLocal() as price_db:
            for ticker in sorted(all_tickers):
                await ensure_price_history(
                    price_db,
                    ticker,
                    force=force_prices,
                    synthetic_only=synthetic_only,
                )
            await price_db.commit()
        print()

        if not skip_risk:
            print("Computing risk snapshots (may take 1–2 minutes)…")
            async with SessionLocal() as risk_db:
                for portfolio in portfolios:
                    print(f"  {portfolio.name}")
                    fresh = (
                        await risk_db.execute(select(Portfolio).where(Portfolio.id == portfolio.id))
                    ).scalar_one()
                    await _persist_risk_seed(risk_db, fresh)
                await risk_db.commit()
        else:
            print("Skipped risk computation (--skip-risk).")

    print("\nDone. Test logins (JWT at /auth):")
    for email, password, _ in TEST_USERS:
        print(f"  {email}  /  {password}")
    print("\nOpen the dashboard — tester@quantrisk.com has five portfolios with VaR and charts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed QuantRisk DB with portfolios, prices, and risk.")
    parser.add_argument(
        "--force-prices",
        action="store_true",
        help="Replace existing price_snapshots for seeded tickers before re-downloading.",
    )
    parser.add_argument(
        "--skip-risk",
        action="store_true",
        help="Only users, positions, and prices (no risk_computations).",
    )
    parser.add_argument(
        "--synthetic-prices",
        action="store_true",
        help="Skip yfinance; write deterministic synthetic OHLC history (works offline).",
    )
    args = parser.parse_args()
    asyncio.run(
        run_seed(
            force_prices=args.force_prices,
            skip_risk=args.skip_risk,
            synthetic_only=args.synthetic_prices,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Seed failed: {exc}", file=sys.stderr)
        sys.exit(1)
