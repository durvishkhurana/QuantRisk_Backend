"""
Create test users and a demo portfolio in PostgreSQL.

Run from backend/ (after alembic upgrade head):
  python scripts/seed_test_data.py

For portfolios, stock positions, price history, and risk rows (full UI):
  python scripts/seed_database_full.py

Uses DATABASE_URL or SUPABASE_DATABASE_URL from the repo root .env.
"""
from __future__ import annotations

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

_BACKEND = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_BACKEND / ".env")

sys.path.insert(0, str(_BACKEND))

from app.auth import hash_password  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import Portfolio, Position, User  # noqa: E402

TEST_USERS: list[tuple[str, str, str]] = [
    ("demo@quantrisk.com", "QuantRisk2025!", "Primary demo account with sample portfolio"),
    ("analyst@quantrisk.com", "Analyst2025!", "Secondary account for testing"),
    ("tester@quantrisk.com", "Tester2025!", "Showcase account with five themed portfolios"),
]

DEMO_PORTFOLIO_NAME = "Demo Portfolio"
DEMO_POSITIONS: list[tuple[str, float, float]] = [
    ("AAPL", 120, 185.00),
    ("MSFT", 80, 390.00),
    ("GOOGL", 45, 175.00),
    ("JPM", 150, 195.00),
    ("GS", 60, 480.00),
]

TESTER_PORTFOLIOS: list[tuple[str, list[tuple[str, float, float]]]] = [
    (
        "Mega Cap Technology",
        [("AAPL", 150, 178.0), ("MSFT", 90, 380.0), ("GOOGL", 60, 165.0), ("META", 45, 480.0), ("NVDA", 120, 95.0)],
    ),
    (
        "Global Financials",
        [("JPM", 200, 185.0), ("BAC", 400, 32.0), ("GS", 75, 450.0), ("MS", 110, 88.0), ("V", 85, 270.0)],
    ),
    (
        "Consumer & Retail",
        [("AMZN", 80, 165.0), ("WMT", 120, 68.0), ("COST", 35, 720.0), ("HD", 55, 340.0), ("MCD", 90, 285.0)],
    ),
    (
        "Healthcare Defensive",
        [("JNJ", 140, 155.0), ("UNH", 50, 520.0), ("PFE", 300, 28.0), ("ABBV", 70, 165.0), ("MRK", 95, 115.0)],
    ),
    (
        "ETF Core Allocation",
        [("SPY", 180, 480.0), ("QQQ", 100, 420.0), ("IWM", 150, 195.0), ("VTI", 130, 240.0), ("AGG", 220, 98.0)],
    ),
]


async def _get_or_create_user(db, email: str, password: str) -> User:
    result = await db.execute(select(User).where(User.email == email.lower()))
    user = result.scalar_one_or_none()
    if user:
        user.password_hash = hash_password(password)
        print(f"  User already exists (password reset): {email}")
        return user
    user = User(email=email.lower(), password_hash=hash_password(password))
    db.add(user)
    await db.flush()
    print(f"  Created user: {email}")
    return user


async def _seed_named_portfolio(
    db,
    user: User,
    name: str,
    positions: list[tuple[str, float, float]],
) -> str:
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == user.id, Portfolio.name == name)
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        portfolio = Portfolio(
            user_id=user.id,
            name=name,
            margin_limit=Decimal("0.05"),
            is_active=True,
        )
        db.add(portfolio)
        await db.flush()
        print(f"  Created portfolio {name} ({portfolio.id})")
    else:
        print(f"  Portfolio already exists: {name} ({portfolio.id})")

    for ticker, qty, px in positions:
        ticker = ticker.upper()
        existing = await db.execute(
            select(Position).where(Position.portfolio_id == portfolio.id, Position.ticker == ticker)
        )
        pos = existing.scalar_one_or_none()
        if ticker in {"AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD"}:
            sector = "Technology"
        elif ticker in {"SPY", "QQQ", "IWM", "VTI", "AGG"}:
            sector = "ETF"
        elif ticker in {"JPM", "BAC", "GS", "MS", "V"}:
            sector = "Financials"
        elif ticker in {"JNJ", "UNH", "PFE", "ABBV", "MRK"}:
            sector = "Healthcare"
        else:
            sector = "Consumer Cyclical"
        if pos:
            pos.quantity = Decimal(str(qty))
            pos.purchase_price = Decimal(str(px))
            pos.sector = sector
        else:
            db.add(
                Position(
                    portfolio_id=portfolio.id,
                    ticker=ticker,
                    quantity=Decimal(str(qty)),
                    purchase_price=Decimal(str(px)),
                    sector=sector,
                )
            )
    print(f"  Ensured {len(positions)} positions in {name}")
    return str(portfolio.id)


async def _seed_demo_portfolio(db, user: User) -> str | None:
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == user.id, Portfolio.name == DEMO_PORTFOLIO_NAME)
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        portfolio = Portfolio(
            user_id=user.id,
            name=DEMO_PORTFOLIO_NAME,
            margin_limit=Decimal("0.05"),
            is_active=True,
        )
        db.add(portfolio)
        await db.flush()
        print(f"  Created portfolio {DEMO_PORTFOLIO_NAME} ({portfolio.id})")
    else:
        print(f"  Portfolio already exists: {DEMO_PORTFOLIO_NAME} ({portfolio.id})")

    for ticker, qty, px in DEMO_POSITIONS:
        ticker = ticker.upper()
        existing = await db.execute(
            select(Position).where(Position.portfolio_id == portfolio.id, Position.ticker == ticker)
        )
        pos = existing.scalar_one_or_none()
        sector = "Technology" if ticker in {"AAPL", "MSFT", "GOOGL"} else "Financials"
        if pos:
            pos.quantity = Decimal(str(qty))
            pos.purchase_price = Decimal(str(px))
            pos.sector = sector
        else:
            db.add(
                Position(
                    portfolio_id=portfolio.id,
                    ticker=ticker,
                    quantity=Decimal(str(qty)),
                    purchase_price=Decimal(str(px)),
                    sector=sector,
                )
            )
    print(f"  Ensured {len(DEMO_POSITIONS)} positions in {DEMO_PORTFOLIO_NAME}")
    print("  Tip: run scripts/seed_database_full.py for prices + risk metrics on the dashboard.")
    return str(portfolio.id)


async def main() -> None:
    print("Seeding test data…")
    async with SessionLocal() as db:
        demo_user: User | None = None
        tester_user: User | None = None
        for email, password, note in TEST_USERS:
            user = await _get_or_create_user(db, email, password)
            if email == "demo@quantrisk.com":
                demo_user = user
            if email == "tester@quantrisk.com":
                tester_user = user
            print(f"    ({note})")

        if demo_user:
            pid = await _seed_demo_portfolio(db, demo_user)
            if pid:
                print(f"\nDemo portfolio id: {pid}")

        if tester_user:
            print("\nTester portfolios:")
            for name, positions in TESTER_PORTFOLIOS:
                await _seed_named_portfolio(db, tester_user, name, positions)
            print("  Tip: run scripts/seed_database_full.py for price history + risk metrics.")

        await db.commit()

    print("\nTest logins (FastAPI JWT auth — use /auth in the app):")
    for email, password, _ in TEST_USERS:
        print(f"  Email: {email}")
        print(f"  Password: {password}")
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Seed failed: {exc}", file=sys.stderr)
        sys.exit(1)
