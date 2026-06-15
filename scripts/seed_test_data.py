"""
Create test users and a demo portfolio in PostgreSQL.

Run from backend/ (after alembic upgrade head):
  python scripts/seed_test_data.py

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
]

DEMO_PORTFOLIO_NAME = "Demo Portfolio"
DEMO_POSITIONS: list[tuple[str, float, float]] = [
    ("AAPL", 120, 185.00),
    ("MSFT", 80, 390.00),
    ("GOOGL", 45, 175.00),
    ("JPM", 150, 195.00),
    ("GS", 60, 480.00),
]


async def _get_or_create_user(db, email: str, password: str) -> User:
    result = await db.execute(select(User).where(User.email == email.lower()))
    user = result.scalar_one_or_none()
    if user:
        print(f"  User already exists: {email}")
        return user
    user = User(email=email.lower(), password_hash=hash_password(password))
    db.add(user)
    await db.flush()
    print(f"  Created user: {email}")
    return user


async def _seed_demo_portfolio(db, user: User) -> str | None:
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == user.id, Portfolio.name == DEMO_PORTFOLIO_NAME)
    )
    portfolio = result.scalar_one_or_none()
    if portfolio:
        print(f"  Portfolio already exists: {DEMO_PORTFOLIO_NAME} ({portfolio.id})")
        return str(portfolio.id)

    portfolio = Portfolio(
        user_id=user.id,
        name=DEMO_PORTFOLIO_NAME,
        margin_limit=Decimal("0.05"),
        is_active=True,
    )
    db.add(portfolio)
    await db.flush()

    for ticker, qty, px in DEMO_POSITIONS:
        db.add(
            Position(
                portfolio_id=portfolio.id,
                ticker=ticker,
                quantity=Decimal(str(qty)),
                purchase_price=Decimal(str(px)),
                sector="Technology" if ticker in {"AAPL", "MSFT", "GOOGL"} else "Financials",
            )
        )
    print(f"  Created portfolio {DEMO_PORTFOLIO_NAME} with {len(DEMO_POSITIONS)} positions")
    return str(portfolio.id)


async def main() -> None:
    print("Seeding test data…")
    async with SessionLocal() as db:
        demo_user: User | None = None
        for email, password, note in TEST_USERS:
            user = await _get_or_create_user(db, email, password)
            if email == "demo@quantrisk.com":
                demo_user = user
            print(f"    ({note})")

        if demo_user:
            pid = await _seed_demo_portfolio(db, demo_user)
            if pid:
                print(f"\nDemo portfolio id: {pid}")

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
