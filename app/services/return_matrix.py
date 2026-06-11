import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import get_settings
from app.models import PriceSnapshot

settings = get_settings()


async def build_returns_matrix(session: AsyncSession, tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    result = await session.execute(select(PriceSnapshot).where(PriceSnapshot.ticker.in_(tickers)))
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()

    data = [{"ticker": row.ticker, "date": row.date, "close": float(row.close)} for row in rows]
    prices = pd.DataFrame(data).pivot_table(index="date", columns="ticker", values="close").sort_index()
    returns = prices.pct_change().dropna()
    return returns.tail(settings.lookback_days)
