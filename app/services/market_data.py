from datetime import date
from decimal import Decimal
import os
import httpx
import yfinance as yf
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import PriceSnapshot
from app.middleware.metrics_middleware import CACHE_HIT_RATE, CACHE_MISS_RATE
from app.services.redis_client import cache_get_json, cache_set_json


async def get_latest_price(ticker: str) -> Decimal:
    cached = await cache_get_json(f"price:{ticker}")
    if cached and "price" in cached:
        CACHE_HIT_RATE.labels("price").inc()
        return Decimal(str(cached["price"]))
    CACHE_MISS_RATE.labels("price").inc()

    ticker_obj = yf.Ticker(ticker)
    try:
        price = ticker_obj.fast_info.get("lastPrice")
    except Exception:  # noqa: BLE001
        price = None
    if price is None:
        try:
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
        except Exception:  # noqa: BLE001
            price = None

    if price is None and os.getenv("ALPHA_VANTAGE_KEY"):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "GLOBAL_QUOTE",
                        "symbol": ticker,
                        "apikey": os.getenv("ALPHA_VANTAGE_KEY"),
                    },
                )
                data = resp.json()
                price = float(data.get("Global Quote", {}).get("05. price"))
        except Exception:  # noqa: BLE001
            price = None

    if price is None:
        raise ValueError(f"Unable to fetch latest price for {ticker}")

    out = Decimal(str(round(float(price), 4)))
    await cache_set_json(f"price:{ticker}", {"price": float(out)}, ttl_seconds=60)
    return out


async def get_sector(ticker: str) -> str | None:
    cached = await cache_get_json(f"sector:{ticker}")
    if cached and "sector" in cached:
        CACHE_HIT_RATE.labels("sector").inc()
        return cached["sector"]
    CACHE_MISS_RATE.labels("sector").inc()
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info or {}
    sector = info.get("sector")
    if sector:
        await cache_set_json(f"sector:{ticker}", {"sector": sector}, ttl_seconds=86400)
    return sector


async def backfill_history(session: AsyncSession, ticker: str, lookback_days: int = 252) -> None:
    result = await session.execute(select(PriceSnapshot).where(PriceSnapshot.ticker == ticker).limit(1))
    if result.scalar_one_or_none():
        return

    history = yf.download(ticker, period="1y", auto_adjust=False, progress=False)
    if history.empty:
        return

    history = history.tail(lookback_days)
    rows: list[PriceSnapshot] = []
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
    session.add_all(rows)
    await session.commit()


async def get_last_snapshot_price(session: AsyncSession, ticker: str) -> Decimal | None:
    result = await session.execute(
        select(PriceSnapshot).where(PriceSnapshot.ticker == ticker).order_by(PriceSnapshot.date.desc()).limit(1)
    )
    row = result.scalar_one_or_none()
    return Decimal(row.close) if row else None
