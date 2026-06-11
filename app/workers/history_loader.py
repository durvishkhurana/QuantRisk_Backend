import asyncio
from app.database import SessionLocal
from app.services.market_data import backfill_history
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.history_loader.backfill_ticker_history")
def backfill_ticker_history(ticker: str) -> dict:
    return asyncio.run(_backfill(ticker))


async def _backfill(ticker: str) -> dict:
    async with SessionLocal() as db:
        await backfill_history(db, ticker)
        await backfill_history(db, "SPY")
    return {"ticker": ticker, "status": "ok"}
