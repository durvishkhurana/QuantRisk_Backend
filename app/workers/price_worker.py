import asyncio
from sqlalchemy import select
from app.database import SessionLocal
from app.models import Position
from app.services.market_data import backfill_history, get_latest_price
from app.services.redis_client import cache_set_json
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.price_worker.fetch_and_cache_prices")
def fetch_and_cache_prices() -> dict:
    return asyncio.run(_fetch())


async def _fetch() -> dict:
    async with SessionLocal() as db:
        rows = (await db.execute(select(Position.ticker).distinct())).all()
        tickers = sorted({r[0] for r in rows} | {"SPY"})
        updated = 0
        failed = 0
        for ticker in tickers:
            try:
                price = await get_latest_price(ticker)
                await cache_set_json(f"price:{ticker}", {"price": float(price)}, ttl_seconds=60)
                await backfill_history(db, ticker)
                updated += 1
            except Exception:  # noqa: BLE001
                failed += 1
        return {"updated": updated, "failed": failed}
