import asyncio
import time
from uuid import UUID

from sqlalchemy import select

from app.database import SessionLocal
from app.middleware.metrics_middleware import MARGIN_BREACH_COUNT, RISK_COMPUTATION_COUNT, RISK_COMPUTATION_LATENCY
from app.models import Portfolio, RiskComputation
from app.routers.risk import _get_shap_map_for_row, _map_risk_row_from_shap, _persist_computation
from app.services.nlp_report_service import generate_risk_narrative
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.risk_worker.compute_all_portfolios")
def compute_all_portfolios() -> dict:
    return asyncio.run(_compute_all())


async def _compute_all() -> dict:
    async with SessionLocal() as db:
        portfolios = (await db.execute(select(Portfolio).where(Portfolio.is_active.is_(True)))).scalars().all()
        success = 0
        failed = 0
        for p in portfolios:
            start = time.perf_counter()
            try:
                row = await _persist_computation(db, p, triggered_by="scheduler")
                elapsed_ms = (time.perf_counter() - start) * 1000
                RISK_COMPUTATION_LATENCY.labels("scheduler").observe(elapsed_ms)
                RISK_COMPUTATION_COUNT.labels(str(p.id), "success").inc()
                if row.margin_status == "BREACH":
                    MARGIN_BREACH_COUNT.labels(str(p.id)).inc()
                success += 1
            except Exception:  # noqa: BLE001
                RISK_COMPUTATION_COUNT.labels(str(p.id), "failed").inc()
                failed += 1
        return {"success": success, "failed": failed}


@celery_app.task(name="app.workers.risk_worker.generate_risk_narrative_task")
def generate_risk_narrative_task(risk_computation_id: str) -> dict:
    return asyncio.run(_generate_narrative(risk_computation_id))


async def _generate_narrative(risk_computation_id: str) -> dict:
    async with SessionLocal() as db:
        row = (
            await db.execute(select(RiskComputation).where(RiskComputation.id == UUID(risk_computation_id)))
        ).scalar_one_or_none()
        if not row:
            return {"status": "not_found"}
        shap_map = await _get_shap_map_for_row(db, row)
        mapped = _map_risk_row_from_shap(row, shap_map)
        narrative = await generate_risk_narrative(mapped.model_dump(mode="json"))
        row.risk_narrative = narrative
        await db.commit()
        return {"status": "ok", "risk_computation_id": risk_computation_id}
