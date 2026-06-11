from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "quantrisk",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.risk_worker",
        "app.workers.price_worker",
        "app.workers.history_loader",
        "app.workers.shap_worker",
    ],
)

celery_app.conf.beat_schedule = {
    "compute-risk-every-minute": {
        "task": "app.workers.risk_worker.compute_all_portfolios",
        "schedule": settings.risk_compute_interval_seconds,
    },
    "refresh-prices-every-minute": {
        "task": "app.workers.price_worker.fetch_and_cache_prices",
        "schedule": settings.risk_compute_interval_seconds,
    },
}
celery_app.conf.timezone = "UTC"
