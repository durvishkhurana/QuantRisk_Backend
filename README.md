# QuantRisk Backend

FastAPI service that ingests equity positions, computes portfolio risk on a schedule, persists audit-friendly history, and exposes REST, WebSocket, and Prometheus metrics.

## Features

- JWT authentication and user-scoped portfolio CRUD
- Historical simulation VaR (95% / 99%) and CVaR on a 252-day lookback
- Stress tests (mild / moderate / severe) with OLS betas vs SPY
- Monte Carlo simulation (10k paths), correlation regime detection, Markowitz-style optimizer
- Margin utilization vs per-portfolio limits with WARNING / BREACH events
- Append-only `risk_computations` and `margin_events` for time-travel queries
- Celery workers for scheduled recomputation (~60s), price backfill, async SHAP kernel attribution, optional risk narrative
- Alerts via REST (filter, paginate, CSV export, acknowledge) and WebSocket with Redis Streams replay (`?since=`)
- Kupiec VaR backtest when sufficient history exists
- Market data via yfinance with Redis caching; optional Alpha Vantage

## Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Pydantic v2, SQLAlchemy async |
| Auth | JWT (python-jose), bcrypt |
| Database | PostgreSQL 16 (TimescaleDB optional for hypertables) |
| Queue / cache | Redis 7, Celery 5 |
| Compute | NumPy, Pandas, SciPy, scikit-learn, SHAP |

## Prerequisites

- Python 3.12+
- PostgreSQL 16+
- Redis 7+

## Configuration

Copy `.env.example` to `.env` and set at minimum:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | `postgresql+asyncpg://...` (Render `postgresql://` URLs are normalized automatically) |
| `REDIS_URL` | Redis connection |
| `JWT_SECRET_KEY` | Signing secret for access tokens |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | Celery broker and result backend |
| `FRONTEND_URLS` | Comma-separated CORS origins for the dashboard |

Optional: `ALPHA_VANTAGE_KEY`, `ANTHROPIC_API_KEY`.

## Local development

```bash
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

In separate terminals:

```bash
celery -A app.workers.celery_app.celery_app worker --loglevel=info
celery -A app.workers.celery_app.celery_app beat --loglevel=info
```

- OpenAPI: `http://localhost:8000/api/docs`
- Health: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`

## Testing

```bash
export DATABASE_URL=postgresql+asyncpg://quantrisk:password@localhost:5432/quantrisk_test
export REDIS_URL=redis://localhost:6379/15
export JWT_SECRET_KEY=test-secret
alembic upgrade head
pytest -q
```

CI runs on push to `main` via `.github/workflows/test.yml`.

## Deployment

Use `render.yaml` in this repository root for a Render Blueprint (web service, Celery worker, Celery beat, Postgres, Redis). Set `FRONTEND_URL` and `FRONTEND_URLS` to your deployed dashboard origin.

See [../md/deploy-without-docker.md](../md/deploy-without-docker.md) for split frontend/backend repos and environment wiring.

## Project layout

```
app/
  main.py          # FastAPI app, CORS, metrics
  routers/         # auth, portfolios, risk, alerts, websocket
  services/        # VaR, stress, margin, market data, optimizer, …
  workers/         # Celery tasks (risk, prices, SHAP)
alembic/           # Schema migrations
tests/             # pytest suite
scripts/           # benchmarks, demo seeding
```

## Related documentation

- [API reference](../md/api_reference.md)
- [Architecture](../md/architecture.md)
- [Design decisions](../md/DESIGN_DECISIONS.md)

## Scope and limitations

- Equity cash positions only (no derivatives or fixed income)
- Authorization enforced in application code (no PostgreSQL RLS)
- Kupiec backtest needs at least 30 days of stored risk history
