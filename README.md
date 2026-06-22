# QuantRisk Backend

FastAPI service that ingests equity positions, computes portfolio risk on a schedule, persists audit-friendly history, and exposes REST, WebSocket, and Prometheus metrics.

> This is its **own git repository**, deployed independently to Render. The sibling
> `frontend/` is a separate repo (Vercel). There is no root-level monorepo.

## Features

- JWT authentication and user-scoped portfolio CRUD
- Historical simulation VaR (95% / 99%) and CVaR on a 252-day lookback
- Stress tests (mild / moderate / severe) with OLS betas vs SPY
- Monte Carlo simulation (10k paths via historical bootstrap), correlation regime detection, Markowitz-style optimizer
- Per-ticker LSTM-vs-GARCH volatility forecasting, trained asynchronously off the hot path
- Margin utilization vs per-portfolio limits with WARNING / BREACH events
- Append-only `risk_computations` and `margin_events` for time-travel queries
- Celery workers for scheduled recomputation (~60s), price backfill, async SHAP kernel attribution, async volatility training, optional risk narrative
- Alerts via REST (filter, paginate, CSV export, acknowledge) and JWT-authenticated WebSocket with Redis Streams replay (`?since=&token=`)
- Kupiec VaR backtest when sufficient history exists
- Market data via yfinance (run off the event loop) with Redis caching; optional Alpha Vantage

The 60s scheduler (`compute_all_portfolios`) recomputes active portfolios sequentially; this is adequate by design because the heavy work (model training, kernel SHAP, narrative) runs in separate Celery tasks, keeping each portfolio's synchronous step small.

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

**Do not commit `.env` or paste secrets into documentation.**

1. Copy `.env.example` to `.env`.
2. Fill in your own values for database, Redis, JWT secret, and CORS origins.
3. Never commit API keys (`ALPHA_VANTAGE_KEY`, `ANTHROPIC_API_KEY`) to the repository.

**Risk narrative:** With `ANTHROPIC_API_KEY` on Render, summaries use Claude Haiku. Without it, the API builds a rule-based narrative from VaR, SHAP, and stress metrics (no xAI/Grok integration).

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Async Postgres (or set `SUPABASE_DATABASE_URL` for Supabase) |
| `SUPABASE_DATABASE_URL` | Overrides `DATABASE_URL`. On **Render/Vercel**, use Supabase **connection pooler** (port **6543**), not direct `db.*.supabase.co:5432`. |
| `SUPABASE_SECRET_KEY` | Supabase secret key (Render env only, never frontend) |
| `REDIS_URL` | Redis connection |
| `JWT_SECRET_KEY` | Long random signing secret |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | Celery broker and result backend |
| `FRONTEND_URLS` | Comma-separated allowed browser origins (CORS) |

Render-managed Postgres URLs using `postgresql://` are normalized to `postgresql+asyncpg://` in application settings.

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

> ⚠️ **Never run the test suite against the production database.** `tests/conftest.py`
> runs `TRUNCATE users RESTART IDENTITY CASCADE` on setup — pointing it at the live
> Supabase URL would wipe all real data. Use a **dedicated, throwaway test database**
> and a separate Redis index. Note that `config.py` overrides `DATABASE_URL` with
> `SUPABASE_DATABASE_URL` when the latter is set, so **unset `SUPABASE_DATABASE_URL`**
> (and any prod `.env`) before running tests.

Test layers:

- **Unit / engine tests** (`test_engines.py`, `test_var_engine.py`) need no DB or Redis.
- **Integration tests** (`test_api.py`) require a live Postgres + Redis.

GitHub Actions supplies isolated CI services in `.github/workflows/test.yml`.

Locally:

```bash
pip install -r requirements.txt
# Point at a DISPOSABLE test DB/Redis — not production:
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/quantrisk_test
export REDIS_URL=redis://localhost:6379/15
unset SUPABASE_DATABASE_URL
export JWT_SECRET_KEY=pytest-secret
alembic upgrade head

pytest -q                                   # full suite (needs DB + Redis)
pytest -q tests/test_engines.py tests/test_var_engine.py   # unit only, no infra
```

## Deployment

**Render (backend):** New → Web Service → connect this repo.

**Python version:** Render defaults to 3.14, which breaks pinned `pandas`/`numpy` builds. Set **Environment → `PYTHON_VERSION`** = `3.12.12` (or rely on repo `.python-version` = `3.12`).

**Start command:**

```bash
PYTHONPATH=. alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Build command:** `pip install -r requirements.txt`

**Supabase Postgres instead of Render DB:** set `SUPABASE_DATABASE_URL` on all backend services.

**Render + Supabase:** In [Supabase](https://supabase.com/dashboard) → **Project Settings → Database → Connection string**, choose **URI** and turn on **Use connection pooling** (Session mode, port **6543**). Paste that into `SUPABASE_DATABASE_URL` on Render. The direct host `db.<ref>.supabase.co:5432` often fails from Render with `Network is unreachable` (IPv6). URL-encode special characters in the password (e.g. `.` → `%2E`).

Keep **Upstash** (or Render) Redis for Celery.

**Vercel (frontend):** import the frontend repo; set `VITE_API_URL` to the Render API URL; set `NEXT_PUBLIC_SUPABASE_*` if using the Supabase client.

On the API service set `FRONTEND_URL` and `FRONTEND_URLS` to your Vercel origin.

Seed data (after DB is reachable):

```bash
python -m alembic upgrade head
python scripts/seed_test_data.py
python scripts/seed_database_full.py   # prices, risk rows, and full tester showcase
```

Test logins: `demo@quantrisk.com` / `QuantRisk2025!`, `analyst@quantrisk.com` / `Analyst2025!`, `tester@quantrisk.com` / `Tester2025!` (five portfolios after full seed).

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

## Scope and limitations

- Equity cash positions only (no derivatives or fixed income)
- Authorization enforced in application code (no PostgreSQL RLS)
- Kupiec backtest needs at least 30 days of stored risk history
