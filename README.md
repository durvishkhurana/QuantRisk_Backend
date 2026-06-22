<div align="center">

# QuantRisk Engine — Backend

**Institutional‑style portfolio risk computation service.**
VaR · CVaR · Monte Carlo · Stress Testing · SHAP Attribution · LSTM/GARCH Volatility · Margin Alerts

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-CA2C2C)](https://www.sqlalchemy.org/)
[![Celery](https://img.shields.io/badge/Celery-5.4-37814A?logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Postgres](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)](https://redis.io/)

</div>

---

## Overview

QuantRisk Engine is a FastAPI service that ingests equity portfolios, continuously
computes a battery of risk metrics, stores an **append‑only** audit history, and pushes
real‑time alerts over JWT‑authenticated WebSockets. CPU‑heavy work (kernel SHAP,
LSTM/GARCH training, AI narrative) runs in **Celery** workers, keeping the API fast.

It is a **risk measurement** service, not a trading platform — it never places orders
or moves money. Frontend lives in a separate repo: **[QuantRisk_Frontend](https://github.com/durvishkhurana/QuantRisk_Frontend)**.

**Live API:** https://quantrisk-backend.onrender.com · **Docs:** `/api/docs`

## Features

- **JWT auth** (HS256) with bcrypt password hashing; user‑scoped portfolio & position CRUD
- **Historical‑simulation VaR** (95% / 99%) and **CVaR** on a 252‑day lookback
- **Monte Carlo VaR** via historical bootstrap (preserves correlation, skew, fat tails)
- **Stress tests** (mild/moderate/severe) using OLS betas vs SPY
- **Risk attribution** — fast linear inline + async **SHAP KernelExplainer**
- **Volatility forecasting** — per‑ticker **LSTM vs GARCH(1,1)**, trained asynchronously
- **Correlation‑regime** detection, **Markowitz optimizer**, **Kupiec** VaR backtest
- **Margin** utilization vs per‑portfolio limit → WARNING / BREACH events
- **Append‑only** `risk_computations` & `margin_events` for time‑travel queries
- **Real‑time** alerts over WebSocket backed by **Redis Streams** (`?since=` replay)
- **Observability** — Prometheus metrics at `/metrics`

## Tech stack

FastAPI · Uvicorn · SQLAlchemy 2 (async) · asyncpg · Alembic · Pydantic v2 ·
Celery · Redis · NumPy · Pandas · SciPy · scikit‑learn · SHAP · PyTorch (LSTM) ·
arch (GARCH) · yfinance · python‑jose · bcrypt · prometheus‑client · pytest.

## Architecture

```
routers/ ─┐
          ├─► services/risk_pipeline.py ─► services/risk.py + engines
workers/ ─┘        (persistence + serialization; routers & workers depend downward)

PostgreSQL (append-only history) · Redis (cache, task state, alert streams)
External: yfinance · Alpha Vantage (opt) · Anthropic (opt)
```

- **Sync (request / 60s beat):** VaR, CVaR, bootstrap MC, stress, correlation, linear SHAP, margin.
- **Async (Celery):** kernel SHAP, LSTM/GARCH training, AI narrative.
- **Schema:** Alembic owns it in production; `create_all` runs only outside prod.

## Project structure

```
app/
├── main.py            # app, CORS, error handlers, /health, /metrics
├── config.py          # typed settings (env/.env, Supabase/Upstash normalization)
├── database.py        # async engine + session
├── models.py          # 8 ORM tables
├── schemas.py         # Pydantic I/O
├── auth.py            # JWT + password hashing
├── task_state.py      # owner-scoped Redis task status
├── middleware/        # Prometheus metrics
├── routers/           # auth, portfolios, risk, alerts, websocket
├── services/          # risk pipeline + engines (var, stress, shap, vol, optimizer, …)
└── workers/           # celery_app + risk/price/history/shap/vol tasks
alembic/versions/      # 001–006, 010 (Timescale), 011 (vol forecasts)
tests/                 # unit (no infra) + integration (needs Postgres+Redis)
```

## Quick start

Requires PostgreSQL and Redis (local or cloud).

```bash
cp .env.example .env          # set DATABASE_URL, REDIS_URL, JWT_SECRET_KEY, FRONTEND_URLS
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Workers (separate terminals):

```bash
celery -A app.workers.celery_app.celery_app worker --loglevel=info
celery -A app.workers.celery_app.celery_app beat   --loglevel=info
```

- API docs: http://localhost:8000/api/docs · Health: `/health` · Metrics: `/metrics`

## Configuration

Key env vars: `DATABASE_URL` (or `SUPABASE_DATABASE_URL`, which overrides) ·
`REDIS_URL` (or `UPSTASH_REDIS_REST_URL`/`_TOKEN`) · `CELERY_BROKER_URL` /
`CELERY_RESULT_BACKEND` · `JWT_SECRET_KEY` · `FRONTEND_URL` / `FRONTEND_URLS` (CORS) ·
`ENVIRONMENT` (`production` gates `create_all` and error verbosity) · optional
`ALPHA_VANTAGE_KEY`, `ANTHROPIC_API_KEY`. See `.env.example`. **Never commit `.env`.**

## API summary

`POST /auth/{register,login}` · `GET/POST /portfolios` · `GET /portfolios/aggregate` ·
`PATCH/DELETE /portfolios/{id}` · `.../positions` CRUD · `GET /portfolios/{id}/risk`
(+ `/history`, `/volatility-forecast`, `/optimize`, `/correlation`, `/backtest`) ·
`POST .../risk/compute` · `GET /tasks/{id}` (owner‑scoped) · `GET /alerts` (+ summary,
detail, csv, acknowledge) · `WS /ws/portfolios/{id}?since=&token=` · `/health` · `/metrics`.

## Testing

```bash
pytest -q tests/test_engines.py tests/test_var_engine.py   # unit only, no infra
pytest -q                                                  # full suite (needs Postgres + Redis)
```

> ⚠️ **Never run the suite against production.** `conftest.py` runs
> `TRUNCATE users ... CASCADE`. Point at a disposable test DB and **unset
> `SUPABASE_DATABASE_URL`** first (it overrides `DATABASE_URL`).

## Deployment (Render, no Docker)

Render Blueprint from `render.yaml` → API web service + Celery worker + beat. Start
command runs `alembic upgrade head` then Uvicorn; Render's `postgresql://` URL is
auto‑normalized to `postgresql+asyncpg://`. Set `FRONTEND_URLS` for CORS.

## License

Educational / portfolio project by Durvish Khurana.
