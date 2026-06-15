"""
Seed demo portfolio via the HTTP API (requires API + Redis + Celery worker).

  API_BASE_URL=https://your-api.onrender.com python scripts/seed_demo_portfolio.py
"""
from __future__ import annotations

import os
import sys
import time

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
EMAIL = os.getenv("DEMO_EMAIL", "demo@quantrisk.com")
PASSWORD = os.getenv("DEMO_PASSWORD", "QuantRisk2025!")
DEMO_PORTFOLIO_NAME = "Demo Portfolio"
DEMO_POSITIONS = [
    ("AAPL", 120, 185.00),
    ("MSFT", 80, 390.00),
    ("GOOGL", 45, 175.00),
    ("JPM", 150, 195.00),
    ("GS", 60, 480.00),
]


def poll_task(session: requests.Session, task_id: str, timeout_sec: int = 180) -> dict:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        resp = session.get(f"{API_BASE_URL}/tasks/{task_id}")
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status in {"SUCCESS", "FAILED"}:
            return data
        time.sleep(2)
    raise TimeoutError(f"Task {task_id} did not complete within {timeout_sec}s")


def compute_and_wait(session: requests.Session, portfolio_id: str) -> None:
    resp = session.post(f"{API_BASE_URL}/portfolios/{portfolio_id}/risk/compute")
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    result = poll_task(session, task_id)
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Risk compute failed: {result}")


def print_summary(session: requests.Session, portfolio_id: str) -> None:
    risk = session.get(f"{API_BASE_URL}/portfolios/{portfolio_id}/risk").json()
    stress = risk.get("stress_tests") or {}
    severe_loss = stress.get("severe", {}).get("loss")
    shap = risk.get("shap_attribution") or []
    top = max(shap, key=lambda x: abs(float(x.get("contribution", 0))), default=None)

    var_95 = float(risk.get("var_95", 0))
    cvar_95 = float(risk.get("cvar_95", 0))
    severe = float(severe_loss or 0)

    print()
    print(f"  Portfolio: {DEMO_PORTFOLIO_NAME}")
    print(f"  VaR 95%: ${var_95:,.0f}")
    print(f"  CVaR 95%: ${cvar_95:,.0f}")
    print(f"  Stress (Severe): ${severe:,.0f}")
    if top:
        ticker = top.get("ticker", "—")
        pct = float(top.get("pct_of_var", 0))
        print(f"  Top risk contributor: {ticker} ({pct:.1f}%)")
    else:
        print("  Top risk contributor: —")
    print()


def main() -> None:
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    register = session.post(f"{API_BASE_URL}/auth/register", json={"email": EMAIL, "password": PASSWORD})
    if register.status_code >= 400:
        login = session.post(f"{API_BASE_URL}/auth/login", json={"email": EMAIL, "password": PASSWORD})
        login.raise_for_status()
        token = login.json()["token"]
        print("Logged in (existing demo user).")
    else:
        register.raise_for_status()
        token = register.json()["token"]
        print("Registered demo user.")

    session.headers.update({"Authorization": f"Bearer {token}"})

    created = session.post(
        f"{API_BASE_URL}/portfolios/",
        json={"name": DEMO_PORTFOLIO_NAME, "margin_limit": 0.05},
    )
    created.raise_for_status()
    portfolio_id = created.json()["portfolio_id"]
    print(f"Created portfolio: {portfolio_id}")

    for ticker, qty, px in DEMO_POSITIONS:
        r = session.post(
            f"{API_BASE_URL}/portfolios/{portfolio_id}/positions",
            json={"ticker": ticker, "quantity": qty, "purchase_price": px},
        )
        r.raise_for_status()
        print(f"  Added {ticker} × {qty}")

    print("Running risk computation (may take a minute while history loads)…")
    compute_and_wait(session, portfolio_id)
    print("Risk computation complete.")
    print_summary(session, portfolio_id)
    print(f"Portfolio ID: {portfolio_id}")
    print("Open the dashboard to explore charts and alerts.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Seed failed: {exc}", file=sys.stderr)
        sys.exit(1)
