from __future__ import annotations

import asyncio
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models import MarginEvent, Portfolio, RiskComputation, User

pytestmark = pytest.mark.usefixtures("_clean_database", "market_mocks")


@pytest.mark.asyncio
async def test_register_creates_user_and_token(client: AsyncClient) -> None:
    email = f"register-{uuid4()}@example.com"
    resp = await client.post("/auth/register", json={"email": email, "password": "password123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == email
    assert body["token"]
    assert body["user_id"]


@pytest.mark.asyncio
async def test_login_success_and_wrong_password(client: AsyncClient) -> None:
    email = f"login-{uuid4()}@example.com"
    password = "password123"
    await client.post("/auth/register", json={"email": email, "password": password})

    ok = await client.post("/auth/login", json={"email": email, "password": password})
    assert ok.status_code == 200
    assert ok.json()["token"]

    bad = await client.post("/auth/login", json={"email": email, "password": "wrong-password"})
    assert bad.status_code == 401


@pytest.mark.asyncio
async def test_create_portfolio_requires_auth(client: AsyncClient) -> None:
    unauth = await client.post("/portfolios/", json={"name": "No Auth", "margin_limit": 0.05})
    assert unauth.status_code == 403

    email = f"portfolio-{uuid4()}@example.com"
    reg = await client.post("/auth/register", json={"email": email, "password": "password123"})
    headers = {"Authorization": f"Bearer {reg.json()['token']}"}

    created = await client.post(
        "/portfolios/",
        json={"name": "Auth Portfolio", "margin_limit": 0.05},
        headers=headers,
    )
    assert created.status_code == 200
    assert created.json()["name"] == "Auth Portfolio"
    assert created.json()["portfolio_id"]


@pytest.mark.asyncio
async def test_add_position(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    create = await client.post(
        "/portfolios/",
        json={"name": "Positions", "margin_limit": 0.05},
        headers=auth_headers,
    )
    portfolio_id = create.json()["portfolio_id"]

    add = await client.post(
        f"/portfolios/{portfolio_id}/positions",
        json={"ticker": "AAPL", "quantity": 5, "purchase_price": 120},
        headers=auth_headers,
    )
    assert add.status_code == 200
    body = add.json()
    assert body["ticker"] == "AAPL"
    assert Decimal(str(body["quantity"])) == Decimal("5")
    assert body["current_price"]


@pytest.mark.asyncio
async def test_get_portfolio_risk(client: AsyncClient, portfolio_with_position: str, auth_headers: dict[str, str]) -> None:
    portfolio_id = portfolio_with_position
    compute = await client.post(f"/portfolios/{portfolio_id}/risk/compute", headers=auth_headers)
    assert compute.status_code == 200
    task_id = compute.json()["task_id"]

    status = "PENDING"
    last_body: dict = {}
    for _ in range(120):
        task = await client.get(f"/tasks/{task_id}")
        assert task.status_code == 200
        last_body = task.json()
        status = last_body["status"]
        if status in {"SUCCESS", "FAILED"}:
            break
        await asyncio.sleep(0.1)

    assert status == "SUCCESS", last_body

    risk = await client.get(f"/portfolios/{portfolio_id}/risk", headers=auth_headers)
    assert risk.status_code == 200
    body = risk.json()
    assert "var_95" in body
    assert "cvar_95" in body
    assert "margin_status" in body
    assert "shap_attribution" in body


@pytest.mark.asyncio
async def test_list_alerts_empty(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    resp = await client.get("/alerts", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["limit"] == 20


@pytest.mark.asyncio
async def test_acknowledge_alert(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    create = await client.post(
        "/portfolios/",
        json={"name": "Alerts", "margin_limit": 0.05},
        headers=auth_headers,
    )
    portfolio_id = UUID(create.json()["portfolio_id"])

    from app.database import SessionLocal

    async with SessionLocal() as db:
        user = (
            await db.execute(select(User).order_by(User.created_at.desc()).limit(1))
        ).scalar_one()
        portfolio = (
            await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id))
        ).scalar_one()
        assert portfolio.user_id == user.id

        event = MarginEvent(
            portfolio_id=portfolio.id,
            event_type="WARNING",
            var_95=Decimal("5000"),
            margin_limit=Decimal("0.05"),
            margin_utilization=Decimal("0.9000"),
        )
        db.add(event)
        await db.commit()
        await db.refresh(event)
        event_id = event.id

    listed = await client.get("/alerts", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["acknowledged"] is False

    ack = await client.post(f"/alerts/{event_id}/acknowledge", headers=auth_headers)
    assert ack.status_code == 200
    assert ack.json()["acknowledged"] is True
    assert ack.json()["acknowledged_at"]

    listed_after = await client.get("/alerts", headers=auth_headers)
    assert listed_after.json()["items"][0]["acknowledged"] is True


@pytest.mark.asyncio
async def test_alert_detail_links_closest_risk(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    create = await client.post(
        "/portfolios/",
        json={"name": "Detail", "margin_limit": 0.05},
        headers=auth_headers,
    )
    portfolio_id = UUID(create.json()["portfolio_id"])

    from datetime import datetime, timedelta, timezone

    from app.database import SessionLocal

    triggered = datetime.now(timezone.utc)
    async with SessionLocal() as db:
        risk = RiskComputation(
            portfolio_id=portfolio_id,
            computed_at=triggered - timedelta(seconds=2),
            portfolio_value=Decimal("100000"),
            var_95=Decimal("4200"),
            var_99=Decimal("6100"),
            cvar_95=Decimal("5100"),
            margin_utilization=Decimal("0.8800"),
            margin_status="WARNING",
            stress_mild=Decimal("1000"),
            stress_moderate=Decimal("2500"),
            stress_severe=Decimal("4000"),
            shap_json={"AAPL": -1200.0, "MSFT": 3400.0},
            n_positions=2,
        )
        db.add(risk)
        await db.flush()
        event = MarginEvent(
            portfolio_id=portfolio_id,
            event_type="WARNING",
            triggered_at=triggered,
            var_95=Decimal("4200"),
            margin_limit=Decimal("0.05"),
            margin_utilization=Decimal("0.8800"),
            risk_computation_id=risk.id,
        )
        db.add(event)
        await db.commit()
        await db.refresh(event)
        event_id = event.id

    detail = await client.get(f"/alerts/{event_id}/detail", headers=auth_headers)
    assert detail.status_code == 200
    body = detail.json()
    assert body["cvar_95"] == 5100.0 or float(body["cvar_95"]) == 5100.0
    assert float(body["stress_loss_moderate"]) == 2500.0
    assert len(body["shap_attributions"]) >= 1
    assert body["var_95"] == 4200.0 or float(body["var_95"]) == 4200.0


@pytest.mark.asyncio
async def test_task_status_persists_in_redis() -> None:
    from uuid import uuid4

    from app.task_state import get_task_status, set_task_status

    task_id = str(uuid4())
    await set_task_status(task_id, "RUNNING")
    running = await get_task_status(task_id)
    assert running is not None
    assert running["status"] == "RUNNING"
    assert running.get("result") is None

    sample_result = {"var_95": 12500.0, "margin_status": "NORMAL"}
    await set_task_status(task_id, "SUCCESS", result=sample_result)
    done = await get_task_status(task_id)
    assert done is not None
    assert done["status"] == "SUCCESS"
    assert done["result"] == sample_result
