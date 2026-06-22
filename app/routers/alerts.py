import csv
import io
from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import MarginEvent, Portfolio, RiskComputation, ShapAttribution, User
from app.schemas import AlertDetailOut, AlertEventOut, AlertShapAttributionOut, AlertsListResponse

router = APIRouter(prefix="/alerts", tags=["alerts"])
portfolio_alert_router = APIRouter(prefix="/portfolios/{portfolio_id}/alerts", tags=["alerts"])

EVENT_TYPE_API_TO_DB: dict[str, str] = {
    "MARGIN_WARNING": "WARNING",
    "MARGIN_BREACH": "BREACH",
    "CORRELATION_ALERT": "CORRELATION_ALERT",
    "WARNING": "WARNING",
    "BREACH": "BREACH",
}

DEFAULT_LIMIT = 20
MAX_LIMIT = 200
CSV_MAX_ROWS = 10_000


def _normalize_event_types(event_types: list[str] | None) -> list[str] | None:
    if not event_types:
        return None
    mapped: list[str] = []
    for raw in event_types:
        key = raw.strip().upper()
        db_type = EVENT_TYPE_API_TO_DB.get(key)
        if db_type and db_type not in mapped:
            mapped.append(db_type)
    return mapped or None


def _parse_date_param(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    if not value:
        return None
    try:
        if "T" in value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        day = date.fromisoformat(value)
        if end_of_day:
            return datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)
        return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date: {value}") from exc


def _event_message(event_type: str, margin_utilization: float, var_95: float) -> str:
    util_pct = margin_utilization * 100
    if event_type == "CORRELATION_ALERT":
        return f"Correlation regime stress detected — margin utilization {util_pct:.1f}%, VaR ${var_95:,.0f}"
    if event_type == "BREACH":
        return f"Margin limit breached — utilization {util_pct:.1f}%, VaR ${var_95:,.0f}"
    if event_type == "WARNING":
        return f"Approaching margin limit — utilization {util_pct:.1f}%, VaR ${var_95:,.0f}"
    return f"Alert — utilization {util_pct:.1f}%, VaR ${var_95:,.0f}"


async def _shap_attributions_for_risk(db: AsyncSession, row: RiskComputation) -> list[AlertShapAttributionOut]:
    kernel_rows = (
        await db.execute(
            select(ShapAttribution).where(
                ShapAttribution.risk_computation_id == row.id,
                ShapAttribution.method == "kernel",
            )
        )
    ).scalars().all()
    shap_map = {r.ticker: float(r.shap_value) for r in kernel_rows} if kernel_rows else {k: float(v) for k, v in (row.shap_json or {}).items()}
    if not shap_map:
        return []
    var_base = Decimal(row.var_95) if Decimal(row.var_95) != 0 else Decimal("1")
    items = sorted(shap_map.items(), key=lambda x: abs(x[1]), reverse=True)
    return [
        AlertShapAttributionOut(
            ticker=ticker,
            contribution=Decimal(str(value)),
            pct_of_var=(Decimal(str(value)) / var_base * Decimal("100")).quantize(Decimal("0.01")),
        )
        for ticker, value in items
    ]


async def _risk_computation_for_event(db: AsyncSession, event: MarginEvent) -> RiskComputation | None:
    if event.risk_computation_id:
        linked = (
            await db.execute(
                select(RiskComputation).where(
                    RiskComputation.id == event.risk_computation_id,
                    RiskComputation.portfolio_id == event.portfolio_id,
                )
            )
        ).scalar_one_or_none()
        if linked:
            return linked

    return (
        await db.execute(
            select(RiskComputation)
            .where(RiskComputation.portfolio_id == event.portfolio_id)
            .order_by(func.abs(func.extract("epoch", RiskComputation.computed_at - event.triggered_at)))
            .limit(1)
        )
    ).scalar_one_or_none()


def _serialize_event(event: MarginEvent, portfolio_name: str) -> AlertEventOut:
    return AlertEventOut(
        id=event.id,
        portfolio_id=event.portfolio_id,
        portfolio_name=portfolio_name,
        event_type=event.event_type,
        triggered_at=event.triggered_at,
        var_95=event.var_95,
        margin_limit=event.margin_limit,
        margin_utilization=event.margin_utilization,
        message=_event_message(event.event_type, float(event.margin_utilization), float(event.var_95)),
        acknowledged_at=event.acknowledged_at,
        acknowledged=event.acknowledged_at is not None,
    )


def _filtered_select(
    user_id: UUID,
    portfolio_id: str | None,
    event_types: list[str] | None,
    from_dt: datetime | None,
    to_dt: datetime | None,
):
    stmt = (
        select(MarginEvent, Portfolio.name)
        .join(Portfolio, Portfolio.id == MarginEvent.portfolio_id)
        .where(Portfolio.user_id == user_id)
    )
    if portfolio_id:
        stmt = stmt.where(MarginEvent.portfolio_id == portfolio_id)
    if event_types:
        stmt = stmt.where(MarginEvent.event_type.in_(event_types))
    if from_dt:
        stmt = stmt.where(MarginEvent.triggered_at >= from_dt)
    if to_dt:
        stmt = stmt.where(MarginEvent.triggered_at <= to_dt)
    return stmt


def _count_filtered(
    user_id: UUID,
    portfolio_id: str | None,
    event_types: list[str] | None,
    from_dt: datetime | None,
    to_dt: datetime | None,
):
    stmt = (
        select(func.count(MarginEvent.id))
        .select_from(MarginEvent)
        .join(Portfolio, Portfolio.id == MarginEvent.portfolio_id)
        .where(Portfolio.user_id == user_id)
    )
    if portfolio_id:
        stmt = stmt.where(MarginEvent.portfolio_id == portfolio_id)
    if event_types:
        stmt = stmt.where(MarginEvent.event_type.in_(event_types))
    if from_dt:
        stmt = stmt.where(MarginEvent.triggered_at >= from_dt)
    if to_dt:
        stmt = stmt.where(MarginEvent.triggered_at <= to_dt)
    return stmt


@router.get("/export/csv")
async def export_alerts_csv(
    portfolio_id: str | None = None,
    event_type: list[str] | None = Query(None),
    from_date: str | None = None,
    to_date: str | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    if portfolio_id:
        owned = (
            await db.execute(
                select(Portfolio.id).where(Portfolio.id == portfolio_id, Portfolio.user_id == user.id).limit(1)
            )
        ).scalar_one_or_none()
        if not owned:
            raise HTTPException(status_code=404, detail="Portfolio not found")

    db_event_types = _normalize_event_types(event_type)
    from_dt = _parse_date_param(from_date)
    to_dt = _parse_date_param(to_date, end_of_day=True)

    stmt = _filtered_select(user.id, portfolio_id, db_event_types, from_dt, to_dt)
    stmt = stmt.order_by(MarginEvent.triggered_at.desc()).limit(CSV_MAX_ROWS)
    rows = (await db.execute(stmt)).all()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "triggered_at",
            "portfolio_id",
            "portfolio_name",
            "event_type",
            "var_95",
            "margin_limit",
            "margin_utilization",
            "message",
            "acknowledged_at",
        ]
    )
    for event, portfolio_name in rows:
        serialized = _serialize_event(event, portfolio_name)
        writer.writerow(
            [
                str(serialized.id),
                serialized.triggered_at.isoformat(),
                str(serialized.portfolio_id),
                serialized.portfolio_name,
                serialized.event_type,
                float(serialized.var_95),
                float(serialized.margin_limit),
                float(serialized.margin_utilization),
                serialized.message,
                serialized.acknowledged_at.isoformat() if serialized.acknowledged_at else "",
            ]
        )

    buffer.seek(0)
    filename = f"quantrisk-alerts-{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("", response_model=AlertsListResponse)
async def list_alerts(
    portfolio_id: str | None = None,
    event_type: list[str] | None = Query(None),
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AlertsListResponse:
    if portfolio_id:
        owned = (
            await db.execute(
                select(Portfolio.id).where(Portfolio.id == portfolio_id, Portfolio.user_id == user.id).limit(1)
            )
        ).scalar_one_or_none()
        if not owned:
            raise HTTPException(status_code=404, detail="Portfolio not found")

    db_event_types = _normalize_event_types(event_type)
    from_dt = _parse_date_param(from_date)
    to_dt = _parse_date_param(to_date, end_of_day=True)

    base = _filtered_select(user.id, portfolio_id, db_event_types, from_dt, to_dt)
    total = int((await db.execute(_count_filtered(user.id, portfolio_id, db_event_types, from_dt, to_dt))).scalar_one())

    page_stmt = base.order_by(MarginEvent.triggered_at.desc()).offset(offset).limit(limit)
    rows = (await db.execute(page_stmt)).all()
    items = [_serialize_event(event, portfolio_name) for event, portfolio_name in rows]
    return AlertsListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/summary")
async def alerts_summary(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    # Latest event per portfolio for this user in one query (DISTINCT ON avoids N+1).
    rows = (
        await db.execute(
            select(MarginEvent, Portfolio.name)
            .join(Portfolio, Portfolio.id == MarginEvent.portfolio_id)
            .where(Portfolio.user_id == user.id)
            .order_by(MarginEvent.portfolio_id, MarginEvent.triggered_at.desc())
            .distinct(MarginEvent.portfolio_id)
        )
    ).all()
    return [
        {
            "portfolio_id": str(event.portfolio_id),
            "portfolio_name": portfolio_name,
            "event_type": event.event_type,
            "triggered_at": event.triggered_at,
            "utilization": float(event.margin_utilization),
        }
        for event, portfolio_name in rows
    ]


@router.get("/portfolios/{portfolio_id}")
async def portfolio_alerts(
    portfolio_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[dict]:
    p = (
        await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id, Portfolio.user_id == user.id).limit(1))
    ).scalar_one_or_none()
    if not p:
        return []
    events = (
        await db.execute(
            select(MarginEvent).where(MarginEvent.portfolio_id == portfolio_id).order_by(MarginEvent.triggered_at.desc()).limit(limit)
        )
    ).scalars().all()
    return [
        {
            "id": str(e.id),
            "event_type": e.event_type,
            "triggered_at": e.triggered_at,
            "var_95": float(e.var_95),
            "margin_limit": float(e.margin_limit),
            "margin_utilization": float(e.margin_utilization),
        }
        for e in events
    ]


@router.get("/{event_id}/detail", response_model=AlertDetailOut)
async def alert_detail(
    event_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AlertDetailOut:
    row = (
        await db.execute(
            select(MarginEvent, Portfolio.name)
            .join(Portfolio, Portfolio.id == MarginEvent.portfolio_id)
            .where(MarginEvent.id == event_id, Portfolio.user_id == user.id)
        )
    ).one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    event, portfolio_name = row
    base = _serialize_event(event, portfolio_name)

    risk_row = await _risk_computation_for_event(db, event)
    cvar_95: Decimal | None = None
    stress_loss_moderate: Decimal | None = None
    shap_attributions: list[AlertShapAttributionOut] = []
    risk_computed_at = None
    detail_var_95 = base.var_95

    if risk_row:
        cvar_95 = Decimal(risk_row.cvar_95)
        stress_loss_moderate = Decimal(risk_row.stress_moderate)
        shap_attributions = await _shap_attributions_for_risk(db, risk_row)
        risk_computed_at = risk_row.computed_at
        detail_var_95 = Decimal(risk_row.var_95)

    return AlertDetailOut(
        id=base.id,
        portfolio_id=base.portfolio_id,
        portfolio_name=base.portfolio_name,
        event_type=base.event_type,
        triggered_at=base.triggered_at,
        var_95=detail_var_95,
        margin_limit=base.margin_limit,
        margin_utilization=base.margin_utilization,
        message=base.message,
        acknowledged_at=base.acknowledged_at,
        acknowledged=base.acknowledged,
        cvar_95=cvar_95,
        shap_attributions=shap_attributions,
        stress_loss_moderate=stress_loss_moderate,
        risk_computed_at=risk_computed_at,
    )


@router.post("/{event_id}/acknowledge", response_model=AlertEventOut)
async def acknowledge_alert(
    event_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AlertEventOut:
    row = (
        await db.execute(
            select(MarginEvent, Portfolio.name)
            .join(Portfolio, Portfolio.id == MarginEvent.portfolio_id)
            .where(MarginEvent.id == event_id, Portfolio.user_id == user.id)
        )
    ).one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    event, portfolio_name = row
    if event.acknowledged_at is None:
        event.acknowledged_at = datetime.now(timezone.utc)
        event.acknowledged_by = user.id
        await db.commit()
        await db.refresh(event)
    return _serialize_event(event, portfolio_name)


@portfolio_alert_router.get("")
async def portfolio_alerts_alias(
    portfolio_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[dict]:
    return await portfolio_alerts(portfolio_id=portfolio_id, limit=limit, db=db, user=user)
