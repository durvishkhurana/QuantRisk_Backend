import uuid
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy import Date, DateTime, Float, ForeignKey, Numeric, String, Boolean, BigInteger, JSON, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


def _uuid_col() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = _uuid_col()
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    portfolios: Mapped[list["Portfolio"]] = relationship(back_populates="user", cascade="all, delete")


class Portfolio(Base):
    __tablename__ = "portfolios"
    id: Mapped[uuid.UUID] = _uuid_col()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    margin_limit: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=Decimal("0.05"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    user: Mapped[User] = relationship(back_populates="portfolios")
    positions: Mapped[list["Position"]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("portfolio_id", "ticker", name="uq_portfolio_ticker"),)
    id: Mapped[uuid.UUID] = _uuid_col()
    portfolio_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), index=True)
    ticker: Mapped[str] = mapped_column(String(10))
    quantity: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    purchase_price: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    sector: Mapped[str | None] = mapped_column(String(50), nullable=True)
    asset_class: Mapped[str] = mapped_column(String(30), default="equity")
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    portfolio: Mapped[Portfolio] = relationship(back_populates="positions")


class PriceSnapshot(Base):
    __tablename__ = "price_snapshots"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[date] = mapped_column(Date)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="yfinance")
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class RiskComputation(Base):
    __tablename__ = "risk_computations"
    id: Mapped[uuid.UUID] = _uuid_col()
    portfolio_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("portfolios.id"), index=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    portfolio_value: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    var_95: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    var_99: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    cvar_95: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    margin_utilization: Mapped[Decimal] = mapped_column(Numeric(6, 4))
    margin_status: Mapped[str] = mapped_column(String(10))
    stress_mild: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    stress_moderate: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    stress_severe: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    shap_json: Mapped[dict] = mapped_column(JSON, default=dict)
    n_positions: Mapped[int] = mapped_column(default=0)
    lookback_days: Mapped[int] = mapped_column(default=252)
    computation_ms: Mapped[int | None] = mapped_column(nullable=True)
    triggered_by: Mapped[str] = mapped_column(String(20), default="scheduler")
    mc_var_95: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    mc_var_99: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    mc_cvar_95: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    mc_skewness: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    mc_kurtosis: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    mc_histogram: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    correlation_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    risk_narrative: Mapped[str | None] = mapped_column(Text, nullable=True)


class MarginEvent(Base):
    __tablename__ = "margin_events"
    id: Mapped[uuid.UUID] = _uuid_col()
    portfolio_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("portfolios.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(30))
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    var_95: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    margin_limit: Mapped[Decimal] = mapped_column(Numeric(5, 4))
    margin_utilization: Mapped[Decimal] = mapped_column(Numeric(6, 4))
    risk_computation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("risk_computations.id"), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)


class ShapAttribution(Base):
    __tablename__ = "shap_attributions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    risk_computation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("risk_computations.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(10))
    shap_value: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    method: Mapped[str] = mapped_column(String(20), default="linear")
    pct_of_var: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    position_weight: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)


class VolatilityForecast(Base):
    __tablename__ = "volatility_forecasts"
    id: Mapped[uuid.UUID] = _uuid_col()
    portfolio_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), index=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    predicted_vol: Mapped[float] = mapped_column(Float)
    garch_vol: Mapped[float] = mapped_column(Float)
    lstm_mae: Mapped[float] = mapped_column(Float)
    lstm_rmse: Mapped[float] = mapped_column(Float)
    garch_mae: Mapped[float] = mapped_column(Float)
    garch_rmse: Mapped[float] = mapped_column(Float)
    direction_accuracy: Mapped[float] = mapped_column(Float)
    vol_regime: Mapped[str] = mapped_column(String(10))
    adjusted_var_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    improvement_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)
