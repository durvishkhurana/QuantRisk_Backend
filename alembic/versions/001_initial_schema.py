"""Initial schema for QuantRisk tables."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "portfolios",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("margin_limit", sa.Numeric(5, 4), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "positions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("portfolio_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("quantity", sa.Numeric(12, 4), nullable=False),
        sa.Column("purchase_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("sector", sa.String(length=50)),
        sa.Column("asset_class", sa.String(length=30)),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("portfolio_id", "ticker", name="uq_portfolio_ticker"),
    )
    op.create_table(
        "price_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("close", sa.Numeric(12, 4), nullable=False),
        sa.Column("volume", sa.BigInteger()),
        sa.Column("source", sa.String(length=20)),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("ticker", "date"),
    )
    op.create_table(
        "risk_computations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("portfolio_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("portfolio_value", sa.Numeric(14, 2), nullable=False),
        sa.Column("var_95", sa.Numeric(14, 2), nullable=False),
        sa.Column("var_99", sa.Numeric(14, 2), nullable=False),
        sa.Column("cvar_95", sa.Numeric(14, 2), nullable=False),
        sa.Column("margin_utilization", sa.Numeric(6, 4), nullable=False),
        sa.Column("margin_status", sa.String(length=10), nullable=False),
        sa.Column("stress_mild", sa.Numeric(14, 2), nullable=False),
        sa.Column("stress_moderate", sa.Numeric(14, 2), nullable=False),
        sa.Column("stress_severe", sa.Numeric(14, 2), nullable=False),
        sa.Column("shap_json", sa.JSON(), nullable=False),
        sa.Column("n_positions", sa.Integer(), nullable=False),
        sa.Column("lookback_days", sa.Integer(), nullable=False),
        sa.Column("computation_ms", sa.Integer()),
        sa.Column("triggered_by", sa.String(length=20)),
    )
    op.create_table(
        "margin_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("portfolio_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("event_type", sa.String(length=10), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("var_95", sa.Numeric(14, 2), nullable=False),
        sa.Column("margin_limit", sa.Numeric(5, 4), nullable=False),
        sa.Column("margin_utilization", sa.Numeric(6, 4), nullable=False),
        sa.Column("risk_computation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("risk_computations.id")),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True)),
        sa.Column("acknowledged_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id")),
    )


def downgrade() -> None:
    op.drop_table("margin_events")
    op.drop_table("risk_computations")
    op.drop_table("price_snapshots")
    op.drop_table("positions")
    op.drop_table("portfolios")
    op.drop_table("users")
