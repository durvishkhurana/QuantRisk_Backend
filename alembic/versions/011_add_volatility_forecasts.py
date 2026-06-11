"""Add volatility_forecasts table for LSTM/GARCH forecasts."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "011_add_volatility_forecasts"
down_revision = "010_timescaledb_hypertables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "volatility_forecasts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("portfolio_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("predicted_vol", sa.Float(), nullable=False),
        sa.Column("garch_vol", sa.Float(), nullable=False),
        sa.Column("lstm_mae", sa.Float(), nullable=False),
        sa.Column("lstm_rmse", sa.Float(), nullable=False),
        sa.Column("garch_mae", sa.Float(), nullable=False),
        sa.Column("garch_rmse", sa.Float(), nullable=False),
        sa.Column("direction_accuracy", sa.Float(), nullable=False),
        sa.Column("vol_regime", sa.String(length=10), nullable=False),
        sa.Column("adjusted_var_95", sa.Float(), nullable=True),
        sa.Column("improvement_pct", sa.Float(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_volatility_forecasts_portfolio_ticker", "volatility_forecasts", ["portfolio_id", "ticker"])
    op.create_index("ix_volatility_forecasts_computed_at", "volatility_forecasts", ["computed_at"])


def downgrade() -> None:
    op.drop_index("ix_volatility_forecasts_computed_at", table_name="volatility_forecasts")
    op.drop_index("ix_volatility_forecasts_portfolio_ticker", table_name="volatility_forecasts")
    op.drop_table("volatility_forecasts")
