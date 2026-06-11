"""Add correlation_json to risk_computations and widen margin event types."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "005_add_correlation_json"
down_revision = "004_add_monte_carlo_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("risk_computations", sa.Column("correlation_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.alter_column("margin_events", "event_type", existing_type=sa.String(length=10), type_=sa.String(length=30))
    op.execute(
        "ALTER TABLE margin_events DROP CONSTRAINT IF EXISTS ck_margin_events_event_type"
    )
    op.create_check_constraint(
        "ck_margin_events_event_type",
        "margin_events",
        "event_type IN ('BREACH', 'WARNING', 'CORRELATION_ALERT')",
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE margin_events DROP CONSTRAINT IF EXISTS ck_margin_events_event_type"
    )
    op.alter_column("margin_events", "event_type", existing_type=sa.String(length=30), type_=sa.String(length=10))
    op.drop_column("risk_computations", "correlation_json")
