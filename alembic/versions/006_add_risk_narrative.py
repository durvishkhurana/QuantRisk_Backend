"""Add risk_narrative text column to risk_computations."""
from alembic import op
import sqlalchemy as sa

revision = "006_add_risk_narrative"
down_revision = "005_add_correlation_json"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("risk_computations", sa.Column("risk_narrative", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("risk_computations", "risk_narrative")
