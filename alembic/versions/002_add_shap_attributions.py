"""Add shap_attributions table."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002_add_shap_attributions"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shap_attributions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("risk_computation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("risk_computations.id"), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("shap_value", sa.Numeric(14, 2), nullable=False),
        sa.Column("pct_of_var", sa.Numeric(6, 2)),
        sa.Column("position_weight", sa.Numeric(6, 4)),
    )


def downgrade() -> None:
    op.drop_table("shap_attributions")
