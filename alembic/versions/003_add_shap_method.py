"""Add method column to shap_attributions."""
from alembic import op
import sqlalchemy as sa

revision = "003_add_shap_method"
down_revision = "002_add_shap_attributions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("shap_attributions", sa.Column("method", sa.String(length=20), nullable=False, server_default="linear"))
    op.alter_column("shap_attributions", "method", server_default=None)


def downgrade() -> None:
    op.drop_column("shap_attributions", "method")
