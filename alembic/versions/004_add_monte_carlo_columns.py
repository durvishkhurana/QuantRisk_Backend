"""Add Monte Carlo VaR columns and histogram JSON to risk_computations."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "004_add_monte_carlo_columns"
down_revision = "003_add_shap_method"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("risk_computations", sa.Column("mc_var_95", sa.Numeric(14, 2), nullable=True))
    op.add_column("risk_computations", sa.Column("mc_var_99", sa.Numeric(14, 2), nullable=True))
    op.add_column("risk_computations", sa.Column("mc_cvar_95", sa.Numeric(14, 2), nullable=True))
    op.add_column("risk_computations", sa.Column("mc_skewness", sa.Numeric(8, 4), nullable=True))
    op.add_column("risk_computations", sa.Column("mc_kurtosis", sa.Numeric(8, 4), nullable=True))
    op.add_column("risk_computations", sa.Column("mc_histogram", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("risk_computations", "mc_histogram")
    op.drop_column("risk_computations", "mc_kurtosis")
    op.drop_column("risk_computations", "mc_skewness")
    op.drop_column("risk_computations", "mc_cvar_95")
    op.drop_column("risk_computations", "mc_var_99")
    op.drop_column("risk_computations", "mc_var_95")
