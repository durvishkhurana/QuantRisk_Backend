"""Enable TimescaleDB hypertables for time-series tables."""

import logging

from alembic import op

revision = "010_timescaledb_hypertables"
down_revision = "006_add_risk_narrative"
branch_labels = None
depends_on = None

logger = logging.getLogger(__name__)


def upgrade() -> None:
    try:
        with op.get_context().autocommit_block():
            op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            op.execute(
                "SELECT create_hypertable('price_snapshots', 'fetched_at', "
                "if_not_exists => TRUE, migrate_data => TRUE)"
            )
            op.execute(
                "SELECT create_hypertable('risk_computations', 'computed_at', "
                "if_not_exists => TRUE, migrate_data => TRUE)"
            )
            op.execute("SELECT add_compression_policy('price_snapshots', INTERVAL '7 days')")
            op.execute("SELECT add_retention_policy('price_snapshots', INTERVAL '400 days')")
    except Exception as exc:  # noqa: BLE001 — plain Postgres (e.g. Render, CI) has no TimescaleDB
        msg = (
            "TimescaleDB not available; skipping hypertable, compression, and retention setup. "
            "Plain PostgreSQL tables work without time-series optimizations. "
            f"({exc})"
        )
        logger.warning(msg)
        print(f"WARNING: {msg}")


def downgrade() -> None:
    print(
        "WARNING: TimescaleDB hypertable downgrade is a no-op; "
        "manual intervention required if rollback is needed."
    )
