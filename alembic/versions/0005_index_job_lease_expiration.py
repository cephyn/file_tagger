"""Persist expiration deadlines for indexing job leases.

Revision ID: 0005_index_job_lease_expiration
Revises: 0004_index_job_lease
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0005_index_job_lease_expiration"
down_revision: str | None = "0004_index_job_lease"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "index_jobs",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        """
        UPDATE index_jobs
        SET lease_expires_at = datetime(
            COALESCE(updated_at, started_at, CURRENT_TIMESTAMP),
            '+300 seconds'
        )
        WHERE status = 'running' AND lease_expires_at IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("index_jobs", "lease_expires_at")
