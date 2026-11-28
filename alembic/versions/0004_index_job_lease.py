"""Add ownership tokens for indexing job leases.

Revision ID: 0004_index_job_lease
Revises: 0003_index_checkpoint
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0004_index_job_lease"
down_revision: str | None = "0003_index_checkpoint"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "index_jobs",
        sa.Column("lease_token", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("index_jobs", "lease_token")
