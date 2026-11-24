"""Persist resumable indexing checkpoint data.

Revision ID: 0003_index_checkpoint
Revises: 0002_legacy_cache
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0003_index_checkpoint"
down_revision: str | None = "0002_legacy_cache"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("index_jobs", sa.Column("checkpoint", sa.JSON(), nullable=True))
    op.create_table(
        "index_job_checkpoint_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("source_ordinal", sa.Integer(), nullable=False),
        sa.Column("embedded_chunks", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "source_ordinal >= 0",
            name="ck_checkpoint_source_ordinal_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["index_jobs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id", "source_ordinal", name="uq_checkpoint_job_source_ordinal"
        ),
    )


def downgrade() -> None:
    op.drop_table("index_job_checkpoint_chunks")
    op.drop_column("index_jobs", "checkpoint")
