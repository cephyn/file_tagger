"""Separate committed index availability from indexing job state.

Revision ID: 0010_index_availability
Revises: 0009_tag_cache_document
"""

from collections.abc import Sequence

from alembic import op


revision: str = "0010_index_availability"
down_revision: str | None = "0009_tag_cache_document"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JOB_STATES = ("failed", "paused", "pending", "queued", "running")


def upgrade() -> None:
    states = ", ".join(f"'{state}'" for state in _JOB_STATES)
    op.execute(
        f"""
        UPDATE files
        SET indexing_state = CASE
            WHEN EXISTS (
                SELECT 1
                FROM documents
                WHERE documents.file_id = files.id
            ) THEN 'indexed'
            ELSE 'not_indexed'
        END
        WHERE indexing_state IN ({states})
        """
    )


def downgrade() -> None:
    pass
