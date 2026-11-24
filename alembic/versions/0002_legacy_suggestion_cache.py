"""Store migration-safe legacy tag suggestion metadata.

Revision ID: 0002_legacy_cache
Revises: 0001_unified_search
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0002_legacy_cache"
down_revision: str | None = "0001_unified_search"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tag_suggestion_cache",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("file_id", sa.Integer(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_hash", sa.String(length=128), nullable=False),
        sa.Column("cached_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("suggestions", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_id"),
    )


def downgrade() -> None:
    op.drop_table("tag_suggestion_cache")
