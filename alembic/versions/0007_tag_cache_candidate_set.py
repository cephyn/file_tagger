"""Add candidate-set identity to tag recommendation cache entries.

Revision ID: 0007_tag_cache_candidate_set
Revises: 0006_typed_tag_cache
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0007_tag_cache_candidate_set"
down_revision: str | None = "0006_typed_tag_cache"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("tag_suggestion_cache") as batch:
        batch.drop_index("ix_tag_suggestion_cache_identity")
        batch.add_column(
            sa.Column("candidate_set_digest", sa.String(length=64), nullable=True)
        )
        batch.create_index(
            "ix_tag_suggestion_cache_identity",
            [
                "file_hash",
                "extraction_version",
                "candidate_model_digest",
                "decision_model_digest",
                "candidate_set_digest",
            ],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("tag_suggestion_cache") as batch:
        batch.drop_index("ix_tag_suggestion_cache_identity")
        batch.drop_column("candidate_set_digest")
        batch.create_index(
            "ix_tag_suggestion_cache_identity",
            [
                "file_hash",
                "extraction_version",
                "candidate_model_digest",
                "decision_model_digest",
            ],
            unique=False,
        )
