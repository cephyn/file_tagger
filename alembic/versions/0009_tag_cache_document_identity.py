"""Scope cached tag recommendations to the target document state.

Revision ID: 0009_tag_cache_document
Revises: 0008_tag_cache_params
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0009_tag_cache_document"
down_revision: str | None = "0008_tag_cache_params"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("tag_suggestion_cache") as batch:
        batch.drop_index("ix_tag_suggestion_cache_identity")
        batch.add_column(
            sa.Column("document_state_digest", sa.String(length=64), nullable=True)
        )
        batch.create_index(
            "ix_tag_suggestion_cache_identity",
            [
                "file_id",
                "file_path",
                "file_hash",
                "document_state_digest",
                "extraction_version",
                "candidate_model_digest",
                "decision_model_digest",
                "candidate_set_digest",
                "max_candidates",
                "confirmation_threshold",
            ],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("tag_suggestion_cache") as batch:
        batch.drop_index("ix_tag_suggestion_cache_identity")
        batch.drop_column("document_state_digest")
        batch.create_index(
            "ix_tag_suggestion_cache_identity",
            [
                "file_hash",
                "extraction_version",
                "candidate_model_digest",
                "decision_model_digest",
                "candidate_set_digest",
                "max_candidates",
                "confirmation_threshold",
            ],
            unique=False,
        )
