"""Include recommendation request parameters in cache identity.

Revision ID: 0008_tag_cache_params
Revises: 0007_tag_cache_candidate_set
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0008_tag_cache_params"
down_revision: str | None = "0007_tag_cache_candidate_set"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("tag_suggestion_cache") as batch:
        batch.drop_index("ix_tag_suggestion_cache_identity")
        batch.add_column(sa.Column("max_candidates", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("confirmation_threshold", sa.Float(), nullable=True)
        )
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


def downgrade() -> None:
    with op.batch_alter_table("tag_suggestion_cache") as batch:
        batch.drop_index("ix_tag_suggestion_cache_identity")
        batch.drop_column("confirmation_threshold")
        batch.drop_column("max_candidates")
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
