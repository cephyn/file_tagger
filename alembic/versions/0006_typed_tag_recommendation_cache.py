"""Add typed identities to tag recommendation cache entries.

Revision ID: 0006_typed_tag_cache
Revises: 0005_index_job_lease_expiration
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0006_typed_tag_cache"
down_revision: str | None = "0005_index_job_lease_expiration"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("tag_suggestion_cache") as batch:
        batch.add_column(
            sa.Column("extraction_version", sa.String(length=64), nullable=True)
        )
        batch.add_column(
            sa.Column(
                "candidate_model_digest",
                sa.String(length=255),
                nullable=True,
            )
        )
        batch.add_column(
            sa.Column(
                "decision_model_digest",
                sa.String(length=255),
                nullable=True,
            )
        )
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


def downgrade() -> None:
    with op.batch_alter_table("tag_suggestion_cache") as batch:
        batch.drop_index("ix_tag_suggestion_cache_identity")
        batch.drop_column("decision_model_digest")
        batch.drop_column("candidate_model_digest")
        batch.drop_column("extraction_version")
