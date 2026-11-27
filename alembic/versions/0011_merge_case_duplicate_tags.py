"""Merge case-insensitive duplicate tags.

Revision ID: 0011_case_duplicate_tags
Revises: 0010_index_availability
"""

from collections.abc import Sequence
import unicodedata

from alembic import op
from sqlalchemy import text


revision: str = "0011_case_duplicate_tags"
down_revision: str | None = "0010_index_availability"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _tag_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def upgrade() -> None:
    bind = op.get_bind()
    tags = bind.execute(
        text("SELECT id, name FROM tags ORDER BY id")
    ).all()

    canonical_by_key: dict[str, int] = {}
    duplicate_ids: dict[int, int] = {}
    for tag_id, name in tags:
        key = _tag_key(str(name))
        canonical_id = canonical_by_key.setdefault(key, int(tag_id))
        if int(tag_id) != canonical_id:
            duplicate_ids[int(tag_id)] = canonical_id

    for duplicate_id, canonical_id in duplicate_ids.items():
        file_ids = bind.execute(
            text(
                "SELECT file_id FROM file_tags WHERE tag_id = :duplicate_id"
            ),
            {"duplicate_id": duplicate_id},
        ).scalars()
        for file_id in tuple(file_ids):
            canonical_exists = bind.execute(
                text(
                    "SELECT 1 FROM file_tags "
                    "WHERE file_id = :file_id AND tag_id = :canonical_id"
                ),
                {"file_id": file_id, "canonical_id": canonical_id},
            ).first()
            if canonical_exists is not None:
                bind.execute(
                    text(
                        "DELETE FROM file_tags "
                        "WHERE file_id = :file_id AND tag_id = :duplicate_id"
                    ),
                    {"file_id": file_id, "duplicate_id": duplicate_id},
                )
            else:
                bind.execute(
                    text(
                        "UPDATE file_tags SET tag_id = :canonical_id "
                        "WHERE file_id = :file_id AND tag_id = :duplicate_id"
                    ),
                    {
                        "file_id": file_id,
                        "duplicate_id": duplicate_id,
                        "canonical_id": canonical_id,
                    },
                )
        bind.execute(
            text("DELETE FROM tags WHERE id = :duplicate_id"),
            {"duplicate_id": duplicate_id},
        )

    op.execute(
        "CREATE UNIQUE INDEX uq_tags_normalized_name "
        "ON tags (unicode_normalize_tag(name))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_tags_normalized_name")
