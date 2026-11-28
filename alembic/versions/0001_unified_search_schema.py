"""Create the unified local search schema.

Revision ID: 0001_unified_search
Revises:
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0001_unified_search"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "files",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("canonical_path", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("mtime_ns", sa.BigInteger(), nullable=True),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("indexing_state", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("canonical_path"),
    )
    op.create_table(
        "tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("color", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "file_tags",
        sa.Column("file_id", sa.Integer(), nullable=False),
        sa.Column("tag_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tag_id"], ["tags.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("file_id", "tag_id"),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("file_id", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("file_hash", sa.String(length=128), nullable=False),
        sa.Column("extraction_version", sa.String(length=64), nullable=False),
        sa.Column("parser", sa.String(length=64), nullable=False),
        sa.Column("parser_version", sa.String(length=64), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("extraction_status", sa.String(length=32), nullable=False),
        sa.Column("diagnostics", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("revision > 0", name="ck_documents_revision_positive"),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "file_id", "revision", name="uq_documents_file_revision"
        ),
    )
    op.create_table(
        "pages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column("headings", sa.JSON(), nullable=False),
        sa.Column("quality", sa.String(length=32), nullable=False),
        sa.CheckConstraint("page_number > 0", name="ck_pages_number_positive"),
        sa.CheckConstraint("ordinal >= 0", name="ck_pages_ordinal_nonnegative"),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id", "page_number", name="uq_pages_document_page"
        ),
        sa.UniqueConstraint(
            "document_id", "ordinal", name="uq_pages_document_ordinal"
        ),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("stable_id", sa.String(length=512), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("page_id", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("heading", sa.Text(), nullable=True),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("model_digest", sa.String(length=255), nullable=False),
        sa.Column("index_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_nonnegative"),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["page_id"], ["pages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id", "ordinal", name="uq_chunks_document_ordinal"
        ),
        sa.UniqueConstraint(
            "stable_id", "index_version", name="uq_chunks_id_version"
        ),
    )
    op.create_table(
        "index_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("file_id", sa.Integer(), nullable=True),
        sa.Column("job_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("total_count", sa.Integer(), nullable=False),
        sa.Column("processed_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("failure", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "processed_count >= 0", name="ck_jobs_processed_nonnegative"
        ),
        sa.CheckConstraint("failed_count >= 0", name="ck_jobs_failed_nonnegative"),
        sa.CheckConstraint("retry_count >= 0", name="ck_jobs_retry_nonnegative"),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "model_profiles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("digest", sa.String(length=255), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=True),
        sa.Column("context_length", sa.Integer(), nullable=False),
        sa.Column("index_version", sa.String(length=64), nullable=False),
        sa.Column("capabilities", sa.JSON(), nullable=False),
        sa.Column("benchmark_data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "role",
            "model",
            "digest",
            "dimensions",
            name="uq_model_profiles_identity",
        ),
    )
    op.create_index(
        "ix_model_profiles_role_index",
        "model_profiles",
        ["role", "index_version"],
        unique=False,
    )

    op.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5("
        "text, heading, section, content='chunks', content_rowid='id'"
        ")"
    )
    for dimension in (256, 512, 768):
        op.execute(
            f"CREATE VIRTUAL TABLE chunk_embeddings_{dimension} USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding float[{dimension}])"
        )

    op.execute(
        """
        CREATE TRIGGER chunks_fts_after_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, heading, section)
            VALUES (new.id, new.text, coalesce(new.heading, ''), coalesce(new.section, ''));
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER chunks_fts_after_delete AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, heading, section)
            VALUES ('delete', old.id, old.text, coalesce(old.heading, ''), coalesce(old.section, ''));
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER chunks_fts_after_update AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, heading, section)
            VALUES ('delete', old.id, old.text, coalesce(old.heading, ''), coalesce(old.section, ''));
            INSERT INTO chunks_fts(rowid, text, heading, section)
            VALUES (new.id, new.text, coalesce(new.heading, ''), coalesce(new.section, ''));
        END
        """
    )


def downgrade() -> None:
    for trigger in (
        "chunks_fts_after_update",
        "chunks_fts_after_delete",
        "chunks_fts_after_insert",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    for table in (
        "chunk_embeddings_768",
        "chunk_embeddings_512",
        "chunk_embeddings_256",
        "chunks_fts",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table}")
    op.drop_index("ix_model_profiles_role_index", table_name="model_profiles")
    op.drop_table("model_profiles")
    op.drop_table("index_jobs")
    op.drop_table("chunks")
    op.drop_table("pages")
    op.drop_table("documents")
    op.drop_table("file_tags")
    op.drop_table("tags")
    op.drop_table("files")
