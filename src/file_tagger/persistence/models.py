from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class File(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_path: Mapped[str] = mapped_column(Text, unique=True)
    content_hash: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    mtime_ns: Mapped[int | None] = mapped_column(BigInteger)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    indexing_state: Mapped[str] = mapped_column(String(32), default="not_indexed")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    tags: Mapped[list["FileTag"]] = relationship(
        back_populates="file", cascade="all, delete-orphan", passive_deletes=True
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="file", cascade="all, delete-orphan", passive_deletes=True
    )
    index_jobs: Mapped[list["IndexJob"]] = relationship(
        back_populates="file", cascade="all, delete-orphan", passive_deletes=True
    )
    suggestion_cache: Mapped["TagSuggestionCache | None"] = relationship(
        back_populates="file", cascade="all, delete-orphan", passive_deletes=True
    )


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    color: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    files: Mapped[list["FileTag"]] = relationship(
        back_populates="tag", cascade="all, delete-orphan", passive_deletes=True
    )


class FileTag(Base):
    __tablename__ = "file_tags"

    file_id: Mapped[int] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(
        ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    file: Mapped[File] = relationship(back_populates="tags")
    tag: Mapped[Tag] = relationship(back_populates="files")


class TagSuggestionCache(Base):
    __tablename__ = "tag_suggestion_cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_id: Mapped[int] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    cached_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    extraction_version: Mapped[str | None] = mapped_column(String(64))
    candidate_model_digest: Mapped[str | None] = mapped_column(String(255))
    decision_model_digest: Mapped[str | None] = mapped_column(String(255))
    suggestions: Mapped[dict[str, Any] | list[Any]] = mapped_column(
        JSON, nullable=False
    )

    file: Mapped[File] = relationship(back_populates="suggestion_cache")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("file_id", "revision", name="uq_documents_file_revision"),
        CheckConstraint("revision > 0", name="ck_documents_revision_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    file_id: Mapped[int] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    extraction_version: Mapped[str] = mapped_column(String(64), nullable=False)
    parser: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_version: Mapped[str | None] = mapped_column(String(64))
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    extraction_status: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False
    )
    diagnostics: Mapped[dict[str, Any] | list[Any]] = mapped_column(
        JSON, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    file: Mapped[File] = relationship(back_populates="documents")
    pages: Mapped[list["Page"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_pages_document_page"),
        UniqueConstraint("document_id", "ordinal", name="uq_pages_document_ordinal"),
        CheckConstraint("page_number > 0", name="ck_pages_number_positive"),
        CheckConstraint("ordinal >= 0", name="ck_pages_ordinal_nonnegative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, default="")
    markdown: Mapped[str] = mapped_column(Text, default="")
    headings: Mapped[list[str]] = mapped_column(JSON, default=list)
    quality: Mapped[str] = mapped_column(String(32), default="good")

    document: Mapped[Document] = relationship(back_populates="pages")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="page", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        UniqueConstraint("stable_id", "index_version", name="uq_chunks_id_version"),
        CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_nonnegative"),
        ForeignKeyConstraint(["page_id"], ["pages.id"], ondelete="CASCADE"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    stable_id: Mapped[str] = mapped_column(String(512), nullable=False)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_id: Mapped[int] = mapped_column(nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    heading: Mapped[str | None] = mapped_column(Text)
    section: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    model_digest: Mapped[str] = mapped_column(String(255), nullable=False)
    index_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
    page: Mapped[Page] = relationship(back_populates="chunks")


class IndexJob(Base):
    __tablename__ = "index_jobs"
    __table_args__ = (
        CheckConstraint("processed_count >= 0", name="ck_jobs_processed_nonnegative"),
        CheckConstraint("failed_count >= 0", name="ck_jobs_failed_nonnegative"),
        CheckConstraint("retry_count >= 0", name="ck_jobs_retry_nonnegative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    file_id: Mapped[int | None] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE")
    )
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    total_count: Mapped[int] = mapped_column(Integer, default=0)
    processed_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    lease_token: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    failure: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    checkpoint: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    file: Mapped[File | None] = relationship(back_populates="index_jobs")


class IndexJobCheckpointChunk(Base):
    __tablename__ = "index_job_checkpoint_chunks"
    __table_args__ = (
        UniqueConstraint(
            "job_id", "source_ordinal", name="uq_checkpoint_job_source_ordinal"
        ),
        CheckConstraint(
            "source_ordinal >= 0", name="ck_checkpoint_source_ordinal_nonnegative"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("index_jobs.id", ondelete="CASCADE"), nullable=False
    )
    source_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    embedded_chunks: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False
    )


class ModelProfileRecord(Base):
    __tablename__ = "model_profiles"
    __table_args__ = (
        UniqueConstraint(
            "role",
            "model",
            "digest",
            "dimensions",
            name="uq_model_profiles_identity",
        ),
        Index("ix_model_profiles_role_index", "role", "index_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    digest: Mapped[str] = mapped_column(String(255), nullable=False)
    dimensions: Mapped[int | None] = mapped_column(Integer)
    context_length: Mapped[int] = mapped_column(Integer, nullable=False)
    index_version: Mapped[str] = mapped_column(String(64), nullable=False)
    capabilities: Mapped[list[str]] = mapped_column(JSON, default=list)
    benchmark_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
