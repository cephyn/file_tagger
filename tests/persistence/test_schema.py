from pathlib import Path

import pytest
from sqlalchemy import select, text

from file_tagger.infrastructure.database import create_session_factory, upgrade_database
from file_tagger.persistence.models import Document, File


@pytest.fixture
def schema_database(tmp_path: Path) -> Path:
    database_path = tmp_path / "schema.sqlite3"
    upgrade_database(database_path)
    return database_path


def test_new_database_has_unified_search_tables(schema_database: Path) -> None:
    session_factory = create_session_factory(schema_database)

    with session_factory() as session:
        names = set(
            session.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type IN ('table', 'view')"
                )
            ).scalars()
        )

    assert {
        "files",
        "tags",
        "file_tags",
        "documents",
        "pages",
        "chunks",
        "index_jobs",
        "index_job_checkpoint_chunks",
        "model_profiles",
        "chunks_fts",
        "chunk_embeddings_256",
        "chunk_embeddings_512",
        "chunk_embeddings_768",
    } <= names


def test_file_deletion_cascades_to_documents(schema_database: Path) -> None:
    session_factory = create_session_factory(schema_database)

    with session_factory.begin() as session:
        file = File(
            canonical_path=r"C:\docs\guide.pdf",
            content_hash="abc123",
            size_bytes=100,
            mtime_ns=123,
            mime_type="application/pdf",
            indexing_state="pending",
        )
        document = Document(
            file=file,
            revision=1,
            file_hash="abc123",
            extraction_version="docling-1",
            parser="docling",
            page_count=1,
            extraction_status="complete",
        )
        session.add_all([file, document])

    with session_factory.begin() as session:
        file = session.scalar(select(File))
        assert file is not None
        session.delete(file)

    with session_factory() as session:
        assert session.scalar(select(Document)) is None
