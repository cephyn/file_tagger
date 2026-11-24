from pathlib import Path
import sqlite3

from alembic import command
from alembic.config import Config
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


def test_index_job_lease_migration_grants_preexisting_running_job_a_deadline(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-running.sqlite3"
    config_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    config = Config(str(config_path))
    config.attributes["database_path"] = database_path
    command.upgrade(config, "0004_index_job_lease")
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            "INSERT INTO files "
            "(canonical_path, indexing_state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (r"C:\docs\guide.md", "running", "2026-11-24 11:00:00", "2026-11-24 11:00:00"),
        )
        connection.execute(
            "INSERT INTO index_jobs "
            "(file_id, job_type, status, total_count, processed_count, "
            "failed_count, retry_count, lease_token, started_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cursor.lastrowid,
                "index",
                "running",
                1,
                0,
                0,
                0,
                "legacy-owner",
                "2026-11-24 11:00:00",
                "2026-11-24 11:00:00",
                "2026-11-24 11:00:00",
            ),
        )

    command.upgrade(config, "head")

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT status, lease_token, lease_expires_at FROM index_jobs"
        ).fetchone()
    assert row is not None
    assert row[0:2] == ("running", "legacy-owner")
    assert isinstance(row[2], str)
    assert row[2]


def test_tag_cache_parameter_migration_invalidates_existing_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "tag-cache-v6.sqlite3"
    config_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    config = Config(str(config_path))
    config.attributes["database_path"] = database_path
    command.upgrade(config, "0007_tag_cache_candidate_set")
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO files "
            "(canonical_path, indexing_state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (
                r"C:\docs\report.md",
                "indexed",
                "2026-11-24 11:00:00",
                "2026-11-24 11:00:00",
            ),
        )
        connection.execute(
            "INSERT INTO tag_suggestion_cache "
            "(file_id, file_path, file_hash, cached_at, extraction_version, "
            "candidate_model_digest, decision_model_digest, suggestions) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                r"C:\docs\report.md",
                "abc123",
                "2026-11-24 11:00:00",
                "docling-v1",
                "candidate-v1",
                "decision-v1",
                "{}",
            ),
        )

    command.upgrade(config, "head")

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(tag_suggestion_cache)")
        }
        row = connection.execute(
            "SELECT file_hash, candidate_set_digest, max_candidates, "
            "confirmation_threshold FROM tag_suggestion_cache"
        ).fetchone()
    assert "candidate_set_digest" in columns
    assert "max_candidates" in columns
    assert "confirmation_threshold" in columns
    assert row == ("abc123", None, None, None)


def test_tag_cache_identity_migration_adds_document_state_digest(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "tag-cache-v8.sqlite3"
    config_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    config = Config(str(config_path))
    config.attributes["database_path"] = database_path
    command.upgrade(config, "0008_tag_cache_params")

    command.upgrade(config, "head")

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(tag_suggestion_cache)")
        }
        index_columns = [
            row[2]
            for row in connection.execute(
                "PRAGMA index_info(ix_tag_suggestion_cache_identity)"
            )
        ]

    assert "document_state_digest" in columns
    assert "file_path" in index_columns
    assert "document_state_digest" in index_columns
