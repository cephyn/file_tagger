import json
from pathlib import Path
import sqlite3
import struct

import pytest
from sqlalchemy import func, select, text

from file_tagger.domain.errors import IndexTransactionError
from file_tagger.infrastructure.database import create_session_factory, upgrade_database
from file_tagger.migration.sqlite_migration import migrate_legacy_sqlite
from file_tagger.persistence.models import (
    Chunk,
    Document,
    File,
    FileTag,
    Page,
    Tag,
    TagSuggestionCache,
)


def _create_legacy_database(path: Path, *, suggestions: bool = True) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE files (id INTEGER PRIMARY KEY, path VARCHAR UNIQUE);
            CREATE TABLE tags (
                id INTEGER PRIMARY KEY,
                name VARCHAR UNIQUE,
                color VARCHAR
            );
            CREATE TABLE file_tags (file_id INTEGER, tag_id INTEGER);
            """
        )
        connection.executemany(
            "INSERT INTO files (id, path) VALUES (?, ?)",
            [(1, r"C:\library\source.pdf")],
        )
        connection.executemany(
            "INSERT INTO tags (id, name, color) VALUES (?, ?, ?)",
            [(1, "research", "#123456"), (2, "reference", "#abcdef")],
        )
        connection.executemany(
            "INSERT INTO file_tags (file_id, tag_id) VALUES (?, ?)",
            [(1, 1), (1, 2)],
        )
        if suggestions:
            connection.execute(
                """
                CREATE TABLE tag_suggestion_cache (
                    id INTEGER PRIMARY KEY,
                    file_path VARCHAR UNIQUE,
                    file_hash VARCHAR,
                    timestamp DATETIME,
                    suggestions JSON,
                    provider VARCHAR
                )
                """
            )
            connection.execute(
                """
                INSERT INTO tag_suggestion_cache
                    (file_path, file_hash, timestamp, suggestions, provider)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    r"C:\library\source.pdf",
                    "abc123",
                    "2026-11-24 12:00:00",
                    json.dumps({"new_tags": [["research", 0.9]]}),
                    "openai",
                ),
            )


@pytest.fixture
def target_session(tmp_path):
    database = tmp_path / "target.sqlite3"
    upgrade_database(database)
    session_factory = create_session_factory(database)
    with session_factory() as session:
        yield session


def _add_indexed_file(
    session, path: str, *, stable_id: str
) -> tuple[int, int]:
    file = File(
        canonical_path=path,
        content_hash="hash",
        indexing_state="indexed",
    )
    document = Document(
        file=file,
        revision=1,
        file_hash="hash",
        extraction_version="test-1",
        parser="test",
        page_count=1,
        extraction_status="complete",
    )
    page = Page(
        document=document,
        page_number=1,
        ordinal=0,
        text="text",
        markdown="text",
    )
    chunk = Chunk(
        document=document,
        page=page,
        stable_id=stable_id,
        ordinal=0,
        text="text",
        content_hash="chunk-hash",
        model_digest="model",
        index_version="index-1",
    )
    session.add_all([file, document, page, chunk])
    session.flush()
    session.execute(
        text(
            "INSERT INTO chunk_embeddings_256 (chunk_id, embedding) "
            "VALUES (:chunk_id, :embedding)"
        ),
        {
            "chunk_id": chunk.id,
            "embedding": struct.pack("<256f", *([0.0] * 256)),
        },
    )
    return int(document.id), int(chunk.id)


def test_legacy_file_tags_are_preserved(tmp_path, target_session) -> None:
    legacy_db = tmp_path / "file_tags.db"
    _create_legacy_database(legacy_db)

    report = migrate_legacy_sqlite(
        legacy_db, target_session, backup_dir=tmp_path / "backups"
    )

    assert report.files_migrated == 1
    assert report.tags_migrated == 2
    assert report.associations_migrated == 2
    assert report.suggestions_preserved == 1
    assert report.errors == ()
    assert report.backup_path.is_file()
    assert target_session.scalar(select(func.count(Tag.id))) == 2
    assert target_session.scalar(select(func.count(FileTag.file_id))) == 2
    assert target_session.scalar(select(func.count(TagSuggestionCache.id))) == 1
    target_session.commit()


def test_legacy_minimal_schema_and_duplicate_migration_are_supported(
    tmp_path, target_session
) -> None:
    legacy_db = tmp_path / "file_tags.db"
    with sqlite3.connect(legacy_db) as connection:
        connection.executescript(
            """
            CREATE TABLE files (id INTEGER PRIMARY KEY, path VARCHAR UNIQUE);
            CREATE TABLE tags (id INTEGER PRIMARY KEY, name VARCHAR UNIQUE);
            CREATE TABLE file_tags (file_id INTEGER, tag_id INTEGER);
            INSERT INTO files VALUES (1, 'C:\\library\\one.txt');
            INSERT INTO tags VALUES (1, 'one');
            INSERT INTO file_tags VALUES (1, 1);
            """
        )

    first = migrate_legacy_sqlite(legacy_db, target_session)
    second = migrate_legacy_sqlite(legacy_db, target_session)

    assert first.files_migrated == 1
    assert second.files_migrated == 0
    assert target_session.scalar(select(func.count(File.id))) == 1
    assert target_session.scalar(select(func.count(Tag.id))) == 1


def test_legacy_snapshot_reconciles_stale_rows_and_converges(
    tmp_path, target_session
) -> None:
    legacy_db = tmp_path / "file_tags.db"
    with sqlite3.connect(legacy_db) as connection:
        connection.executescript(
            """
            CREATE TABLE files (id INTEGER PRIMARY KEY, path VARCHAR UNIQUE);
            CREATE TABLE tags (
                id INTEGER PRIMARY KEY,
                name VARCHAR UNIQUE,
                color VARCHAR
            );
            CREATE TABLE file_tags (file_id INTEGER, tag_id INTEGER);
            INSERT INTO files VALUES (1, 'C:\\library\\kept.pdf');
            INSERT INTO files VALUES (2, 'C:\\library\\renamed.pdf');
            INSERT INTO tags VALUES (1, 'research', '#123456');
            INSERT INTO tags VALUES (2, 'reference', '#abcdef');
            INSERT INTO file_tags VALUES (1, 1);
            INSERT INTO file_tags VALUES (2, 2);
            """
        )

    research = Tag(name="research", color="#ffffff")
    reference = Tag(name="reference", color="#abcdef")
    stale_tag = Tag(name="modern-only", color="#999999")
    target_session.add_all([research, reference, stale_tag])
    target_session.flush()
    kept_document_id, kept_chunk_id = _add_indexed_file(
        target_session,
        r"C:\library\kept.pdf",
        stable_id="kept-chunk",
    )
    stale_document_id, stale_chunk_id = _add_indexed_file(
        target_session,
        r"C:\library\old-name.pdf",
        stable_id="stale-chunk",
    )
    kept_file = target_session.scalar(
        select(File).where(File.canonical_path == r"C:\library\kept.pdf")
    )
    stale_file = target_session.scalar(
        select(File).where(File.canonical_path == r"C:\library\old-name.pdf")
    )
    assert kept_file is not None
    assert stale_file is not None
    target_session.add_all(
        [
            FileTag(file_id=kept_file.id, tag_id=research.id),
            FileTag(file_id=kept_file.id, tag_id=reference.id),
            FileTag(file_id=stale_file.id, tag_id=stale_tag.id),
        ]
    )
    target_session.commit()

    first = migrate_legacy_sqlite(
        legacy_db, target_session, backup_dir=tmp_path / "backups"
    )
    target_session.commit()
    second = migrate_legacy_sqlite(
        legacy_db, target_session, backup_dir=tmp_path / "backups"
    )

    assert first.files_migrated == 1
    assert first.tags_migrated == 0
    assert first.associations_migrated == 1
    assert second.files_migrated == 0
    assert second.tags_migrated == 0
    assert second.associations_migrated == 0
    assert set(target_session.execute(select(File.canonical_path)).scalars()) == {
        r"C:\library\kept.pdf",
        r"C:\library\renamed.pdf",
    }
    assert dict(target_session.execute(select(Tag.name, Tag.color)).all()) == {
        "reference": "#abcdef",
        "research": "#123456",
    }
    assert set(
        target_session.execute(
            select(File.canonical_path, Tag.name)
            .join(FileTag, FileTag.file_id == File.id)
            .join(Tag, Tag.id == FileTag.tag_id)
        ).all()
    ) == {
        (r"C:\library\kept.pdf", "research"),
        (r"C:\library\renamed.pdf", "reference"),
    }
    assert target_session.get(Document, kept_document_id) is not None
    assert target_session.get(Chunk, kept_chunk_id) is not None
    assert target_session.get(Document, stale_document_id) is None
    assert target_session.get(Chunk, stale_chunk_id) is None
    assert target_session.scalar(
        text("SELECT count(*) FROM chunk_embeddings_256 WHERE chunk_id = :chunk_id"),
        {"chunk_id": kept_chunk_id},
    ) == 1
    assert target_session.scalar(
        text("SELECT count(*) FROM chunk_embeddings_256 WHERE chunk_id = :chunk_id"),
        {"chunk_id": stale_chunk_id},
    ) == 0
    assert target_session.scalar(
        text("SELECT count(*) FROM chunks_fts WHERE rowid = :chunk_id"),
        {"chunk_id": stale_chunk_id},
    ) == 0


def test_failed_target_write_keeps_backup(
    tmp_path, target_session, monkeypatch
) -> None:
    legacy_db = tmp_path / "file_tags.db"
    _create_legacy_database(legacy_db, suggestions=False)
    backup_dir = tmp_path / "backups"
    stale_document_id, stale_chunk_id = _add_indexed_file(
        target_session,
        r"C:\library\stale.pdf",
        stable_id="rollback-chunk",
    )
    stale_tag = Tag(name="stale", color="#999999")
    stale_file = target_session.scalar(
        select(File).where(File.canonical_path == r"C:\library\stale.pdf")
    )
    assert stale_file is not None
    target_session.add(stale_tag)
    target_session.flush()
    target_session.add(FileTag(file_id=stale_file.id, tag_id=stale_tag.id))
    target_session.commit()
    original_flush = target_session.flush

    def fail_flush(*args, **kwargs):
        monkeypatch.undo()
        raise RuntimeError("forced target failure")

    monkeypatch.setattr(target_session, "flush", fail_flush)

    with pytest.raises(IndexTransactionError, match="backup"):
        migrate_legacy_sqlite(legacy_db, target_session, backup_dir=backup_dir)

    monkeypatch.setattr(target_session, "flush", original_flush)
    backups = list(backup_dir.glob("file_tags-*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("SELECT count(*) FROM files").fetchone() == (1,)
        assert backup.execute("SELECT count(*) FROM tags").fetchone() == (2,)
    target_session.expire_all()
    assert target_session.scalar(
        select(File.canonical_path).where(
            File.canonical_path == r"C:\library\stale.pdf"
        )
    ) == r"C:\library\stale.pdf"
    assert target_session.scalar(
        select(Tag.color).where(Tag.name == "stale")
    ) == "#999999"
    assert target_session.get(Document, stale_document_id) is not None
    assert target_session.get(Chunk, stale_chunk_id) is not None
    assert target_session.scalar(
        select(func.count(FileTag.file_id)).where(
            FileTag.file_id == stale_file.id
        )
    ) == 1
    assert target_session.scalar(
        text("SELECT count(*) FROM chunk_embeddings_256 WHERE chunk_id = :chunk_id"),
        {"chunk_id": stale_chunk_id},
    ) == 1
    assert target_session.scalar(
        text("SELECT count(*) FROM chunks_fts WHERE rowid = :chunk_id"),
        {"chunk_id": stale_chunk_id},
    ) == 1
    assert target_session.scalar(select(func.count(File.id))) == 1
    assert target_session.scalar(select(func.count(Tag.id))) == 1

