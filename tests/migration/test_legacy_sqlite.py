import json
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import func, select

from file_tagger.domain.errors import IndexTransactionError
from file_tagger.infrastructure.database import create_session_factory, upgrade_database
from file_tagger.migration.sqlite_migration import migrate_legacy_sqlite
from file_tagger.persistence.models import File, FileTag, Tag, TagSuggestionCache


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


def test_failed_target_write_keeps_backup(
    tmp_path, target_session, monkeypatch
) -> None:
    legacy_db = tmp_path / "file_tags.db"
    _create_legacy_database(legacy_db, suggestions=False)
    backup_dir = tmp_path / "backups"
    original_flush = target_session.flush

    def fail_flush(*args, **kwargs):
        monkeypatch.undo()
        raise RuntimeError("forced target failure")

    monkeypatch.setattr(target_session, "flush", fail_flush)

    with pytest.raises(IndexTransactionError, match="backup"):
        migrate_legacy_sqlite(legacy_db, target_session, backup_dir=backup_dir)

    monkeypatch.setattr(target_session, "flush", original_flush)
    assert list(backup_dir.glob("file_tags-*.sqlite3"))
    assert target_session.scalar(select(func.count(File.id))) == 0

