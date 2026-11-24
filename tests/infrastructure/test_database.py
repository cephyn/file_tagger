import sqlite3

import pytest
from sqlalchemy import text

from file_tagger.domain.errors import DatabaseExtensionError
from file_tagger.infrastructure.database import create_session_factory


def test_database_has_fts5_and_vector_extensions(tmp_path) -> None:
    session_factory = create_session_factory(tmp_path / "app.sqlite3")

    with session_factory() as session:
        session.execute(text("CREATE VIRTUAL TABLE search_probe USING fts5(content)"))
        session.execute(
            text("INSERT INTO search_probe(content) VALUES (:content)"),
            {"content": "ready"},
        )
        assert session.scalar(
            text("SELECT count(*) FROM search_probe WHERE search_probe MATCH 'ready'")
        ) == 1
        assert session.scalar(text("SELECT vec_version() IS NOT NULL")) == 1


def test_database_configures_every_connection(tmp_path) -> None:
    session_factory = create_session_factory(tmp_path / "app.sqlite3")

    with session_factory() as session:
        assert session.scalar(text("PRAGMA foreign_keys")) == 1
        assert session.scalar(text("PRAGMA busy_timeout")) == 5_000
        assert session.scalar(text("PRAGMA journal_mode")) == "wal"


def test_database_reports_missing_vector_extension(tmp_path, monkeypatch) -> None:
    from file_tagger.infrastructure import database

    def fail_load(connection) -> None:
        raise sqlite3.OperationalError("extension unavailable")

    monkeypatch.setattr(database.sqlite_vec, "load", fail_load)

    with pytest.raises(DatabaseExtensionError, match="sqlite-vec"):
        database.create_session_factory(tmp_path / "missing.sqlite3")
