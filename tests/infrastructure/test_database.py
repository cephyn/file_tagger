import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from file_tagger.domain.errors import DatabaseExtensionError
from file_tagger.infrastructure import database as database_module
from file_tagger.infrastructure.database import create_session_factory, upgrade_database
from file_tagger.persistence.models import Chunk, Document, File, Page


def test_upgrade_database_resolves_migrations_from_pyinstaller_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "_MEIPASS"
    (bundle / "alembic" / "versions").mkdir(parents=True)
    config_path = bundle / "alembic.ini"
    config_path.write_text("[alembic]\nscript_location = alembic\n", encoding="utf-8")
    database_path = tmp_path / "frozen.sqlite3"
    captured: dict[str, object] = {}

    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)

    def capture_upgrade(config, revision: str) -> None:
        captured["config"] = config
        captured["revision"] = revision

    monkeypatch.setattr(database_module.command, "upgrade", capture_upgrade)

    upgrade_database(database_path)

    config = captured["config"]
    assert isinstance(config, database_module.Config)
    assert config.config_file_name == str(config_path)
    assert config.get_main_option("script_location") == str(bundle / "alembic")
    assert config.attributes["database_path"] == database_path.expanduser().resolve()
    assert captured["revision"] == "head"


def _add_indexed_file(session) -> int:
    file = File(canonical_path=r"C:\library\guide.md", indexing_state="indexed")
    session.add(file)
    session.flush()
    document = Document(
        file_id=file.id,
        revision=1,
        file_hash="source-hash",
        extraction_version="test",
        parser="test",
    )
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, ordinal=0, text="source")
    session.add(page)
    session.flush()
    chunk = Chunk(
        stable_id="stable-chunk",
        document_id=document.id,
        page_id=page.id,
        ordinal=0,
        text="source",
        content_hash="content-hash",
        model_digest="embedding-model",
        index_version="index-version",
    )
    session.add(chunk)
    session.flush()
    return int(chunk.id)


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


def test_database_registers_deterministic_unicode_normalization_on_connections(
    tmp_path,
) -> None:
    session_factory = create_session_factory(tmp_path / "app.sqlite3")

    with session_factory() as first_session:
        assert (
            first_session.scalar(text("SELECT unicode_normalize('Straße')"))
            == "strasse"
        )
        with session_factory() as second_session:
            assert (
                second_session.scalar(text("SELECT unicode_normalize('STRASSE')"))
                == "strasse"
            )
            assert second_session.scalar(text("SELECT vec_version() IS NOT NULL")) == 1

    with session_factory() as session:
        function_flags = session.scalar(
            text(
                "SELECT flags FROM pragma_function_list "
                "WHERE name = 'unicode_normalize'"
            )
        )
        assert isinstance(function_flags, int)
        assert function_flags & 0x800


def test_database_initialization_purges_orphaned_vectors_before_id_reuse(
    tmp_path,
) -> None:
    database_path = tmp_path / "orphan-cleanup.sqlite3"
    upgrade_database(database_path)
    original_factory = create_session_factory(database_path)

    with original_factory.begin() as session:
        original_chunk_id = _add_indexed_file(session)
        for dimensions in (256, 512, 768):
            session.execute(
                text(
                    f"INSERT INTO chunk_embeddings_{dimensions} "
                    "(chunk_id, embedding) VALUES (:chunk_id, :embedding)"
                ),
                {
                    "chunk_id": original_chunk_id,
                    "embedding": bytes(dimensions * 4),
                },
            )

    with original_factory.begin() as session:
        session.delete(session.get_one(File, 1))

    initialized_factory = create_session_factory(database_path)

    with initialized_factory() as session:
        for dimensions in (256, 512, 768):
            assert (
                session.scalar(
                    text(f"SELECT count(*) FROM chunk_embeddings_{dimensions}")
                )
                == 0
            )

    with initialized_factory.begin() as session:
        replacement_chunk_id = _add_indexed_file(session)
        session.execute(
            text(
                "INSERT INTO chunk_embeddings_256 "
                "(chunk_id, embedding) VALUES (:chunk_id, :embedding)"
            ),
            {
                "chunk_id": replacement_chunk_id,
                "embedding": bytes(256 * 4),
            },
        )

    assert replacement_chunk_id == original_chunk_id


def test_database_reports_missing_vector_extension(tmp_path, monkeypatch) -> None:
    from file_tagger.infrastructure import database

    def fail_load(connection) -> None:
        raise sqlite3.OperationalError("extension unavailable")

    monkeypatch.setattr(database.sqlite_vec, "load", fail_load)

    with pytest.raises(DatabaseExtensionError, match="sqlite-vec"):
        database.create_session_factory(tmp_path / "missing.sqlite3")
