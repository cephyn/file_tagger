from pathlib import Path
import sqlite3

import sqlite_vec  # type: ignore[import-untyped]
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ..domain.errors import DatabaseExtensionError

BUSY_TIMEOUT_MS = 5_000


def _create_engine(database_path: Path) -> Engine:
    resolved_path = database_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{resolved_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite_connection(
        dbapi_connection: sqlite3.Connection, connection_record: object
    ) -> None:
        del connection_record
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            cursor.close()
            dbapi_connection.enable_load_extension(True)
            try:
                sqlite_vec.load(dbapi_connection)
            finally:
                dbapi_connection.enable_load_extension(False)
        except sqlite3.Error as exc:
            raise DatabaseExtensionError(
                "The required sqlite-vec extension could not be loaded."
            ) from exc

    return engine


def _verify_search_extensions(engine: Engine) -> None:
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql(
                "CREATE VIRTUAL TABLE temp.file_tagger_fts_probe USING fts5(content)"
            )
            version = connection.scalar(text("SELECT vec_version()"))
    except (DatabaseExtensionError, SQLAlchemyError) as exc:
        engine.dispose()
        raise DatabaseExtensionError(
            "Required SQLite search extension is unavailable: FTS5 or sqlite-vec."
        ) from exc

    if not isinstance(version, str) or not version:
        engine.dispose()
        raise DatabaseExtensionError("The required sqlite-vec extension returned no version.")


def create_session_factory(database_path: Path) -> sessionmaker[Session]:
    engine = _create_engine(database_path)
    _verify_search_extensions(engine)
    return sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
    )
