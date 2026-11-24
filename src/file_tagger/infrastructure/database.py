from pathlib import Path
import sqlite3
import unicodedata

from alembic import command
from alembic.config import Config
import sqlite_vec  # type: ignore[import-untyped]
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ..domain.errors import DatabaseExtensionError

BUSY_TIMEOUT_MS = 5_000


def normalize_unicode_key(value: str | None) -> str | None:
    if value is None:
        return None
    return unicodedata.normalize("NFKC", value).casefold()


def configure_sqlite_engine(engine: Engine) -> Engine:
    event.listen(engine, "connect", configure_sqlite_connection)
    return engine


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
        dbapi_connection.create_function(
            "unicode_normalize",
            1,
            normalize_unicode_key,
            deterministic=True,
        )
    except sqlite3.Error as exc:
        raise DatabaseExtensionError(
            "The required sqlite-vec extension could not be loaded."
        ) from exc


def create_database_engine(database_path: Path) -> Engine:
    resolved_path = database_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{resolved_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    return configure_sqlite_engine(engine)


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
    engine = create_database_engine(database_path)
    _verify_search_extensions(engine)
    return sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
    )


def upgrade_database(database_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[3] / "alembic.ini"
    if not config_path.is_file():
        raise FileNotFoundError(f"Alembic configuration not found: {config_path}")

    alembic_config = Config(str(config_path))
    alembic_config.attributes["database_path"] = database_path.expanduser().resolve()
    command.upgrade(alembic_config, "head")
