from logging.config import fileConfig
from pathlib import Path

from alembic import context

from file_tagger.infrastructure.app_paths import AppPaths
from file_tagger.infrastructure.database import create_database_engine
from file_tagger.persistence.models import Base


config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations() -> None:
    database_path = config.attributes.get("database_path")
    if database_path is None:
        database_path = AppPaths.discover().database
    engine = create_database_engine(Path(database_path))

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()

    engine.dispose()


run_migrations()
