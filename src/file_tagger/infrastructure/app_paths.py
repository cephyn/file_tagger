from dataclasses import dataclass
import os
from pathlib import Path

from platformdirs import user_cache_path, user_config_path, user_data_path


@dataclass(frozen=True, slots=True)
class AppPaths:
    data_dir: Path
    config_dir: Path
    logs_dir: Path
    cache_dir: Path
    temp_dir: Path
    database: Path
    config_file: Path
    salt_file: Path
    password_hash_file: Path
    password_key_file: Path
    recovery_file: Path
    legacy_database: Path
    legacy_chroma: Path

    @classmethod
    def discover(cls) -> "AppPaths":
        override = os.environ.get("FILE_TAGGER_DATA_DIR")
        if override:
            data_dir = Path(override).expanduser().resolve()
            config_dir = data_dir / "config"
            cache_dir = data_dir / "cache"
        else:
            data_dir = user_data_path("file_tagger", appauthor=False).resolve()
            config_dir = user_config_path("file_tagger", appauthor=False).resolve()
            cache_dir = user_cache_path("file_tagger", appauthor=False).resolve()

        legacy_root = Path.cwd().resolve()
        return cls(
            data_dir=data_dir,
            config_dir=config_dir,
            logs_dir=data_dir / "logs",
            cache_dir=cache_dir,
            temp_dir=cache_dir / "extraction",
            database=data_dir / "file_tagger.sqlite3",
            config_file=config_dir / "config.encrypted",
            salt_file=config_dir / ".salt",
            password_hash_file=config_dir / ".passhash",
            password_key_file=config_dir / ".password-key",
            recovery_file=config_dir / ".recovery",
            legacy_database=legacy_root / "file_tags.db",
            legacy_chroma=legacy_root / ".chroma",
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.config_dir,
            self.logs_dir,
            self.cache_dir,
            self.temp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def data(self) -> Path:
        return self.data_dir

    @property
    def config(self) -> Path:
        return self.config_dir

    @property
    def logs(self) -> Path:
        return self.logs_dir

    @property
    def cache(self) -> Path:
        return self.cache_dir

    @property
    def temp(self) -> Path:
        return self.temp_dir
