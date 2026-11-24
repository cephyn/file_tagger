from pathlib import Path

from file_tagger.infrastructure.app_paths import AppPaths


def test_paths_are_built_under_platformdirs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    assert paths.database == tmp_path / "file_tagger.sqlite3"
    assert paths.legacy_database != paths.database


def test_portable_override_contains_runtime_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))

    paths = AppPaths.discover()

    for path in (
        paths.database,
        paths.config_file,
        paths.logs_dir,
        paths.cache_dir,
        paths.temp_dir,
    ):
        assert path.is_relative_to(tmp_path)
    assert paths.legacy_database == Path.cwd().resolve() / "file_tags.db"
    assert paths.legacy_chroma == Path.cwd().resolve() / ".chroma"


def test_config_state_uses_discovered_config_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    from file_tagger.config import store_password_hash

    store_password_hash("password")

    assert (tmp_path / "config" / ".passhash").is_file()


def test_main_log_path_uses_discovered_logs_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    from file_tagger.main import setup_logging

    log_file = setup_logging()

    assert log_file.parent == tmp_path / "logs"
