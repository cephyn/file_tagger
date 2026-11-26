import os
import subprocess
import sys
from pathlib import Path

import pytest

from file_tagger.infrastructure import database as database_module
from file_tagger.infrastructure.database import upgrade_database


def test_packaged_upgrade_resolves_migrations_from_meipass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "_MEIPASS"
    (bundle / "alembic" / "versions").mkdir(parents=True)
    config_path = bundle / "alembic.ini"
    config_path.write_text("[alembic]\nscript_location = alembic\n", encoding="utf-8")
    database_path = tmp_path / "packaged.sqlite3"
    captured: dict[str, object] = {}

    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(
        database_module.command,
        "upgrade",
        lambda config, revision: captured.update(
            config=config,
            revision=revision,
        ),
    )

    upgrade_database(database_path)

    config = captured["config"]
    assert isinstance(config, database_module.Config)
    assert config.config_file_name == str(config_path)
    assert config.get_main_option("script_location") == str(bundle / "alembic")
    assert config.attributes["database_path"] == database_path.expanduser().resolve()
    assert captured["revision"] == "head"


def test_installed_app_can_create_modern_database(tmp_path: Path) -> None:
    data_dir = tmp_path / "user-data"
    env = os.environ.copy()
    env["FILE_TAGGER_DATA_DIR"] = str(data_dir)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

    result = subprocess.run(
        [sys.executable, "-m", "file_tagger", "--print-modern-database"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "modern_database=ok" in result.stdout
    assert (data_dir / "file_tagger.sqlite3").is_file()


def test_installed_app_uses_per_user_data_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "user-data"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    env = os.environ.copy()
    env["FILE_TAGGER_DATA_DIR"] = str(data_dir)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

    result = subprocess.run(
        [sys.executable, "-m", "file_tagger", "--print-paths"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "file_tagger.sqlite3" in result.stdout
    assert str(data_dir.resolve()) in result.stdout
    modern_line = next(
        line for line in result.stdout.splitlines() if line.startswith("modern_database=")
    )
    assert cwd.resolve().as_posix() not in modern_line
    assert "legacy_database=" in result.stdout
    assert "legacy_chroma=" in result.stdout
    assert "config_file=" in result.stdout
    assert "logs_dir=" in result.stdout
    assert not (cwd / "file_tags.db").exists()
    assert not (cwd / "file_tagger.sqlite3").exists()
    assert not (cwd / ".chroma").exists()
    assert not (cwd / "config.encrypted").exists()
    assert not (cwd / ".salt").exists()
    assert not (cwd / ".passhash").exists()
    assert not (cwd / ".recovery").exists()
    assert not list(cwd.glob("file_tagger_*.log"))
