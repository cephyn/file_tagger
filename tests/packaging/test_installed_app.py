import os
import subprocess
import sys
from pathlib import Path


def test_installed_app_uses_per_user_data_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "user-data"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    env = os.environ.copy()
    env["FILE_TAGGER_DATA_DIR"] = str(data_dir)

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
    assert cwd.resolve().as_posix() not in result.stdout
    assert not (cwd / "file_tags.db").exists()
    assert not (cwd / "file_tagger.sqlite3").exists()
