from pathlib import Path
import tomllib


def test_project_targets_python_313_and_has_one_dependency_manifest() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert project["requires-python"].startswith(">=3.13")
    assert not Path("requirements.txt").exists()
