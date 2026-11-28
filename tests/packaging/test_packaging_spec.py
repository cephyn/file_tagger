import subprocess
import sys
import tarfile
from pathlib import Path


SPEC_PATH = Path(__file__).resolve().parents[2] / "packaging" / "file-tagger.spec"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_packaging_spec_collects_required_packages_and_fails_closed() -> None:
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "find_spec(" in spec
    assert '"PySide6"' in spec
    assert '"sqlite_vec"' in spec
    assert '"docling"' in spec
    assert '"docling_parse"' in spec
    assert '"pypdfium2"' in spec
    assert '"latex2mathml"' in spec
    assert '    "alembic",\n' in spec
    assert '    "sqlalchemy",\n' in spec
    assert '"logging.config"' in spec
    assert "HookError" in spec
    assert "collect_all(package)" in spec
    assert "except Exception:\n        pass" not in spec
    assert "PROJECT_ROOT" in spec
    assert "SPECPATH" in spec
    assert 'pathex=[str(SRC_ROOT)]' in spec
    assert 'str(PACKAGING_ROOT / "launcher.py")' in spec
    assert 'datas.append((str(alembic_ini), "."))' in spec
    assert 'datas.append((str(alembic_dir), "alembic"))' in spec


def test_sdist_contains_migrations_and_python_package(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--outdir",
            str(dist_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    archives = list(dist_dir.glob("*.tar.gz"))
    assert len(archives) == 1

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with tarfile.open(archives[0], "r:gz") as archive:
        archive.extractall(extract_dir, filter="data")

    roots = list(extract_dir.iterdir())
    assert len(roots) == 1
    source_root = roots[0]
    assert (source_root / "src" / "file_tagger").is_dir()
    assert (source_root / "src" / "file_tagger" / "__init__.py").is_file()
    assert (source_root / "alembic.ini").is_file()
    assert (source_root / "alembic" / "env.py").is_file()
    assert (source_root / "alembic" / "script.py.mako").is_file()
    assert (source_root / "alembic" / "versions").is_dir()
    assert list((source_root / "alembic" / "versions").glob("*.py"))
