from pathlib import Path


SPEC_PATH = Path(__file__).resolve().parents[2] / "packaging" / "file-tagger.spec"


def test_packaging_spec_collects_required_packages_and_fails_closed() -> None:
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "find_spec(" in spec
    assert '"PySide6"' in spec
    assert '"sqlite_vec"' in spec
    assert '"docling"' in spec
    assert "HookError" in spec
    assert "collect_all(package)" in spec
    assert "except Exception:\n        pass" not in spec
    assert "PROJECT_ROOT" in spec
    assert "SPECPATH" in spec
    assert 'pathex=[str(SRC_ROOT)]' in spec
    assert 'str(PACKAGING_ROOT / "launcher.py")' in spec
    assert 'datas.append((str(alembic_ini), "."))' in spec
    assert 'datas.append((str(alembic_dir), "alembic"))' in spec
