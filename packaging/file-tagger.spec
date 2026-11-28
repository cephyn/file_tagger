# -*- mode: python ; coding: utf-8 -*-
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.exceptions import HookError
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

try:
    SPEC_DIR = Path(__file__).resolve().parent
except NameError:
    spec_path = Path(SPECPATH).resolve()
    SPEC_DIR = spec_path if spec_path.is_dir() else spec_path.parent
PROJECT_ROOT = SPEC_DIR.parent
PACKAGING_ROOT = SPEC_DIR
SRC_ROOT = PROJECT_ROOT / "src"

hiddenimports = []
hiddenimports += [
    "logging.config",
    "logging.handlers",
]
datas = []
binaries = []

required_packages = (
    "PySide6",
    "sqlite_vec",
    "sqlalchemy",
    "alembic",
    "docling",
    "docling_parse",
    "pypdfium2",
    "latex2mathml",
)

optional_packages = (
    "docling_core",
    "docling_ibm_models",
)

for package in required_packages:
    if find_spec(package) is None:
        raise HookError(
            f"Required package '{package}' is not installed; complete packaging dependencies "
            "before running PyInstaller."
        )
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
    except Exception as exc:
        raise HookError(
            f"PyInstaller collection failed for required package '{package}': {exc}"
        ) from exc
    if not (
        package_datas or package_binaries or package_hiddenimports
    ):
        raise HookError(
            f"PyInstaller collection was incomplete for required package '{package}': "
            "no datas, binaries, or hidden imports were resolved."
        )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

for package in optional_packages:
    try:
        if find_spec(package) is None:
            continue
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hiddenimports
    except Exception as exc:
        raise HookError(
            f"PyInstaller collection failed for optional package '{package}': {exc}"
        ) from exc

hiddenimports += collect_submodules("file_tagger")
hiddenimports += collect_submodules("sqlite_vec")

try:
    package_datas, package_binaries, package_hiddenimports = collect_all("file_tagger")
except Exception as exc:
    raise HookError(
        f"PyInstaller collection failed for required package 'file_tagger': {exc}"
    ) from exc
datas += package_datas
binaries += package_binaries
hiddenimports += package_hiddenimports

alembic_ini = PROJECT_ROOT / "alembic.ini"
alembic_dir = PROJECT_ROOT / "alembic"
if alembic_ini.is_file():
    datas.append((str(alembic_ini), "."))
if alembic_dir.is_dir():
    datas.append((str(alembic_dir), "alembic"))

a = Analysis(
    [str(PACKAGING_ROOT / "launcher.py")],
    pathex=[str(SRC_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch.distributed", "tensorflow"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="file-tagger",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    a.zipped_data,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="file-tagger",
)
