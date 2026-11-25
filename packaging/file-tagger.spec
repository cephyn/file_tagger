# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

hiddenimports = []
datas = []
binaries = []

for package in (
    "PySide6",
    "sqlite_vec",
    "docling",
    "docling_core",
    "docling_ibm_models",
    "pypdfium2",
):
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hiddenimports
    except Exception:
        # Individual optional resource packages may not exist in every supported environment.
        pass

hiddenimports += collect_submodules("file_tagger")
hiddenimports += collect_submodules("sqlite_vec")

a = Analysis(
    ["launcher.py"],
    pathex=["../src"],
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
