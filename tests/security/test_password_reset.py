from pathlib import Path

import pytest

from file_tagger.config import Config
from file_tagger.domain.errors import CredentialRecoveryError
from file_tagger.infrastructure.app_paths import AppPaths
from file_tagger.security.password_reset import reset_encrypted_configuration
from file_tagger.security.password_vault import PasswordVault


def _configured_paths(tmp_path: Path, monkeypatch) -> AppPaths:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    return paths


def test_reset_quarantines_encrypted_files_but_preserves_database(
    tmp_path, monkeypatch
) -> None:
    paths = _configured_paths(tmp_path, monkeypatch)
    PasswordVault.create(paths, "forgotten")
    paths.config_file.write_bytes(b"encrypted settings")
    paths.database.write_bytes(b"tag database")
    original_database = paths.database.read_bytes()

    result = reset_encrypted_configuration(paths, "replacement")

    assert result.backup_dir.is_dir()
    assert (result.backup_dir / "config.encrypted").read_bytes() == (
        b"encrypted settings"
    )
    assert paths.database.read_bytes() == original_database
    PasswordVault.open(paths, "replacement")
    assert Config("replacement").get_home_directory() == str(Path.home())
    assert result.recovery_key


def test_reset_restores_files_if_new_credentials_cannot_be_created(
    tmp_path, monkeypatch
) -> None:
    paths = _configured_paths(tmp_path, monkeypatch)
    PasswordVault.create(paths, "forgotten")
    paths.config_file.write_bytes(b"encrypted settings")
    originals = {
        path: path.read_bytes()
        for path in (
            paths.config_file,
            paths.salt_file,
            paths.password_hash_file,
            paths.password_key_file,
            paths.recovery_file,
        )
        if path.is_file()
    }

    monkeypatch.setattr(
        "file_tagger.security.password_reset.PasswordVault.create",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CredentialRecoveryError("simulated failure")
        ),
    )

    with pytest.raises(CredentialRecoveryError):
        reset_encrypted_configuration(paths, "replacement")

    assert all(path.read_bytes() == content for path, content in originals.items())
    PasswordVault.open(paths, "forgotten")


def test_reset_quarantines_pending_cwd_legacy_credentials(
    tmp_path, monkeypatch
) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    monkeypatch.chdir(legacy_root)
    paths = _configured_paths(tmp_path / "data", monkeypatch)
    credential_names = (
        "config.encrypted",
        ".salt",
        ".passhash",
        ".recovery",
    )
    for name in credential_names:
        (legacy_root / name).write_bytes(f"legacy-{name}".encode())
    (legacy_root / "file_tags.db").write_bytes(b"legacy database")

    result = reset_encrypted_configuration(paths, "replacement")

    assert all(
        (result.backup_dir / name).read_bytes() == f"legacy-{name}".encode()
        for name in credential_names
    )
    assert all(not (legacy_root / name).exists() for name in credential_names)
    assert (legacy_root / "file_tags.db").read_bytes() == b"legacy database"
    PasswordVault.open(paths, "replacement")
