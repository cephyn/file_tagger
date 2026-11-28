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
    monkeypatch.chdir(tmp_path)
    paths = _configured_paths(tmp_path, monkeypatch)
    PasswordVault.create(paths, "forgotten")
    paths.config_file.write_bytes(b"encrypted settings")
    paths.database.write_bytes(b"tag database")
    cache_data = paths.cache_dir / "search-data.bin"
    cache_data.write_bytes(b"search cache")
    original_database = paths.database.read_bytes()

    result = reset_encrypted_configuration(paths, "replacement")

    assert result.backup_dir.is_dir()
    assert (result.backup_dir / "config.encrypted").read_bytes() == (
        b"encrypted settings"
    )
    assert paths.database.read_bytes() == original_database
    assert cache_data.read_bytes() == b"search cache"
    PasswordVault.open(paths, "replacement")
    assert Config("replacement").get_home_directory() == str(Path.home())
    assert result.recovery_key


def test_reset_restores_files_if_new_credentials_cannot_be_created(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
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


def test_reset_aborts_when_quarantine_backup_fails(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = _configured_paths(tmp_path, monkeypatch)
    PasswordVault.create(paths, "forgotten")
    paths.config_file.write_bytes(b"encrypted settings")
    originals = {
        path: path.read_bytes()
        for path in (
            paths.config_file,
            paths.password_hash_file,
            paths.password_key_file,
            paths.recovery_file,
        )
    }
    monkeypatch.setattr(
        "file_tagger.security.password_reset.shutil.copy2",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("backup unavailable")
        ),
    )

    with pytest.raises(CredentialRecoveryError):
        reset_encrypted_configuration(paths, "replacement")

    assert all(path.read_bytes() == content for path, content in originals.items())
    PasswordVault.open(paths, "forgotten")


def test_reset_failure_preserves_target_credentials_not_selected_for_quarantine(
    tmp_path, monkeypatch
) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    monkeypatch.chdir(legacy_root)
    paths = _configured_paths(tmp_path / "data", monkeypatch)
    PasswordVault.create(paths, "target-password")
    target_files = {
        path: path.read_bytes()
        for path in (
            paths.password_hash_file,
            paths.password_key_file,
            paths.recovery_file,
        )
    }
    (legacy_root / "config.encrypted").write_bytes(b"legacy encrypted config")
    (legacy_root / ".salt").write_bytes(b"legacy salt")
    monkeypatch.setattr(
        "file_tagger.security.password_reset.PasswordVault.create",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CredentialRecoveryError("simulated failure")
        ),
    )

    with pytest.raises(CredentialRecoveryError):
        reset_encrypted_configuration(paths, "replacement")

    assert all(path.read_bytes() == content for path, content in target_files.items())
    assert (legacy_root / "config.encrypted").read_bytes() == b"legacy encrypted config"
    assert (legacy_root / ".salt").read_bytes() == b"legacy salt"
    PasswordVault.open(paths, "target-password")


def test_reset_clears_pending_legacy_files_and_stale_target_salt(
    tmp_path, monkeypatch
) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    monkeypatch.chdir(legacy_root)
    paths = _configured_paths(tmp_path / "data", monkeypatch)
    legacy_files = {
        "config.encrypted": b"legacy config",
        ".salt": b"legacy salt",
        ".passhash": b"legacy passhash",
        ".recovery": b"legacy recovery",
    }
    for name, content in legacy_files.items():
        (legacy_root / name).write_bytes(content)
    paths.salt_file.write_bytes(b"stale target salt")

    result = reset_encrypted_configuration(paths, "replacement")

    assert all(not (legacy_root / name).exists() for name in legacy_files)
    assert not paths.salt_file.exists()
    assert all(
        (result.backup_dir / name).read_bytes() == content
        for name, content in legacy_files.items()
    )
    PasswordVault.open(paths, "replacement")
