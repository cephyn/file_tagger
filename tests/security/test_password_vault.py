import pytest

from file_tagger.domain.errors import (
    CredentialFormatError,
    CredentialRecoveryError,
)
from file_tagger.infrastructure.app_paths import AppPaths
from file_tagger.security.password_vault import CredentialStatus, PasswordVault


def test_password_vault_discovers_first_run_and_envelope_paths(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()

    assert paths.password_key_file == tmp_path / "config" / ".password-key"
    assert PasswordVault.status(paths) is CredentialStatus.UNINITIALIZED


def test_password_vault_identifies_legacy_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    paths.salt_file.write_bytes(b"salt")
    paths.password_hash_file.write_text("{}")

    assert PasswordVault.status(paths) is CredentialStatus.LEGACY


def test_password_vault_encrypts_config_and_changes_password(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()

    vault, recovery_key = PasswordVault.create(paths, "old-password")
    plaintext = b'{"home_directory":"C:\\\\Users\\\\owner"}'
    ciphertext = vault.encrypt(plaintext)
    paths.config_file.write_bytes(ciphertext)

    assert recovery_key
    assert vault.decrypt(ciphertext) == plaintext
    vault.change_password("old-password", "new-password")
    assert PasswordVault.open(paths, "new-password").decrypt(
        paths.config_file.read_bytes()
    ) == plaintext
    assert paths.config_file.read_bytes() == ciphertext

    with pytest.raises(CredentialRecoveryError):
        PasswordVault.open(paths, "old-password")


def test_password_vault_rejects_corrupt_envelope(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    PasswordVault.create(paths, "old-password")
    paths.password_key_file.write_text("{invalid", encoding="utf-8")

    assert PasswordVault.status(paths) is CredentialStatus.CORRUPT
    with pytest.raises(CredentialFormatError):
        PasswordVault.open(paths, "old-password")
