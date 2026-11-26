import base64
import hashlib
import json

import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from file_tagger.config import Config
from file_tagger.domain.errors import (
    CredentialFormatError,
    CredentialRecoveryError,
)
from file_tagger.infrastructure.app_paths import AppPaths
from file_tagger.security.password_vault import CredentialStatus, PasswordVault


def _legacy_fernet(password: str, salt: bytes) -> Fernet:
    material = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    ).derive(password.encode())
    return Fernet(base64.urlsafe_b64encode(material))


def _write_legacy_credentials(paths, password: str, config: dict) -> None:
    encryption_salt = b"legacy-encryption-salt"
    password_salt = b"legacy-password-salt"
    legacy_fernet = _legacy_fernet(password, encryption_salt)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), password_salt, 100_000
    )
    paths.salt_file.write_bytes(encryption_salt)
    paths.password_hash_file.write_text(
        json.dumps(
            {
                "hash": base64.b64encode(password_hash).decode(),
                "salt": base64.b64encode(password_salt).decode(),
            }
        ),
        encoding="utf-8",
    )
    paths.recovery_file.write_bytes(legacy_fernet.encrypt(b"old-recovery-key"))
    paths.config_file.write_bytes(
        legacy_fernet.encrypt(json.dumps(config).encode())
    )


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


def test_recovery_key_resets_password_without_reencrypting_config(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    vault, recovery_key = PasswordVault.create(paths, "forgotten")
    ciphertext = vault.encrypt(b"config")
    paths.config_file.write_bytes(ciphertext)
    original_recovery = paths.recovery_file.read_bytes()

    recovered = PasswordVault.recover(paths, recovery_key, "replacement")

    assert recovered.decrypt(paths.config_file.read_bytes()) == b"config"
    assert paths.config_file.read_bytes() == ciphertext
    assert paths.recovery_file.read_bytes() == original_recovery
    with pytest.raises(CredentialRecoveryError):
        PasswordVault.open(paths, "forgotten")


def test_rotating_recovery_key_invalidates_only_old_recovery_key(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    vault, old_key = PasswordVault.create(paths, "current-password")

    new_key = vault.rotate_recovery_key("current-password")

    assert new_key != old_key
    with pytest.raises(CredentialRecoveryError):
        PasswordVault.recover(paths, old_key, "invalid")
    PasswordVault.recover(paths, new_key, "replacement")


def test_legacy_config_migrates_after_authenticated_login(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    _write_legacy_credentials(
        paths,
        "legacy-password",
        {
            "home_directory": r"C:\Users\owner",
            "system_message": "Keep this setting",
        },
    )

    config = Config("legacy-password")

    assert PasswordVault.status(paths) is CredentialStatus.ENVELOPE
    assert config.get_home_directory() == r"C:\Users\owner"
    assert config.pending_recovery_key
    assert Config("legacy-password").get_system_message() == "Keep this setting"
    with pytest.raises(CredentialRecoveryError):
        PasswordVault.open(paths, "old-wrong-password")


def test_legacy_migration_preserves_files_when_password_is_wrong(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    _write_legacy_credentials(paths, "legacy-password", {"setting": "value"})
    originals = {
        path: path.read_bytes()
        for path in (
            paths.salt_file,
            paths.password_hash_file,
            paths.recovery_file,
            paths.config_file,
        )
    }

    with pytest.raises(CredentialRecoveryError):
        Config("wrong-password")

    assert all(path.read_bytes() == content for path, content in originals.items())
