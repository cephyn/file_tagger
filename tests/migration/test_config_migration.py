import base64
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from file_tagger.infrastructure.app_paths import AppPaths
from file_tagger.domain.errors import PasswordEncryptionError
from file_tagger.migration.config_migration import migrate_legacy_config
from file_tagger.security.password_vault import CredentialStatus, PasswordVault

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def _legacy_key(password: str, salt: bytes) -> Fernet:
    key = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    ).derive(password.encode())
    return Fernet(base64.urlsafe_b64encode(key))


def test_config_migration_removes_cloud_fields_without_exposing_values(
    tmp_path, monkeypatch
) -> None:
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    password = "migration-password"
    salt = b"legacy-salt-1234"
    fernet = _legacy_key(password, salt)
    (legacy_dir / ".salt").write_bytes(salt)
    password_salt = b"password-salt"
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), password_salt, 100_000
    )
    (legacy_dir / ".passhash").write_text(
        json.dumps(
            {
                "hash": base64.b64encode(password_hash).decode(),
                "salt": base64.b64encode(password_salt).decode(),
            }
        )
    )
    (legacy_dir / ".recovery").write_bytes(fernet.encrypt(b"recovery"))
    (legacy_dir / "config.encrypted").write_bytes(
        fernet.encrypt(
            json.dumps(
                {
                    "api_keys": {
                        "openai": "must-not-migrate",
                        "gemini": "must-not-migrate",
                    },
                    "selected_provider": "openai",
                    "home_directory": r"C:\Users\owner",
                    "local_models": {"model_path": "old.gguf", "model_type": "llama"},
                    "system_message": "Keep this setting",
                }
            ).encode()
        )
    )

    target_root = tmp_path / "target"
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(target_root))
    monkeypatch.chdir(legacy_dir)
    from file_tagger.config import Config

    active_config = Config(password)
    paths = AppPaths.discover()
    report = migrate_legacy_config(
        legacy_dir, paths, password, backup_dir=tmp_path / "backups"
    )

    assert report.cloud_fields_removed == 3
    assert report.errors == ()
    decrypted = json.loads(fernet.decrypt(paths.config_file.read_bytes()))
    assert decrypted == {
        "home_directory": r"C:\Users\owner",
        "system_message": "Keep this setting",
    }
    assert paths.salt_file.read_bytes() == salt
    assert paths.password_hash_file.is_file()
    assert paths.recovery_file.is_file()
    assert PasswordVault.status(paths) is CredentialStatus.LEGACY
    assert "must-not-migrate" not in paths.config_file.read_bytes().decode(
        errors="ignore"
    )

    active_config.reload_after_migration()

    assert PasswordVault.status(paths) is CredentialStatus.ENVELOPE
    assert active_config.pending_recovery_key
    assert active_config.get_home_directory() == r"C:\Users\owner"
    assert active_config.get_system_message() == "Keep this setting"


def test_missing_legacy_config_is_a_no_op(tmp_path, monkeypatch) -> None:
    legacy_dir = tmp_path / "missing"
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path / "target"))

    report = migrate_legacy_config(
        legacy_dir, AppPaths.discover(), "password", backup_dir=tmp_path / "backups"
    )

    assert report.skipped is True
    assert report.cloud_fields_removed == 0


def test_config_migration_rolls_back_partial_target_writes(
    tmp_path, monkeypatch
) -> None:
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    password = "migration-password"
    salt = b"legacy-salt-1234"
    fernet = _legacy_key(password, salt)
    (legacy_dir / ".salt").write_bytes(salt)
    password_salt = b"password-salt"
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), password_salt, 100_000
    )
    (legacy_dir / ".passhash").write_text(
        json.dumps(
            {
                "hash": base64.b64encode(password_hash).decode(),
                "salt": base64.b64encode(password_salt).decode(),
            }
        )
    )
    (legacy_dir / "config.encrypted").write_bytes(
        fernet.encrypt(b'{"home_directory":"C:\\\\Users\\\\owner"}')
    )
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path / "target"))
    paths = AppPaths.discover()
    original_replace = Path.replace
    replacement_count = 0

    def fail_second_replacement(source, target):
        nonlocal replacement_count
        if source.name.endswith(".migration"):
            replacement_count += 1
            if replacement_count == 2:
                raise OSError("simulated second replacement failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_replacement)

    with pytest.raises(PasswordEncryptionError):
        migrate_legacy_config(
            legacy_dir,
            paths,
            password,
            backup_dir=tmp_path / "backups",
        )

    assert not paths.config_file.exists()
    assert not paths.salt_file.exists()
    assert not paths.password_hash_file.exists()
    assert not paths.recovery_file.exists()


def test_config_migration_aborts_when_backup_verification_fails(
    tmp_path, monkeypatch
) -> None:
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    password = "migration-password"
    salt = b"legacy-salt-1234"
    fernet = _legacy_key(password, salt)
    (legacy_dir / ".salt").write_bytes(salt)
    password_salt = b"password-salt"
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), password_salt, 100_000
    )
    (legacy_dir / ".passhash").write_text(
        json.dumps(
            {
                "hash": base64.b64encode(password_hash).decode(),
                "salt": base64.b64encode(password_salt).decode(),
            }
        )
    )
    (legacy_dir / "config.encrypted").write_bytes(
        fernet.encrypt(b'{"setting":"value"}')
    )
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path / "target"))
    paths = AppPaths.discover()
    original_copy = shutil.copy2

    def corrupt_backup(source, destination):
        result = original_copy(source, destination)
        Path(destination).write_bytes(b"corrupt backup")
        return result

    monkeypatch.setattr(
        "file_tagger.migration.config_migration.shutil.copy2",
        corrupt_backup,
    )

    with pytest.raises(PasswordEncryptionError):
        migrate_legacy_config(
            legacy_dir,
            paths,
            password,
            backup_dir=tmp_path / "backups",
        )

    assert not paths.config_file.exists()
