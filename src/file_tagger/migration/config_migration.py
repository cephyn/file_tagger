import base64
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
import shutil

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..domain.errors import PasswordEncryptionError
from ..infrastructure.app_paths import AppPaths
from ..security.file_transactions import replace_files_transactionally


CLOUD_FIELDS = {"api_keys", "selected_provider", "local_models"}


@dataclass(frozen=True, slots=True)
class ConfigMigrationReport:
    cloud_fields_removed: int
    backup_dir: Path | None
    skipped: bool = False
    errors: tuple[str, ...] = ()


def _derive_fernet(password: str, salt: bytes) -> Fernet:
    key = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    ).derive(password.encode())
    return Fernet(base64.urlsafe_b64encode(key))


def _verify_password(password: str, password_hash_file: Path) -> None:
    try:
        stored = json.loads(password_hash_file.read_text())
        salt = base64.b64decode(stored["salt"])
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt, 100_000
        )
        expected = base64.b64decode(stored["hash"])
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise PasswordEncryptionError(
            "Legacy password metadata is invalid."
        ) from exc
    if not hmac.compare_digest(actual, expected):
        raise PasswordEncryptionError("Legacy password verification failed.")


def _backup_legacy_config(legacy_root: Path, backup_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    destination = backup_dir / f"config-{timestamp}"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.encrypted",
        ".salt",
        ".passhash",
        ".recovery",
    ):
        source = legacy_root / name
        if source.is_file():
            backup = destination / name
            content = source.read_bytes()
            shutil.copy2(source, backup)
            if backup.read_bytes() != content:
                raise OSError("Legacy configuration backup verification failed.")
    return destination


def migrate_legacy_config(
    legacy_root: Path,
    target_paths: AppPaths,
    password: str,
    *,
    backup_dir: Path | None = None,
) -> ConfigMigrationReport:
    legacy_root = legacy_root.expanduser().resolve()
    legacy_config = legacy_root / "config.encrypted"
    if not legacy_config.is_file():
        return ConfigMigrationReport(
            cloud_fields_removed=0,
            backup_dir=None,
            skipped=True,
        )

    if target_paths.config_file.is_file():
        return ConfigMigrationReport(
            cloud_fields_removed=0,
            backup_dir=None,
            skipped=True,
            errors=("Target configuration already exists; migration was skipped.",),
        )

    required = (legacy_root / ".salt", legacy_root / ".passhash")
    if any(not path.is_file() for path in required):
        raise PasswordEncryptionError("Legacy encryption metadata is incomplete.")

    salt = required[0].read_bytes()
    _verify_password(password, required[1])
    fernet = _derive_fernet(password, salt)
    try:
        config = json.loads(fernet.decrypt(legacy_config.read_bytes()))
    except (InvalidToken, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PasswordEncryptionError(
            "Legacy configuration could not be decrypted safely."
        ) from exc

    sanitized = {key: value for key, value in config.items() if key not in CLOUD_FIELDS}
    removed = len(config.keys() & CLOUD_FIELDS)
    backups = backup_dir or legacy_root / "migration-backups"
    try:
        backup = _backup_legacy_config(legacy_root, backups)
    except OSError as exc:
        raise PasswordEncryptionError(
            "Legacy configuration backup could not be completed safely."
        ) from exc

    try:
        target_paths.ensure_directories()
        replacements = {
            target_paths.config_file: fernet.encrypt(
                json.dumps(sanitized).encode()
            ),
            target_paths.salt_file: salt,
            target_paths.password_hash_file: required[1].read_bytes(),
        }
        recovery_file = legacy_root / ".recovery"
        if recovery_file.is_file():
            replacements[target_paths.recovery_file] = recovery_file.read_bytes()
        replace_files_transactionally(replacements)
    except OSError as exc:
        raise PasswordEncryptionError(
            f"Configuration migration failed; backup remains at {backup}."
        ) from exc

    return ConfigMigrationReport(
        cloud_fields_removed=removed,
        backup_dir=backup,
    )
