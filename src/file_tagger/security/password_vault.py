from __future__ import annotations

import base64
from datetime import UTC, datetime
import hashlib
import hmac
from enum import Enum, auto
import json
import os
from pathlib import Path
import secrets
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..domain.errors import (
    CredentialFormatError,
    CredentialRecoveryError,
)
from ..infrastructure.app_paths import AppPaths
from .file_transactions import replace_files_transactionally

PASSWORD_ENVELOPE_VERSION = 1
PBKDF2_ITERATIONS = 310_000
SALT_BYTES = 32


class CredentialStatus(Enum):
    UNINITIALIZED = auto()
    ENVELOPE = auto()
    LEGACY = auto()
    CORRUPT = auto()


class PasswordVault:
    def __init__(self, paths: AppPaths, data_key: bytes) -> None:
        self.paths = paths
        self.data_key = data_key

    @classmethod
    def status(cls, paths: AppPaths) -> CredentialStatus:
        envelope_files = (
            paths.password_key_file,
            paths.password_hash_file,
            paths.recovery_file,
        )
        if all(path.is_file() for path in envelope_files):
            try:
                for path in envelope_files:
                    json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return CredentialStatus.CORRUPT
            return CredentialStatus.ENVELOPE
        if paths.password_key_file.exists():
            return CredentialStatus.CORRUPT
        if (
            any(path.exists() for path in envelope_files)
            or paths.salt_file.exists()
        ):
            return CredentialStatus.LEGACY
        if paths.config_file.exists():
            return CredentialStatus.CORRUPT
        return CredentialStatus.UNINITIALIZED

    @classmethod
    def create(
        cls, paths: AppPaths, password: str
    ) -> tuple["PasswordVault", str]:
        paths.ensure_directories()
        if any(
            path.exists()
            for path in (
                paths.password_key_file,
                paths.password_hash_file,
                paths.recovery_file,
                paths.salt_file,
                paths.config_file,
            )
        ):
            raise CredentialFormatError("Credential metadata already exists.")

        data_key = Fernet.generate_key()
        recovery_key = secrets.token_urlsafe(24)
        password_material = _password_material(password, data_key)
        recovery_material = _recovery_material(recovery_key, data_key)
        _replace_transaction(
            {
                paths.password_hash_file: _json_bytes(password_material.verifier),
                paths.password_key_file: _json_bytes(password_material.envelope),
                paths.recovery_file: _json_bytes(recovery_material),
            }
        )
        return cls(paths, data_key), recovery_key

    @classmethod
    def open(cls, paths: AppPaths, password: str) -> "PasswordVault":
        verifier = _read_json(paths.password_hash_file)
        _verify_password(password, verifier)
        envelope = _read_envelope(paths.password_key_file)
        data_key = _unwrap_data_key(envelope, password)
        return cls(paths, data_key)

    def encrypt(self, plaintext: bytes) -> bytes:
        return Fernet(self.data_key).encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        try:
            return Fernet(self.data_key).decrypt(ciphertext)
        except (InvalidToken, ValueError) as exc:
            raise CredentialRecoveryError(
                "Configuration could not be decrypted safely."
            ) from exc

    def change_password(self, old_password: str, new_password: str) -> None:
        verifier = _read_json(self.paths.password_hash_file)
        _verify_password(old_password, verifier)
        envelope = _read_envelope(self.paths.password_key_file)
        if not hmac.compare_digest(_unwrap_data_key(envelope, old_password), self.data_key):
            raise CredentialRecoveryError("Credential metadata is inconsistent.")

        material = _password_material(new_password, self.data_key)
        _replace_transaction(
            {
                self.paths.password_hash_file: _json_bytes(material.verifier),
                self.paths.password_key_file: _json_bytes(material.envelope),
            }
        )

    @classmethod
    def recover(
        cls, paths: AppPaths, recovery_key: str, new_password: str
    ) -> "PasswordVault":
        recovery_envelope = _read_envelope(paths.recovery_file)
        data_key = _unwrap_data_key(recovery_envelope, recovery_key)
        _validate_data_key_for_config(paths.config_file, data_key)
        material = _password_material(new_password, data_key)
        _replace_transaction(
            {
                paths.password_hash_file: _json_bytes(material.verifier),
                paths.password_key_file: _json_bytes(material.envelope),
            }
        )
        return cls(paths, data_key)

    def rotate_recovery_key(self, current_password: str) -> str:
        verifier = _read_json(self.paths.password_hash_file)
        _verify_password(current_password, verifier)
        envelope = _read_envelope(self.paths.password_key_file)
        if not hmac.compare_digest(
            _unwrap_data_key(envelope, current_password), self.data_key
        ):
            raise CredentialRecoveryError("Credential metadata is inconsistent.")

        recovery_key = secrets.token_urlsafe(24)
        _replace_transaction(
            {
                self.paths.recovery_file: _json_bytes(
                    _recovery_material(recovery_key, self.data_key)
                )
            }
        )
        return recovery_key

    @classmethod
    def migrate_legacy(
        cls, paths: AppPaths, password: str
    ) -> tuple["PasswordVault", str]:
        if cls.status(paths) is not CredentialStatus.LEGACY:
            raise CredentialFormatError("Legacy credentials are not available.")

        verifier = _read_json(paths.password_hash_file)
        _verify_legacy_password(password, verifier)
        try:
            legacy_fernet = _derive_legacy_fernet(
                password, paths.salt_file.read_bytes()
            )
        except OSError as exc:
            raise CredentialFormatError("Legacy encryption metadata is missing.") from exc

        replacement_config: bytes | None = None
        if paths.config_file.is_file():
            try:
                plaintext = legacy_fernet.decrypt(paths.config_file.read_bytes())
                config = json.loads(plaintext)
            except (InvalidToken, UnicodeError, json.JSONDecodeError) as exc:
                raise CredentialFormatError(
                    "Legacy configuration could not be decrypted safely."
                ) from exc
            if not isinstance(config, dict):
                raise CredentialFormatError("Legacy configuration is invalid.")

        data_key = Fernet.generate_key()
        recovery_key = secrets.token_urlsafe(24)
        material = _password_material(password, data_key)
        replacements = {
            paths.password_hash_file: _json_bytes(material.verifier),
            paths.password_key_file: _json_bytes(material.envelope),
            paths.recovery_file: _json_bytes(
                _recovery_material(recovery_key, data_key)
            ),
        }
        if paths.config_file.is_file():
            replacement_config = Fernet(data_key).encrypt(plaintext)
            replacements[paths.config_file] = replacement_config

        _backup_credential_files(paths)
        try:
            _replace_transaction(replacements)
        except CredentialRecoveryError:
            raise
        return cls(paths, data_key), recovery_key


class _PasswordMaterial:
    def __init__(self, verifier: dict[str, Any], envelope: dict[str, Any]) -> None:
        self.verifier = verifier
        self.envelope = envelope


def _derive_fernet(secret: str, salt: bytes) -> Fernet:
    material = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    ).derive(secret.encode("utf-8"))
    return Fernet(base64.urlsafe_b64encode(material))


def _derive_legacy_fernet(password: str, salt: bytes) -> Fernet:
    material = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    ).derive(password.encode("utf-8"))
    return Fernet(base64.urlsafe_b64encode(material))


def _password_material(password: str, data_key: bytes) -> _PasswordMaterial:
    verifier_salt = os.urandom(SALT_BYTES)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        verifier_salt,
        PBKDF2_ITERATIONS,
    )
    envelope_salt = os.urandom(SALT_BYTES)
    return _PasswordMaterial(
        verifier={
            "version": PASSWORD_ENVELOPE_VERSION,
            "algorithm": "pbkdf2-sha256",
            "iterations": PBKDF2_ITERATIONS,
            "hash": base64.b64encode(password_hash).decode("ascii"),
            "salt": base64.b64encode(verifier_salt).decode("ascii"),
        },
        envelope={
            "version": PASSWORD_ENVELOPE_VERSION,
            "salt": base64.b64encode(envelope_salt).decode("ascii"),
            "wrapped_key": _derive_fernet(password, envelope_salt)
            .encrypt(data_key)
            .decode("ascii"),
        },
    )


def _recovery_material(recovery_key: str, data_key: bytes) -> dict[str, Any]:
    salt = os.urandom(SALT_BYTES)
    return {
        "version": PASSWORD_ENVELOPE_VERSION,
        "salt": base64.b64encode(salt).decode("ascii"),
        "wrapped_key": _derive_fernet(recovery_key, salt)
        .encrypt(data_key)
        .decode("ascii"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CredentialFormatError("Credential metadata is missing.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CredentialFormatError("Credential metadata is invalid.") from exc
    if not isinstance(value, dict):
        raise CredentialFormatError("Credential metadata is invalid.")
    return value


def _read_envelope(path: Path) -> dict[str, Any]:
    envelope = _read_json(path)
    if (
        envelope.get("version") != PASSWORD_ENVELOPE_VERSION
        or not isinstance(envelope.get("salt"), str)
        or not isinstance(envelope.get("wrapped_key"), str)
    ):
        raise CredentialFormatError("Credential envelope is unsupported.")
    return envelope


def _verify_password(password: str, verifier: dict[str, Any]) -> None:
    try:
        if (
            verifier.get("version") != PASSWORD_ENVELOPE_VERSION
            or verifier.get("algorithm") != "pbkdf2-sha256"
            or verifier.get("iterations") != PBKDF2_ITERATIONS
            or not isinstance(verifier.get("salt"), str)
            or not isinstance(verifier.get("hash"), str)
        ):
            raise CredentialFormatError("Password verifier is unsupported.")
        salt = base64.b64decode(verifier["salt"], validate=True)
        expected = base64.b64decode(verifier["hash"], validate=True)
    except (TypeError, ValueError) as exc:
        raise CredentialFormatError("Password verifier is invalid.") from exc

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    if not hmac.compare_digest(actual, expected):
        raise CredentialRecoveryError("Password authentication failed.")


def _verify_legacy_password(password: str, verifier: dict[str, Any]) -> None:
    try:
        salt = base64.b64decode(verifier["salt"], validate=True)
        expected = base64.b64decode(verifier["hash"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise CredentialFormatError("Legacy password metadata is invalid.") from exc

    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 100_000
    )
    if not hmac.compare_digest(actual, expected):
        raise CredentialRecoveryError("Legacy password authentication failed.")


def _unwrap_data_key(envelope: dict[str, Any], secret: str) -> bytes:
    try:
        salt = base64.b64decode(envelope["salt"], validate=True)
        wrapped = envelope["wrapped_key"].encode("ascii")
        data_key = _derive_fernet(secret, salt).decrypt(wrapped)
        Fernet(data_key)
        return data_key
    except InvalidToken as exc:
        raise CredentialRecoveryError("Credential authentication failed.") from exc
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise CredentialFormatError("Credential envelope is invalid.") from exc


def _validate_data_key_for_config(config_file: Path, data_key: bytes) -> None:
    if not config_file.is_file():
        return
    try:
        Fernet(data_key).decrypt(config_file.read_bytes())
    except (InvalidToken, OSError, ValueError) as exc:
        raise CredentialRecoveryError(
            "Recovery credentials do not match the active configuration."
        ) from exc


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _backup_credential_files(paths: AppPaths) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = (
        paths.data_dir
        / "password-envelope-backups"
        / f"credentials-{timestamp}-{secrets.token_hex(3)}"
    )
    try:
        backup_dir.mkdir(parents=True)
        for source in (
            paths.config_file,
            paths.salt_file,
            paths.password_hash_file,
            paths.password_key_file,
            paths.recovery_file,
        ):
            if not source.is_file():
                continue
            destination = backup_dir / source.name
            content = source.read_bytes()
            destination.write_bytes(content)
            if destination.read_bytes() != content:
                raise OSError("Credential backup verification failed.")
    except OSError as exc:
        raise CredentialRecoveryError(
            "Credential migration backup could not be completed."
        ) from exc
    return backup_dir


def _replace_transaction(files: dict[Path, bytes]) -> None:
    try:
        replace_files_transactionally(files)
    except OSError as exc:
        raise CredentialRecoveryError("Credential update failed safely.") from exc
