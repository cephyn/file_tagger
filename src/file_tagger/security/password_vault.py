from __future__ import annotations

import base64
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


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _replace_transaction(files: dict[Path, bytes]) -> None:
    token = secrets.token_hex(8)
    staged: dict[Path, Path] = {}
    originals: dict[Path, bytes | None] = {}
    replaced: list[Path] = []
    try:
        for destination, content in files.items():
            temporary = destination.with_name(f".{destination.name}.{token}.tmp")
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            staged[destination] = temporary
            originals[destination] = (
                destination.read_bytes() if destination.is_file() else None
            )
        for destination, temporary in staged.items():
            temporary.replace(destination)
            replaced.append(destination)
    except OSError as exc:
        rollback_error: OSError | None = None
        for destination in reversed(replaced):
            try:
                original = originals[destination]
                if original is None:
                    destination.unlink(missing_ok=True)
                else:
                    rollback = destination.with_name(
                        f".{destination.name}.{token}.rollback"
                    )
                    rollback.write_bytes(original)
                    rollback.replace(destination)
            except OSError as error:
                rollback_error = error
        if rollback_error is not None:
            raise CredentialRecoveryError(
                "Credential update failed and rollback was incomplete."
            ) from rollback_error
        raise CredentialRecoveryError("Credential update failed safely.") from exc
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
