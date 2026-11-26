from __future__ import annotations

from enum import Enum, auto
import json

from ..infrastructure.app_paths import AppPaths


class CredentialStatus(Enum):
    UNINITIALIZED = auto()
    ENVELOPE = auto()
    LEGACY = auto()
    CORRUPT = auto()


class PasswordVault:
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
