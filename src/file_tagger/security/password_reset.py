from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
import secrets
import shutil

from ..domain.errors import CredentialRecoveryError
from ..infrastructure.app_paths import AppPaths
from .file_transactions import replace_files_transactionally
from .password_vault import PasswordVault

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PasswordResetResult:
    backup_dir: Path
    recovery_key: str


def reset_encrypted_configuration(
    paths: AppPaths,
    new_password: str,
) -> PasswordResetResult:
    if not new_password:
        raise CredentialRecoveryError("A new password is required.")

    paths.ensure_directories()
    target_snapshot = {
        path: path.read_bytes() if path.is_file() else None
        for path in _target_credential_files(paths)
    }
    sources = _active_credential_files(paths)
    backup_dir, quarantined = _quarantine_files(paths, sources)
    try:
        _, recovery_key = PasswordVault.create(paths, new_password)
        from ..config import Config

        config = Config(new_password)
        config.save()
    except Exception as exc:
        logger.error(
            "Password reset failed during credential initialization (%s); backup=%s",
            exc.__class__.__name__,
            backup_dir,
        )
        _rollback_reset(target_snapshot, quarantined, backup_dir)
        raise CredentialRecoveryError(
            "The configuration reset could not be completed safely."
        ) from exc

    return PasswordResetResult(backup_dir, recovery_key)


def _active_credential_files(paths: AppPaths) -> tuple[Path, ...]:
    legacy_root = paths.legacy_database.parent
    if (legacy_root / "config.encrypted").is_file():
        sources = [
            legacy_root / name
            for name in (
                "config.encrypted",
                ".salt",
                ".passhash",
                ".password-key",
                ".recovery",
            )
        ]
        sources.extend(_target_credential_files(paths))
        unique: list[Path] = []
        for source in sources:
            if source not in unique:
                unique.append(source)
        return tuple(unique)

    if paths.config_file.is_file():
        return _target_credential_files(paths)
    return _target_credential_files(paths)


def _target_credential_files(paths: AppPaths) -> tuple[Path, ...]:
    return (
        paths.config_file,
        paths.salt_file,
        paths.password_hash_file,
        paths.password_key_file,
        paths.recovery_file,
    )


def _quarantine_files(
    paths: AppPaths, sources: tuple[Path, ...]
) -> tuple[Path, list[tuple[Path, Path]]]:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = (
        paths.data_dir
        / "password-reset-backups"
        / f"reset-{timestamp}-{secrets.token_hex(3)}"
    )
    existing = [source for source in sources if source.is_file()]
    try:
        backup_dir.mkdir(parents=True)
        staged: list[tuple[Path, Path]] = []
        for source in existing:
            destination = backup_dir / source.name
            if destination.exists() and source.parent != backup_dir:
                target_dir = backup_dir / "target"
                target_dir.mkdir(parents=True, exist_ok=True)
                destination = target_dir / source.name
            original = source.read_bytes()
            shutil.copy2(source, destination)
            if destination.read_bytes() != original:
                raise OSError("Password reset backup verification failed.")
            staged.append((source, destination))
    except OSError as exc:
        raise CredentialRecoveryError(
            "Configuration quarantine could not be backed up safely."
        ) from exc

    quarantined: list[tuple[Path, Path]] = []
    try:
        for source, destination in staged:
            source.unlink()
            quarantined.append((source, destination))
    except OSError as exc:
        _restore_quarantined(quarantined)
        raise CredentialRecoveryError(
            "Configuration quarantine could not be completed safely."
        ) from exc
    return backup_dir, quarantined


def _rollback_reset(
    target_snapshot: dict[Path, bytes | None],
    quarantined: list[tuple[Path, Path]],
    backup_dir: Path,
) -> None:
    rollback_error: OSError | None = None
    try:
        replace_files_transactionally(
            {
                path: content
                for path, content in target_snapshot.items()
                if content is not None
            }
        )
    except OSError as exc:
        rollback_error = exc
    for path, content in target_snapshot.items():
        if content is not None:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            rollback_error = exc
    try:
        _restore_quarantined(
            quarantined
        )
    except OSError as exc:
        rollback_error = exc
    if rollback_error is not None:
        raise CredentialRecoveryError(
            f"Password reset rollback was incomplete; backup remains at {backup_dir}."
        ) from rollback_error


def _restore_quarantined(quarantined: list[tuple[Path, Path]]) -> None:
    first_error: OSError | None = None
    for source, backup in reversed(quarantined):
        try:
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, source)
            if source.read_bytes() != backup.read_bytes():
                raise OSError("Password reset rollback verification failed.")
        except OSError as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
