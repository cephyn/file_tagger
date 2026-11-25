from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from sqlalchemy import bindparam, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..domain.errors import IndexTransactionError
from ..persistence.models import File, FileTag, Tag, TagSuggestionCache


@dataclass(frozen=True, slots=True)
class LegacyMigrationReport:
    files_seen: int
    files_migrated: int
    tags_migrated: int
    associations_migrated: int
    suggestions_preserved: int
    backup_path: Path
    errors: tuple[str, ...] = ()


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _backup_database(source_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    backup_path = backup_dir / f"file_tags-{timestamp}.sqlite3"
    with sqlite3.connect(source_path) as source:
        with sqlite3.connect(backup_path) as destination:
            source.backup(destination)
    return backup_path


def _normalized_path(path: str) -> str:
    return os.path.normpath(path)


def _load_legacy_rows(
    source_path: Path,
) -> tuple[
    list[tuple[int, str]],
    list[tuple[int, str, str | None]],
    list[tuple[int, int]],
    list[dict[str, Any]],
    tuple[str, ...],
]:
    errors: list[str] = []
    with sqlite3.connect(source_path) as connection:
        file_columns = _table_columns(connection, "files")
        tag_columns = _table_columns(connection, "tags")
        if not {"id", "path"} <= file_columns:
            raise IndexTransactionError("Legacy files table has an unsupported schema.")
        if not {"id", "name"} <= tag_columns:
            raise IndexTransactionError("Legacy tags table has an unsupported schema.")

        files = [
            (int(row[0]), _normalized_path(str(row[1])))
            for row in connection.execute("SELECT id, path FROM files")
            if row[1]
        ]
        color_sql = "color" if "color" in tag_columns else "NULL"
        tags = [
            (int(row[0]), str(row[1]), str(row[2]) if row[2] else None)
            for row in connection.execute(f"SELECT id, name, {color_sql} FROM tags")
            if row[1]
        ]
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        associations = (
            [
                (int(row[0]), int(row[1]))
                for row in connection.execute(
                    "SELECT file_id, tag_id FROM file_tags"
                )
            ]
            if "file_tags" in tables
            else []
        )

        suggestions: list[dict[str, Any]] = []
        if "tag_suggestion_cache" in tables:
            columns = _table_columns(connection, "tag_suggestion_cache")
            required = {"file_path", "file_hash", "suggestions"}
            if required <= columns:
                timestamp_sql = (
                    "timestamp" if "timestamp" in columns else "CURRENT_TIMESTAMP"
                )
                for path, file_hash, timestamp, raw_suggestions in connection.execute(
                    "SELECT file_path, file_hash, "
                    f"{timestamp_sql}, suggestions FROM tag_suggestion_cache"
                ):
                    if not path or not file_hash or raw_suggestions is None:
                        errors.append("Skipped an incomplete legacy suggestion row.")
                        continue
                    try:
                        parsed_suggestions = (
                            json.loads(raw_suggestions)
                            if isinstance(raw_suggestions, str)
                            else raw_suggestions
                        )
                    except (json.JSONDecodeError, TypeError):
                        errors.append("Skipped a malformed legacy suggestion row.")
                        continue
                    suggestions.append(
                        {
                            "file_path": _normalized_path(str(path)),
                            "file_hash": str(file_hash),
                            "cached_at": str(timestamp),
                            "suggestions": parsed_suggestions,
                        }
                    )
    return files, tags, associations, suggestions, tuple(errors)


def _cache_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(UTC)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _delete_file_embeddings(session: Session, file_ids: set[int]) -> None:
    if not file_ids:
        return
    statement_parameters = {"file_ids": sorted(file_ids)}
    for dimensions in (256, 512, 768):
        statement = text(
            f"""
            DELETE FROM chunk_embeddings_{dimensions}
            WHERE chunk_id IN (
                SELECT chunks.id
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE documents.file_id IN :file_ids
            )
            """
        ).bindparams(bindparam("file_ids", expanding=True))
        session.execute(statement, statement_parameters)


def migrate_legacy_sqlite(
    legacy_database: Path,
    target_session: Session,
    *,
    backup_dir: Path | None = None,
) -> LegacyMigrationReport:
    source_path = legacy_database.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy database not found: {source_path}")

    backups = backup_dir or source_path.parent / "migration-backups"
    backup_path = _backup_database(source_path, backups)
    files, tags, associations, suggestions, legacy_errors = _load_legacy_rows(
        source_path
    )
    errors = list(legacy_errors)
    source_files = {path for _, path in files}
    source_tags = {name for _, name, _ in tags}
    source_tag_colors = {
        name: (color or "#808080") for _, name, color in tags
    }
    source_file_ids = {legacy_id: path for legacy_id, path in files}
    source_tag_ids = {legacy_id: name for legacy_id, name, _ in tags}
    source_associations = {
        (source_file_ids[file_id], source_tag_ids[tag_id])
        for file_id, tag_id in associations
        if file_id in source_file_ids and tag_id in source_tag_ids
    }
    migrated_files = 0
    migrated_tags = 0
    migrated_associations = 0
    preserved_suggestions = 0

    try:
        with target_session.begin_nested():
            current_files = list(target_session.scalars(select(File)))
            target_files = {file.canonical_path: file for file in current_files}
            stale_files = [
                file for file in current_files if file.canonical_path not in source_files
            ]
            _delete_file_embeddings(
                target_session, {int(file.id) for file in stale_files}
            )
            for file in stale_files:
                target_session.delete(file)

            current_tags = list(target_session.scalars(select(Tag)))
            target_tags = {tag.name: tag for tag in current_tags}
            for tag in current_tags:
                if tag.name not in source_tags:
                    target_session.delete(tag)

            for path in sorted(source_files):
                if path not in target_files:
                    target_file = File(
                        canonical_path=path,
                        indexing_state="not_indexed",
                    )
                    target_session.add(target_file)
                    target_files[path] = target_file
                    migrated_files += 1

            for name, color in sorted(source_tag_colors.items()):
                target_tag = target_tags.get(name)
                if target_tag is None:
                    target_tag = Tag(name=name, color=color)
                    target_session.add(target_tag)
                    target_tags[name] = target_tag
                    migrated_tags += 1
                else:
                    target_tag.color = color

            target_session.flush()

            file_paths_by_id = {
                int(file.id): path for path, file in target_files.items()
            }
            tag_names_by_id = {
                int(tag.id): name for name, tag in target_tags.items()
            }
            current_associations = {
                (
                    file_paths_by_id[association.file_id],
                    tag_names_by_id[association.tag_id],
                ): association
                for association in target_session.scalars(select(FileTag))
            }
            for association_key, association in current_associations.items():
                if association_key not in source_associations:
                    target_session.delete(association)
            for path, name in source_associations:
                if (path, name) not in current_associations:
                    target_session.add(
                        FileTag(
                            file_id=int(target_files[path].id),
                            tag_id=int(target_tags[name].id),
                        )
                    )
                    migrated_associations += 1

            for suggestion in suggestions:
                suggestion_file = target_files.get(str(suggestion["file_path"]))
                if suggestion_file is None:
                    errors.append("Skipped suggestion data for an unknown legacy file.")
                    continue
                if target_session.scalar(
                    select(TagSuggestionCache).where(
                        TagSuggestionCache.file_id == suggestion_file.id
                    )
                ):
                    continue
                target_session.add(
                    TagSuggestionCache(
                        file_id=suggestion_file.id,
                        file_path=str(suggestion["file_path"]),
                        file_hash=str(suggestion["file_hash"]),
                        cached_at=_cache_timestamp(str(suggestion["cached_at"])),
                        suggestions=suggestion["suggestions"],
                    )
                )
                preserved_suggestions += 1

            target_session.flush()

            target_file_paths = set(
                target_session.scalars(select(File.canonical_path))
            )
            target_tag_values = set(
                target_session.execute(select(Tag.name, Tag.color))
            )
            target_associations = set(
                target_session.execute(
                    select(File.canonical_path, Tag.name)
                    .select_from(FileTag)
                    .join(File, File.id == FileTag.file_id)
                    .join(Tag, Tag.id == FileTag.tag_id)
                )
            )
            if (
                len(target_file_paths) != len(source_files)
                or target_file_paths != source_files
                or _checksum(sorted(target_file_paths))
                != _checksum(sorted(source_files))
                or len(target_tag_values) != len(source_tags)
                or target_tag_values != set(source_tag_colors.items())
                or len(target_associations) != len(source_associations)
                or target_associations != source_associations
            ):
                raise IndexTransactionError(
                    "Legacy migration verification failed; "
                    f"backup remains at {backup_path}."
                )
    except (SQLAlchemyError, RuntimeError) as exc:
        raise IndexTransactionError(
            f"Legacy migration failed; original database backup remains at {backup_path}."
        ) from exc

    return LegacyMigrationReport(
        files_seen=len(files),
        files_migrated=migrated_files,
        tags_migrated=migrated_tags,
        associations_migrated=migrated_associations,
        suggestions_preserved=preserved_suggestions,
        backup_path=backup_path,
        errors=tuple(errors),
    )
