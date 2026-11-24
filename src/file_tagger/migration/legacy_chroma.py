from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Protocol


PATH_METADATA_KEYS = ("path", "file_path", "source")


class LegacyCollection(Protocol):
    def get(self, *, include: list[str]) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class LegacyChromaPlan:
    paths: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    found: bool = True

    @property
    def reuse_embeddings(self) -> bool:
        return False

    @property
    def requires_full_reindex(self) -> bool:
        return bool(self.paths)


def _inventory_metadatas(metadatas: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(metadatas, list):
        return (), ("Legacy collection returned malformed metadata.",)

    paths: set[str] = set()
    warnings: list[str] = []
    for index, metadata in enumerate(metadatas):
        if not isinstance(metadata, Mapping):
            warnings.append(f"Legacy metadata row {index} is malformed.")
            continue
        path = next(
            (
                metadata[key]
                for key in PATH_METADATA_KEYS
                if isinstance(metadata.get(key), str) and metadata[key].strip()
            ),
            None,
        )
        if path is None:
            warnings.append(f"Legacy metadata row {index} has no usable file path.")
            continue
        paths.add(path)
    return tuple(sorted(paths)), tuple(warnings)


def _inspect_collection(collection: LegacyCollection) -> LegacyChromaPlan:
    try:
        result = collection.get(include=["metadatas"])
    except (AttributeError, TypeError, ValueError) as exc:
        return LegacyChromaPlan(
            paths=(),
            warnings=(f"Legacy collection metadata could not be read: {type(exc).__name__}.",),
        )
    paths, warnings = _inventory_metadatas(result.get("metadatas"))
    return LegacyChromaPlan(paths=paths, warnings=warnings)


def _inspect_directory(chroma_dir: Path) -> LegacyChromaPlan:
    if not chroma_dir.exists():
        return LegacyChromaPlan(paths=(), found=False)
    database = chroma_dir / "chroma.sqlite3"
    if not database.is_file():
        return LegacyChromaPlan(
            paths=(),
            warnings=("Legacy Chroma directory has no metadata database.",),
        )

    try:
        connection = sqlite3.connect(
            f"{database.resolve().as_uri()}?mode=ro",
            uri=True,
        )
    except sqlite3.Error as exc:
        return LegacyChromaPlan(
            paths=(),
            warnings=(f"Legacy Chroma metadata database is unreadable: {exc.__class__.__name__}.",),
        )

    try:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(embedding_metadata)"
            ).fetchall()
        }
        if not {"id", "key", "string_value"} <= columns:
            return LegacyChromaPlan(
                paths=(),
                warnings=("Legacy Chroma metadata schema is unsupported.",),
            )
        rows = connection.execute(
            "SELECT id, key, string_value FROM embedding_metadata"
        ).fetchall()
    except sqlite3.Error as exc:
        return LegacyChromaPlan(
            paths=(),
            warnings=(f"Legacy Chroma metadata could not be read: {exc.__class__.__name__}.",),
        )
    finally:
        connection.close()

    by_id: dict[int, dict[str, str]] = {}
    for raw_id, key, value in rows:
        try:
            record_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if isinstance(key, str) and isinstance(value, str):
            by_id.setdefault(record_id, {})[key] = value

    paths, warnings = _inventory_metadatas(list(by_id.values()))
    return LegacyChromaPlan(paths=paths, warnings=warnings)


def inspect_legacy_chroma(
    source: Path | LegacyCollection,
) -> LegacyChromaPlan:
    if isinstance(source, Path):
        return _inspect_directory(source.expanduser())
    return _inspect_collection(source)


def cleanup_legacy_chroma(chroma_dir: Path, *, confirmed: bool) -> bool:
    if not confirmed:
        raise ValueError("Legacy Chroma cleanup requires explicit confirmation.")
    resolved = chroma_dir.expanduser().resolve()
    if resolved.name != ".chroma":
        raise ValueError("Cleanup is limited to a legacy .chroma directory.")
    if not resolved.exists():
        return False
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError("Legacy Chroma cleanup path is not a regular directory.")
    shutil.rmtree(resolved)
    return True
