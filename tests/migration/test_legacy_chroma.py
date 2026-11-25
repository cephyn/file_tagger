import os
from pathlib import Path

import pytest

from file_tagger.migration.legacy_chroma import (
    cleanup_legacy_chroma,
    inspect_legacy_chroma,
)


class LegacyCollection:
    def __init__(self, metadatas) -> None:
        self.metadatas = metadatas

    def get(self, *, include):
        assert include == ["metadatas"]
        return {
            "ids": [str(index) for index, _ in enumerate(self.metadatas)],
            "metadatas": self.metadatas,
        }


def test_chroma_paths_are_reindexed_not_vector_copied() -> None:
    plan = inspect_legacy_chroma(
        LegacyCollection(
            [
                {"path": "C:/library/source.pdf"},
                {"path": "C:/library/source.pdf", "chunk_id": 1},
            ]
        )
    )

    assert plan.paths == ("C:/library/source.pdf",)
    assert plan.reuse_embeddings is False
    assert plan.requires_full_reindex is True
    assert plan.warnings == ()


def test_malformed_and_missing_chroma_metadata_are_explicit(tmp_path) -> None:
    malformed = inspect_legacy_chroma(
        LegacyCollection([None, {"filename": "unknown.pdf"}, {"path": ""}])
    )
    missing = inspect_legacy_chroma(tmp_path / "missing-chroma")

    assert malformed.paths == ()
    assert len(malformed.warnings) == 3
    assert malformed.requires_full_reindex is False
    assert missing.found is False
    assert missing.warnings == ()


def test_chroma_directory_requires_confirmation_for_cleanup(tmp_path) -> None:
    chroma_dir = tmp_path / ".chroma"
    chroma_dir.mkdir()
    (chroma_dir / "chroma.sqlite3").write_bytes(b"legacy")

    with pytest.raises(ValueError, match="confirmation"):
        cleanup_legacy_chroma(chroma_dir, confirmed=False)

    assert cleanup_legacy_chroma(chroma_dir, confirmed=True) is True
    assert not chroma_dir.exists()


def test_chroma_cleanup_missing_directory_returns_false(tmp_path: Path) -> None:
    chroma_dir = tmp_path / ".chroma"

    assert cleanup_legacy_chroma(chroma_dir, confirmed=True) is False


def test_chroma_cleanup_rejects_non_chroma_directory(tmp_path: Path) -> None:
    chroma_dir = tmp_path / "other"
    chroma_dir.mkdir()

    with pytest.raises(ValueError, match="limited to a legacy .chroma"):
        cleanup_legacy_chroma(chroma_dir, confirmed=True)

    assert chroma_dir.is_dir()


def test_chroma_cleanup_rejects_regular_file(tmp_path: Path) -> None:
    chroma_file = tmp_path / ".chroma"
    chroma_file.write_bytes(b"not a directory")

    with pytest.raises(ValueError, match="not a regular directory"):
        cleanup_legacy_chroma(chroma_file, confirmed=True)

    assert chroma_file.is_file()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform does not support symlinks")
def test_chroma_cleanup_rejects_symlink_and_preserves_target(tmp_path: Path) -> None:
    target = tmp_path / "legacy-data"
    target.mkdir()
    (target / "chroma.sqlite3").write_bytes(b"legacy")
    chroma_link = tmp_path / ".chroma"

    try:
        chroma_link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="not a regular directory"):
        cleanup_legacy_chroma(chroma_link, confirmed=True)

    assert chroma_link.is_symlink()
    assert target.is_dir()


def test_chroma_sqlite_inventory_reads_only_path_metadata(tmp_path: Path) -> None:
    import sqlite3

    chroma_dir = tmp_path / ".chroma"
    chroma_dir.mkdir()
    with sqlite3.connect(chroma_dir / "chroma.sqlite3") as connection:
        connection.executescript(
            """
            CREATE TABLE embedding_metadata (
                id INTEGER,
                key TEXT,
                string_value TEXT
            );
            INSERT INTO embedding_metadata VALUES (1, 'path', 'C:/library/a.pdf');
            INSERT INTO embedding_metadata VALUES (2, 'path', 'C:/library/a.pdf');
            INSERT INTO embedding_metadata VALUES (3, 'chunk_id', '0');
            INSERT INTO embedding_metadata VALUES (4, 'summary', 'do not read');
            """
        )

    plan = inspect_legacy_chroma(chroma_dir)

    assert plan.paths == ("C:/library/a.pdf",)
    assert plan.found is True
