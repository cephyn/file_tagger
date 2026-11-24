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
