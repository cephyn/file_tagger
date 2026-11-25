from dataclasses import replace
from hashlib import sha256
import logging
from pathlib import Path
import sqlite3
import sys
import threading
from threading import Event
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select, text
import pytest

from file_tagger.domain.contracts import HardwareReport, ModelProfile
from file_tagger.extraction.docling_adapter import ExtractedDocument, ExtractedPage
from file_tagger.indexing.repository import IndexingRepository
from file_tagger.indexing.service import IndexingService
from file_tagger.infrastructure.database import (
    create_session_factory,
    upgrade_database,
)
from file_tagger.modern_context import (
    ModernIndexingUnavailable,
    ModernSearchUnavailable,
    ModernRuntime,
    initialize_modern_runtime,
)
from file_tagger.persistence.models import File, FileTag, Tag
from file_tagger.ollama.readiness import ReadinessReport, RequiredModels
from file_tagger.domain.contracts import OllamaReadiness


def _create_legacy_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE files (id INTEGER PRIMARY KEY, path VARCHAR UNIQUE);
            CREATE TABLE tags (
                id INTEGER PRIMARY KEY,
                name VARCHAR UNIQUE,
                color VARCHAR
            );
            CREATE TABLE file_tags (file_id INTEGER, tag_id INTEGER);
            """
        )
        connection.execute("INSERT INTO files VALUES (1, ?)", (r"C:\library\a.pdf",))
        connection.execute("INSERT INTO tags VALUES (1, 'research', '#123456')")
        connection.execute("INSERT INTO file_tags VALUES (1, 1)")


def _paths(tmp_path: Path, monkeypatch: Any):
    from file_tagger.infrastructure.app_paths import AppPaths

    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path / "modern"))
    discovered = AppPaths.discover()
    return replace(
        discovered,
        legacy_database=tmp_path / "file_tags.db",
        legacy_chroma=tmp_path / ".chroma",
    )


def _manager_class(monkeypatch):
    from types import ModuleType

    vector_search = ModuleType("file_tagger.vector_search")
    vector_search.VectorSearch = object
    content_extractor = ModuleType("file_tagger.vector_search.content_extractor")
    content_extractor.ContentExtractor = object
    monkeypatch.setitem(sys.modules, "file_tagger.vector_search", vector_search)
    monkeypatch.setitem(
        sys.modules,
        "file_tagger.vector_search.content_extractor",
        content_extractor,
    )
    sys.modules.pop("file_tagger.file_tag_manager", None)
    from file_tagger.file_tag_manager import FileTagManager

    return FileTagManager


@pytest.fixture(autouse=True)
def _clear_legacy_ui_module_cache():
    yield
    sys.modules.pop("file_tagger.file_tag_manager", None)
    sys.modules.pop("file_tagger.tag_suggestion", None)
    package = sys.modules.get("file_tagger")
    if package is not None:
        for name in ("file_tag_manager", "tag_suggestion"):
            if hasattr(package, name):
                delattr(package, name)


class ReloadableConfig:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.reloads = 0

    def reload_after_migration(self) -> None:
        self.reloads += 1


def test_startup_upgrades_and_idempotently_migrates_legacy_tags(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _create_legacy_database(paths.legacy_database)
    config = ReloadableConfig(paths)

    first = initialize_modern_runtime(
        paths,
        config,
        "password",
        start_readiness=False,
    )
    second = initialize_modern_runtime(
        paths,
        config,
        "password",
        start_readiness=False,
    )

    assert first.session_factory is not None
    assert first.sqlite_migration is not None
    assert second.sqlite_migration is not None
    assert second.sqlite_migration.files_migrated == 0
    assert second.sqlite_migration.tags_migrated == 0
    assert second.sqlite_migration.associations_migrated == 0
    with first.session_factory() as session:
        file = session.scalar(select(File))
        tag = session.scalar(select(Tag))
        association = session.scalar(select(FileTag))
        assert file is not None and file.canonical_path == r"C:\library\a.pdf"
        assert tag is not None and tag.name == "research"
        assert association is not None


def test_startup_reconciles_legacy_snapshot_changes(tmp_path, monkeypatch) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _create_legacy_database(paths.legacy_database)
    config = ReloadableConfig(paths)

    initialize_modern_runtime(
        paths,
        config,
        "password",
        start_readiness=False,
    )
    with sqlite3.connect(paths.legacy_database) as connection:
        connection.executescript(
            """
            UPDATE tags SET color = '#654321' WHERE name = 'research';
            DELETE FROM file_tags;
            DELETE FROM files;
            INSERT INTO files VALUES (2, 'C:\\library\\renamed.pdf');
            INSERT INTO file_tags VALUES (2, 1);
            """
        )

    reconciled = initialize_modern_runtime(
        paths,
        config,
        "password",
        start_readiness=False,
    )
    converged = initialize_modern_runtime(
        paths,
        config,
        "password",
        start_readiness=False,
    )

    assert reconciled.sqlite_migration is not None
    assert reconciled.sqlite_migration.files_migrated == 1
    assert converged.sqlite_migration is not None
    assert converged.sqlite_migration.files_migrated == 0
    assert converged.sqlite_migration.tags_migrated == 0
    assert converged.sqlite_migration.associations_migrated == 0
    assert reconciled.session_factory is not None
    with reconciled.session_factory() as session:
        files = set(session.execute(select(File.canonical_path)).scalars())
        tag = session.scalar(select(Tag))
        association = session.execute(
            select(File.canonical_path, Tag.name)
            .join(FileTag, FileTag.file_id == File.id)
            .join(Tag, Tag.id == FileTag.tag_id)
        ).one()
        assert files == {r"C:\library\renamed.pdf"}
        assert tag is not None and tag.color == "#654321"
        assert association == (r"C:\library\renamed.pdf", "research")


def test_startup_reloads_config_only_after_a_real_legacy_config_migration(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    config = ReloadableConfig(paths)
    monkeypatch.setattr(
        "file_tagger.modern_context.migrate_legacy_config",
        lambda *args, **kwargs: SimpleNamespace(skipped=False, errors=()),
    )

    initialize_modern_runtime(
        paths,
        config,
        "password",
        start_readiness=False,
    )

    assert config.reloads == 1

    monkeypatch.setattr(
        "file_tagger.modern_context.migrate_legacy_config",
        lambda *args, **kwargs: SimpleNamespace(skipped=True, errors=()),
    )
    initialize_modern_runtime(
        paths,
        config,
        "password",
        start_readiness=False,
    )
    assert config.reloads == 1


def test_pending_legacy_config_does_not_bootstrap_a_modern_password(
    tmp_path, monkeypatch
) -> None:
    from file_tagger.config import Config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path / "modern"))
    (tmp_path / "config.encrypted").write_bytes(b"legacy")

    Config("candidate-password")

    paths = _paths(tmp_path, monkeypatch)
    assert not paths.password_hash_file.exists()
    assert not paths.recovery_file.exists()


def test_bad_pending_legacy_password_fails_instead_of_poisoning_startup(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    (paths.legacy_database.parent / "config.encrypted").write_bytes(b"legacy")
    monkeypatch.setattr(
        "file_tagger.modern_context.migrate_legacy_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("wrong password")),
    )

    try:
        initialize_modern_runtime(
            paths,
            ReloadableConfig(paths),
            "wrong-password",
            start_readiness=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A pending legacy config must authenticate successfully.")


def test_manager_commit_synchronizes_only_after_legacy_success(
    monkeypatch,
) -> None:
    events: list[str] = []

    class LegacySession:
        def commit(self) -> None:
            events.append("legacy-commit")

    class Modern:
        def sync_tag(self, name: str, color: str | None) -> None:
            events.append(f"modern-tag:{name}")

        def sync_file(self, path: str, tags: list[tuple[str, str | None]]) -> None:
            events.append(f"modern-file:{path}")

        def delete_tag(self, name: str) -> None:
            events.append(f"modern-delete-tag:{name}")

        def delete_file(self, path: str) -> None:
            events.append(f"modern-delete-file:{path}")

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.db_session = LegacySession()
    manager.modern_runtime = Modern()
    manager.logger = logging.getLogger("startup-sync-test")
    manager.modern_index_status_text = ""

    manager._commit_legacy_changes(
        tag_upserts=[("research", "#123456")],
        file_paths=[("C:\\library\\a.pdf", [("research", "#123456")])],
    )

    assert events == [
        "legacy-commit",
        "modern-tag:research",
        "modern-file:C:\\library\\a.pdf",
    ]


def test_manager_uses_atomic_modern_snapshot_when_supported(monkeypatch) -> None:
    events: list[str] = []

    class LegacySession:
        def commit(self) -> None:
            events.append("legacy-commit")

    class Modern:
        def sync_legacy_changes(self, **changes) -> None:
            events.append(
                f"modern:{changes['tag_upserts'][0][0]}:"
                f"{changes['file_paths'][0][0]}"
            )

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.db_session = LegacySession()
    manager.modern_runtime = Modern()
    manager.logger = logging.getLogger("startup-atomic-sync-test")

    assert manager._commit_legacy_changes(
        tag_upserts=[("research", "#123456")],
        file_paths=[("C:\\library\\a.pdf", [("research", "#123456")])],
    )
    assert events == ["legacy-commit", "modern:research:C:\\library\\a.pdf"]


def test_manager_reports_unavailable_modern_index_without_affecting_legacy(
    tmp_path, monkeypatch
) -> None:
    class Status:
        def start(self, path: str) -> None:
            raise ModernIndexingUnavailable("Ollama embedding profile is not ready.")

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.indexing_status = Status()
    manager.modern_runtime = object()
    manager.logger = logging.getLogger("startup-reindex-test")
    manager.modern_index_status_text = ""

    assert manager._index_modern_file(str(tmp_path / "file.txt")) is False
    assert manager.modern_index_status_text == (
        "Modern indexing unavailable: Ollama embedding profile is not ready."
    )


def test_manager_delegates_available_reindex_to_threaded_ui_status(
    tmp_path, monkeypatch
) -> None:
    started: list[str] = []

    class Available:
        def index_file(self, path: str) -> None:
            raise AssertionError("the UI thread must not perform modern indexing")

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.modern_runtime = Available()
    manager.indexing_status = SimpleNamespace(start=started.append)
    manager.logger = logging.getLogger("startup-reindex-success-test")
    manager.modern_index_status_text = ""
    path = str(tmp_path / "file.txt")

    assert manager._index_modern_file(path) is True
    assert started == [path]
    assert manager.modern_index_status_text == "Modern indexing started."


def test_manager_never_falls_back_to_synchronous_modern_index(
    tmp_path, monkeypatch
) -> None:
    calls: list[str] = []

    class SynchronousOnly:
        def index_file(self, path: str):
            calls.append(path)
            return SimpleNamespace(status="completed")

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.modern_runtime = SynchronousOnly()
    manager.logger = logging.getLogger("startup-reindex-failed-test")
    manager.modern_index_status_text = ""

    assert manager._index_modern_file(str(tmp_path / "file.txt")) is False
    assert calls == []


def test_manager_never_calls_runtime_async_when_status_widget_is_missing(
    tmp_path, monkeypatch
) -> None:
    calls: list[str] = []

    class AsyncOnly:
        def index_file_async(self, path: str) -> None:
            calls.append(path)
            raise AssertionError("the manager must use the IndexingStatus QThread")

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.modern_runtime = AsyncOnly()
    manager.logger = logging.getLogger("startup-reindex-missing-status-test")
    manager.modern_index_status_text = ""
    path = str(tmp_path / "file.txt")

    assert manager._index_modern_file(path) is False
    assert calls == []
    assert manager.modern_index_status_text == (
        "Modern indexing unavailable: indexing status UI is not initialized."
    )


@pytest.mark.parametrize("had_existing_tags", [False, True])
def test_tag_suggestion_never_uses_legacy_vector_search_for_indexing(
    monkeypatch, had_existing_tags: bool
) -> None:
    from types import ModuleType

    from file_tagger import tag_suggestion

    class ItemList:
        def __init__(self, selected: bool) -> None:
            self.selected = selected

        def selectedItems(self) -> list[object]:
            return [object()] if self.selected else []

        def itemWidget(self, item: object) -> SimpleNamespace:
            del item
            return SimpleNamespace(tag_name="research")

    class FileRecord:
        def __init__(self) -> None:
            self.tags = (
                [SimpleNamespace(name="existing", color="#111111")]
                if had_existing_tags
                else []
            )

    class Session:
        def __init__(self) -> None:
            self.file = FileRecord()
            self.tag = SimpleNamespace(name="research", color="#123456")
            self.queries = 0

        def query(self, model: Any):
            del model
            self.queries += 1
            selected = self.file if self.queries == 1 else self.tag
            return SimpleNamespace(
                filter_by=lambda **kwargs: SimpleNamespace(
                    first=lambda: selected
                )
            )

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            raise AssertionError("tag application unexpectedly failed")

        def add(self, value: object) -> None:
            del value

    legacy_calls: list[str] = []
    queued: list[str] = []
    accepted: list[bool] = []
    runtime_syncs: list[str] = []

    class LegacySearch:
        def index_file(self, path: str, content: str) -> None:
            del content
            legacy_calls.append(path)

        def update_metadata(self, path: str) -> None:
            legacy_calls.append(path)

    class Runtime:
        def sync_file(self, path: str, tags: Any) -> None:
            del tags
            runtime_syncs.append(path)

    parent = SimpleNamespace(
        vector_search=LegacySearch(),
        _index_modern_file=lambda path: queued.append(path),
    )
    session = Session()
    dialog = SimpleNamespace(
        existing_list=ItemList(True),
        new_list=ItemList(False),
        file_path="C:\\library\\guide.md",
        db_session=session,
        modern_runtime=Runtime(),
        config=SimpleNamespace(),
        parent=lambda: parent,
        accept=lambda: accepted.append(True),
    )
    extractor_module = ModuleType("file_tagger.vector_search.content_extractor")

    class CompatExtractor:
        @staticmethod
        def extract_file_content_async(
            path, complete, progress, *, pdf_extractor
        ) -> None:
            del path, progress, pdf_extractor
            complete({"content": "guide"})

    extractor_module.ContentExtractor = CompatExtractor
    monkeypatch.setitem(
        sys.modules,
        "file_tagger.vector_search.content_extractor",
        extractor_module,
    )

    tag_suggestion.TagSuggestionDialog.apply_tags(dialog)

    assert runtime_syncs == ["C:\\library\\guide.md"]
    assert queued == ["C:\\library\\guide.md"]
    assert legacy_calls == []
    assert accepted == [True]


def test_runtime_indexes_queued_files_through_one_worker(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = initialize_modern_runtime(
        paths,
        ReloadableConfig(paths),
        "password",
        start_readiness=False,
    )
    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    runtime.sync_file(str(first), ())
    runtime.sync_file(str(second), ())
    calls: list[str] = []
    worker_ids: set[int] = set()

    class RecordingService:
        def __init__(self) -> None:
            self._next_job_id = 1

        def queue(self, path: str) -> int:
            job_id = self._next_job_id
            self._next_job_id += 1
            return job_id

        def reindex(
            self,
            path: str,
            *,
            cancellation_check: Any = None,
            job_id: int | None = None,
        ):
            del cancellation_check, job_id
            calls.append(path)
            worker_ids.add(threading.get_ident())
            return SimpleNamespace(status="completed")

    runtime._indexing_service = RecordingService()

    runtime.index_file_async(first)
    runtime.index_file_async(second)
    runtime._index_queue.join()

    assert calls == [str(first.resolve()), str(second.resolve())]
    assert len(worker_ids) == 1


@pytest.mark.parametrize("deletion_method", ("delete_file", "sync_legacy_changes"))
def test_runtime_removes_pending_queue_entry_when_file_is_deleted(
    tmp_path, monkeypatch, deletion_method: str
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = initialize_modern_runtime(
        paths,
        ReloadableConfig(paths),
        "password",
        start_readiness=False,
    )
    first = tmp_path / "first.txt"
    deleted = tmp_path / "deleted.txt"
    first.write_text("first", encoding="utf-8")
    deleted.write_text("deleted", encoding="utf-8")
    runtime.sync_file(str(first), ())
    runtime.sync_file(str(deleted), ())
    first_started = Event()
    release_first = Event()
    calls: list[str] = []

    class BlockingService:
        def __init__(self) -> None:
            self._next_job_id = 1

        def queue(self, path: str) -> int:
            job_id = self._next_job_id
            self._next_job_id += 1
            return job_id

        def reindex(
            self,
            path: str,
            *,
            cancellation_check: Any = None,
            job_id: int | None = None,
        ):
            del cancellation_check, job_id
            calls.append(str(path))
            if Path(path).resolve() == first.resolve():
                first_started.set()
                assert release_first.wait(5)
            return SimpleNamespace(status="completed")

    runtime._indexing_service = BlockingService()

    runtime.index_file_async(first)
    assert first_started.wait(5)
    runtime.index_file_async(deleted)
    if deletion_method == "delete_file":
        runtime.delete_file(str(deleted))
    else:
        runtime.sync_legacy_changes(file_deletes=(str(deleted),))
    release_first.set()
    runtime._index_queue.join()

    assert calls == [str(first.resolve())]
    assert runtime.session_factory is not None
    with runtime.session_factory() as session:
        assert session.scalar(
            select(File).where(File.canonical_path == str(deleted.resolve()))
        ) is None


def test_runtime_deletion_during_worker_does_not_resurrect_file(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = initialize_modern_runtime(
        paths,
        ReloadableConfig(paths),
        "password",
        start_readiness=False,
    )
    source = tmp_path / "racing.txt"
    source.write_text("source", encoding="utf-8")
    runtime.sync_file(str(source), ())
    worker_started = Event()
    release_worker = Event()

    class QueueRaceService:
        def queue(self, path: str) -> int:
            return 1

        def reindex(
            self,
            path: str,
            *,
            cancellation_check: Any = None,
            job_id: int | None = None,
        ):
            del cancellation_check
            worker_started.set()
            assert release_worker.wait(5)
            if job_id is None:
                runtime.sync_file(path, ())
            return SimpleNamespace(status="completed")

    runtime._indexing_service = QueueRaceService()

    runtime.index_file_async(source)
    assert worker_started.wait(5)
    runtime.delete_file(str(source))
    release_worker.set()
    runtime._index_queue.join()

    assert runtime.session_factory is not None
    with runtime.session_factory() as session:
        assert session.scalar(
            select(File).where(File.canonical_path == str(source.resolve()))
        ) is None


def test_runtime_deletion_cancels_slow_active_embedding(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = initialize_modern_runtime(
        paths,
        ReloadableConfig(paths),
        "password",
        start_readiness=False,
    )
    source = tmp_path / "active.txt"
    source.write_text("source text", encoding="utf-8")
    runtime.sync_file(str(source), ())
    assert runtime.session_factory is not None
    document = ExtractedDocument(
        file_path=str(source.resolve()),
        file_hash=sha256(source.read_bytes()).hexdigest(),
        parser_version="test",
        status="complete",
        pages=(
            ExtractedPage(
                number=1,
                ordinal=0,
                text="source text",
                markdown="source text",
            ),
        ),
    )

    class Extractor:
        def extract(self, file_path: str) -> ExtractedDocument:
            assert Path(file_path) == source.resolve()
            return document

    embed_started = Event()
    release_embed = Event()

    class BlockingGateway:
        async def embed(
            self,
            model: str,
            text: str,
            *,
            dimensions: int | None = None,
            truncate: bool = False,
            keep_alive: str = "5m",
        ) -> tuple[tuple[float, ...],]:
            del model, truncate, keep_alive
            assert text == "source text"
            embed_started.set()
            assert release_embed.wait(5)
            assert dimensions is not None
            return (tuple(0.0 for _ in range(dimensions)),)

    repository = IndexingRepository(runtime.session_factory)
    checkpoints_after_delete = 0
    deleted = False
    original_checkpoint = repository.checkpoint

    def tracked_checkpoint(*args, **kwargs):
        nonlocal checkpoints_after_delete
        if deleted:
            checkpoints_after_delete += 1
        return original_checkpoint(*args, **kwargs)

    repository.checkpoint = tracked_checkpoint  # type: ignore[method-assign]
    service = IndexingService(
        repository,
        Extractor(),
        embedding_gateway=BlockingGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
        recover_on_startup=False,
    )
    active_cancellation_check = None
    original_reindex = service.reindex

    def tracked_reindex(
        path: str,
        *,
        cancellation_check: Any = None,
        job_id: int | None = None,
    ):
        nonlocal active_cancellation_check
        active_cancellation_check = cancellation_check
        return original_reindex(
            path,
            cancellation_check=cancellation_check,
            job_id=job_id,
        )

    service.reindex = tracked_reindex  # type: ignore[method-assign]
    runtime._indexing_service = service

    runtime.index_file_async(source)
    assert embed_started.wait(5)
    runtime.delete_file(str(source))
    deleted = True
    cancellation_detected = bool(
        active_cancellation_check is not None and active_cancellation_check()
    )
    release_embed.set()
    runtime._index_queue.join()

    assert cancellation_detected
    assert checkpoints_after_delete == 0
    with runtime.session_factory() as session:
        assert session.scalar(
            select(File).where(File.canonical_path == str(source.resolve()))
        ) is None
        vector_count = session.execute(
            text("SELECT COUNT(*) FROM chunk_embeddings_256")
        ).scalar_one()
        assert vector_count == 0


def test_legacy_commit_failure_does_not_touch_modern_database(monkeypatch) -> None:
    events: list[str] = []

    class LegacySession:
        def commit(self) -> None:
            events.append("legacy-commit")
            raise RuntimeError("commit failed")

    class Modern:
        def sync_tag(self, name: str, color: str | None) -> None:
            events.append(f"modern-tag:{name}")

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.db_session = LegacySession()
    manager.modern_runtime = Modern()
    manager.logger = logging.getLogger("startup-commit-failure-test")

    try:
        manager._commit_legacy_changes(tag_upserts=[("research", "#123456")])
    except RuntimeError:
        pass
    else:
        raise AssertionError("The legacy commit error should propagate.")

    assert events == ["legacy-commit"]


def test_modern_runtime_sync_updates_tags_and_exact_file_associations(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = initialize_modern_runtime(
        paths,
        ReloadableConfig(paths),
        "password",
        start_readiness=False,
    )

    runtime.sync_tag("research", "#123456")
    runtime.sync_tag("reference", "#abcdef")
    runtime.sync_file(
        r"C:\library\a.pdf",
        [("research", "#123456"), ("reference", "#abcdef")],
    )
    runtime.sync_file(
        r"C:\library\a.pdf",
        [("reference", "#999999")],
    )
    runtime.delete_tag("reference")

    assert runtime.session_factory is not None
    with runtime.session_factory() as session:
        file = session.scalar(select(File))
        assert file is not None
        assert [row[0] for row in session.execute(
            select(Tag.name).join(FileTag).where(FileTag.file_id == file.id)
        )] == []
        assert session.scalar(select(Tag).where(Tag.name == "reference")) is None


def test_readiness_thread_does_not_block_runtime_initialization(
    tmp_path, monkeypatch
) -> None:
    started: list[bool] = []
    paths = _paths(tmp_path, monkeypatch)

    class NonBlockingRuntime(ModernRuntime):
        def start_readiness(self) -> None:
            started.append(True)

    monkeypatch.setattr(
        "file_tagger.modern_context.ModernRuntime",
        NonBlockingRuntime,
    )

    runtime = initialize_modern_runtime(
        paths,
        ReloadableConfig(paths),
        "password",
    )

    assert runtime is not None
    assert started == [True]


def test_runtime_persists_model_selections_and_refreshes(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = ModernRuntime(paths, object())  # type: ignore[arg-type]
    refreshed: list[RequiredModels | None] = []
    profiles = (
        ModelProfile(
            role="embedding",
            model="embeddinggemma-2:440m",
            digest="embedding",
            context_length=8192,
            dimensions=512,
            capabilities=("embedding",),
        ),
        ModelProfile(
            role="decision",
            model="tev1:4b",
            digest="decision",
            context_length=8192,
            capabilities=("systemone",),
        ),
        ModelProfile(
            role="generation",
            model="gemma4:e2b",
            digest="generation",
            context_length=8192,
            capabilities=("completion",),
        ),
    )
    runtime.readiness = ReadinessReport(
        ready=True,
        profiles=profiles,
        ollama=OllamaReadiness(
            available=True,
            version="0.35.0",
            base_url="http://127.0.0.1:11434",
        ),
    )

    def start_readiness():
        refreshed.append(runtime.required_models)
        return None

    monkeypatch.setattr(runtime, "start_readiness", start_readiness)
    selected = {
        "embedding": "embeddinggemma-2:440m",
        "decision": "tev1:4b",
        "generation": "gemma4:e2b",
    }

    runtime.apply_model_selections(selected)
    restarted = ModernRuntime(paths, object())  # type: ignore[arg-type]

    assert refreshed == [
        RequiredModels(
            embedding="embeddinggemma-2:440m",
            decision="tev1:4b",
            generation="gemma4:e2b",
        )
    ]
    assert restarted.model_overrides == selected


def test_runtime_resolves_active_profile_by_selection_not_catalog_order(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = ModernRuntime(paths, object())  # type: ignore[arg-type]
    selected = RequiredModels(
        embedding="embedding-a",
        decision="decision-a",
        generation="generation-a",
    )
    embedding_a = ModelProfile(
        role="embedding",
        model="embedding-a",
        digest="embedding-a-digest",
        context_length=8192,
        dimensions=256,
        capabilities=("embedding",),
    )
    embedding_b = ModelProfile(
        role="embedding",
        model="embedding-b",
        digest="embedding-b-digest",
        context_length=8192,
        dimensions=512,
        capabilities=("embedding",),
    )
    runtime.required_models = selected
    runtime.readiness = ReadinessReport(
        ready=True,
        profiles=(embedding_a, embedding_b),
        selected_models=selected,
    )

    assert runtime.embedding_profile == embedding_a
    assert runtime.embedding_profile != embedding_b


def test_runtime_does_not_expose_blocked_embedding_profile(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = ModernRuntime(paths, object())  # type: ignore[arg-type]
    selected = RequiredModels(
        embedding="embedding-unsupported",
        decision="decision-a",
        generation="generation-a",
    )
    runtime.required_models = selected
    runtime.readiness = ReadinessReport(
        ready=False,
        blocked_roles=("indexing", "search"),
        blocked_model_roles=("embedding",),
        profiles=(
            ModelProfile(
                role="embedding",
                model="embedding-unsupported",
                digest="embedding-unsupported-digest",
                context_length=8192,
                dimensions=512,
                capabilities=("embedding",),
            ),
        ),
        selected_models=selected,
    )

    assert runtime.embedding_profile is None


def test_modern_runtime_search_uses_active_profile_and_embedding_gateway(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    session = SimpleNamespace(close=lambda: None)

    class Factory:
        def __call__(self):
            return session

    runtime = ModernRuntime(paths, Factory())  # type: ignore[arg-type]
    profile = ModelProfile(
        role="embedding",
        model="embedding-active",
        digest="embedding-active-digest",
        context_length=8192,
        dimensions=256,
        capabilities=("embedding",),
    )
    selected = RequiredModels(
        embedding=profile.model,
        decision="decision-active",
        generation="generation-active",
    )
    runtime.required_models = selected
    runtime.readiness = ReadinessReport(
        ready=True,
        profiles=(profile,),
        selected_models=selected,
    )
    gateway = object()
    runtime._embedding_gateway = gateway
    calls: list[tuple[Any, ...]] = []
    expected = (SimpleNamespace(path="C:\\library\\guide.md"),)

    class FakeHybridSearch:
        def __init__(self, selected_session, *, embedding_profile):
            calls.append(("hybrid", selected_session, embedding_profile))

    monkeypatch.setattr(
        "file_tagger.modern_context.HybridSearch",
        FakeHybridSearch,
    )

    def execute(
        plan,
        search,
        *,
        embedding_gateway,
        embedding_profile,
        model_digest,
        index_version,
        per_query_limit,
        limit,
        max_sections_per_file,
    ):
        calls.append(
            (
                "execute",
                plan,
                search,
                embedding_gateway,
                embedding_profile,
                model_digest,
                index_version,
                per_query_limit,
                limit,
                max_sections_per_file,
            )
        )
        return expected

    monkeypatch.setattr(
        "file_tagger.modern_context.execute_search_plan",
        execute,
    )
    monkeypatch.setattr(
        "file_tagger.modern_context.merge_results",
        lambda values, *, limit: tuple(values),
    )

    results = runtime.search_content(
        "query",
        tags=("research",),
        use_and=True,
        limit=7,
    )

    assert results == expected
    assert calls[0] == ("hybrid", session, profile)
    assert calls[1][3:5] == (gateway, profile)
    assert calls[1][5:7] == (
        profile.digest,
        profile.default_index_version,
    )


def test_modern_runtime_reuses_contextual_rag_service_for_conversation(
    tmp_path,
    monkeypatch,
) -> None:
    created = []

    class FakeSession:
        def close(self) -> None:
            pass

    class FakeFactory:
        def __call__(self):
            return FakeSession()

    class FakeSearch:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class FakeRag:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            created.append(self)

    monkeypatch.setattr("file_tagger.modern_context.HybridSearch", FakeSearch)
    monkeypatch.setattr("file_tagger.modern_context.RagService", FakeRag)

    runtime = ModernRuntime(_paths(tmp_path, monkeypatch), FakeFactory())
    profile = ModelProfile(
        role="embedding",
        model="embedding-active",
        digest="sha256:embedding",
        context_length=4096,
        capabilities=("embedding",),
    )
    generation = ModelProfile(
        role="generation",
        model="generation-active",
        digest="sha256:generation",
        context_length=4096,
        capabilities=("completion",),
    )

    first_id, first = runtime._get_contextual_rag_service(
        "conversation-1",
        profile=profile,
        generation_profile=generation,
    )
    second_id, second = runtime._get_contextual_rag_service(
        "conversation-1",
        profile=profile,
        generation_profile=generation,
    )

    assert first_id == second_id == "conversation-1"
    assert first is second
    assert len(created) == 1


def test_modern_runtime_search_reports_unavailable_without_active_profile(
    tmp_path, monkeypatch
) -> None:
    runtime = ModernRuntime(_paths(tmp_path, monkeypatch), object())  # type: ignore[arg-type]

    with pytest.raises(ModernSearchUnavailable):
        runtime.search_content("query")


def test_manager_content_search_never_constructs_legacy_vector_search(
    qtbot, monkeypatch, tmp_path
) -> None:
    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    file_tag_manager = sys.modules["file_tagger.file_tag_manager"]
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    source = tmp_path / "search-fixture.txt"
    source.write_text("modern", encoding="utf-8")
    calls: list[tuple[str, tuple[str, ...], bool, int]] = []

    class Runtime:
        def search_content(self, query, *, tags=(), use_and=True, limit=20):
            calls.append((query, tuple(tags), use_and, limit))
            return (
                {
                    "path": str(source),
                    "score": 0.9,
                    "summary": "modern result",
                    "snippets": ["modern"],
                    "tags": [],
                },
            )

    manager.modern_runtime = Runtime()
    manager.vector_search = SimpleNamespace(
        search=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy VectorSearch must not be used")
        )
    )
    tab = manager.create_rag_search_tab()
    qtbot.addWidget(tab)
    manager.query_input.setText("modern")

    try:
        manager.search_by_content()
        qtbot.waitUntil(lambda: bool(calls))
        qtbot.waitUntil(lambda: bool(manager.current_search_results))
    finally:
        source.unlink(missing_ok=True)

    assert calls == [("modern", (), True, 20)]
    assert manager.current_search_results[0]["summary"] == "modern result"
    assert manager.rag_search_results.count() > 0


def test_manager_content_search_runs_runtime_work_on_qthread(
    qtbot, monkeypatch, tmp_path
) -> None:
    from PySide6.QtCore import QThread

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    file_tag_manager = sys.modules["file_tagger.file_tag_manager"]
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    source = tmp_path / "threaded-search.txt"
    source.write_text("modern", encoding="utf-8")
    entered = Event()
    release = Event()
    worker_threads: list[object] = []

    class Runtime:
        def search_content(
            self,
            query,
            *,
            tags=(),
            use_and=True,
            limit=20,
            cancellation_check=None,
        ):
            del query, tags, use_and, limit, cancellation_check
            worker_threads.append(QThread.currentThread())
            entered.set()
            assert release.wait(2)
            return (
                {
                    "path": str(source),
                    "score": 0.9,
                    "summary": "threaded result",
                    "snippets": ["modern"],
                    "tags": [],
                },
            )

    manager.modern_runtime = Runtime()
    manager.vector_search = SimpleNamespace(
        search=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy VectorSearch must not be used")
        )
    )
    tab = manager.create_rag_search_tab()
    qtbot.addWidget(tab)
    manager.query_input.setText("modern")

    manager.search_by_content()
    assert entered.wait(1)
    assert manager.current_search_results == []
    assert worker_threads[0] is not QThread.currentThread()

    release.set()
    qtbot.waitUntil(
        lambda: bool(manager.current_search_results)
        and manager.current_search_results[0]["summary"] == "threaded result"
    )


def test_manager_content_search_cancellation_does_not_publish_results(
    qtbot, monkeypatch
) -> None:
    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    file_tag_manager = sys.modules["file_tagger.file_tag_manager"]
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    entered = Event()

    class Runtime:
        def search_content(self, *args, cancellation_check=None, **kwargs):
            del args, kwargs
            entered.set()
            assert cancellation_check is not None
            while not cancellation_check():
                Event().wait(0.01)
            return ({"path": "cancelled"},)

    manager.modern_runtime = Runtime()
    tab = manager.create_rag_search_tab()
    qtbot.addWidget(tab)
    manager.query_input.setText("cancel")

    manager.search_by_content()
    assert entered.wait(1)
    manager.cancel_content_search()
    qtbot.waitUntil(lambda: manager._content_search_worker is None)

    assert manager.current_search_results == []
    assert manager.cancel_content_search_btn.isEnabled() is False
    assert manager.search_content_btn.isEnabled() is True


def test_manager_content_search_reports_modern_unavailable(
    qtbot, monkeypatch
) -> None:
    from file_tagger.file_tag_manager import FileTagManager

    manager = FileTagManager.__new__(FileTagManager)
    from PySide6.QtWidgets import QMainWindow

    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    warnings: list[str] = []
    tab = manager.create_rag_search_tab()
    qtbot.addWidget(tab)
    manager.modern_runtime = None
    manager.query_input.setText("modern")
    monkeypatch.setattr(
        "file_tagger.file_tag_manager.QMessageBox.warning",
        lambda parent, title, message: warnings.append(message),
    )

    manager.search_by_content()

    assert warnings
    assert "unavailable" in warnings[0].lower()
    assert manager.current_search_results == []


def test_runtime_blocks_search_until_profile_full_reindex_is_complete(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    database = tmp_path / "runtime-profile-reindex.sqlite3"
    upgrade_database(database)
    session_factory = create_session_factory(database)
    source = tmp_path / "old-index.md"
    source.write_text("source", encoding="utf-8")
    old_profile = ModelProfile(
        role="embedding",
        model="embedding-old",
        digest="sha256:old",
        context_length=1_000,
        dimensions=256,
    )
    new_profile = ModelProfile(
        role="embedding",
        model="embedding-new",
        digest="sha256:new",
        context_length=1_000,
        dimensions=512,
    )

    class OldGateway:
        async def embed(self, *args, **kwargs):
            del args, kwargs
            return (tuple(float(index) for index in range(256)),)

    old_service = IndexingService(
        IndexingRepository(session_factory),
        extractor=SimpleNamespace(
            extract=lambda path: ExtractedDocument(
                file_path=str(path),
                file_hash=sha256(Path(path).read_bytes()).hexdigest(),
                parser="test",
                parser_version="1",
                status="complete",
                pages=(
                    ExtractedPage(
                        number=1,
                        ordinal=0,
                        text="old indexed content",
                        markdown="old indexed content",
                    ),
                ),
            )
        ),
        embedding_gateway=OldGateway(),
        embedding_profile=old_profile,
    )
    assert old_service.reindex(source).status == "completed"

    runtime = ModernRuntime(paths, session_factory)
    runtime.required_models = RequiredModels(
        embedding=new_profile.model,
        decision="decision",
        generation="generation",
    )
    runtime.readiness = ReadinessReport(
        ready=True,
        profiles=(new_profile,),
        selected_models=runtime.required_models,
    )
    runtime._embedding_gateway = object()

    with pytest.raises(ModernSearchUnavailable, match="reindex"):
        runtime.search_content("old indexed content")
    assert runtime.readiness is not None
    assert runtime.readiness.ready is False
    assert "search" in runtime.readiness.blocked_roles


def test_runtime_readiness_completion_ignores_stale_overlapping_result(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = ModernRuntime(paths, object())  # type: ignore[arg-type]
    started = (Event(), Event())
    release = (Event(), Event())
    call_count = 0
    callbacks: list[ReadinessReport | None] = []

    hardware = HardwareReport(
        cpu_architecture="AMD64",
        logical_processors=8,
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=24 * 1024**3,
        available_disk_bytes=100 * 1024**3,
    )
    monkeypatch.setattr(
        "file_tagger.modern_context.profile_hardware",
        lambda: hardware,
    )
    monkeypatch.setattr(
        "file_tagger.modern_context.IndexingService",
        lambda *args, **kwargs: SimpleNamespace(
            recover_abandoned=lambda **kwargs: ()
        ),
    )

    async def check_readiness(gateway, required):
        nonlocal call_count
        call_index = call_count
        call_count += 1
        started[call_index].set()
        release[call_index].wait()
        return ReadinessReport(
            ready=True,
            profiles=(
                ModelProfile(
                    role="embedding",
                    model=required.embedding,
                    digest=f"embedding-{call_index}",
                    context_length=8192,
                    dimensions=768,
                    capabilities=("embedding",),
                ),
            ),
            ollama=OllamaReadiness(
                available=True,
                version="0.35.0",
                base_url="http://127.0.0.1:11434",
            ),
            selected_models=required,
            warnings=(f"result-{call_index}",),
        )

    monkeypatch.setattr(runtime, "_check_readiness", check_readiness)

    first = runtime.start_readiness(callbacks.append)
    assert started[0].wait(2)
    second = runtime.start_readiness(callbacks.append)
    assert started[1].wait(2)

    release[1].set()
    assert second.result(timeout=2) is runtime.readiness
    release[0].set()
    assert first.result(timeout=2) is None

    assert runtime.readiness is not None
    assert runtime.readiness.warnings == ("result-1",)
    assert callbacks[-1] is None
    assert any(report and report.warnings == ("result-1",) for report in callbacks)


def test_overlapping_readiness_only_recovers_published_service(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = ModernRuntime(paths, object())  # type: ignore[arg-type]
    started = (Event(), Event())
    release = (Event(), Event())
    call_count = 0
    recovered_digests: list[str] = []

    hardware = HardwareReport(
        cpu_architecture="AMD64",
        logical_processors=8,
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=24 * 1024**3,
        available_disk_bytes=100 * 1024**3,
    )
    monkeypatch.setattr(
        "file_tagger.modern_context.profile_hardware",
        lambda: hardware,
    )

    class FakeIndexingService:
        def __init__(
            self,
            *args,
            embedding_profile,
            recover_on_startup=True,
            **kwargs,
        ) -> None:
            self.embedding_profile = embedding_profile
            if recover_on_startup:
                self.recover_abandoned()

        def recover_abandoned(self, *, claim_guard=None) -> tuple[()]:
            assert claim_guard is None or claim_guard()
            recovered_digests.append(self.embedding_profile.digest)
            return ()

    monkeypatch.setattr(
        "file_tagger.modern_context.IndexingService",
        FakeIndexingService,
    )

    async def check_readiness(gateway, required):
        nonlocal call_count
        call_index = call_count
        call_count += 1
        started[call_index].set()
        release[call_index].wait()
        return ReadinessReport(
            ready=True,
            profiles=(
                ModelProfile(
                    role="embedding",
                    model=required.embedding,
                    digest=f"embedding-{call_index}",
                    context_length=8192,
                    dimensions=768,
                    capabilities=("embedding",),
                ),
            ),
            ollama=OllamaReadiness(
                available=True,
                version="0.35.0",
                base_url="http://127.0.0.1:11434",
            ),
            selected_models=required,
        )

    monkeypatch.setattr(runtime, "_check_readiness", check_readiness)

    first = runtime.start_readiness()
    assert started[0].wait(2)
    second = runtime.start_readiness()
    assert started[1].wait(2)

    release[1].set()
    assert second.result(timeout=2) is runtime.readiness
    release[0].set()
    assert first.result(timeout=2) is None

    assert recovered_digests == ["embedding-1"]


def test_slow_recovery_does_not_hold_modern_runtime_lock(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    runtime = ModernRuntime(paths, object())  # type: ignore[arg-type]
    recovery_started = Event()
    release_recovery = Event()
    lock_probe_finished = Event()
    hardware = HardwareReport(
        cpu_architecture="AMD64",
        logical_processors=8,
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=24 * 1024**3,
        available_disk_bytes=100 * 1024**3,
    )
    monkeypatch.setattr(
        "file_tagger.modern_context.profile_hardware",
        lambda: hardware,
    )

    class SlowRecoveryService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def recover_abandoned(self, *, claim_guard=None) -> tuple[()]:
            assert claim_guard is None or claim_guard()
            recovery_started.set()
            assert release_recovery.wait(5)
            return ()

    monkeypatch.setattr(
        "file_tagger.modern_context.IndexingService",
        SlowRecoveryService,
    )

    async def check_readiness(gateway, required):
        return ReadinessReport(
            ready=True,
            profiles=(
                ModelProfile(
                    role="embedding",
                    model=required.embedding,
                    digest="embedding",
                    context_length=8192,
                    dimensions=768,
                    capabilities=("embedding",),
                ),
            ),
            ollama=OllamaReadiness(
                available=True,
                version="0.35.0",
                base_url="http://127.0.0.1:11434",
            ),
            selected_models=required,
        )

    monkeypatch.setattr(runtime, "_check_readiness", check_readiness)
    completion = runtime.start_readiness()
    assert recovery_started.wait(2)

    def inspect_published_runtime() -> None:
        assert runtime.selected_models is not None
        lock_probe_finished.set()

    probe = threading.Thread(target=inspect_published_runtime)
    probe.start()
    try:
        assert lock_probe_finished.wait(0.5)
    finally:
        release_recovery.set()
        probe.join(2)
        completion.result(timeout=2)


def test_initialization_worker_exposes_modern_runtime_to_login(
    monkeypatch,
) -> None:
    from types import ModuleType

    vector_search_module = ModuleType("file_tagger.vector_search")
    vector_search_module.VectorSearch = lambda session, config: (_ for _ in ()).throw(
        AssertionError("startup must not construct legacy VectorSearch")
    )
    monkeypatch.setitem(sys.modules, "file_tagger.vector_search", vector_search_module)
    from file_tagger import initialization

    legacy_session = object()
    config = SimpleNamespace(paths=SimpleNamespace())
    runtime = object()
    monkeypatch.setattr(initialization, "init_db", lambda: legacy_session)
    monkeypatch.setattr(initialization, "Config", lambda password: config)
    monkeypatch.setattr(
        initialization,
        "initialize_modern_runtime",
        lambda paths, active_config, password: runtime,
    )
    monkeypatch.setattr(initialization.time, "sleep", lambda duration: None)
    results = []

    worker = initialization.InitializationWorker("password")
    worker.finished_signal.connect(results.append)
    worker.run()

    assert len(results) == 1
    assert results[0] == (legacy_session, config, None, runtime)


def test_login_accepts_modern_runtime_without_legacy_search(
    qtbot, monkeypatch
) -> None:
    from types import ModuleType

    from PySide6.QtWidgets import QWidget

    from file_tagger.login import LoginScreen

    constructed: list[tuple[object, object, object, object]] = []

    class Window(QWidget):
        def __init__(self, *args: object) -> None:
            super().__init__()
            constructed.append((args[0], args[1], args[2], args[3]))

    manager_module = ModuleType("file_tagger.file_tag_manager")
    manager_module.FileTagManager = Window
    monkeypatch.setitem(sys.modules, "file_tagger.file_tag_manager", manager_module)
    screen = LoginScreen()
    qtbot.addWidget(screen)
    monkeypatch.setattr(screen, "hide", lambda: None)
    monkeypatch.setattr(screen, "close", lambda: None)
    modern_runtime = object()
    legacy_session = object()
    config = object()

    screen.on_initialization_finished(
        (legacy_session, config, None, modern_runtime)
    )

    assert constructed == [
        (legacy_session, config, None, modern_runtime)
    ]
