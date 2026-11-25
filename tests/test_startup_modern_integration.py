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
from file_tagger.modern_context import (
    ModernIndexingUnavailable,
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
    class Unavailable:
        def index_file(self, path: str) -> None:
            raise ModernIndexingUnavailable("Ollama embedding profile is not ready.")

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.modern_runtime = Unavailable()
    manager.logger = logging.getLogger("startup-reindex-test")
    manager.modern_index_status_text = ""

    assert manager._index_modern_file(str(tmp_path / "file.txt")) is False
    assert manager.modern_index_status_text == (
        "Modern indexing unavailable: Ollama embedding profile is not ready."
    )


def test_manager_delegates_available_reindex_to_modern_service(
    tmp_path, monkeypatch
) -> None:
    indexed: list[str] = []

    class Available:
        def index_file(self, path: str) -> None:
            indexed.append(path)

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.modern_runtime = Available()
    manager.logger = logging.getLogger("startup-reindex-success-test")
    manager.modern_index_status_text = ""
    path = str(tmp_path / "file.txt")

    assert manager._index_modern_file(path) is True
    assert indexed == [path]
    assert manager.modern_index_status_text == "Modern index updated."


def test_manager_reports_failed_modern_index_result(tmp_path, monkeypatch) -> None:
    class Failed:
        def index_file(self, path: str):
            return SimpleNamespace(status="failed")

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.modern_runtime = Failed()
    manager.logger = logging.getLogger("startup-reindex-failed-test")
    manager.modern_index_status_text = ""

    assert manager._index_modern_file(str(tmp_path / "file.txt")) is False
    assert "ended with status failed" in manager.modern_index_status_text


def test_manager_queues_modern_index_work_without_calling_sync_path(
    tmp_path, monkeypatch
) -> None:
    queued: list[str] = []

    class Queued:
        def index_file_async(self, path: str) -> None:
            queued.append(path)

        def index_file(self, path: str) -> None:
            raise AssertionError("the UI thread must not perform modern indexing")

    manager_class = _manager_class(monkeypatch)
    manager = manager_class.__new__(manager_class)
    manager.modern_runtime = Queued()
    manager.logger = logging.getLogger("startup-reindex-queued-test")
    manager.modern_index_status_text = ""
    path = str(tmp_path / "file.txt")

    assert manager._index_modern_file(path) is True
    assert queued == [path]
    assert manager.modern_index_status_text == "Modern indexing queued."


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
        lambda *args, **kwargs: object(),
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


def test_initialization_worker_exposes_modern_runtime_to_login(
    monkeypatch,
) -> None:
    from types import ModuleType

    vector_search_module = ModuleType("file_tagger.vector_search")
    vector_search_module.VectorSearch = lambda session, config: object()
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
    assert results[0] == (legacy_session, config, results[0][2], runtime)
