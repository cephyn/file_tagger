import logging
from threading import Event
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMessageBox

from file_tagger.ui.indexing_status import IndexingStatus


class CancellationService:
    cancel_requested = False
    previous_revision_available = True


def test_index_cancel_does_not_delete_previous_revision(qtbot) -> None:
    service = CancellationService()
    status = IndexingStatus(service)
    qtbot.addWidget(status)

    status.cancel()

    assert service.cancel_requested is True
    assert service.previous_revision_available is True


def test_indexing_runs_off_the_ui_thread_and_commits_after_success(qtbot) -> None:
    entered = Event()
    release = Event()

    class Service:
        cancel_requested = False
        previous_revision_available = True

        def queue(self, path: str) -> int:
            assert path == "C:\\library\\guide.pdf"
            return 7

        def state(self, job_id: int):
            assert job_id == 7
            return SimpleNamespace(
                job_id=job_id,
                status="running",
                file_path="C:\\library\\guide.pdf",
                revision=None,
                total_count=2,
                processed_count=1,
                failed_count=0,
                retry_count=0,
                failure=None,
            )

        def resume(self, job_id: int, *, cancellation_check=None):
            entered.set()
            assert release.wait(2)
            assert cancellation_check is not None
            return SimpleNamespace(
                job_id=job_id,
                status="completed",
                file_path="C:\\library\\guide.pdf",
                revision=2,
                total_count=2,
                processed_count=2,
                failed_count=0,
                retry_count=0,
                failure=None,
            )

    status = IndexingStatus(Service())
    qtbot.addWidget(status)

    assert status.start("C:\\library\\guide.pdf") == 7
    assert entered.wait(1)
    assert "embed" in status.phase_label.text().lower()

    release.set()
    qtbot.waitUntil(
        lambda: "completed" in status.phase_label.text().lower()
    )

    assert status.progress_bar.value() == 100
    assert status.previous_revision_available is True


def test_indexing_queues_a_second_ui_request_while_worker_is_running(qtbot) -> None:
    release_first = Event()
    first_started = Event()
    resumed: list[int] = []

    class Service:
        cancel_requested = False
        previous_revision_available = True

        def __init__(self) -> None:
            self.next_job_id = 1

        def queue(self, path: str) -> int:
            del path
            job_id = self.next_job_id
            self.next_job_id += 1
            return job_id

        def state(self, job_id: int):
            return SimpleNamespace(
                status="running",
                total_count=1,
                processed_count=0,
            )

        def resume(self, job_id: int, *, cancellation_check=None):
            del cancellation_check
            resumed.append(job_id)
            if job_id == 1:
                first_started.set()
                assert release_first.wait(2)
            return SimpleNamespace(status="completed")

    status = IndexingStatus(Service())
    qtbot.addWidget(status)

    assert status.start("C:\\library\\one.md") == 1
    assert first_started.wait(1)
    assert status.start("C:\\library\\two.md") is None

    release_first.set()
    qtbot.waitUntil(lambda: resumed == [1, 2])


def test_indexing_status_does_not_call_runtime_async_queue(qtbot) -> None:
    calls: list[str] = []

    class AsyncOnlyService:
        def index_file_async(self, path: str) -> None:
            calls.append(path)
            raise AssertionError("indexing must run in the QThread worker")

    status = IndexingStatus(AsyncOnlyService())
    qtbot.addWidget(status)

    with pytest.raises(
        RuntimeError, match="Modern indexing is not available"
    ):
        status.start("C:\\library\\guide.md")

    assert calls == []


def test_stale_qthread_callbacks_do_not_clear_replacement_worker(qtbot) -> None:
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    release_second = Event()
    completed: list[int] = []

    class Service:
        cancel_requested = False
        previous_revision_available = True

        def __init__(self) -> None:
            self.next_job_id = 1

        def queue(self, path: str) -> int:
            del path
            job_id = self.next_job_id
            self.next_job_id += 1
            return job_id

        def state(self, job_id: int):
            return SimpleNamespace(
                status="running",
                total_count=1,
                processed_count=0,
            )

        def resume(self, job_id: int, *, cancellation_check=None):
            del cancellation_check
            if job_id == 1:
                first_entered.set()
                assert release_first.wait(2)
                return SimpleNamespace(job_id=job_id, status="failed")
            second_entered.set()
            assert release_second.wait(2)
            completed.append(job_id)
            return SimpleNamespace(job_id=job_id, status="completed")

    status = IndexingStatus(Service())
    qtbot.addWidget(status)

    assert status.start("C:\\library\\first.md") == 1
    assert first_entered.wait(1)
    first_worker = status._worker
    assert first_worker is not None
    release_first.set()
    assert first_worker.wait(1_000)

    assert status.start("C:\\library\\second.md") == 2
    replacement = status._worker
    assert replacement is not None
    assert replacement is not first_worker
    assert second_entered.wait(1)

    status._worker_finished(first_worker, 1)
    assert status._worker is replacement
    assert "failed" not in status.phase_label.text().lower()

    release_second.set()
    qtbot.waitUntil(lambda: completed == [2])
    qtbot.waitUntil(
        lambda: "completed" in status.phase_label.text().lower()
    )


def test_shutdown_waits_for_real_qthread_and_manager_rejects_timeout(
    qtbot, monkeypatch
) -> None:
    entered = Event()
    release = Event()

    class Service:
        cancel_requested = False
        previous_revision_available = True

        def queue(self, path: str) -> int:
            del path
            return 9

        def resume(self, job_id: int, *, cancellation_check=None):
            entered.set()
            while not release.wait(0.01):
                if cancellation_check is not None and cancellation_check():
                    pass
            return SimpleNamespace(job_id=job_id, status="completed")

        def state(self, job_id: int):
            return SimpleNamespace(
                status="running",
                total_count=1,
                processed_count=0,
            )

    status = IndexingStatus(Service())
    qtbot.addWidget(status)
    status.start("C:\\library\\slow.md")
    assert entered.wait(1)

    assert status.shutdown(25) is False

    release.set()
    assert status.shutdown(1_000) is True
    assert status.wait(0) is True
    assert status._worker is None

    from file_tagger import file_tag_manager

    manager = file_tag_manager.FileTagManager.__new__(
        file_tag_manager.FileTagManager
    )
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.logger = logging.getLogger("indexing-shutdown-test")
    manager.indexing_status = SimpleNamespace(shutdown=lambda: False)
    event = QCloseEvent()

    manager.closeEvent(event)

    assert event.isAccepted() is False


def test_new_status_restores_and_resumes_persisted_latest_job(qtbot) -> None:
    resumed: list[int] = []
    persisted = SimpleNamespace(
        id=17,
        job_id=17,
        status="running",
        file_path="C:\\library\\persisted.md",
        total_count=4,
        processed_count=2,
        failed_count=1,
        retry_count=1,
        revision=1,
        failure={
            "code": "cancel_requested",
            "summary": "resume this checkpoint",
        },
    )

    class Service:
        cancel_requested = False
        previous_revision_available = True

        def latest_resumable_job(self):
            return persisted

        def resume(self, job_id: int, *, cancellation_check=None):
            resumed.append(job_id)
            return SimpleNamespace(
                job_id=job_id,
                status="completed",
                revision=2,
            )

    status = IndexingStatus(Service())
    qtbot.addWidget(status)

    assert status._job_id == 17
    assert status.resume_button.isEnabled()
    assert "cancelled" in status.phase_label.text().lower()
    assert status.resume() == 17
    qtbot.waitUntil(lambda: resumed == [17])


def test_failed_job_is_resumable_without_losing_previous_revision(qtbot) -> None:
    attempts = 0

    class Service:
        cancel_requested = False
        previous_revision_available = True

        def queue(self, path: str) -> int:
            del path
            return 3

        def state(self, job_id: int):
            return SimpleNamespace(
                job_id=job_id,
                status="failed",
                file_path="C:\\library\\guide.pdf",
                revision=1,
                total_count=3,
                processed_count=1,
                failed_count=1,
                retry_count=0,
                failure={
                    "code": "index_transaction_failed",
                    "summary": "The commit did not replace the previous revision.",
                },
            )

        def resume(self, job_id: int, *, cancellation_check=None):
            nonlocal attempts
            del cancellation_check
            attempts += 1
            return SimpleNamespace(
                job_id=job_id,
                status="failed" if attempts == 1 else "completed",
                file_path="C:\\library\\guide.pdf",
                revision=1,
                total_count=3,
                processed_count=1 if attempts == 1 else 3,
                failed_count=1,
                retry_count=attempts - 1,
                failure=(
                    {
                        "code": "index_transaction_failed",
                        "summary": "The commit did not replace the previous revision.",
                    }
                    if attempts == 1
                    else None
                ),
            )

    status = IndexingStatus(Service())
    qtbot.addWidget(status)
    status.start("C:\\library\\guide.pdf")
    qtbot.waitUntil(lambda: status.resume_button.isEnabled())

    status.resume()
    qtbot.waitUntil(
        lambda: "completed" in status.phase_label.text().lower()
    )

    assert attempts == 2
    assert status.progress_bar.value() == 100
    assert status.previous_revision_available is True


def test_manager_reindex_all_uses_modern_status_only(qtbot, monkeypatch) -> None:
    from file_tagger import file_tag_manager

    queued: list[list[str]] = []
    manager = file_tag_manager.FileTagManager.__new__(
        file_tag_manager.FileTagManager
    )
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.logger = logging.getLogger("indexing-status-test")
    manager.db_session = SimpleNamespace(
        query=lambda model: SimpleNamespace(
            all=lambda: [
                SimpleNamespace(path="C:\\library\\one.md"),
                SimpleNamespace(path="C:\\library\\two.md"),
            ]
        )
    )
    manager.vector_search = SimpleNamespace(
        reindex_all=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy VectorSearch must not reindex production data")
        )
    )
    manager.indexing_status = SimpleNamespace(start_many=queued.append)
    monkeypatch.setattr(
        file_tag_manager.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    manager.reindex_files()

    assert queued == [[
        "C:\\library\\one.md",
        "C:\\library\\two.md",
    ]]


def test_manager_reindex_all_reports_missing_status_without_runtime_fallback(
    qtbot, monkeypatch
) -> None:
    from file_tagger import file_tag_manager

    calls: list[str] = []

    class Runtime:
        def index_file_async(self, path: str) -> None:
            calls.append(path)
            raise AssertionError("the manager must not queue runtime indexing")

    manager = file_tag_manager.FileTagManager.__new__(
        file_tag_manager.FileTagManager
    )
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.logger = logging.getLogger("indexing-status-test")
    manager.db_session = SimpleNamespace(
        query=lambda model: SimpleNamespace(
            all=lambda: [SimpleNamespace(path="C:\\library\\one.md")]
        )
    )
    manager.modern_runtime = Runtime()
    manager.modern_index_status_text = ""
    shown: list[str] = []
    monkeypatch.setattr(
        file_tag_manager.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        file_tag_manager.QMessageBox,
        "warning",
        lambda parent, title, message: shown.append(message),
    )

    manager.reindex_files()

    assert calls == []
    assert manager.modern_index_status_text == (
        "Modern indexing unavailable: indexing status UI is not initialized."
    )
    assert shown == [manager.modern_index_status_text]


def test_batch_tagging_queues_paths_as_one_indexing_status_job(
    qtbot, monkeypatch
) -> None:
    from file_tagger import file_tag_manager

    manager = file_tag_manager.FileTagManager.__new__(
        file_tag_manager.FileTagManager
    )
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.logger = logging.getLogger("indexing-status-test")
    paths = ["C:\\library\\one.md", "C:\\library\\two.md"]
    queued: list[list[str]] = []
    manager.db_session = SimpleNamespace(
        query=lambda model: SimpleNamespace(
            filter_by=lambda **kwargs: SimpleNamespace(
                one=lambda: SimpleNamespace(tags=[])
            )
        )
    )
    manager.apply_tags_to_file = lambda path, tags: None
    manager._commit_legacy_changes = lambda **kwargs: None
    manager.indexing_status = SimpleNamespace(
        start_many=lambda values: queued.append(list(values))
    )
    manager.modern_runtime = SimpleNamespace(
        index_file_async=lambda path: (_ for _ in ()).throw(
            AssertionError("the manager must use the IndexingStatus QThread")
        )
    )
    manager.refresh_tags = lambda: None
    manager.refresh_file_tags = lambda: None
    monkeypatch.setattr(
        file_tag_manager.QMessageBox,
        "information",
        lambda *args, **kwargs: None,
    )

    manager.apply_tags_to_files(paths, ["research"])

    assert queued == [paths]


def test_suggestion_batching_queues_paths_as_one_indexing_status_job(
    qtbot, monkeypatch
) -> None:
    from file_tagger import file_tag_manager

    manager = file_tag_manager.FileTagManager.__new__(
        file_tag_manager.FileTagManager
    )
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.logger = logging.getLogger("indexing-status-test")
    paths = ["C:\\library\\one.md", "C:\\library\\two.md"]
    queued: list[list[str]] = []
    manager.file_suggestions_map = {
        path: {"research": 0.9}
        for path in paths
    }
    manager.db_session = SimpleNamespace(
        query=lambda model: SimpleNamespace(
            filter_by=lambda **kwargs: SimpleNamespace(
                one=lambda: SimpleNamespace(tags=[])
            )
        )
    )
    manager.apply_tags_to_file = lambda path, tags: None
    manager._commit_legacy_changes = lambda **kwargs: None
    manager.indexing_status = SimpleNamespace(
        start_many=lambda values: queued.append(list(values))
    )
    manager.modern_runtime = SimpleNamespace(
        index_file_async=lambda path: (_ for _ in ()).throw(
            AssertionError("the manager must use the IndexingStatus QThread")
        )
    )
    monkeypatch.setattr(
        file_tag_manager.QMessageBox,
        "information",
        lambda *args, **kwargs: None,
    )

    manager.apply_all_suggestions_to_files(paths)

    assert queued == [paths]


def test_safe_legacy_cleanup_requires_confirmation(qtbot, monkeypatch, tmp_path) -> None:
    from file_tagger import file_tag_manager

    legacy_chroma = tmp_path / ".chroma"
    legacy_chroma.mkdir()
    manager = file_tag_manager.FileTagManager.__new__(
        file_tag_manager.FileTagManager
    )
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.logger = logging.getLogger("indexing-status-test")
    manager.modern_runtime = SimpleNamespace(
        paths=SimpleNamespace(legacy_chroma=legacy_chroma)
    )
    shown: list[str] = []
    monkeypatch.setattr(
        file_tag_manager.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )
    monkeypatch.setattr(
        file_tag_manager.QMessageBox,
        "information",
        lambda parent, title, message: shown.append(message),
    )

    assert manager.cleanup_legacy_search_data() is False
    assert legacy_chroma.exists()
    assert shown


def test_migration_inventory_reports_modern_migration_counts(
    qtbot, tmp_path
) -> None:
    from file_tagger import file_tag_manager

    chroma = tmp_path / ".chroma"
    chroma.mkdir()
    manager = file_tag_manager.FileTagManager.__new__(
        file_tag_manager.FileTagManager
    )
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.logger = logging.getLogger("indexing-status-test")
    manager.modern_runtime = SimpleNamespace(
        paths=SimpleNamespace(legacy_chroma=chroma),
        sqlite_migration=SimpleNamespace(
            files_seen=4,
            files_migrated=2,
            tags_migrated=1,
            associations_migrated=3,
            suggestions_preserved=0,
            errors=(),
        ),
        config_migration=SimpleNamespace(
            skipped=False,
            cloud_fields_removed=1,
            errors=(),
        ),
    )

    inventory = manager.migration_inventory()

    assert inventory["legacy_chroma_found"] is True
    assert inventory["files_seen"] == 4
    assert inventory["files_migrated"] == 2
    assert inventory["requires_full_reindex"] is False


def test_single_file_reindex_does_not_require_legacy_vector_search(
    qtbot, tmp_path, monkeypatch
) -> None:
    from file_tagger import file_tag_manager

    source = tmp_path / "guide.md"
    source.write_text("guide", encoding="utf-8")
    manager = file_tag_manager.FileTagManager.__new__(
        file_tag_manager.FileTagManager
    )
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.logger = logging.getLogger("indexing-status-test")
    manager.db_session = SimpleNamespace(
        query=lambda model: SimpleNamespace(
            filter_by=lambda **kwargs: SimpleNamespace(
                first=lambda: object()
            )
        )
    )
    started: list[str] = []
    manager.indexing_status = SimpleNamespace(start=started.append)
    manager.modern_runtime = SimpleNamespace(
        index_file=lambda path: (_ for _ in ()).throw(
            AssertionError("the UI thread must not perform modern indexing")
        )
    )
    manager.vector_search = SimpleNamespace()

    assert manager.force_reindex_file(str(source)) is True
    assert started == [str(source)]


def test_single_file_reindex_requires_status_without_runtime_fallback(
    qtbot, tmp_path
) -> None:
    from file_tagger import file_tag_manager

    source = tmp_path / "guide.md"
    source.write_text("guide", encoding="utf-8")
    calls: list[str] = []

    class Runtime:
        def index_file_async(self, path: str) -> None:
            calls.append(path)
            raise AssertionError("the manager must not queue runtime indexing")

    manager = file_tag_manager.FileTagManager.__new__(
        file_tag_manager.FileTagManager
    )
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.logger = logging.getLogger("indexing-status-test")
    manager.db_session = SimpleNamespace(
        query=lambda model: SimpleNamespace(
            filter_by=lambda **kwargs: SimpleNamespace(
                first=lambda: object()
            )
        )
    )
    manager.modern_runtime = Runtime()
    manager.modern_index_status_text = ""

    assert manager.force_reindex_file(str(source)) is False
    assert calls == []
    assert manager.modern_index_status_text == (
        "Modern indexing unavailable: indexing status UI is not initialized."
    )


def test_indexing_menu_exposes_resume_inventory_and_safe_cleanup(qtbot) -> None:
    from file_tagger import file_tag_manager

    manager = file_tag_manager.FileTagManager.__new__(
        file_tag_manager.FileTagManager
    )
    file_tag_manager.QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.logger = logging.getLogger("indexing-status-test")
    manager.config = SimpleNamespace(get_pdf_extractor=lambda: "accurate")

    manager.setup_menus()

    labels = [
        action.text()
        for menu_action in manager.menuBar().actions()
        for action in (
            menu_action.menu().actions() if menu_action.menu() else ()
        )
        if menu_action.text() == "Indexing"
    ]
    assert "Resume Indexing" in labels
    assert "Migration Inventory" in labels
    assert "Clean Legacy Search Data" in labels
    assert "Reindex All Files" in labels
