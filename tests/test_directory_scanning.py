from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import (
    QDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
)
from PySide6.QtGui import QCloseEvent

from file_tagger.file_tag_manager import FileTagManager, hide_scan_progress
from file_tagger.persistence.models import File, Tag
from file_tagger.tag_suggestion import TagSuggester
from file_tagger.utils import is_hidden_path, iter_visible_files


def test_iter_visible_files_includes_root_and_nested_files(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.md").write_text("child", encoding="utf-8")

    assert set(iter_visible_files(tmp_path)) == {
        str(tmp_path / "root.txt"),
        str(tmp_path / "nested" / "child.md"),
    }


def test_iter_visible_files_skips_hidden_files(tmp_path: Path) -> None:
    (tmp_path / ".hidden.txt").write_text("hidden", encoding="utf-8")
    visible = tmp_path / "visible.txt"
    visible.write_text("visible", encoding="utf-8")

    assert list(iter_visible_files(tmp_path)) == [str(visible)]


def test_iter_visible_files_skips_hidden_directories(tmp_path: Path) -> None:
    visible = tmp_path / "visible.txt"
    visible.write_text("visible", encoding="utf-8")
    hidden_directory = tmp_path / ".hidden"
    hidden_directory.mkdir()
    (hidden_directory / "child.txt").write_text("hidden", encoding="utf-8")

    assert list(iter_visible_files(tmp_path)) == [str(visible)]


def test_iter_visible_files_skips_hidden_root(tmp_path: Path) -> None:
    hidden_root = tmp_path / ".hidden-root"
    hidden_root.mkdir()
    (hidden_root / "child.txt").write_text("hidden", encoding="utf-8")
    assert list(iter_visible_files(hidden_root)) == []


def test_iter_visible_files_cancellation_suppresses_walk_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cancelled = False

    def cancelled_walk(*args, **kwargs):
        nonlocal cancelled
        del args
        cancelled = True
        kwargs["onerror"](PermissionError("cancelled scan"))
        return iter(())

    monkeypatch.setattr("file_tagger.utils.os.walk", cancelled_walk)
    assert list(iter_visible_files(tmp_path, should_stop=lambda: cancelled)) == []


def test_is_hidden_path_detects_windows_hidden_attribute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = tmp_path / "hidden.txt"
    hidden.write_text("hidden", encoding="utf-8")
    monkeypatch.setattr("file_tagger.utils.os.name", "nt")
    monkeypatch.setattr(
        "file_tagger.utils.os.stat",
        lambda path: SimpleNamespace(st_file_attributes=0x2),
    )

    assert is_hidden_path(hidden)


def test_iter_visible_files_stops_when_cancelled(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("first", encoding="utf-8")
    (tmp_path / "second.txt").write_text("second", encoding="utf-8")
    should_stop = False

    iterator = iter(
        iter_visible_files(
            tmp_path,
            should_stop=lambda: should_stop,
        )
    )
    first = next(iterator)
    assert first in {str(tmp_path / "first.txt"), str(tmp_path / "second.txt")}

    should_stop = True
    assert list(iterator) == []


def test_iter_visible_files_skips_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    file_link = tmp_path / "file-link.txt"
    directory_target = tmp_path / "directory-target"
    directory_target.mkdir()
    (directory_target / "inside.txt").write_text("inside", encoding="utf-8")
    directory_link = tmp_path / "directory-link"

    try:
        file_link.symlink_to(target)
        directory_link.symlink_to(directory_target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links require platform privileges")

    visible_files = set(iter_visible_files(tmp_path))
    assert str(target) in visible_files
    assert str(directory_target / "inside.txt") in visible_files
    assert str(directory_link / "inside.txt") not in visible_files
    assert str(file_link) not in visible_files


def test_iter_visible_files_skips_junction_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    junction_root = tmp_path / "junction-root"
    junction_root.mkdir()
    (junction_root / "child.txt").write_text("hidden", encoding="utf-8")
    monkeypatch.setattr(
        "file_tagger.utils.os.path.isjunction",
        lambda path: Path(path).name == "junction-root",
        raising=False,
    )

    assert list(iter_visible_files(junction_root)) == []


def test_iter_visible_files_skips_junction_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    junction_directory = tmp_path / "junction-directory"
    junction_directory.mkdir()
    (junction_directory / "inside.txt").write_text("hidden", encoding="utf-8")
    monkeypatch.setattr(
        "file_tagger.utils.os.path.isjunction",
        lambda path: Path(path).name == "junction-directory",
        raising=False,
    )

    assert list(iter_visible_files(tmp_path)) == []


def test_iter_visible_files_skips_junction_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    junction_file = tmp_path / "junction-file.txt"
    junction_file.write_text("hidden", encoding="utf-8")
    monkeypatch.setattr(
        "file_tagger.utils.os.path.isjunction",
        lambda path: Path(path).name == "junction-file.txt",
        raising=False,
    )

    assert list(iter_visible_files(tmp_path)) == []


def test_iter_visible_files_propagates_walk_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied_walk(*args, **kwargs):
        del args
        kwargs["onerror"](PermissionError("directory denied"))
        return iter(())

    monkeypatch.setattr("file_tagger.utils.os.walk", denied_walk)

    with pytest.raises(PermissionError, match="directory denied"):
        list(iter_visible_files(tmp_path))


def test_hide_scan_progress_does_not_emit_cancelled(qtbot) -> None:
    progress = QProgressDialog()
    qtbot.addWidget(progress)
    cancelled = False

    def mark_cancelled() -> None:
        nonlocal cancelled
        cancelled = True

    progress.canceled.connect(mark_cancelled)
    hide_scan_progress(progress)

    assert cancelled is False


def test_scan_directory_for_untagged_rejects_overlapping_scan(
    tmp_path: Path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.path_display = QLineEdit(str(tmp_path))
    manager.db_session = SimpleNamespace()
    manager.config = SimpleNamespace()
    manager.modern_runtime = None
    manager.scan_thread = SimpleNamespace(isRunning=lambda: True)

    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    manager.scan_directory_for_untagged()

    assert len(warnings) == 1


def test_close_event_stops_running_scan_and_waits(
    qtbot,
) -> None:
    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)

    waited_for = []
    manager.scan_thread = SimpleNamespace(
        stop_requested=False,
        isRunning=lambda: True,
        requestInterruption=lambda: None,
        wait=lambda timeout: waited_for.append(timeout) or False,
    )
    event = QCloseEvent()

    manager.closeEvent(event)

    assert manager.scan_thread.stop_requested is True
    assert waited_for == [1000]
    assert event.isAccepted() is False


def test_show_untagged_files_dialog_lists_nested_file_and_suggestion(
    tmp_path: Path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    nested_file = nested_directory / "nested.txt"
    nested_file.write_text("nested", encoding="utf-8")

    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.db_session = SimpleNamespace(
        query=lambda model: SimpleNamespace(
            all=lambda: [] if model is Tag else pytest.fail(
                f"unexpected query model: {model}"
            )
        )
    )
    monkeypatch.setattr(QDialog, "exec", lambda self: 0)

    manager.show_untagged_files_dialog(
        {str(nested_file): {"nested-tag": 0.9}},
        str(tmp_path),
    )

    assert manager.untagged_files_list.count() == 1
    file_item = manager.untagged_files_list.item(0)
    assert file_item.toolTip() == str(nested_file)

    manager.untagged_files_list.setCurrentRow(-1)
    manager.untagged_files_list.setCurrentRow(0)
    assert manager.file_suggestions_list.count() == 1
    assert "nested-tag" in manager.file_suggestions_list.item(0).text()


def test_scan_directory_for_untagged_runs_qthread_and_finds_nested_file(
    tmp_path: Path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    nested_file = nested_directory / "nested.txt"
    nested_file.write_text("nested", encoding="utf-8")
    nested_path = str(nested_file.resolve())

    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.path_display = QLineEdit(str(tmp_path))
    manager.db_session = SimpleNamespace(
        query=lambda model: (
            SimpleNamespace(
                filter_by=lambda **kwargs: SimpleNamespace(first=lambda: None)
            )
            if model is File
            else SimpleNamespace(all=lambda: [])
            if model is Tag
            else pytest.fail(f"unexpected query model: {model}")
        )
    )
    manager.config = SimpleNamespace()
    manager.modern_runtime = None

    monkeypatch.setattr(TagSuggester, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        TagSuggester,
        "suggest_tags_for_file",
        lambda self, file_path: {"nested-tag": 0.9},
    )

    captured = {}

    def capture_files(files, directory):
        captured["files"] = files
        captured["directory"] = directory

    monkeypatch.setattr(manager, "show_untagged_files_dialog", capture_files)

    manager.scan_directory_for_untagged()

    qtbot.waitUntil(
        lambda: nested_path in captured.get("files", {}),
        timeout=5000,
    )
    assert nested_path in captured["files"]
    assert captured["directory"] == str(tmp_path)


def test_scan_directory_uses_dedicated_bound_session(
    tmp_path: Path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    nested_file = nested_directory / "nested.txt"
    nested_file.write_text("nested", encoding="utf-8")

    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.path_display = QLineEdit(str(tmp_path))
    manager.config = SimpleNamespace()
    manager.modern_runtime = None

    bind = object()
    manager.db_session = SimpleNamespace(
        get_bind=lambda: bind,
        query=lambda model: pytest.fail("scan used the shared database session"),
    )

    dedicated_session = SimpleNamespace(
        closed=False,
        query=lambda model: SimpleNamespace(
            filter_by=lambda **kwargs: SimpleNamespace(first=lambda: None)
        ),
        close=lambda: setattr(dedicated_session, "closed", True),
    )
    factory_calls = []

    import file_tagger.file_tag_manager as file_tag_manager_module

    def fake_sessionmaker(*, bind):
        factory_calls.append(bind)
        return lambda: dedicated_session

    monkeypatch.setattr(
        file_tag_manager_module,
        "sessionmaker",
        fake_sessionmaker,
        raising=False,
    )
    monkeypatch.setattr(TagSuggester, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        TagSuggester,
        "suggest_tags_for_file",
        lambda self, file_path: {},
    )

    captured = {}
    monkeypatch.setattr(
        manager,
        "show_untagged_files_dialog",
        lambda files, directory: captured.update(files=files, directory=directory),
    )

    manager.scan_directory_for_untagged()

    qtbot.waitUntil(
        lambda: "files" in captured,
        timeout=5000,
    )
    assert factory_calls == [bind]
    assert dedicated_session.closed is True


def test_scan_result_uses_captured_thread_not_replaced_manager_thread(
    tmp_path: Path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_file = tmp_path / "nested.txt"
    nested_file.write_text("nested", encoding="utf-8")

    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.path_display = QLineEdit(str(tmp_path))
    manager.db_session = SimpleNamespace(
        query=lambda model: SimpleNamespace(
            filter_by=lambda **kwargs: SimpleNamespace(first=lambda: None)
        )
    )
    manager.config = SimpleNamespace()
    manager.modern_runtime = None

    monkeypatch.setattr(TagSuggester, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        TagSuggester,
        "suggest_tags_for_file",
        lambda self, file_path: {},
    )

    captured = {}
    monkeypatch.setattr(
        manager,
        "show_untagged_files_dialog",
        lambda files, directory: captured.update(files=files, directory=directory),
    )

    manager.scan_directory_for_untagged()
    manager.scan_thread = SimpleNamespace(stop_requested=True)

    qtbot.waitUntil(
        lambda: "files" in captured,
        timeout=5000,
    )
    assert str(nested_file.resolve()) in captured["files"]
