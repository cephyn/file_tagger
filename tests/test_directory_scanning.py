from pathlib import Path
from types import SimpleNamespace

import pytest

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
