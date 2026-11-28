# Recursive Folder Scanning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make **Scan Directory** recursively enumerate visible files in nested subfolders while preserving cancellation and error handling.

**Architecture:** Extract recursive traversal into a testable utility in `src/file_tagger/utils.py`, using `os.walk(..., followlinks=False)` to prune hidden directories and symlinks. Replace the `ScanThread` top-level `os.scandir` enumeration with this helper, retaining the existing database lookup, suggestions, progress, and dialog flow.

**Tech Stack:** Python 3.13, PySide6, pytest, SQLAlchemy, Windows hidden-file attributes.

---

## Task 1: Add Recursive Enumeration Tests

**Files:**
- Test: `tests/test_directory_scanning.py`
- Modify later: `src/file_tagger/utils.py`

**Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from file_tagger.utils import iter_visible_files


def test_iter_visible_files_includes_root_and_nested_files(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.md").write_text("child", encoding="utf-8")

    assert set(iter_visible_files(tmp_path)) == {
        str(tmp_path / "root.txt"),
        str(tmp_path / "nested" / "child.md"),
    }


def test_iter_visible_files_skips_hidden_files_and_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / ".hidden.txt").write_text("hidden", encoding="utf-8")
    visible = tmp_path / "visible.txt"
    visible.write_text("visible", encoding="utf-8")
    hidden_directory = tmp_path / ".hidden"
    hidden_directory.mkdir()
    (hidden_directory / "child.txt").write_text("hidden", encoding="utf-8")

    assert list(iter_visible_files(tmp_path)) == [str(visible)]


def test_iter_visible_files_stops_when_cancelled(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")

    assert list(iter_visible_files(tmp_path, should_stop=lambda: True)) == []


def test_iter_visible_files_skips_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    file_link = tmp_path / "file-link.txt"
    directory_target = tmp_path / "directory-target"
    directory_target.mkdir()
    directory_link = tmp_path / "directory-link"

    try:
        file_link.symlink_to(target)
        directory_link.symlink_to(directory_target, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links require platform privileges")

    assert list(iter_visible_files(tmp_path)) == [str(target)]


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
```

**Step 2: Run tests and verify red**

Run:

```powershell
uv run pytest tests/test_directory_scanning.py -v
```

Expected: collection fails because `iter_visible_files` is not exported, or the
tests fail for an implementation that does not exist.

## Task 2: Implement Recursive Enumeration

**Files:**
- Modify: `src/file_tagger/utils.py`

**Step 1: Write the minimal helper implementation**

Add imports:

```python
import stat
from collections.abc import Callable, Iterator
from pathlib import Path
```

Add helpers below the existing imports:

```python
def is_hidden_path(path: str | os.PathLike[str]) -> bool:
    path_obj = Path(path)
    if path_obj.name.startswith("."):
        return True
    if os.name != "nt":
        return False

    attributes = getattr(os.stat(path_obj), "st_file_attributes", 0)
    return bool(attributes & stat.FILE_ATTRIBUTE_HIDDEN)


def iter_visible_files(
    directory: str | os.PathLike[str],
    *,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[str]:
    root = os.fspath(directory)
    if should_stop is not None and should_stop():
        return
    if os.path.islink(root):
        return

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current_root, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        if should_stop is not None and should_stop():
            return

        directory_names[:] = [
            name
            for name in directory_names
            if not os.path.islink(os.path.join(current_root, name))
            and not is_hidden_path(os.path.join(current_root, name))
        ]

        for name in file_names:
            if should_stop is not None and should_stop():
                return

            file_path = os.path.join(current_root, name)
            if (
                os.path.islink(file_path)
                or not os.path.isfile(file_path)
                or is_hidden_path(file_path)
            ):
                continue
            yield file_path
```

**Step 2: Run tests and verify green**

Run:

```powershell
uv run pytest tests/test_directory_scanning.py -v
```

Expected: all five tests pass.

**Step 3: Commit**

```powershell
git add src/file_tagger/utils.py tests/test_directory_scanning.py
git diff --cached --check
git commit -m "feat: add recursive visible-file enumeration" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

## Task 3: Wire Recursive Enumeration into Scan Directory

**Files:**
- Modify: `src/file_tagger/file_tag_manager.py:1901-2004`
- Existing import: `src/file_tagger/utils.py`

**Step 1: Import the helper**

At the existing utility import:

```python
from .utils import (
    is_dark_color,
    get_score_color,
    iter_visible_files,
    open_file,
    open_containing_folder,
)
```

**Step 2: Replace top-level enumeration**

Replace:

```python
# List all files in directory (not subdirectories)
files = []
with os.scandir(self.directory) as entries:
    for entry in entries:
        if entry.is_file() and not entry.name.startswith("."):
            files.append(entry.path)
```

With:

```python
files = list(
    iter_visible_files(
        self.directory,
        should_stop=lambda: self.stop_requested,
    )
)
if self.stop_requested:
    return
```

Do not call `scan_finished` for cancellation. Keep the existing zero-file check
and no-change database/suggestion loop intact.

**Step 3: Verify focused behavior**

Run:

```powershell
uv run pytest tests/test_directory_scanning.py -v
uv run pytest tests/ui/test_modern_tag_ui.py tests/ui/test_indexing_status.py -v
```

Expected: recursive enumeration tests pass and existing tagging/indexing UI
tests remain green.

**Step 4: Commit**

```powershell
git add src/file_tagger/file_tag_manager.py
git diff --cached --check
git commit -m "feat: scan nested folders recursively"
```

Include the required Co-authored-by trailer.

## Task 4: Final Verification

**Files:** No production changes expected.

**Step 1: Run focused and relevant suites**

```powershell
uv run pytest tests/test_directory_scanning.py tests/ui/test_modern_tag_ui.py tests/ui/test_indexing_status.py -v
uv run ruff check src/file_tagger/utils.py src/file_tagger/file_tag_manager.py tests/test_directory_scanning.py
```

**Step 2: Review worktree**

```powershell
git --no-pager diff --stat
git --no-pager diff --check
git --no-pager status --short
```

Verify only the intended implementation files are added on top of any unrelated
pre-existing changes. Do not revert or overwrite unrelated work.
