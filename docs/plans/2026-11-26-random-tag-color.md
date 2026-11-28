# Random Tag Color Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a shared create/edit tag dialog with an **Assign random color** action and no saved change until OK is confirmed.

**Architecture:** Add a small `TagEditDialog` beside `FileTagManager`, plus one pure `generate_random_tag_color()` helper for repeatable HSV generation and existing-color collision avoidance. Replace the separate name and color prompts in `add_tag()` and `edit_tag()` with the shared dialog while preserving `_commit_legacy_changes()` and list refresh behavior.

**Tech Stack:** Python 3.13, PySide6 Qt widgets, SQLAlchemy, pytest-qt, uv

---

### Task 1: Add failing dialog and persistence tests

**Files:**
- Modify: `tests/ui/test_modern_tag_ui.py`

**Step 1: Add imports and test helpers**

In `tests/ui/test_modern_tag_ui.py`, add:

```python
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog
```

Keep the existing `Path`, `QListWidget`, `QMainWindow`, SQLAlchemy imports.

**Step 2: Add a collision test for random generation**

Add this test:

```python
def test_generate_random_tag_color_avoids_existing_color(monkeypatch) -> None:
    from file_tagger import file_tag_manager

    values = iter((0, 255, 255, 120, 255, 255))
    monkeypatch.setattr(
        file_tag_manager.random,
        "randint",
        lambda _low, _high: next(values),
    )

    color = file_tag_manager.generate_random_tag_color({"#ff0000"})

    assert color.name() == "#00ff00"
```

**Step 3: Add a dialog preview test**

Add this test:

```python
def test_tag_edit_dialog_assign_random_updates_preview_only(qtbot, monkeypatch) -> None:
    from file_tagger.file_tag_manager import TagEditDialog

    dialog = TagEditDialog(
        tag_name="work",
        initial_color=QColor("#123456"),
        existing_colors={"#123456"},
    )
    qtbot.addWidget(dialog)

    monkeypatch.setattr(
        dialog,
        "generate_random_color",
        lambda: QColor("#00ff00"),
    )
    dialog.assign_random_color()

    assert dialog.selected_color == QColor("#00ff00")
```

**Step 4: Add create and edit integration tests**

Create a small fake accepted dialog in each test and monkeypatch
`file_tagger.file_tag_manager.TagEditDialog`.

For create, verify:

- `tag_name` passed by the manager starts empty.
- The dialog's chosen name and color are persisted.
- `_commit_legacy_changes()` receives `tag_upserts=[("contracts", "#123456")]`.
- `refresh_tags()` is called.

For edit, verify:

- The selected tag's name and color are passed to the dialog.
- OK updates the tag name and color.
- Rename commits include `tag_deletes=["old-name"]`.
- `refresh_tags()` and `refresh_file_tags()` are called.

**Step 5: Run the focused test file**

Run:

```powershell
uv run pytest tests/ui/test_modern_tag_ui.py -v
```

Expected: FAIL because `generate_random_tag_color` and `TagEditDialog` do not
exist yet.

**Step 6: Commit the tests**

```powershell
git add tests/ui/test_modern_tag_ui.py
git commit -m "test: cover random tag color dialog"
```

### Task 2: Add random-color helper and shared dialog

**Files:**
- Modify: `src/file_tagger/file_tag_manager.py`
- Modify: `src/file_tagger/file_tag_manager.py` imports

**Step 1: Add imports**

Add `import random` near the existing imports.

Update `PySide6.QtWidgets` imports to include:

```python
from PySide6.QtWidgets import (
    # existing imports...
    QDialogButtonBox,
)
```

**Step 2: Add the pure random-color helper**

Place this before `HTMLDelegate`:

```python
def generate_random_tag_color(
    existing_colors: Iterable[str] = (),
    excluded_color: str | None = None,
) -> QColor:
    existing = {
        QColor(color).name().lower()
        for color in existing_colors
        if QColor(color).isValid()
    }
    if excluded_color:
        existing.discard(QColor(excluded_color).name().lower())

    generated = QColor()
    for _ in range(16):
        hue = random.randint(0, 359)
        saturation = random.randint(128, 255)
        value = random.randint(180, 255)
        generated = QColor.fromHsv(hue, saturation, value)
        if generated.name().lower() not in existing:
            break
    return generated
```

**Step 3: Add `TagEditDialog`**

Place the dialog after `hide_scan_progress()` and before `HTMLDelegate`.

Required fields and behavior:

```python
class TagEditDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        tag_name: str = "",
        initial_color: QColor | None = None,
        existing_colors: Iterable[str] = (),
    ) -> None:
        super().__init__(parent)
        self.existing_colors = tuple(existing_colors)
        self.selected_color = (
            QColor(initial_color)
            if initial_color and initial_color.isValid()
            else generate_random_tag_color(self.existing_colors)
        )
```

The dialog must include:

- Window title `"Add Tag"` when `tag_name` is empty and `"Edit Tag"` otherwise.
- `name_input = QLineEdit(tag_name)`.
- A preview `QLabel` with fixed minimum height and a tooltip such as
  `"Selected tag color"`.
- A `QPushButton("Choose color")` connected to `_choose_color`.
- A `QPushButton("Assign random color")` connected to `assign_random_color`.
- `QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)`.
- `accepted` connected to `accept`; `rejected` connected to `reject`.

Implement:

```python
def generate_random_color(self) -> QColor:
    own_color = self.selected_color.name() if self.initially_existing else None
    return generate_random_tag_color(
        self.existing_colors,
        excluded_color=own_color,
    )

def assign_random_color(self) -> None:
    color = self.generate_random_color()
    if color.isValid():
        self.selected_color = QColor(color)
        self._refresh_preview()

def _choose_color(self) -> None:
    color = QColorDialog.getColor(
        initial=self.selected_color,
        parent=self,
        title="Select Tag Color",
    )
    if color.isValid():
        self.selected_color = QColor(color)
        self._refresh_preview()

def accept(self) -> None:
    if not self.name_input.text().strip():
        QMessageBox.warning(self, "Error", "Tag name cannot be blank!")
        return
    super().accept()

def get_tag_name(self) -> str:
    return self.name_input.text().strip()

def get_selected_color(self) -> QColor:
    return QColor(self.selected_color)
```

Set `initially_existing` to `initial_color is not None` in `__init__`.
`_refresh_preview()` updates `preview.setStyleSheet` with
`background-color: {color.name()}; border: 1px solid #777;`.

**Step 4: Run the focused tests**

Run:

```powershell
uv run pytest tests/ui/test_modern_tag_ui.py -v
```

Expected: PASS for helper and dialog tests.

**Step 5: Commit the helper and dialog**

```powershell
git add src/file_tagger/file_tag_manager.py tests/ui/test_modern_tag_ui.py
git commit -m "feat: add shared random tag color dialog"
```

### Task 3: Wire create and edit to the shared dialog

**Files:**
- Modify: `src/file_tagger/file_tag_manager.py:1138-1210`

**Step 1: Replace `add_tag()`**

```python
def add_tag(self):
    """Add a new tag."""
    existing_colors = [
        tag.color
        for tag in self.db_session.query(Tag).all()
        if tag.color
    ]
    dialog = TagEditDialog(
        self,
        tag_name="",
        existing_colors=existing_colors,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return

    tag_name = dialog.get_tag_name()
    selected_color = dialog.get_selected_color()
    if not selected_color.isValid():
        return

    existing = self.db_session.query(Tag).filter_by(name=tag_name).first()
    if existing:
        QMessageBox.warning(
            self,
            "Error",
            "A tag with this name already exists!",
        )
        return

    new_tag = Tag(name=tag_name, color=selected_color.name())
    self.db_session.add(new_tag)
    self._commit_legacy_changes(
        tag_upserts=[(tag_name, selected_color.name())]
    )
    self.refresh_tags()
```

**Step 2: Replace `edit_tag()`**

Keep the existing selected-item lookup and database lookup.

Pass the tag's current name and color to the dialog:

```python
existing_colors = [
    existing_tag.color
    for existing_tag in self.db_session.query(Tag).all()
    if existing_tag.color
]
dialog = TagEditDialog(
    self,
    tag_name=tag.name,
    initial_color=QColor(tag.color),
    existing_colors=existing_colors,
)
if dialog.exec() != QDialog.DialogCode.Accepted:
    return

old_name = tag.name
new_name = dialog.get_tag_name()
selected_color = dialog.get_selected_color()
if not selected_color.isValid():
    return

if new_name != tag.name:
    existing = self.db_session.query(Tag).filter_by(name=new_name).first()
    if existing:
        QMessageBox.warning(
            self,
            "Error",
            "A tag with this name already exists!",
        )
        return

tag.name = new_name
tag.color = selected_color.name()
self._commit_legacy_changes(
    tag_upserts=[(new_name, selected_color.name())],
    tag_deletes=[old_name] if old_name != new_name else [],
    file_paths=[
        (
            association.file.canonical_path,
            self._tag_data(association.file),
        )
        for association in tag.files
    ],
)
self.refresh_tags()
self.refresh_file_tags()
```

**Step 3: Run create/edit tests**

Run:

```powershell
uv run pytest tests/ui/test_modern_tag_ui.py -v
```

Expected: PASS.

**Step 4: Commit create/edit wiring**

```powershell
git add src/file_tagger/file_tag_manager.py tests/ui/test_modern_tag_ui.py
git commit -m "feat: wire random color dialog into tag create and edit"
```

### Task 4: Run focused and broader verification

**Files:** None

**Step 1: Run the UI test suite**

```powershell
uv run pytest tests/ui -v
```

**Step 2: Run lint**

```powershell
uv run ruff check src/file_tagger/file_tag_manager.py tests/ui/test_modern_tag_ui.py
```

**Step 3: Run relevant startup integration test**

```powershell
uv run pytest tests/test_startup_modern_integration.py tests/ui/test_modern_tag_ui.py -v
```

**Step 4: Confirm no regression in tag persistence**

If tests expose a modern/legacy synchronization issue, fix that in
`FileTagManager` only; do not change the database model or migration schema.

**Step 5: Commit verification fixes, if needed**

```powershell
git add src/file_tagger/file_tag_manager.py tests/ui/test_modern_tag_ui.py
git commit -m "fix: finalize random tag color behavior"
```
