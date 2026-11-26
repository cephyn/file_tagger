import random
from pathlib import Path

from PySide6.QtWidgets import QListWidget, QMainWindow
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from file_tagger.infrastructure.database import create_database_engine, upgrade_database
from file_tagger.persistence.models import File, FileTag, Tag


def test_refresh_file_tags_uses_modern_canonical_path(tmp_path: Path, qtbot) -> None:
    from file_tagger.file_tag_manager import FileTagManager

    database_path = tmp_path / "file-tagger.sqlite3"
    upgrade_database(database_path)
    engine = create_database_engine(database_path)
    session_factory = sessionmaker(bind=engine)

    with session_factory.begin() as session:
        file = File(
            canonical_path=r"E:\Google Drive",
            indexing_state="not_indexed",
        )
        tag = Tag(name="research", color="#123456")
        session.add_all([file, tag])
        session.flush()
        session.add(FileTag(file=file, tag=tag))

    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.db_session = session_factory()
    manager.current_file_path = r"E:\Google Drive"
    manager.file_tags_list = QListWidget()

    manager.refresh_file_tags()

    assert manager.file_tags_list.count() == 1
    assert manager.file_tags_list.item(0).text() == "research"


def test_add_tag_to_file_creates_modern_association(tmp_path: Path, qtbot) -> None:
    from file_tagger.file_tag_manager import FileTagManager

    database_path = tmp_path / "file-tagger.sqlite3"
    upgrade_database(database_path)
    engine = create_database_engine(database_path)
    session_factory = sessionmaker(bind=engine)

    with session_factory.begin() as session:
        file = File(
            canonical_path=r"E:\Google Drive",
            indexing_state="not_indexed",
        )
        tag = Tag(name="research", color="#123456")
        session.add_all([file, tag])

    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.db_session = session_factory()
    manager.current_file_path = r"E:\Google Drive"
    manager.tag_list = QListWidget()
    manager.tag_list.addItem("research")
    manager.tag_list.item(0).setSelected(True)
    manager.file_tags_list = QListWidget()
    manager.modern_runtime = None
    manager.logger = __import__("logging").getLogger("modern-tag-add-test")
    manager._index_modern_file = lambda _path: True

    manager.add_tag_to_file()

    with session_factory() as session:
        association = session.scalar(select(FileTag))
        assert association is not None
        assert association.tag.name == "research"
        assert association.file.canonical_path == r"E:\Google Drive"


def test_remove_tag_from_file_removes_modern_association(tmp_path: Path, qtbot) -> None:
    from file_tagger.file_tag_manager import FileTagManager

    database_path = tmp_path / "file-tagger.sqlite3"
    upgrade_database(database_path)
    engine = create_database_engine(database_path)
    session_factory = sessionmaker(bind=engine)

    with session_factory.begin() as session:
        file = File(
            canonical_path=r"E:\Google Drive",
            indexing_state="not_indexed",
        )
        tag = Tag(name="research", color="#123456")
        session.add_all([file, tag])
        session.flush()
        session.add(FileTag(file=file, tag=tag))

    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.db_session = session_factory()
    manager.current_file_path = r"E:\Google Drive"
    manager.file_tags_list = QListWidget()
    manager.file_tags_list.addItem("research")
    manager.file_tags_list.item(0).setSelected(True)
    manager.modern_runtime = None
    manager.logger = __import__("logging").getLogger("modern-tag-remove-test")

    manager.remove_tag_from_file()

    with session_factory() as session:
        assert session.scalar(select(FileTag)) is None


def test_generate_random_tag_color_avoids_existing_color(monkeypatch) -> None:
    from file_tagger import file_tag_manager

    values = iter((0, 255, 255, 120, 255, 255))
    monkeypatch.setattr(
        random,
        "randint",
        lambda _low, _high: next(values),
    )

    color = file_tag_manager.generate_random_tag_color({"#ff0000"})

    assert color.name() == "#00ff00"


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

    assert dialog.selected_color.name() == "#00ff00"


def test_tag_edit_dialog_random_preserves_manual_color_collision(
    qtbot,
    monkeypatch,
) -> None:
    from file_tagger.file_tag_manager import TagEditDialog

    dialog = TagEditDialog(
        tag_name="work",
        initial_color=QColor("#123456"),
        existing_colors={"#123456", "#ff0000"},
    )
    qtbot.addWidget(dialog)
    dialog.selected_color = QColor("#ff0000")

    values = iter((0, 255, 255, 120, 255, 255))
    monkeypatch.setattr(
        random,
        "randint",
        lambda _low, _high: next(values),
    )
    dialog.assign_random_color()

    assert dialog.selected_color.name() == "#00ff00"


def test_add_tag_uses_shared_dialog_and_persists_choice(
    tmp_path: Path,
    qtbot,
    monkeypatch,
) -> None:
    from file_tagger import file_tag_manager
    from file_tagger.file_tag_manager import FileTagManager

    database_path = tmp_path / "file-tagger.sqlite3"
    upgrade_database(database_path)
    engine = create_database_engine(database_path)
    session_factory = sessionmaker(bind=engine)

    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.db_session = session_factory()
    manager.tag_list = QListWidget()
    commit_calls = []
    refresh_calls = []

    def commit_legacy_changes(**kwargs):
        commit_calls.append(kwargs)
        manager.db_session.commit()

    manager._commit_legacy_changes = commit_legacy_changes
    manager.refresh_tags = lambda: refresh_calls.append(True)

    class AcceptedDialog:
        def __init__(self, parent, *, tag_name, existing_colors):
            assert parent is manager
            assert tag_name == ""
            assert not existing_colors

        def exec(self):
            return QDialog.DialogCode.Accepted

        def get_tag_name(self):
            return "contracts"

        def get_selected_color(self):
            return QColor("#123456")

    monkeypatch.setattr(file_tag_manager, "TagEditDialog", AcceptedDialog)

    manager.add_tag()

    with session_factory() as session:
        tag = session.scalar(select(Tag).where(Tag.name == "contracts"))
        assert tag is not None
        assert tag.color == "#123456"

    assert commit_calls == [{"tag_upserts": [("contracts", "#123456")]}]
    assert refresh_calls == [True]


def test_edit_tag_uses_shared_dialog_and_persists_choice(
    tmp_path: Path,
    qtbot,
    monkeypatch,
) -> None:
    from file_tagger import file_tag_manager
    from file_tagger.file_tag_manager import FileTagManager

    database_path = tmp_path / "file-tagger.sqlite3"
    upgrade_database(database_path)
    engine = create_database_engine(database_path)
    session_factory = sessionmaker(bind=engine)

    with session_factory.begin() as session:
        session.add(Tag(name="old-name", color="#123456"))

    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    qtbot.addWidget(manager)
    manager.db_session = session_factory()
    manager.tag_list = QListWidget()
    manager.tag_list.addItem("old-name")
    manager.tag_list.item(0).setSelected(True)
    manager.file_tags_list = QListWidget()
    commit_calls = []
    refresh_calls = []
    file_refresh_calls = []

    def commit_legacy_changes(**kwargs):
        commit_calls.append(kwargs)
        manager.db_session.commit()

    manager._commit_legacy_changes = commit_legacy_changes
    manager.refresh_tags = lambda: refresh_calls.append(True)
    manager.refresh_file_tags = lambda: file_refresh_calls.append(True)

    class AcceptedDialog:
        def __init__(
            self,
            parent,
            *,
            tag_name,
            initial_color,
            existing_colors,
        ):
            assert parent is manager
            assert tag_name == "old-name"
            assert initial_color == QColor("#123456")
            assert set(existing_colors) == {"#123456"}

        def exec(self):
            return QDialog.DialogCode.Accepted

        def get_tag_name(self):
            return "contracts"

        def get_selected_color(self):
            return QColor("#abcdef")

    monkeypatch.setattr(file_tag_manager, "TagEditDialog", AcceptedDialog)

    manager.edit_tag()

    with session_factory() as session:
        tag = session.scalar(select(Tag).where(Tag.name == "contracts"))
        assert tag is not None
        assert tag.color == "#abcdef"
        assert session.scalar(select(Tag).where(Tag.name == "old-name")) is None

    assert commit_calls == [
        {
            "tag_upserts": [("contracts", "#abcdef")],
            "tag_deletes": ["old-name"],
            "file_paths": [],
        }
    ]
    assert refresh_calls == [True]
    assert file_refresh_calls == [True]
