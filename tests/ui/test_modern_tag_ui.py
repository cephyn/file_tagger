from pathlib import Path

from PySide6.QtWidgets import QListWidget, QMainWindow
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
