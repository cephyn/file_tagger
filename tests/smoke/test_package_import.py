def test_application_package_imports() -> None:
    from file_tagger.main import main

    assert callable(main)


def test_login_screen_imports_without_loading_optional_ai_stack() -> None:
    from PySide6.QtWidgets import QWidget

    from file_tagger.login import LoginScreen

    assert issubclass(LoginScreen, QWidget)


def test_search_package_imports_without_legacy_ai_service() -> None:
    from file_tagger.search import ChatWithResultsDialog, markdown_to_html

    assert ChatWithResultsDialog.__module__ == "file_tagger.search.chat"
    assert markdown_to_html("**ready**") == "<b>ready</b>"


def test_search_chat_control_stays_disabled_after_results(
    qtbot, tmp_path
) -> None:
    from PySide6.QtWidgets import QMainWindow

    from file_tagger.file_tag_manager import FileTagManager

    source = tmp_path / "indexed.txt"
    source.write_text("indexed evidence", encoding="utf-8")

    class Search:
        def search_content(self, *args, **kwargs):
            return [
                {
                    "path": str(source),
                    "score": 0.9,
                    "summary": "Indexed document",
                    "snippets": ["indexed evidence"],
                }
            ]

    manager = FileTagManager.__new__(FileTagManager)
    QMainWindow.__init__(manager)
    manager.modern_runtime = Search()
    tab = manager.create_rag_search_tab()
    qtbot.addWidget(tab)
    manager.query_input.setText("evidence")

    manager.search_by_content()

    assert manager.chat_results_btn.isEnabled() is False
