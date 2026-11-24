def test_application_package_imports() -> None:
    from file_tagger.main import main

    assert callable(main)


def test_login_screen_imports_without_loading_optional_ai_stack() -> None:
    from PySide6.QtWidgets import QWidget

    from file_tagger.login import LoginScreen

    assert issubclass(LoginScreen, QWidget)
