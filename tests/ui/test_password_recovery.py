from types import SimpleNamespace
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QMessageBox

from file_tagger.password_management import PasswordRecoveryDialog
from file_tagger.security.password_vault import CredentialStatus


def test_login_exposes_forgot_password_dialog(qtbot, monkeypatch) -> None:
    from file_tagger.login import LoginScreen

    opened = []
    monkeypatch.setattr(
        PasswordRecoveryDialog,
        "exec",
        lambda self: opened.append(self) or QDialog.DialogCode.Rejected,
    )
    screen = LoginScreen()
    qtbot.addWidget(screen)

    qtbot.mouseClick(screen.forgot_password_button, Qt.MouseButton.LeftButton)

    assert len(opened) == 1


def test_recovery_dialog_rejects_mismatched_passwords(
    qtbot, monkeypatch
) -> None:
    warnings = []
    monkeypatch.setattr(
        "file_tagger.password_management.PasswordVault.status",
        lambda paths: CredentialStatus.ENVELOPE,
    )
    monkeypatch.setattr(
        "file_tagger.password_management.PasswordVault.recover",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("recovery must not run")
        ),
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    dialog = PasswordRecoveryDialog()
    qtbot.addWidget(dialog)
    dialog.recovery_key_input.setText("recovery-key")
    dialog.recover_new_password_input.setText("first")
    dialog.recover_confirm_password_input.setText("second")

    dialog.recover_button.click()

    assert warnings
    assert "do not match" in warnings[-1][2].lower()


def test_recovery_dialog_disables_recovery_for_legacy_and_explains_reset(
    qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(
        "file_tagger.password_management.PasswordVault.status",
        lambda paths: CredentialStatus.LEGACY,
    )

    dialog = PasswordRecoveryDialog()
    qtbot.addWidget(dialog)

    assert not dialog.recover_button.isEnabled()
    assert "cannot" in dialog.status_label.text().lower()
    assert "reset" in dialog.status_label.text().lower()


def test_recovery_dialog_detects_pending_cwd_legacy_configuration(
    qtbot, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.encrypted").write_bytes(b"legacy config")
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path / "data"))

    dialog = PasswordRecoveryDialog()
    qtbot.addWidget(dialog)

    assert not dialog.recover_button.isEnabled()
    assert "legacy" in dialog.status_label.text().lower()


def test_recovery_dialog_quarantines_only_after_typed_confirmation(
    qtbot, monkeypatch
) -> None:
    resets = []
    monkeypatch.setattr(
        "file_tagger.password_management.PasswordVault.status",
        lambda paths: CredentialStatus.ENVELOPE,
    )
    monkeypatch.setattr(
        "file_tagger.password_management.reset_encrypted_configuration",
        lambda paths, password: (
            resets.append((paths, password))
            or SimpleNamespace(
                backup_dir=paths.data_dir / "backup",
                recovery_key="one-time-key",
            )
        ),
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    dialog = PasswordRecoveryDialog()
    qtbot.addWidget(dialog)
    dialog.reset_new_password_input.setText("replacement")
    dialog.reset_confirm_password_input.setText("replacement")

    dialog.reset_button.click()
    assert resets == []

    dialog.reset_confirmation_input.setText("RESET")
    dialog.reset_button.click()

    assert len(resets) == 1
    assert dialog.new_password == "replacement"
    assert dialog.new_recovery_key == "one-time-key"


def test_accepted_recovery_retries_login_with_new_password(
    qtbot, monkeypatch
) -> None:
    from file_tagger.login import LoginScreen

    def accept_recovery(dialog):
        dialog._new_password = "replacement"
        dialog._new_recovery_key = "one-time-key"
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(PasswordRecoveryDialog, "exec", accept_recovery)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    screen = LoginScreen()
    qtbot.addWidget(screen)
    submitted = []
    monkeypatch.setattr(screen, "submit_password", lambda: submitted.append(True))

    qtbot.mouseClick(screen.forgot_password_button, Qt.MouseButton.LeftButton)

    assert screen.password_input.text() == "replacement"
    assert submitted == [True]


def test_first_setup_requires_recovery_key_acknowledgement(
    qtbot, monkeypatch
) -> None:
    from file_tagger.login import LoginScreen
    from types import ModuleType
    from PySide6.QtWidgets import QWidget

    events = []
    acknowledgements = []
    config = SimpleNamespace(
        pending_recovery_key="one-time-key",
        acknowledge_recovery_key=lambda: acknowledgements.append(True),
    )

    class Window(QWidget):
        def __init__(self, *args) -> None:
            super().__init__()
            events.append("window")

    manager_module = ModuleType("file_tagger.file_tag_manager")
    manager_module.FileTagManager = Window
    monkeypatch.setitem(sys.modules, "file_tagger.file_tag_manager", manager_module)
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: events.append(("acknowledged", text)) or QMessageBox.StandardButton.Ok,
    )
    screen = LoginScreen()
    qtbot.addWidget(screen)
    screen.password_input.setText("initial-password")
    monkeypatch.setattr(screen, "hide", lambda: None)
    monkeypatch.setattr(screen, "close", lambda: None)

    screen.on_initialization_finished((object(), config, object()))

    assert acknowledgements == [True]
    assert events[0][0] == "acknowledged"
    assert "one-time-key" in events[0][1]
    assert events[1] == "window"
    assert screen.password_input.text() == ""


def test_manager_opens_password_management_with_correct_arguments(
    monkeypatch,
) -> None:
    from file_tagger import file_tag_manager

    config = object()
    manager = SimpleNamespace(config=config)
    constructed = []

    class Dialog:
        def __init__(self, received_config, parent) -> None:
            constructed.append((received_config, parent))

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(file_tag_manager, "PasswordManagementDialog", Dialog)

    file_tag_manager.FileTagManager.show_password_management(manager)

    assert constructed == [(config, manager)]


def test_pending_recovery_key_is_acknowledged_even_after_startup_failure(
    qtbot, monkeypatch
) -> None:
    from file_tagger.login import LoginScreen

    events = []
    config = SimpleNamespace(
        pending_recovery_key="one-time-key",
        acknowledge_recovery_key=lambda: events.append("cleared"),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: events.append(("shown", text))
        or QMessageBox.StandardButton.Ok,
    )
    screen = LoginScreen()
    qtbot.addWidget(screen)
    monkeypatch.setattr(screen, "show_error", lambda message: events.append("error"))
    monkeypatch.setattr(screen, "cleanup_worker", lambda: None)

    screen.on_initialization_finished((None, config, None))

    assert events[0][0] == "shown"
    assert "one-time-key" in events[0][1]
    assert events[1:] == ["cleared", "error"]
