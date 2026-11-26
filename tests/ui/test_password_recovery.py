from types import SimpleNamespace

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
