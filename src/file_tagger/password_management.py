from __future__ import annotations

import logging

from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from .config import Config
from .domain.errors import (
    CredentialFormatError,
    CredentialRecoveryError,
    PasswordEncryptionError,
)
from .infrastructure.app_paths import AppPaths
from .security.password_reset import reset_encrypted_configuration
from .security.password_vault import CredentialStatus, PasswordVault

logger = logging.getLogger(__name__)


def get_password(parent=None) -> str | None:
    password, ok = QInputDialog.getText(
        parent,
        "Password Required",
        "Enter your password:",
        QLineEdit.EchoMode.Password,
    )
    return password if ok and password else None


class PasswordManagementDialog(QDialog):
    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._new_password: str | None = None
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("Password Management")
        self.setModal(True)
        layout = QVBoxLayout(self)

        change_group = QGroupBox("Change Password")
        change_layout = QVBoxLayout(change_group)
        self.old_pass_input = _password_input("Current password")
        self.new_pass_input = _password_input("New password")
        self.confirm_pass_input = _password_input("Confirm new password")
        change_button = QPushButton("Change Password")
        change_button.clicked.connect(self.change_password)
        change_layout.addWidget(self.old_pass_input)
        change_layout.addWidget(self.new_pass_input)
        change_layout.addWidget(self.confirm_pass_input)
        change_layout.addWidget(change_button)
        layout.addWidget(change_group)

        rotate_group = QGroupBox("Recovery Key")
        rotate_layout = QVBoxLayout(rotate_group)
        rotate_layout.addWidget(
            QLabel(
                "Recovery keys are shown only when created or rotated. "
                "Rotating invalidates the previous key."
            )
        )
        rotate_button = QPushButton("Rotate Recovery Key")
        rotate_button.clicked.connect(self.rotate_recovery_key)
        rotate_layout.addWidget(rotate_button)
        layout.addWidget(rotate_group)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        layout.addWidget(close_button)

    def change_password(self) -> None:
        old_password = self.old_pass_input.text()
        new_password = self.new_pass_input.text()
        confirm_password = self.confirm_pass_input.text()
        if not old_password or not new_password or not confirm_password:
            QMessageBox.warning(self, "Error", "All fields are required.")
            return
        if new_password != confirm_password:
            QMessageBox.warning(self, "Error", "New passwords do not match.")
            return
        if self.config.change_password(old_password, new_password):
            QMessageBox.information(self, "Success", "Password changed successfully.")
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Current password is incorrect.")

    def rotate_recovery_key(self) -> None:
        current_password, ok = QInputDialog.getText(
            self,
            "Verify Password",
            "Enter current password to rotate the recovery key:",
            QLineEdit.EchoMode.Password,
        )
        if not ok or not current_password:
            return
        try:
            recovery_key = self.config.rotate_recovery_key(current_password)
        except (
            CredentialFormatError,
            CredentialRecoveryError,
            PasswordEncryptionError,
        ):
            QMessageBox.warning(self, "Error", "Current password is incorrect.")
            return
        QMessageBox.information(
            self,
            "New Recovery Key",
            f"Your new recovery key is:\n\n{recovery_key}\n\n"
            "Store this key in a safe place. It will not be shown again, "
            "and the previous recovery key is now invalid.",
        )


class PasswordRecoveryDialog(QDialog):
    def __init__(self, parent=None, paths: AppPaths | None = None) -> None:
        super().__init__(parent)
        self.paths = paths or AppPaths.discover()
        self._new_password: str | None = None
        self._new_recovery_key: str | None = None
        self._init_ui()

    @property
    def new_password(self) -> str | None:
        return self._new_password

    @property
    def new_recovery_key(self) -> str | None:
        return self._new_recovery_key

    def _init_ui(self) -> None:
        self.setWindowTitle("Password Recovery")
        self.setModal(True)
        layout = QVBoxLayout(self)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        recover_group = QGroupBox("Recover with key")
        recover_layout = QVBoxLayout(recover_group)
        self.recovery_key_input = QLineEdit()
        self.recovery_key_input.setPlaceholderText("Recovery key")
        self.recover_new_password_input = _password_input("New password")
        self.recover_confirm_password_input = _password_input(
            "Confirm new password"
        )
        self.recover_button = QPushButton("Recover Password")
        self.recover_button.clicked.connect(self.recover_password)
        recover_layout.addWidget(self.recovery_key_input)
        recover_layout.addWidget(self.recover_new_password_input)
        recover_layout.addWidget(self.recover_confirm_password_input)
        recover_layout.addWidget(self.recover_button)
        layout.addWidget(recover_group)

        reset_group = QGroupBox("Reset configuration")
        reset_layout = QVBoxLayout(reset_group)
        reset_layout.addWidget(
            QLabel(
                "Reset is destructive to encrypted settings and stored credentials. "
                "Tag databases and search data are preserved. Quarantined files "
                "remain available for manual recovery."
            )
        )
        self.reset_new_password_input = _password_input("New password")
        self.reset_confirm_password_input = _password_input(
            "Confirm new password"
        )
        self.reset_confirmation_input = QLineEdit()
        self.reset_confirmation_input.setPlaceholderText("Type RESET to confirm")
        self.reset_button = QPushButton("Reset Configuration")
        self.reset_button.clicked.connect(self.reset_configuration)
        reset_layout.addWidget(self.reset_new_password_input)
        reset_layout.addWidget(self.reset_confirm_password_input)
        reset_layout.addWidget(self.reset_confirmation_input)
        reset_layout.addWidget(self.reset_button)
        layout.addWidget(reset_group)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)
        self._apply_credential_status()

    def _apply_credential_status(self) -> None:
        status = PasswordVault.status(self.paths)
        if (
            status is CredentialStatus.UNINITIALIZED
            and (
                self.paths.legacy_database.parent / "config.encrypted"
            ).is_file()
        ):
            status = CredentialStatus.LEGACY
        messages = {
            CredentialStatus.ENVELOPE: (
                "Enter the recovery key that was shown when File Tagger was set up "
                "or the key was rotated."
            ),
            CredentialStatus.LEGACY: (
                "The legacy recovery key cannot recover this installation because "
                "it was encrypted with the forgotten password. Use Reset configuration "
                "if the old password is unavailable."
            ),
            CredentialStatus.CORRUPT: (
                "Credential metadata is missing, corrupt, or unsupported. Recovery is "
                "disabled to protect existing files; Reset configuration is available."
            ),
            CredentialStatus.UNINITIALIZED: (
                "No password-protected configuration has been initialized. Recovery "
                "is unavailable, but the configuration can be reset."
            ),
        }
        self.status_label.setText(messages[status])
        self.recover_button.setEnabled(status is CredentialStatus.ENVELOPE)

    def recover_password(self) -> None:
        recovery_key = self.recovery_key_input.text().strip()
        new_password = self.recover_new_password_input.text()
        confirm_password = self.recover_confirm_password_input.text()
        if not recovery_key or not new_password or not confirm_password:
            QMessageBox.warning(self, "Error", "All fields are required.")
            return
        if new_password != confirm_password:
            QMessageBox.warning(self, "Error", "New passwords do not match.")
            return
        try:
            PasswordVault.recover(self.paths, recovery_key, new_password)
        except (CredentialFormatError, CredentialRecoveryError):
            QMessageBox.warning(
                self,
                "Recovery Failed",
                "The recovery key is invalid or the credential metadata cannot be used. "
                "Existing configuration was not changed.",
            )
            return
        self._new_password = new_password
        self.accept()

    def reset_configuration(self) -> None:
        new_password = self.reset_new_password_input.text()
        confirm_password = self.reset_confirm_password_input.text()
        if not new_password or not confirm_password:
            QMessageBox.warning(self, "Error", "All fields are required.")
            return
        if new_password != confirm_password:
            QMessageBox.warning(self, "Error", "New passwords do not match.")
            return
        if self.reset_confirmation_input.text() != "RESET":
            QMessageBox.warning(
                self, "Confirmation Required", 'Type "RESET" exactly to continue.'
            )
            return
        try:
            result = reset_encrypted_configuration(self.paths, new_password)
        except CredentialRecoveryError as exc:
            logger.error(
                "Password reset dialog failed (%s).",
                exc.__class__.__name__,
            )
            QMessageBox.warning(
                self,
                "Reset Failed",
                "The reset could not be completed safely. Existing files were "
                "restored where possible; inspect the password-reset backup before retrying.",
            )
            return
        self._new_password = new_password
        self._new_recovery_key = result.recovery_key
        self.accept()


def _password_input(placeholder: str) -> QLineEdit:
    widget = QLineEdit()
    widget.setPlaceholderText(placeholder)
    widget.setEchoMode(QLineEdit.EchoMode.Password)
    return widget
