# filepath: c:\Users\cephy\Documents\App Development\file_tagger\password_management.py
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QMessageBox,
    QWidget,
    QInputDialog,
)
from PySide6.QtCore import Qt
from config import Config, get_recovery_key


# Function to get the password from the user
def get_password(parent=None):
    """
    Prompts the user for their password or retrieves a stored password.
    Returns the password string or None if canceled.
    """
    password, ok = QInputDialog.getText(
        parent, "Password Required", "Enter your password:", QLineEdit.Password
    )

    if ok and password:
        return password
    return None


class PasswordManagementDialog(QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Password Management")
        self.setModal(True)
        layout = QVBoxLayout(self)

        # Create tab widget
        tab_widget = QTabWidget()

        # Change Password tab
        change_tab = QWidget()
        change_layout = QVBoxLayout(change_tab)

        # Old password
        old_pass_layout = QHBoxLayout()
        old_pass_label = QLabel("Current Password:")
        self.old_pass_input = QLineEdit()
        self.old_pass_input.setEchoMode(QLineEdit.Password)
        old_pass_layout.addWidget(old_pass_label)
        old_pass_layout.addWidget(self.old_pass_input)

        # New password
        new_pass_layout = QHBoxLayout()
        new_pass_label = QLabel("New Password:")
        self.new_pass_input = QLineEdit()
        self.new_pass_input.setEchoMode(QLineEdit.Password)
        new_pass_layout.addWidget(new_pass_label)
        new_pass_layout.addWidget(self.new_pass_input)

        # Confirm new password
        confirm_pass_layout = QHBoxLayout()
        confirm_pass_label = QLabel("Confirm Password:")
        self.confirm_pass_input = QLineEdit()
        self.confirm_pass_input.setEchoMode(QLineEdit.Password)
        confirm_pass_layout.addWidget(confirm_pass_label)
        confirm_pass_layout.addWidget(self.confirm_pass_input)

        # Change password button
        change_btn = QPushButton("Change Password")
        change_btn.clicked.connect(self.change_password)

        # Add to change password tab
        change_layout.addLayout(old_pass_layout)
        change_layout.addLayout(new_pass_layout)
        change_layout.addLayout(confirm_pass_layout)
        change_layout.addWidget(change_btn)
        change_layout.addStretch()

        # Recovery tab
        recovery_tab = QWidget()
        recovery_layout = QVBoxLayout(recovery_tab)

        # Recovery key section
        key_layout = QHBoxLayout()
        key_label = QLabel("Recovery Key:")
        self.key_input = QLineEdit()
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.key_input)

        # New password for recovery
        recovery_pass_layout = QHBoxLayout()
        recovery_pass_label = QLabel("New Password:")
        self.recovery_pass_input = QLineEdit()
        self.recovery_pass_input.setEchoMode(QLineEdit.Password)
        recovery_pass_layout.addWidget(recovery_pass_label)
        recovery_pass_layout.addWidget(self.recovery_pass_input)

        # Confirm new password for recovery
        recovery_confirm_layout = QHBoxLayout()
        recovery_confirm_label = QLabel("Confirm Password:")
        self.recovery_confirm_input = QLineEdit()
        self.recovery_confirm_input.setEchoMode(QLineEdit.Password)
        recovery_confirm_layout.addWidget(recovery_confirm_label)
        recovery_confirm_layout.addWidget(self.recovery_confirm_input)

        # Recover button
        recover_btn = QPushButton("Recover Access")
        recover_btn.clicked.connect(self.recover_password)

        # Show recovery key button
        show_key_btn = QPushButton("Show My Recovery Key")
        show_key_btn.clicked.connect(self.show_recovery_key)

        # Reset recovery key button
        reset_key_btn = QPushButton("Reset Recovery Key")
        reset_key_btn.clicked.connect(self.reset_recovery_key)

        # Add to recovery tab
        recovery_layout.addLayout(key_layout)
        recovery_layout.addLayout(recovery_pass_layout)
        recovery_layout.addLayout(recovery_confirm_layout)
        recovery_layout.addWidget(recover_btn)
        recovery_layout.addWidget(show_key_btn)
        recovery_layout.addWidget(reset_key_btn)
        recovery_layout.addStretch()

        # Add tabs to widget
        tab_widget.addTab(change_tab, "Change Password")
        tab_widget.addTab(recovery_tab, "Recovery")

        # Add tab widget to main layout
        layout.addWidget(tab_widget)

        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)

    def change_password(self):
        old_pass = self.old_pass_input.text()
        new_pass = self.new_pass_input.text()
        confirm_pass = self.confirm_pass_input.text()

        if not old_pass or not new_pass or not confirm_pass:
            QMessageBox.warning(self, "Error", "All fields are required")
            return

        if new_pass != confirm_pass:
            QMessageBox.warning(self, "Error", "New passwords do not match")
            return

        if self.config.change_password(old_pass, new_pass):
            QMessageBox.information(self, "Success", "Password changed successfully")
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Current password is incorrect")

    def recover_password(self):
        recovery_key = self.key_input.text()
        new_pass = self.recovery_pass_input.text()
        confirm_pass = self.recovery_confirm_input.text()

        if not recovery_key or not new_pass or not confirm_pass:
            QMessageBox.warning(self, "Error", "All fields are required")
            return

        if new_pass != confirm_pass:
            QMessageBox.warning(self, "Error", "New passwords do not match")
            return

        if self.config.recover_password(recovery_key, new_pass):
            QMessageBox.information(
                self,
                "Success",
                "Access recovered successfully\n"
                "Please restart the application to continue",
            )
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Invalid recovery key")

    def show_recovery_key(self):
        current_pass, ok = QInputDialog.getText(
            self,
            "Verify Password",
            "Enter current password to view recovery key:",
            QLineEdit.Password,
        )

        if not ok or not current_pass:
            return

        recovery_key = get_recovery_key(current_pass)
        if recovery_key:
            QMessageBox.information(
                self,
                "Recovery Key",
                f"Your recovery key is:\n\n{recovery_key}\n\n"
                "Store this key in a safe place. You will need it if you "
                "forget your password.",
            )
        else:
            QMessageBox.warning(self, "Error", "Incorrect password")

    def reset_recovery_key(self):
        current_pass, ok = QInputDialog.getText(
            self,
            "Verify Password",
            "Enter current password to reset recovery key:",
            QLineEdit.Password,
        )

        if not ok or not current_pass:
            return

        success, new_key = self.config.reset_recovery_key(current_pass)
        if success:
            QMessageBox.information(
                self,
                "Success",
                f"Your new recovery key is:\n\n{new_key}\n\n"
                "Store this key in a safe place. You will need it if you "
                "forget your password. The old recovery key is no longer valid.",
            )
        else:
            QMessageBox.warning(self, "Error", "Incorrect password")
