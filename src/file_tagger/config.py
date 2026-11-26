import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import hashlib
import secrets
import string
from .domain.errors import (
    CredentialFormatError,
    CredentialRecoveryError,
    PasswordEncryptionError,
)
from .infrastructure.app_paths import AppPaths
from .security.password_vault import CredentialStatus, PasswordVault


def _get_app_paths() -> AppPaths:
    paths = AppPaths.discover()
    paths.ensure_directories()
    return paths

# Default system message for AI interactions
DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant for tagging and organizing files.
Your primary task is to analyze files and suggest relevant tags based on their names and contents.
Suggest both existing tags that match and new tags that might be useful."""

def generate_key(password: str, salt: bytes | None = None) -> bytes:
    """Generate an encryption key from password and salt."""
    salt_file = _get_app_paths().salt_file
    if salt is None:
        salt = os.urandom(16)
        # Save salt for future use
        with open(salt_file, 'wb') as f:
            f.write(salt)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def get_encryption_key(password: str) -> Fernet:
    """Get or create an encryption key."""
    salt_file = _get_app_paths().salt_file
    if os.path.exists(salt_file):
        with open(salt_file, 'rb') as f:
            salt = f.read()
    else:
        salt = None
    
    key = generate_key(password, salt)
    return Fernet(key)

def hash_password(password: str, salt: bytes | None = None) -> tuple[str, bytes]:
    """Hash password with a salt using SHA256."""
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return base64.b64encode(key).decode('utf-8'), salt

def verify_password(password: str) -> bool:
    """Verify if the password matches the stored hash."""
    password_hash_file = _get_app_paths().password_hash_file
    if not os.path.exists(password_hash_file):
        return True  # First time setup
    
    try:
        with open(password_hash_file, 'rb') as f:
            stored_data = json.load(f)
            stored_hash = stored_data['hash']
            salt = base64.b64decode(stored_data['salt'])
            
        password_hash, _ = hash_password(password, salt)
        return password_hash == stored_hash
    except Exception:
        return False

def store_password_hash(password: str):
    """Store the hashed password."""
    password_hash_file = _get_app_paths().password_hash_file
    password_hash, salt = hash_password(password)
    with open(password_hash_file, 'w') as f:
        json.dump({
            'hash': password_hash,
            'salt': base64.b64encode(salt).decode('utf-8')
        }, f)

def generate_recovery_key() -> str:
    """Generate a random recovery key."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(16))

def store_recovery_key(recovery_key: str, password: str):
    """Store an encrypted recovery key."""
    recovery_file = _get_app_paths().recovery_file
    fernet = get_encryption_key(password)
    encrypted_key = fernet.encrypt(recovery_key.encode())
    with open(recovery_file, 'wb') as f:
        f.write(encrypted_key)

def get_recovery_key(password: str) -> str | None:
    """Get the recovery key using the current password."""
    recovery_file = _get_app_paths().recovery_file
    if not os.path.exists(recovery_file):
        return None
    try:
        fernet = get_encryption_key(password)
        with open(recovery_file, 'rb') as f:
            encrypted_key = f.read()
        decrypted_key = fernet.decrypt(encrypted_key)
        return decrypted_key.decode()
    except Exception:
        return None

class Config:
    def __init__(self, password: str):
        self.paths = _get_app_paths()
        self._pending_recovery_key: str | None = None
        self._vault: PasswordVault | None = None
        pending_legacy_config = (
            self.paths.legacy_database.parent / "config.encrypted"
        ).is_file() and not self.paths.config_file.is_file()
        status = PasswordVault.status(self.paths)

        if status is CredentialStatus.ENVELOPE:
            self._vault = PasswordVault.open(self.paths, password)
        elif status is CredentialStatus.LEGACY:
            self._vault, self._pending_recovery_key = PasswordVault.migrate_legacy(
                self.paths, password
            )
        elif status is CredentialStatus.CORRUPT:
            raise CredentialFormatError("Credential metadata is corrupt or inconsistent.")
        elif not pending_legacy_config:
            self._vault, self._pending_recovery_key = PasswordVault.create(
                self.paths, password
            )
        self.password = password
        self.config_data = self._load_config()

    @property
    def pending_recovery_key(self) -> str | None:
        return self._pending_recovery_key

    def acknowledge_recovery_key(self) -> None:
        self._pending_recovery_key = None

    def _load_config(self) -> dict:
        if not self.paths.config_file.is_file():
            return self._default_config()
        if self._vault is None:
            return self._default_config()

        try:
            config = json.loads(
                self._vault.decrypt(self.paths.config_file.read_bytes())
            )
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise PasswordEncryptionError("Encrypted configuration is invalid.") from exc
        if not isinstance(config, dict):
            raise PasswordEncryptionError("Encrypted configuration is invalid.")
        config.setdefault("home_directory", str(Path.home()))
        config.setdefault("system_message", DEFAULT_SYSTEM_MESSAGE)
        return config

    @staticmethod
    def _default_config() -> dict:
        return {
            "home_directory": str(Path.home()),
            "system_message": DEFAULT_SYSTEM_MESSAGE,
        }

    def save(self) -> None:
        if self._vault is None:
            raise PasswordEncryptionError(
                "Configuration cannot be saved before legacy migration completes."
            )
        encrypted_data = self._vault.encrypt(json.dumps(self.config_data).encode())
        self.paths.config_file.write_bytes(encrypted_data)

    def reload_after_migration(self) -> None:
        status = PasswordVault.status(self.paths)
        if status is CredentialStatus.ENVELOPE:
            self._vault = PasswordVault.open(self.paths, self.password)
        elif status is CredentialStatus.LEGACY:
            self._vault, self._pending_recovery_key = (
                PasswordVault.migrate_legacy(self.paths, self.password)
            )
        elif status is CredentialStatus.CORRUPT:
            raise CredentialFormatError("Migrated credential metadata is invalid.")
        else:
            raise CredentialFormatError("Migrated credential metadata is missing.")
        self.config_data = self._load_config()

    def change_password(self, old_password: str, new_password: str) -> bool:
        if self._vault is None:
            return False
        try:
            self._vault.change_password(old_password, new_password)
        except (CredentialFormatError, CredentialRecoveryError):
            return False
        self.password = new_password
        return True

    def recover_password(self, recovery_key: str, new_password: str) -> bool:
        try:
            self._vault = PasswordVault.recover(
                self.paths, recovery_key, new_password
            )
        except (CredentialFormatError, CredentialRecoveryError):
            return False
        self.password = new_password
        self.config_data = self._load_config()
        return True

    def reset_recovery_key(self, current_password: str) -> tuple[bool, str]:
        try:
            return True, self.rotate_recovery_key(current_password)
        except (CredentialFormatError, CredentialRecoveryError, PasswordEncryptionError):
            return False, ""

    def rotate_recovery_key(self, current_password: str) -> str:
        if self._vault is None:
            raise PasswordEncryptionError(
                "Recovery keys cannot be rotated before legacy migration completes."
            )
        return self._vault.rotate_recovery_key(current_password)

    def set_home_directory(self, path: str):
        """Set the home directory for the file browser."""
        if os.path.exists(path) and os.path.isdir(path):
            self.config_data['home_directory'] = str(Path(path))
            self.save()
            return True
        return False
    
    def get_home_directory(self) -> str:
        """Get the configured home directory."""
        return self.config_data.get('home_directory', str(Path.home()))

    def set_system_message(self, message: str):
        """Set the custom system message for AI interactions."""
        self.config_data['system_message'] = message
        self.save()
    
    def get_system_message(self) -> str:
        """Get the custom system message for AI interactions."""
        return self.config_data.get('system_message', DEFAULT_SYSTEM_MESSAGE)
    
    def reset_system_message(self):
        """Reset the system message to the default value."""
        self.config_data['system_message'] = DEFAULT_SYSTEM_MESSAGE
        self.save()
