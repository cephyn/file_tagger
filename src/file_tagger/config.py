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
from .infrastructure.app_paths import AppPaths


def _get_app_paths() -> AppPaths:
    paths = AppPaths.discover()
    paths.ensure_directories()
    return paths

# Default system message for AI interactions
DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant for tagging and organizing files.
Your primary task is to analyze files and suggest relevant tags based on their names and contents.
Suggest both existing tags that match and new tags that might be useful."""

def generate_key(password: str, salt: bytes = None) -> bytes:
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

def hash_password(password: str, salt: bytes = None) -> tuple[str, bytes]:
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

def get_recovery_key(password: str) -> str:
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
        """Initialize config with password verification."""
        self.paths = _get_app_paths()
        if not verify_password(password):
            raise ValueError("Invalid password")

        legacy_config_pending = (
            self.paths.legacy_database.parent / "config.encrypted"
        ).is_file() and not self.paths.config_file.is_file()

        # Store password hash if this is first time setup
        if (
            not legacy_config_pending
            and not os.path.exists(self.paths.password_hash_file)
        ):
            store_password_hash(password)
            
        self.password = password
        self.fernet = get_encryption_key(password)
        self.config_data = self._load_config()
        
        # Generate recovery key on first setup
        if (
            not legacy_config_pending
            and not os.path.exists(self.paths.recovery_file)
        ):
            recovery_key = generate_recovery_key()
            store_recovery_key(recovery_key, password)
    
    def _load_config(self) -> dict:
        """Load and decrypt configuration file."""
        if not os.path.exists(self.paths.config_file):
            return {
                'home_directory': str(Path.home()),  # Default to user's home directory
                'system_message': DEFAULT_SYSTEM_MESSAGE
            }
        
        try:
            with open(self.paths.config_file, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            config = json.loads(decrypted_data)

            # Add home_directory if it doesn't exist in older config files
            if 'home_directory' not in config:
                config['home_directory'] = str(Path.home())
            
            # Add system_message if it doesn't exist in older config files
            if 'system_message' not in config:
                config['system_message'] = DEFAULT_SYSTEM_MESSAGE
                
            return config
        except Exception:
            return {
                'home_directory': str(Path.home()),
                'system_message': DEFAULT_SYSTEM_MESSAGE
            }
    
    def save(self):
        """Encrypt and save configuration."""
        encrypted_data = self.fernet.encrypt(json.dumps(self.config_data).encode())
        with open(self.paths.config_file, 'wb') as f:
            f.write(encrypted_data)

    def reload_after_migration(self) -> None:
        """Reload encryption state and settings after legacy config migration."""
        if not verify_password(self.password):
            raise ValueError("Invalid password after configuration migration")
        self.fernet = get_encryption_key(self.password)
        self.config_data = self._load_config()
    
    def change_password(self, old_password: str, new_password: str) -> bool:
        """Change the encryption password."""
        if not verify_password(old_password):
            return False
            
        try:
            # Re-encrypt configuration with new password
            old_fernet = get_encryption_key(old_password)
            new_fernet = get_encryption_key(new_password)
            
            # Re-encrypt config file
            if os.path.exists(self.paths.config_file):
                with open(self.paths.config_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = old_fernet.decrypt(encrypted_data)
                new_encrypted_data = new_fernet.encrypt(decrypted_data)
                with open(self.paths.config_file, 'wb') as f:
                    f.write(new_encrypted_data)
            
            # Re-encrypt recovery key
            if os.path.exists(self.paths.recovery_file):
                with open(self.paths.recovery_file, 'rb') as f:
                    encrypted_key = f.read()
                recovery_key = old_fernet.decrypt(encrypted_key).decode()
                store_recovery_key(recovery_key, new_password)
            
            # Update password hash
            store_password_hash(new_password)
            
            self.password = new_password
            self.fernet = new_fernet
            return True
            
        except Exception:
            return False
    
    def recover_password(self, recovery_key: str, new_password: str) -> bool:
        """Recover access using recovery key and set new password."""
        try:
            # Try each possible encryption key until we find one that works
            if not os.path.exists(self.paths.recovery_file):
                return False
                
            with open(self.paths.recovery_file, 'rb') as f:
                stored_encrypted_key = f.read()
            
            # First verify the recovery key by trying to decrypt stored key
            old_passwords = []  # We'll collect working passwords
            
            for password in self._get_recent_passwords():
                try:
                    fernet = get_encryption_key(password)
                    decrypted_key = fernet.decrypt(stored_encrypted_key).decode()
                    if decrypted_key == recovery_key:
                        old_passwords.append(password)
                except Exception:
                    continue
            
            if not old_passwords:
                return False
            
            # Use the first working password to migrate to new password
            return self.change_password(old_passwords[0], new_password)
            
        except Exception:
            return False
    
    def _get_recent_passwords(self) -> list:
        """Get list of possible recent passwords based on salt file age."""
        # This is a placeholder - in a real implementation, you might want to
        # keep a secure log of recent password hashes or implement a more
        # sophisticated recovery mechanism
        if os.path.exists(self.paths.salt_file):
            return [self.password]  # Current password
        return []

    def reset_recovery_key(self, current_password: str) -> tuple[bool, str]:
        """Reset the recovery key. Returns (success, new_key)."""
        try:
            if not verify_password(current_password):
                return False, ""
                
            # Generate and store new recovery key
            new_key = generate_recovery_key()
            store_recovery_key(new_key, current_password)
            return True, new_key
            
        except Exception:
            return False, ""

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
