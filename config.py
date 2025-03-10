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

CONFIG_FILE = "config.encrypted"
SALT_FILE = ".salt"
PASS_HASH_FILE = ".passhash"
RECOVERY_FILE = ".recovery"

def generate_key(password: str, salt: bytes = None) -> bytes:
    """Generate an encryption key from password and salt."""
    if salt is None:
        salt = os.urandom(16)
        # Save salt for future use
        with open(SALT_FILE, 'wb') as f:
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
    if os.path.exists(SALT_FILE):
        with open(SALT_FILE, 'rb') as f:
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
    if not os.path.exists(PASS_HASH_FILE):
        return True  # First time setup
    
    try:
        with open(PASS_HASH_FILE, 'rb') as f:
            stored_data = json.load(f)
            stored_hash = stored_data['hash']
            salt = base64.b64decode(stored_data['salt'])
            
        password_hash, _ = hash_password(password, salt)
        return password_hash == stored_hash
    except Exception:
        return False

def store_password_hash(password: str):
    """Store the hashed password."""
    password_hash, salt = hash_password(password)
    with open(PASS_HASH_FILE, 'w') as f:
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
    fernet = get_encryption_key(password)
    encrypted_key = fernet.encrypt(recovery_key.encode())
    with open(RECOVERY_FILE, 'wb') as f:
        f.write(encrypted_key)

def get_recovery_key(password: str) -> str:
    """Get the recovery key using the current password."""
    if not os.path.exists(RECOVERY_FILE):
        return None
    try:
        fernet = get_encryption_key(password)
        with open(RECOVERY_FILE, 'rb') as f:
            encrypted_key = f.read()
        decrypted_key = fernet.decrypt(encrypted_key)
        return decrypted_key.decode()
    except Exception:
        return None

class Config:
    def __init__(self, password: str):
        """Initialize config with password verification."""
        if not verify_password(password):
            raise ValueError("Invalid password")
            
        # Store password hash if this is first time setup
        if not os.path.exists(PASS_HASH_FILE):
            store_password_hash(password)
            
        self.password = password
        self.fernet = get_encryption_key(password)
        self.config_data = self._load_config()
        
        # Generate recovery key on first setup
        if not os.path.exists(RECOVERY_FILE):
            recovery_key = generate_recovery_key()
            store_recovery_key(recovery_key, password)
    
    def _load_config(self) -> dict:
        """Load and decrypt configuration file."""
        if not os.path.exists(CONFIG_FILE):
            return {
                'api_keys': {
                    'openai': '',
                    'gemini': '',
                    'anthropic': ''
                },
                'selected_provider': 'openai',
                'home_directory': str(Path.home())  # Default to user's home directory
            }
        
        try:
            with open(CONFIG_FILE, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            config = json.loads(decrypted_data)
            
            # Add home_directory if it doesn't exist in older config files
            if 'home_directory' not in config:
                config['home_directory'] = str(Path.home())
                
            return config
        except Exception:
            return {
                'api_keys': {
                    'openai': '',
                    'gemini': '',
                    'anthropic': ''
                },
                'selected_provider': 'openai',
                'home_directory': str(Path.home())
            }
    
    def save(self):
        """Encrypt and save configuration."""
        encrypted_data = self.fernet.encrypt(json.dumps(self.config_data).encode())
        with open(CONFIG_FILE, 'wb') as f:
            f.write(encrypted_data)
    
    def set_api_key(self, provider: str, key: str):
        """Set API key for a provider."""
        self.config_data['api_keys'][provider] = key
        self.save()
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for a provider."""
        return self.config_data['api_keys'].get(provider, '')
    
    def set_selected_provider(self, provider: str):
        """Set the selected AI provider."""
        self.config_data['selected_provider'] = provider
        self.save()
    
    def get_selected_provider(self) -> str:
        """Get the selected AI provider."""
        return self.config_data.get('selected_provider', 'openai')
    
    def change_password(self, old_password: str, new_password: str) -> bool:
        """Change the encryption password."""
        if not verify_password(old_password):
            return False
            
        try:
            # Re-encrypt configuration with new password
            old_fernet = get_encryption_key(old_password)
            new_fernet = get_encryption_key(new_password)
            
            # Re-encrypt config file
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = old_fernet.decrypt(encrypted_data)
                new_encrypted_data = new_fernet.encrypt(decrypted_data)
                with open(CONFIG_FILE, 'wb') as f:
                    f.write(new_encrypted_data)
            
            # Re-encrypt recovery key
            if os.path.exists(RECOVERY_FILE):
                with open(RECOVERY_FILE, 'rb') as f:
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
            if not os.path.exists(RECOVERY_FILE):
                return False
                
            with open(RECOVERY_FILE, 'rb') as f:
                stored_encrypted_key = f.read()
            
            # First verify the recovery key by trying to decrypt stored key
            old_passwords = []  # We'll collect working passwords
            
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'rb') as f:
                    config_data = f.read()
            
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
        if os.path.exists(SALT_FILE):
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