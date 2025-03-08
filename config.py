import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import hashlib

CONFIG_FILE = "config.encrypted"
SALT_FILE = ".salt"
PASS_HASH_FILE = ".passhash"

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
    
    def _load_config(self) -> dict:
        """Load and decrypt configuration file."""
        if not os.path.exists(CONFIG_FILE):
            return {
                'api_keys': {
                    'openai': '',
                    'gemini': '',
                    'anthropic': ''
                },
                'selected_provider': 'openai'
            }
        
        try:
            with open(CONFIG_FILE, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data)
        except Exception:
            return {
                'api_keys': {
                    'openai': '',
                    'gemini': '',
                    'anthropic': ''
                },
                'selected_provider': 'openai'
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