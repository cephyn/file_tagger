import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

CONFIG_FILE = "config.encrypted"
SALT_FILE = ".salt"

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

class Config:
    def __init__(self, password: str):
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