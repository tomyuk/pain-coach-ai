"""Encryption manager for medical data protection."""

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import base64
import os
import json
from typing import Dict, Any, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Medical data encryption manager with KeyChain integration."""
    
    def __init__(self, app_name: str = "PainCoachAI"):
        self.app_name = app_name
        self.salt = self._get_or_create_salt()
        self._cipher_cache: Dict[str, Fernet] = {}
        
    def _get_or_create_salt(self) -> bytes:
        """Get or create encryption salt."""
        try:
            salt_b64 = keyring.get_password(self.app_name, "encryption_salt")
            if salt_b64:
                return base64.b64decode(salt_b64)
        except Exception as e:
            logger.warning(f"Could not retrieve salt from keyring: {e}")
            
        # Generate new salt
        salt = os.urandom(16)
        try:
            keyring.set_password(
                self.app_name, 
                "encryption_salt", 
                base64.b64encode(salt).decode()
            )
        except Exception as e:
            logger.error(f"Could not store salt in keyring: {e}")
            
        return salt
    
    def _get_cipher(self, user_id: str) -> Fernet:
        """Get user-specific cipher."""
        if user_id in self._cipher_cache:
            return self._cipher_cache[user_id]
            
        try:
            # Try to get key from KeyChain
            key_b64 = keyring.get_password(self.app_name, f"user_key_{user_id}")
            if key_b64:
                key = base64.b64decode(key_b64)
            else:
                # Generate new key
                key = Fernet.generate_key()
                keyring.set_password(
                    self.app_name, 
                    f"user_key_{user_id}", 
                    base64.b64encode(key).decode()
                )
        except Exception as e:
            logger.warning(f"KeyChain access failed: {e}, using PBKDF2 fallback")
            # Fallback to password-based key
            password = f"pain_coach_user_{user_id}"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            
        cipher = Fernet(key)
        self._cipher_cache[user_id] = cipher
        return cipher
    
    def encrypt_data(self, data: Any, user_id: str) -> bytes:
        """Encrypt data for a specific user."""
        cipher = self._get_cipher(user_id)
        
        # Convert to JSON string
        json_str = json.dumps(data, ensure_ascii=False, default=str)
        
        # Encrypt
        encrypted = cipher.encrypt(json_str.encode('utf-8'))
        return encrypted
    
    def decrypt_data(self, encrypted_data: bytes, user_id: str) -> Any:
        """Decrypt data for a specific user."""
        cipher = self._get_cipher(user_id)
        
        try:
            # Decrypt
            decrypted_bytes = cipher.decrypt(encrypted_data)
            json_str = decrypted_bytes.decode('utf-8')
            
            # Parse JSON
            return json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for searchable encryption."""
        return hashlib.sha256(
            (data + base64.b64encode(self.salt).decode()).encode()
        ).hexdigest()
    
    def encrypt_field(self, field_value: str, user_id: str) -> bytes:
        """Encrypt a single field."""
        return self.encrypt_data(field_value, user_id)
    
    def decrypt_field(self, encrypted_field: bytes, user_id: str) -> str:
        """Decrypt a single field."""
        return self.decrypt_data(encrypted_field, user_id)
    
    def rotate_user_key(self, user_id: str, old_data: Dict[str, bytes]) -> Dict[str, bytes]:
        """Rotate encryption key for a user."""
        # Decrypt all data with old key
        old_cipher = self._cipher_cache.get(user_id)
        if not old_cipher:
            raise ValueError("No existing cipher found for user")
            
        decrypted_data = {}
        for field, encrypted_value in old_data.items():
            decrypted_data[field] = self.decrypt_data(encrypted_value, user_id)
        
        # Generate new key
        new_key = Fernet.generate_key()
        try:
            keyring.set_password(
                self.app_name, 
                f"user_key_{user_id}", 
                base64.b64encode(new_key).decode()
            )
        except Exception as e:
            logger.error(f"Could not store new key: {e}")
            raise
        
        # Clear cache to force new cipher
        if user_id in self._cipher_cache:
            del self._cipher_cache[user_id]
        
        # Re-encrypt all data with new key
        reencrypted_data = {}
        for field, value in decrypted_data.items():
            reencrypted_data[field] = self.encrypt_data(value, user_id)
        
        return reencrypted_data
    
    def delete_user_key(self, user_id: str):
        """Delete user's encryption key."""
        try:
            keyring.delete_password(self.app_name, f"user_key_{user_id}")
        except Exception as e:
            logger.warning(f"Could not delete key from keyring: {e}")
        
        # Clear cache
        if user_id in self._cipher_cache:
            del self._cipher_cache[user_id]
    
    def verify_encryption_integrity(self, encrypted_data: bytes, user_id: str) -> bool:
        """Verify that encrypted data can be decrypted."""
        try:
            self.decrypt_data(encrypted_data, user_id)
            return True
        except Exception:
            return False