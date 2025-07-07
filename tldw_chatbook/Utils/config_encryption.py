"""
Config file encryption utilities for protecting sensitive data like API keys.

Uses AES-256 encryption with PBKDF2 key derivation for secure password-based encryption.
"""
import base64
import hashlib
import json
import os
import secrets
from typing import Dict, Optional, Tuple, Any

from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad, unpad
from loguru import logger


class ConfigEncryption:
    """Handles encryption/decryption of configuration values."""
    
    ENCRYPTION_PREFIX = "enc:"
    SALT_SIZE = 32  # 256 bits
    KEY_SIZE = 32   # 256 bits for AES-256
    BLOCK_SIZE = 16  # AES block size
    ITERATIONS = 100000  # PBKDF2 iterations
    
    def __init__(self):
        self._cached_key: Optional[bytes] = None
        self._salt: Optional[bytes] = None
    
    def generate_salt(self) -> bytes:
        """Generate a new random salt for key derivation."""
        return get_random_bytes(self.SALT_SIZE)
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive an encryption key from password and salt using PBKDF2."""
        return PBKDF2(
            password.encode('utf-8'),
            salt,
            dkLen=self.KEY_SIZE,
            count=self.ITERATIONS,
            hmac_hash_module=hashlib.sha256
        )
    
    def hash_password(self, password: str) -> str:
        """Create a hash of the password for verification (not for encryption)."""
        # Use a simple SHA-256 hash for verification
        # The actual encryption key is derived separately with PBKDF2
        return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify if the provided password matches the stored hash."""
        return self.hash_password(password) == stored_hash
    
    def encrypt_value(self, value: str, password: str, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """
        Encrypt a string value using AES-256-CBC.
        
        Args:
            value: The string to encrypt
            password: The password to use for encryption
            salt: Optional salt (will generate new one if not provided)
            
        Returns:
            Tuple of (encrypted_base64_string, salt)
        """
        if salt is None:
            salt = self.generate_salt()
        
        # Derive key from password
        key = self.derive_key(password, salt)
        
        # Generate random IV
        iv = get_random_bytes(self.BLOCK_SIZE)
        
        # Create cipher
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        # Pad and encrypt the data
        padded_data = pad(value.encode('utf-8'), self.BLOCK_SIZE)
        encrypted_data = cipher.encrypt(padded_data)
        
        # Combine IV and encrypted data
        combined = iv + encrypted_data
        
        # Encode to base64 for storage
        encrypted_b64 = base64.b64encode(combined).decode('utf-8')
        
        # Add prefix to indicate encrypted value
        return f"{self.ENCRYPTION_PREFIX}{encrypted_b64}", salt
    
    def decrypt_value(self, encrypted_value: str, password: str, salt: bytes) -> str:
        """
        Decrypt an encrypted string value.
        
        Args:
            encrypted_value: The encrypted string (with or without prefix)
            password: The password to use for decryption
            salt: The salt used during encryption
            
        Returns:
            The decrypted string
            
        Raises:
            ValueError: If decryption fails
        """
        # Remove prefix if present
        if encrypted_value.startswith(self.ENCRYPTION_PREFIX):
            encrypted_value = encrypted_value[len(self.ENCRYPTION_PREFIX):]
        
        try:
            # Decode from base64
            combined = base64.b64decode(encrypted_value)
            
            # Extract IV and encrypted data
            iv = combined[:self.BLOCK_SIZE]
            encrypted_data = combined[self.BLOCK_SIZE:]
            
            # Derive key from password
            key = self.derive_key(password, salt)
            
            # Create cipher and decrypt
            cipher = AES.new(key, AES.MODE_CBC, iv)
            padded_plaintext = cipher.decrypt(encrypted_data)
            
            # Remove padding
            plaintext = unpad(padded_plaintext, self.BLOCK_SIZE)
            
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt value. Invalid password or corrupted data.")
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted based on its prefix."""
        return value.startswith(self.ENCRYPTION_PREFIX)
    
    def encrypt_config_section(self, config_dict: Dict[str, Any], password: str, 
                             salt: Optional[bytes] = None) -> Tuple[Dict[str, Any], bytes]:
        """
        Encrypt all values in a configuration dictionary.
        
        Args:
            config_dict: Dictionary of configuration values
            password: Password to use for encryption
            salt: Optional salt (will generate new one if not provided)
            
        Returns:
            Tuple of (encrypted_dict, salt)
        """
        if salt is None:
            salt = self.generate_salt()
        
        encrypted_dict = {}
        
        for key, value in config_dict.items():
            if isinstance(value, str) and value and not self.is_encrypted(value):
                # Only encrypt non-empty strings that aren't already encrypted
                encrypted_value, _ = self.encrypt_value(value, password, salt)
                encrypted_dict[key] = encrypted_value
            else:
                # Keep non-string values as-is
                encrypted_dict[key] = value
        
        return encrypted_dict, salt
    
    def decrypt_config_section(self, config_dict: Dict[str, Any], password: str, 
                             salt: bytes) -> Dict[str, Any]:
        """
        Decrypt all encrypted values in a configuration dictionary.
        
        Args:
            config_dict: Dictionary with potentially encrypted values
            password: Password to use for decryption
            salt: Salt used during encryption
            
        Returns:
            Dictionary with decrypted values
        """
        decrypted_dict = {}
        
        for key, value in config_dict.items():
            if isinstance(value, str) and self.is_encrypted(value):
                try:
                    decrypted_value = self.decrypt_value(value, password, salt)
                    decrypted_dict[key] = decrypted_value
                except ValueError:
                    # If decryption fails, keep the encrypted value
                    logger.warning(f"Failed to decrypt value for key: {key}")
                    decrypted_dict[key] = value
            else:
                # Keep non-encrypted values as-is
                decrypted_dict[key] = value
        
        return decrypted_dict
    
    def detect_api_keys(self, config_dict: Dict[str, Any]) -> bool:
        """
        Detect if there are API keys in the configuration.
        
        Looks for 'api_settings' sections with non-empty 'api_key' values.
        """
        for section_name, section_value in config_dict.items():
            if section_name.startswith('api_settings.') and isinstance(section_value, dict):
                api_key = section_value.get('api_key', '')
                # Check if API key exists and is not a placeholder
                if api_key and not api_key.startswith('<') and not api_key.endswith('>'):
                    return True
        return False
    
    def clear_cache(self):
        """Clear any cached encryption keys from memory."""
        self._cached_key = None
        self._salt = None


# Global instance for convenience
config_encryption = ConfigEncryption()