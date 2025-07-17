"""
Config file encryption utilities for protecting sensitive data like API keys.

Uses AES-256 encryption with PBKDF2 key derivation for secure password-based encryption.
"""
import base64
import hashlib
import json
import os
import secrets
import time
from typing import Dict, Optional, Tuple, Any

from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad, unpad
from loguru import logger

from ..Metrics.metrics_logger import log_counter, log_histogram


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
        start_time = time.time()
        log_counter("config_encryption_derive_key_attempt")
        
        key = PBKDF2(
            password.encode('utf-8'),
            salt,
            dkLen=self.KEY_SIZE,
            count=self.ITERATIONS,
            hmac_hash_module=hashlib.sha256
        )
        
        duration = time.time() - start_time
        log_histogram("config_encryption_derive_key_duration", duration)
        log_counter("config_encryption_derive_key_success")
        
        return key
    
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
        start_time = time.time()
        log_counter("config_encryption_encrypt_value_attempt", labels={
            "has_salt": str(salt is not None)
        })
        log_histogram("config_encryption_plaintext_length", len(value))
        
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
        result = f"{self.ENCRYPTION_PREFIX}{encrypted_b64}", salt
        
        # Log success
        duration = time.time() - start_time
        log_histogram("config_encryption_encrypt_value_duration", duration)
        log_histogram("config_encryption_ciphertext_length", len(encrypted_b64))
        log_counter("config_encryption_encrypt_value_success")
        
        return result
    
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
        start_time = time.time()
        log_counter("config_encryption_decrypt_value_attempt", labels={
            "has_prefix": str(encrypted_value.startswith(self.ENCRYPTION_PREFIX))
        })
        
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
            
            decrypted = plaintext.decode('utf-8')
            
            # Log success
            duration = time.time() - start_time
            log_histogram("config_encryption_decrypt_value_duration", duration)
            log_counter("config_encryption_decrypt_value_success")
            
            return decrypted
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            log_counter("config_encryption_decrypt_value_error", labels={"error_type": type(e).__name__})
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
        start_time = time.time()
        log_counter("config_encryption_encrypt_section_attempt", labels={
            "keys_count": str(len(config_dict))
        })
        
        if salt is None:
            salt = self.generate_salt()
        
        encrypted_dict = {}
        encrypted_count = 0
        
        for key, value in config_dict.items():
            if isinstance(value, str) and value and not self.is_encrypted(value):
                # Only encrypt non-empty strings that aren't already encrypted
                encrypted_value, _ = self.encrypt_value(value, password, salt)
                encrypted_dict[key] = encrypted_value
                encrypted_count += 1
            else:
                # Keep non-string values as-is
                encrypted_dict[key] = value
        
        # Log completion
        duration = time.time() - start_time
        log_histogram("config_encryption_encrypt_section_duration", duration)
        log_counter("config_encryption_encrypt_section_complete", labels={
            "total_keys": str(len(config_dict)),
            "encrypted_keys": str(encrypted_count)
        })
        
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
        start_time = time.time()
        log_counter("config_encryption_decrypt_section_attempt", labels={
            "keys_count": str(len(config_dict))
        })
        
        decrypted_dict = {}
        decrypted_count = 0
        failed_count = 0
        
        for key, value in config_dict.items():
            if isinstance(value, str) and self.is_encrypted(value):
                try:
                    decrypted_value = self.decrypt_value(value, password, salt)
                    decrypted_dict[key] = decrypted_value
                    decrypted_count += 1
                except ValueError:
                    # If decryption fails, keep the encrypted value
                    logger.warning(f"Failed to decrypt value for key: {key}")
                    decrypted_dict[key] = value
                    failed_count += 1
            else:
                # Keep non-encrypted values as-is
                decrypted_dict[key] = value
        
        # Log completion
        duration = time.time() - start_time
        log_histogram("config_encryption_decrypt_section_duration", duration)
        log_counter("config_encryption_decrypt_section_complete", labels={
            "total_keys": str(len(config_dict)),
            "decrypted_keys": str(decrypted_count),
            "failed_keys": str(failed_count)
        })
        
        return decrypted_dict
    
    def detect_api_keys(self, config_dict: Dict[str, Any]) -> bool:
        """
        Detect if there are API keys in the configuration.
        
        Looks for 'api_settings' sections with non-empty 'api_key' values.
        """
        log_counter("config_encryption_detect_api_keys_attempt")
        api_key_count = 0
        
        for section_name, section_value in config_dict.items():
            if section_name.startswith('api_settings.') and isinstance(section_value, dict):
                api_key = section_value.get('api_key', '')
                # Check if API key exists and is not a placeholder
                if api_key and not api_key.startswith('<') and not api_key.endswith('>'):
                    api_key_count += 1
        
        log_counter("config_encryption_detect_api_keys_result", labels={
            "has_api_keys": str(api_key_count > 0),
            "api_key_count": str(api_key_count)
        })
        
        return api_key_count > 0
    
    def clear_cache(self):
        """Clear any cached encryption keys from memory."""
        self._cached_key = None
        self._salt = None


# Global instance for convenience
config_encryption = ConfigEncryption()