"""
Config file encryption utilities for protecting sensitive data like API keys.

Uses AES-256-GCM encryption with scrypt key derivation for maximum security.
Provides authenticated encryption to ensure both confidentiality and integrity.
"""
import base64
import os
import secrets
from typing import Dict, Any, Optional

from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import scrypt
from loguru import logger


class ConfigEncryption:
    """Handles encryption/decryption of configuration values with authenticated encryption."""
    
    # Constants
    VERSION = 1  # Clean start with new format
    ENCRYPTION_PREFIX = "enc:"
    SALT_SIZE = 32  # 256 bits
    KEY_SIZE = 32   # 256 bits for AES-256
    NONCE_SIZE = 12  # 96 bits for GCM
    TAG_SIZE = 16   # 128 bits for GCM
    
    # scrypt parameters (2^20 = 1048576)
    SCRYPT_N = 1048576  # CPU/memory cost
    SCRYPT_R = 8        # Block size
    SCRYPT_P = 1        # Parallelization
    
    def __init__(self):
        """Initialize the encryption module."""
        pass
    
    def encrypt_value(self, plaintext: str, password: str) -> str:
        """
        Encrypt a string value using AES-256-GCM with scrypt key derivation.
        
        Format: Base64(VERSION || SALT || NONCE || CIPHERTEXT || TAG)
        
        Args:
            plaintext: The string to encrypt
            password: The password to use for encryption
            
        Returns:
            Encrypted string with prefix
        """
        # Generate random salt and nonce
        salt = os.urandom(self.SALT_SIZE)
        nonce = os.urandom(self.NONCE_SIZE)
        
        # Derive key using scrypt
        key = scrypt(
            password.encode('utf-8'),
            salt,
            key_len=self.KEY_SIZE,
            N=self.SCRYPT_N,
            r=self.SCRYPT_R,
            p=self.SCRYPT_P
        )
        
        # Create cipher and encrypt
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))
        
        # Build complete message
        version_bytes = bytes([self.VERSION])
        combined = version_bytes + salt + nonce + ciphertext + tag
        
        # Encode to base64
        encrypted_b64 = base64.b64encode(combined).decode('utf-8')
        
        # Return with prefix
        return f"{self.ENCRYPTION_PREFIX}{encrypted_b64}"
    
    def decrypt_value(self, encrypted_value: str, password: str) -> str:
        """
        Decrypt an encrypted string value with authentication.
        
        Args:
            encrypted_value: The encrypted string (with or without prefix)
            password: The password to use for decryption
            
        Returns:
            The decrypted string
            
        Raises:
            ValueError: If decryption or authentication fails
        """
        # Remove prefix if present
        if encrypted_value.startswith(self.ENCRYPTION_PREFIX):
            encrypted_value = encrypted_value[len(self.ENCRYPTION_PREFIX):]
        
        try:
            # Decode from base64
            combined = base64.b64decode(encrypted_value)
            
            # Check minimum length
            min_length = 1 + self.SALT_SIZE + self.NONCE_SIZE + self.TAG_SIZE
            if len(combined) < min_length:
                raise ValueError("Invalid encrypted data length")
            
            # Extract components
            version = combined[0]
            if version != self.VERSION:
                raise ValueError(f"Unsupported encryption version: {version}")
            
            offset = 1
            salt = combined[offset:offset + self.SALT_SIZE]
            offset += self.SALT_SIZE
            
            nonce = combined[offset:offset + self.NONCE_SIZE]
            offset += self.NONCE_SIZE
            
            # Ciphertext is everything except the last TAG_SIZE bytes
            ciphertext = combined[offset:-self.TAG_SIZE]
            tag = combined[-self.TAG_SIZE:]
            
            # Derive key using scrypt
            key = scrypt(
                password.encode('utf-8'),
                salt,
                key_len=self.KEY_SIZE,
                N=self.SCRYPT_N,
                r=self.SCRYPT_R,
                p=self.SCRYPT_P
            )
            
            # Create cipher and decrypt
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {type(e).__name__}")
            raise ValueError("Failed to decrypt value. Invalid password or corrupted data.")
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted based on its prefix."""
        return isinstance(value, str) and value.startswith(self.ENCRYPTION_PREFIX)
    
    def encrypt_config(self, config: Dict[str, Any], password: str) -> Dict[str, Any]:
        """
        Encrypt all string values in a configuration dictionary.
        
        Args:
            config: Dictionary of configuration values
            password: Password to use for encryption
            
        Returns:
            Dictionary with encrypted values
        """
        encrypted_config = {}
        
        for key, value in config.items():
            if isinstance(value, str) and value and not self.is_encrypted(value):
                # Encrypt non-empty strings that aren't already encrypted
                encrypted_config[key] = self.encrypt_value(value, password)
            elif isinstance(value, dict):
                # Recursively encrypt nested dictionaries
                encrypted_config[key] = self.encrypt_config(value, password)
            else:
                # Keep non-string values as-is
                encrypted_config[key] = value
        
        return encrypted_config
    
    def decrypt_config(self, config: Dict[str, Any], password: str) -> Dict[str, Any]:
        """
        Decrypt all encrypted values in a configuration dictionary.
        
        Args:
            config: Dictionary with potentially encrypted values
            password: Password to use for decryption
            
        Returns:
            Dictionary with decrypted values
        """
        decrypted_config = {}
        
        for key, value in config.items():
            if self.is_encrypted(value):
                try:
                    decrypted_config[key] = self.decrypt_value(value, password)
                except ValueError:
                    logger.warning(f"Failed to decrypt value for key: {key}")
                    decrypted_config[key] = value  # Keep encrypted on failure
            elif isinstance(value, dict):
                # Recursively decrypt nested dictionaries
                decrypted_config[key] = self.decrypt_config(value, password)
            else:
                # Keep non-encrypted values as-is
                decrypted_config[key] = value
        
        return decrypted_config
    
    def create_password_verifier(self, password: str) -> str:
        """
        Create an encrypted test value for password verification.
        
        Args:
            password: The password to verify later
            
        Returns:
            Encrypted verification token
        """
        # Use a known test string with timestamp for uniqueness
        test_value = f"PASSWORD_VERIFICATION_TOKEN_{secrets.token_hex(16)}"
        return self.encrypt_value(test_value, password)
    
    def verify_password(self, password: str, verifier: str) -> bool:
        """
        Verify a password by attempting to decrypt a verification token.
        
        Args:
            password: The password to verify
            verifier: The encrypted verification token
            
        Returns:
            True if password is correct, False otherwise
        """
        try:
            decrypted = self.decrypt_value(verifier, password)
            # Check if it's a valid verification token
            return decrypted.startswith("PASSWORD_VERIFICATION_TOKEN_")
        except ValueError:
            return False
    
    def detect_api_keys(self, config: Dict[str, Any]) -> bool:
        """
        Detect if there are API keys in the configuration that should be encrypted.
        
        Args:
            config: Configuration dictionary to check
            
        Returns:
            True if unencrypted API keys are found
        """
        def check_dict_for_keys(d: Dict[str, Any], parent_key: str = "") -> bool:
            for key, value in d.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                
                # Check for API key patterns
                if key.lower() in ['api_key', 'apikey', 'api-key', 'secret', 'token', 'password']:
                    if isinstance(value, str) and value.strip() and not self.is_encrypted(value):
                        # Skip placeholders
                        if not (value.startswith('<') and value.endswith('>')):
                            return True
                
                # Check nested dictionaries
                if isinstance(value, dict):
                    if check_dict_for_keys(value, full_key):
                        return True
            
            return False
        
        return check_dict_for_keys(config)


# Global instance for convenience
config_encryption = ConfigEncryption()