# security.py
# Description: Security utilities for subscription monitoring
#
# This module provides security features including:
# - Input validation and sanitization
# - Authentication credential encryption
#
# URL/SSRF policy (private IPs, cloud metadata, schemes) is owned by
# ``tldw_chatbook.Utils.egress``; this module no longer carries its own
# validator (TASK-591 removed the caller-less SecurityValidator/SSRFProtector
# that only delegated to it).
#
# Imports
import re
from typing import Optional, List
import hashlib
import secrets
from base64 import b64encode, b64decode

#
# Third-Party Imports
from loguru import logger

# Optional cryptography import
CRYPTOGRAPHY_AVAILABLE = False
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.backends import default_backend

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    logger.warning(
        "cryptography module not available. Credential encryption will be disabled."
    )
#
########################################################################################################################
#
# Security Classes
#
########################################################################################################################


class SecurityError(Exception):
    """Base exception for security-related errors."""

    pass


class CredentialEncryptor:
    """Encrypt/decrypt authentication credentials."""

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryptor.

        Args:
            key: 32-byte encryption key (generates random if not provided)
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning(
                "CredentialEncryptor: cryptography module not available. Credentials will be stored in plain text."
            )
            self.key = None
            self.backend = None
            return

        if key is None:
            # Generate a random key
            self.key = secrets.token_bytes(32)
        else:
            if len(key) != 32:
                raise ValueError("Encryption key must be 32 bytes")
            self.key = key

        self.backend = default_backend()

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string.

        Args:
            plaintext: String to encrypt

        Returns:
            Base64-encoded encrypted string
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError(
                "Cannot encrypt credentials: cryptography module not installed. "
                "Please install it with: pip install cryptography"
            )

        # Generate random IV
        iv = secrets.token_bytes(16)

        # Create cipher
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()

        # Pad plaintext
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext.encode("utf-8")) + padder.finalize()

        # Encrypt
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Combine IV and ciphertext
        encrypted = iv + ciphertext

        # Return base64 encoded
        return b64encode(encrypted).decode("utf-8")

    def decrypt(self, encrypted: str) -> str:
        """
        Decrypt a string.

        Args:
            encrypted: Base64-encoded encrypted string

        Returns:
            Decrypted plaintext
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError(
                "Cannot decrypt credentials: cryptography module not installed. "
                "Please install it with: pip install cryptography"
            )

        # Decode from base64
        encrypted_bytes = b64decode(encrypted.encode("utf-8"))

        # Extract IV and ciphertext
        iv = encrypted_bytes[:16]
        ciphertext = encrypted_bytes[16:]

        # Create cipher
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()

        # Decrypt
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        # Unpad
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

        return plaintext.decode("utf-8")

    @staticmethod
    def derive_key_from_password(
        password: str, salt: bytes = None
    ) -> tuple[bytes, bytes]:
        """
        Derive encryption key from password.

        Args:
            password: Password string
            salt: Salt bytes (generates random if not provided)

        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(16)

        # Use PBKDF2 with SHA256
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)

        return key, salt


class InputValidator:
    """Validate and sanitize user inputs."""

    @staticmethod
    def validate_subscription_name(name: str) -> str:
        """
        Validate subscription name.

        Args:
            name: Name to validate

        Returns:
            Validated name

        Raises:
            ValueError: If invalid
        """
        if not name or not name.strip():
            raise ValueError("Subscription name cannot be empty")

        name = name.strip()

        # Check length
        if len(name) > 200:
            raise ValueError("Subscription name too long (max 200 characters)")

        # Remove control characters
        name = "".join(char for char in name if ord(char) >= 32 or char in "\t\n")

        return name

    @staticmethod
    def validate_check_frequency(frequency: int) -> int:
        """
        Validate check frequency.

        Args:
            frequency: Frequency in seconds

        Returns:
            Validated frequency

        Raises:
            ValueError: If invalid
        """
        # Minimum 1 minute
        if frequency < 60:
            raise ValueError("Check frequency must be at least 60 seconds")

        # Maximum 7 days
        if frequency > 604800:
            raise ValueError("Check frequency cannot exceed 7 days")

        return frequency

    @staticmethod
    def validate_priority(priority: int) -> int:
        """
        Validate priority level.

        Args:
            priority: Priority (1-5)

        Returns:
            Validated priority

        Raises:
            ValueError: If invalid
        """
        if not isinstance(priority, int):
            raise ValueError("Priority must be an integer")

        if priority < 1 or priority > 5:
            raise ValueError("Priority must be between 1 and 5")

        return priority

    @staticmethod
    def validate_tags(tags: List[str]) -> List[str]:
        """
        Validate tag list.

        Args:
            tags: List of tags

        Returns:
            Validated tags
        """
        validated = []

        for tag in tags:
            if not isinstance(tag, str):
                continue

            tag = tag.strip()
            if not tag:
                continue

            # Limit tag length
            if len(tag) > 50:
                tag = tag[:50]

            # Remove special characters
            tag = re.sub(r"[^\w\s-]", "", tag)
            tag = re.sub(r"[-\s]+", "-", tag)

            if tag:
                validated.append(tag.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for tag in validated:
            if tag not in seen:
                seen.add(tag)
                unique.append(tag)

        return unique[:20]  # Limit to 20 tags


# End of security.py
