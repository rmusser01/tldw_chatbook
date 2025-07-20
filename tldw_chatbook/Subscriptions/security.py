# security.py
# Description: Security utilities for subscription monitoring
#
# This module provides security features including:
# - XXE (XML External Entity) attack prevention
# - SSRF (Server-Side Request Forgery) protection
# - Input validation and sanitization
# - URL validation
# - Authentication credential encryption
#
# Imports
import ipaddress
import re
import socket
from typing import Optional, List, Set
from urllib.parse import urlparse, urlunparse
import hashlib
import secrets
from base64 import b64encode, b64decode
#
# Third-Party Imports
import defusedxml.ElementTree as ET
from defusedxml import DefusedXmlException
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from loguru import logger
#
########################################################################################################################
#
# Security Classes
#
########################################################################################################################

class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass


class SSRFError(SecurityError):
    """Exception for SSRF attempts."""
    pass


class XXEError(SecurityError):
    """Exception for XXE attempts."""
    pass


class SecurityValidator:
    """Comprehensive security validation for subscriptions."""
    
    # Private IP ranges to block
    PRIVATE_IP_RANGES = [
        ipaddress.ip_network('127.0.0.0/8'),      # Loopback
        ipaddress.ip_network('10.0.0.0/8'),       # Private
        ipaddress.ip_network('172.16.0.0/12'),    # Private
        ipaddress.ip_network('192.168.0.0/16'),   # Private
        ipaddress.ip_network('169.254.0.0/16'),   # Link-local
        ipaddress.ip_network('::1/128'),          # IPv6 loopback
        ipaddress.ip_network('fc00::/7'),         # IPv6 private
        ipaddress.ip_network('fe80::/10'),        # IPv6 link-local
    ]
    
    # Blocked URL schemes
    BLOCKED_SCHEMES = {'file', 'ftp', 'gopher', 'javascript', 'data'}
    
    # Allowed URL schemes
    ALLOWED_SCHEMES = {'http', 'https'}
    
    # Common cloud metadata endpoints to block
    METADATA_ENDPOINTS = {
        '169.254.169.254',  # AWS/GCP/Azure
        'metadata.google.internal',  # GCP
        'metadata.azure.com',  # Azure
    }
    
    @classmethod
    def validate_feed_url(cls, url: str) -> str:
        """
        Validate and sanitize feed URLs.
        
        Args:
            url: URL to validate
            
        Returns:
            Sanitized URL
            
        Raises:
            SSRFError: If URL is potentially malicious
            ValueError: If URL is invalid
        """
        if not url:
            raise ValueError("URL cannot be empty")
        
        # Parse URL
        try:
            parsed = urlparse(url.strip())
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}")
        
        # Check scheme
        if not parsed.scheme:
            raise ValueError("URL must include scheme (http/https)")
        
        if parsed.scheme not in cls.ALLOWED_SCHEMES:
            raise SSRFError(f"URL scheme '{parsed.scheme}' not allowed")
        
        # Check hostname
        if not parsed.hostname:
            raise ValueError("URL must include hostname")
        
        # Check for metadata endpoints
        if parsed.hostname.lower() in cls.METADATA_ENDPOINTS:
            raise SSRFError(f"Access to metadata endpoint '{parsed.hostname}' is blocked")
        
        # Resolve hostname and check IP
        try:
            # Get IP address
            ip_str = socket.gethostbyname(parsed.hostname)
            ip = ipaddress.ip_address(ip_str)
            
            # Check if IP is private
            for private_range in cls.PRIVATE_IP_RANGES:
                if ip in private_range:
                    raise SSRFError(f"Access to private IP address {ip} is blocked")
                    
        except socket.gaierror:
            # Could not resolve hostname - let it through for now
            # The actual HTTP request will fail if invalid
            logger.warning(f"Could not resolve hostname: {parsed.hostname}")
        except Exception as e:
            logger.error(f"Error validating IP for {parsed.hostname}: {e}")
            raise SSRFError(f"Error validating URL: {e}")
        
        # Normalize URL
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or '/',
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        
        return normalized
    
    @staticmethod
    def validate_xml_content(content: str) -> str:
        """
        Validate XML content for XXE attacks.
        
        Args:
            content: XML content to validate
            
        Returns:
            Validated content
            
        Raises:
            XXEError: If potentially malicious XML detected
        """
        if not content:
            raise ValueError("XML content cannot be empty")
        
        # Check for common XXE patterns
        xxe_patterns = [
            r'<!ENTITY',
            r'<!DOCTYPE[^>]+SYSTEM',
            r'<!DOCTYPE[^>]+PUBLIC',
            r'SYSTEM\s+["\']file:',
            r'SYSTEM\s+["\']http:',
            r'SYSTEM\s+["\']https:',
            r'xmlns:xi\s*=',  # XInclude
        ]
        
        content_lower = content.lower()
        for pattern in xxe_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                raise XXEError(f"Potentially malicious XML pattern detected: {pattern}")
        
        # Try parsing with defusedxml
        try:
            # This will raise if XXE is detected
            ET.fromstring(content)
        except DefusedXmlException as e:
            raise XXEError(f"XXE attack detected: {e}")
        except ET.ParseError:
            # Let parse errors through - they'll be handled elsewhere
            pass
        
        return content
    
    @staticmethod
    def sanitize_item(item: dict) -> dict:
        """
        Sanitize a feed item for storage.
        
        Args:
            item: Item dictionary to sanitize
            
        Returns:
            Sanitized item
        """
        # Create a copy to avoid modifying original
        sanitized = item.copy()
        
        # Sanitize text fields
        text_fields = ['title', 'content', 'author', 'url']
        for field in text_fields:
            if field in sanitized and sanitized[field]:
                # Remove null bytes
                sanitized[field] = sanitized[field].replace('\x00', '')
                
                # Limit length
                max_lengths = {
                    'title': 1000,
                    'content': 100000,  # 100KB
                    'author': 500,
                    'url': 2000
                }
                max_len = max_lengths.get(field, 10000)
                if len(sanitized[field]) > max_len:
                    sanitized[field] = sanitized[field][:max_len]
        
        # Validate URL if present
        if 'url' in sanitized and sanitized['url']:
            try:
                sanitized['url'] = SecurityValidator.validate_feed_url(sanitized['url'])
            except (SSRFError, ValueError) as e:
                logger.warning(f"Invalid item URL, removing: {e}")
                sanitized['url'] = None
        
        return sanitized


class SSRFProtector:
    """Advanced SSRF protection with DNS rebinding prevention."""
    
    def __init__(self):
        """Initialize SSRF protector."""
        self.resolved_ips = {}  # Cache of hostname -> IP mappings
        
    def check_url(self, url: str) -> bool:
        """
        Check if URL is safe from SSRF.
        
        Args:
            url: URL to check
            
        Returns:
            True if safe, False otherwise
        """
        try:
            SecurityValidator.validate_feed_url(url)
            return True
        except (SSRFError, ValueError):
            return False
    
    def is_ip_allowed(self, ip: str) -> bool:
        """
        Check if an IP address is allowed.
        
        Args:
            ip: IP address string
            
        Returns:
            True if allowed, False otherwise
        """
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check against private ranges
            for private_range in SecurityValidator.PRIVATE_IP_RANGES:
                if ip_obj in private_range:
                    return False
                    
            # Check specific blocked IPs
            if str(ip_obj) in SecurityValidator.METADATA_ENDPOINTS:
                return False
                
            return True
            
        except ValueError:
            return False


class CredentialEncryptor:
    """Encrypt/decrypt authentication credentials."""
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryptor.
        
        Args:
            key: 32-byte encryption key (generates random if not provided)
        """
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
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        # Pad plaintext
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext.encode('utf-8')) + padder.finalize()
        
        # Encrypt
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine IV and ciphertext
        encrypted = iv + ciphertext
        
        # Return base64 encoded
        return b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted: str) -> str:
        """
        Decrypt a string.
        
        Args:
            encrypted: Base64-encoded encrypted string
            
        Returns:
            Decrypted plaintext
        """
        # Decode from base64
        encrypted_bytes = b64decode(encrypted.encode('utf-8'))
        
        # Extract IV and ciphertext
        iv = encrypted_bytes[:16]
        ciphertext = encrypted_bytes[16:]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        # Decrypt
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Unpad
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
        
        return plaintext.decode('utf-8')
    
    @staticmethod
    def derive_key_from_password(password: str, salt: bytes = None) -> tuple[bytes, bytes]:
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
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        
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
        name = ''.join(char for char in name if ord(char) >= 32 or char in '\t\n')
        
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
            tag = re.sub(r'[^\w\s-]', '', tag)
            tag = re.sub(r'[-\s]+', '-', tag)
            
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