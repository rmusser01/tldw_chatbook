"""
Input validation utilities for secure user input handling.
"""
import re
import ipaddress
from typing import Union, Optional
from pathlib import Path


def validate_email(email: str) -> bool:
    """Validate email address format."""
    if not email or len(email) > 254:
        return False
    
    # Basic email regex - not perfect but good enough for most cases
    pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(pattern.match(email))


def validate_username(username: str, min_length: int = 3, max_length: int = 50) -> bool:
    """Validate username format."""
    if not username or len(username) < min_length or len(username) > max_length:
        return False
    
    # Allow alphanumeric, underscore, hyphen
    pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
    return bool(pattern.match(username))


def validate_ip_address(ip: str) -> bool:
    """Validate IP address (IPv4 or IPv6)."""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def validate_port(port: Union[str, int]) -> bool:
    """Validate port number."""
    try:
        port_num = int(port)
        return 1 <= port_num <= 65535
    except (ValueError, TypeError):
        return False


def validate_url(url: str) -> bool:
    """Basic URL validation."""
    if not url or len(url) > 2000:
        return False
    
    # Basic URL pattern
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(pattern.match(url))


def validate_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal and dangerous characters."""
    if not filename or len(filename) > 255:
        return False
    
    # Reject dangerous characters and patterns
    dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    if any(char in filename for char in dangerous_chars):
        return False
    
    # Reject reserved Windows filenames
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
        'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
        'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_without_ext = filename.split('.')[0].upper()
    return name_without_ext not in reserved_names


def validate_text_input(text: str, max_length: int = 10000, allow_html: bool = False) -> bool:
    """Validate general text input."""
    if text is None:
        return True  # Allow None/empty
    
    if len(text) > max_length:
        return False
    
    if not allow_html:
        # Check for potential HTML/script injection
        dangerous_patterns = ['<script', '</script', 'javascript:', 'onclick=', 'onerror=']
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in dangerous_patterns):
            return False
    
    return True


def validate_number_range(value: Union[str, int, float], min_val: Optional[float] = None, 
                         max_val: Optional[float] = None) -> bool:
    """Validate numeric value within range."""
    try:
        num_val = float(value)
        
        if min_val is not None and num_val < min_val:
            return False
        
        if max_val is not None and num_val > max_val:
            return False
        
        return True
    except (ValueError, TypeError):
        return False


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """Sanitize string input by removing dangerous characters."""
    if not text:
        return ""
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove null bytes and control characters (except common whitespace)
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    return sanitized


def validate_json_size(json_str: str, max_size: int = 1024 * 1024) -> bool:
    """Validate JSON string size."""
    if not json_str:
        return True
    
    return len(json_str.encode('utf-8')) <= max_size


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_and_raise(condition: bool, message: str) -> None:
    """Validate condition and raise ValidationError if false."""
    if not condition:
        raise ValidationError(message)