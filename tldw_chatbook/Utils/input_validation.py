"""
Input validation utilities for secure user input handling.
"""
import re
import ipaddress
import time
from typing import Union, Optional
from pathlib import Path
from ..Metrics.metrics_logger import log_counter, log_histogram


def validate_email(email: str) -> bool:
    """Validate email address format."""
    start_time = time.time()
    log_counter("input_validation_email_attempt")
    
    if not email or len(email) > 254:
        log_counter("input_validation_email_invalid", labels={"reason": "length"})
        return False
    
    # Basic email regex - not perfect but good enough for most cases
    pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    result = bool(pattern.match(email))
    
    # Log result
    duration = time.time() - start_time
    log_histogram("input_validation_email_duration", duration)
    log_counter("input_validation_email_result", labels={"valid": str(result)})
    
    return result


def validate_username(username: str, min_length: int = 3, max_length: int = 50) -> bool:
    """Validate username format."""
    start_time = time.time()
    log_counter("input_validation_username_attempt", labels={
        "min_length": str(min_length),
        "max_length": str(max_length)
    })
    
    if not username or len(username) < min_length or len(username) > max_length:
        log_counter("input_validation_username_invalid", labels={
            "reason": "empty" if not username else "length"
        })
        return False
    
    # Allow alphanumeric, underscore, hyphen
    pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
    result = bool(pattern.match(username))
    
    # Log result
    duration = time.time() - start_time
    log_histogram("input_validation_username_duration", duration)
    log_counter("input_validation_username_result", labels={"valid": str(result)})
    
    return result


def validate_ip_address(ip: str) -> bool:
    """Validate IP address (IPv4 or IPv6)."""
    start_time = time.time()
    log_counter("input_validation_ip_attempt")
    
    try:
        ip_obj = ipaddress.ip_address(ip)
        
        # Log success with IP version
        duration = time.time() - start_time
        log_histogram("input_validation_ip_duration", duration)
        log_counter("input_validation_ip_result", labels={
            "valid": "true",
            "version": "ipv4" if isinstance(ip_obj, ipaddress.IPv4Address) else "ipv6"
        })
        
        return True
    except ValueError:
        log_counter("input_validation_ip_result", labels={"valid": "false"})
        return False


def validate_port(port: Union[str, int]) -> bool:
    """Validate port number."""
    log_counter("input_validation_port_attempt")
    
    try:
        port_num = int(port)
        result = 1 <= port_num <= 65535
        
        log_counter("input_validation_port_result", labels={
            "valid": str(result),
            "range": "privileged" if result and port_num < 1024 else "unprivileged" if result else "invalid"
        })
        
        return result
    except (ValueError, TypeError):
        log_counter("input_validation_port_result", labels={"valid": "false", "range": "invalid"})
        return False


def validate_url(url: str) -> bool:
    """Basic URL validation."""
    start_time = time.time()
    log_counter("input_validation_url_attempt")
    
    if not url or len(url) > 2000:
        log_counter("input_validation_url_invalid", labels={
            "reason": "empty" if not url else "too_long"
        })
        return False
    
    # Basic URL pattern
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    result = bool(pattern.match(url))
    
    # Log result
    duration = time.time() - start_time
    log_histogram("input_validation_url_duration", duration)
    log_counter("input_validation_url_result", labels={
        "valid": str(result),
        "scheme": "https" if result and url.startswith('https://') else "http" if result else "invalid"
    })
    log_histogram("input_validation_url_length", len(url))
    
    return result


def validate_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal and dangerous characters."""
    log_counter("input_validation_filename_attempt")
    
    if not filename or len(filename) > 255:
        log_counter("input_validation_filename_invalid", labels={
            "reason": "empty" if not filename else "too_long"
        })
        return False
    
    # Reject dangerous characters and patterns
    dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    for char in dangerous_chars:
        if char in filename:
            log_counter("input_validation_filename_invalid", labels={
                "reason": "dangerous_char",
                "char": char.replace('\\', 'backslash')
            })
            return False
    
    # Reject reserved Windows filenames
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
        'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
        'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_without_ext = filename.split('.')[0].upper()
    if name_without_ext in reserved_names:
        log_counter("input_validation_filename_invalid", labels={
            "reason": "reserved_name",
            "name": name_without_ext
        })
        return False
    
    log_counter("input_validation_filename_result", labels={"valid": "true"})
    return True


def validate_text_input(text: str, max_length: int = 10000, allow_html: bool = False) -> bool:
    """Validate general text input."""
    start_time = time.time()
    log_counter("input_validation_text_attempt", labels={
        "max_length": str(max_length),
        "allow_html": str(allow_html)
    })
    
    if text is None:
        log_counter("input_validation_text_result", labels={"valid": "true", "type": "none"})
        return True  # Allow None/empty
    
    if len(text) > max_length:
        log_counter("input_validation_text_invalid", labels={"reason": "too_long"})
        log_histogram("input_validation_text_oversized_length", len(text))
        return False
    
    if not allow_html:
        # Check for potential HTML/script injection
        dangerous_patterns = ['<script', '</script', 'javascript:', 'onclick=', 'onerror=']
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern in text_lower:
                log_counter("input_validation_text_invalid", labels={
                    "reason": "dangerous_pattern",
                    "pattern": pattern
                })
                return False
    
    # Log success
    duration = time.time() - start_time
    log_histogram("input_validation_text_duration", duration)
    log_histogram("input_validation_text_length", len(text))
    log_counter("input_validation_text_result", labels={"valid": "true", "type": "text"})
    
    return True


def validate_number_range(value: Union[str, int, float], min_val: Optional[float] = None, 
                         max_val: Optional[float] = None) -> bool:
    """Validate numeric value within range."""
    log_counter("input_validation_number_range_attempt", labels={
        "has_min": str(min_val is not None),
        "has_max": str(max_val is not None)
    })
    
    try:
        num_val = float(value)
        
        if min_val is not None and num_val < min_val:
            log_counter("input_validation_number_range_invalid", labels={"reason": "below_min"})
            return False
        
        if max_val is not None and num_val > max_val:
            log_counter("input_validation_number_range_invalid", labels={"reason": "above_max"})
            return False
        
        log_counter("input_validation_number_range_result", labels={"valid": "true"})
        return True
    except (ValueError, TypeError):
        log_counter("input_validation_number_range_invalid", labels={"reason": "not_numeric"})
        return False


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """Sanitize string input by removing dangerous characters."""
    log_counter("input_validation_sanitize_attempt")
    
    if not text:
        log_counter("input_validation_sanitize_result", labels={"action": "empty_input"})
        return ""
    
    original_length = len(text)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
        log_counter("input_validation_sanitize_truncated", labels={
            "original_length": str(original_length),
            "max_length": str(max_length)
        })
    
    # Remove null bytes and control characters (except common whitespace)
    removed_chars = 0
    sanitized = ''
    for char in text:
        if ord(char) >= 32 or char in '\t\n\r':
            sanitized += char
        else:
            removed_chars += 1
    
    log_histogram("input_validation_sanitize_removed_chars", removed_chars)
    log_counter("input_validation_sanitize_result", labels={
        "action": "sanitized",
        "had_dangerous_chars": str(removed_chars > 0)
    })
    
    return sanitized


def validate_json_size(json_str: str, max_size: int = 1024 * 1024) -> bool:
    """Validate JSON string size."""
    log_counter("input_validation_json_size_attempt")
    
    if not json_str:
        log_counter("input_validation_json_size_result", labels={"valid": "true", "empty": "true"})
        return True
    
    size_bytes = len(json_str.encode('utf-8'))
    result = size_bytes <= max_size
    
    log_histogram("input_validation_json_size_bytes", size_bytes)
    log_counter("input_validation_json_size_result", labels={
        "valid": str(result),
        "empty": "false"
    })
    
    if not result:
        log_histogram("input_validation_json_oversized_bytes", size_bytes)
    
    return result


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_and_raise(condition: bool, message: str) -> None:
    """Validate condition and raise ValidationError if false."""
    if not condition:
        raise ValidationError(message)