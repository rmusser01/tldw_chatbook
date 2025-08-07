"""
Log sanitizer utilities to prevent sensitive data from being logged.

This module provides functions to scrub API keys, passwords, and other
sensitive information from log messages.
"""

import re
from typing import Any, Dict, List, Union


# Patterns for common sensitive data
SENSITIVE_PATTERNS = [
    # Specific API key formats first (more specific patterns before general ones)
    (r'sk-[a-zA-Z0-9]{20,}', '***OPENAI_KEY***'),  # OpenAI keys
    (r'claude-[a-zA-Z0-9-]+', '***ANTHROPIC_KEY***'),  # Anthropic keys
    (r'AIza[0-9A-Za-z-_]{35}', '***GOOGLE_KEY***'),  # Google API keys
    
    # API Keys and tokens
    (r'(api[_-]?key|apikey|api_secret|access[_-]?token|auth[_-]?token|bearer)\s*[:=]\s*["\']?([^\s"\']+)', r'\1=***REDACTED***'),
    
    # Bearer tokens in headers
    (r'(Bearer\s+)(sk-[a-zA-Z0-9]{20,})', r'\1***OPENAI_KEY***'),  # Specific OpenAI bearer
    (r'(Bearer\s+)([a-zA-Z0-9\-._~+/]+=*)', r'\1***REDACTED***'),
    (r'(Authorization:\s*)(Bearer\s+)?([^\s]+)', r'\1\2***REDACTED***'),
    
    # Token patterns
    (r'(token)\s*[:=]\s*["\']?([^\s"\']+)', r'\1=***REDACTED***'),
    
    # Password patterns
    (r'(password|passwd|pwd)\s*[:=]\s*["\']?([^\s"\']+)', r'\1=***REDACTED***'),
    
    # URLs with embedded credentials
    (r'(https?://)([^:]+):([^@]+)@', r'\1***:***@'),
    
    # JSON/Dict API keys
    (r'["\']?(api[_-]?key|apikey|api_secret|password|secret|token)["\']?\s*:\s*["\']([^"\']+)["\']', r'"\1": "***REDACTED***"'),
    
    # Environment variable style
    (r'(OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|API_KEY|SECRET_KEY|DATABASE_URL)\s*=\s*([^\s]+)', r'\1=***REDACTED***'),
]

# Fields to redact in dictionaries
SENSITIVE_FIELDS = {
    'api_key', 'apikey', 'api-key', 'api_secret', 'api-secret',
    'password', 'passwd', 'pwd', 'secret', 'token', 'auth_token',
    'access_token', 'bearer_token', 'client_secret', 'private_key',
    'openai_api_key', 'anthropic_api_key', 'google_api_key',
    'aws_access_key_id', 'aws_secret_access_key', 'database_url',
    'connection_string', 'credentials'
}


def sanitize_string(text: str) -> str:
    """
    Sanitize a string by removing sensitive data patterns.
    
    Args:
        text: The string to sanitize
        
    Returns:
        Sanitized string with sensitive data redacted
    """
    if not isinstance(text, str):
        return str(text)
    
    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def sanitize_dict(data: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """
    Sanitize a dictionary by redacting sensitive fields.
    
    Args:
        data: The dictionary to sanitize
        deep: Whether to recursively sanitize nested structures
        
    Returns:
        New dictionary with sensitive fields redacted
    """
    if not isinstance(data, dict):
        return data
    
    result = {}
    for key, value in data.items():
        # Check if key is sensitive
        if key.lower() in SENSITIVE_FIELDS:
            result[key] = "***REDACTED***"
        elif deep and isinstance(value, dict):
            result[key] = sanitize_dict(value, deep=True)
        elif deep and isinstance(value, list):
            result[key] = sanitize_list(value, deep=True)
        elif isinstance(value, str):
            # Still sanitize string values for embedded secrets
            result[key] = sanitize_string(value)
        else:
            result[key] = value
    
    return result


def sanitize_list(data: List[Any], deep: bool = True) -> List[Any]:
    """
    Sanitize a list by processing each element.
    
    Args:
        data: The list to sanitize
        deep: Whether to recursively sanitize nested structures
        
    Returns:
        New list with sensitive data redacted
    """
    if not isinstance(data, list):
        return data
    
    result = []
    for item in data:
        if isinstance(item, dict) and deep:
            result.append(sanitize_dict(item, deep=True))
        elif isinstance(item, list) and deep:
            result.append(sanitize_list(item, deep=True))
        elif isinstance(item, str):
            result.append(sanitize_string(item))
        else:
            result.append(item)
    
    return result


def sanitize_log_params(*args, **kwargs) -> tuple:
    """
    Sanitize both positional and keyword arguments for logging.
    
    Returns:
        Tuple of (sanitized_args, sanitized_kwargs)
    """
    clean_args = []
    for arg in args:
        if isinstance(arg, dict):
            clean_args.append(sanitize_dict(arg))
        elif isinstance(arg, list):
            clean_args.append(sanitize_list(arg))
        elif isinstance(arg, str):
            clean_args.append(sanitize_string(arg))
        else:
            clean_args.append(arg)
    
    clean_kwargs = sanitize_dict(kwargs) if kwargs else {}
    
    return tuple(clean_args), clean_kwargs


def create_safe_log_message(template: str, *args, **kwargs) -> str:
    """
    Create a safe log message by sanitizing all parameters.
    
    Args:
        template: The log message template
        *args: Positional arguments for the template
        **kwargs: Keyword arguments for the template
        
    Returns:
        Formatted log message with sensitive data redacted
    """
    # First sanitize all arguments
    clean_args, clean_kwargs = sanitize_log_params(*args, **kwargs)
    
    # Format with sanitized arguments first
    try:
        if clean_args and clean_kwargs:
            formatted = template.format(*clean_args, **clean_kwargs)
        elif clean_args:
            formatted = template.format(*clean_args)
        elif clean_kwargs:
            formatted = template.format(**clean_kwargs)
        else:
            formatted = template
    except Exception:
        # If formatting fails, just use template
        formatted = template
    
    # Then sanitize the final result to catch any embedded secrets
    return sanitize_string(formatted)


# Convenience function for common use case
def safe_log(logger_func, message: str, *args, **kwargs):
    """
    Safely log a message by sanitizing all parameters first.
    
    Args:
        logger_func: The logger function to use (e.g., logger.info)
        message: The log message
        *args: Additional arguments
        **kwargs: Additional keyword arguments
    """
    safe_message = create_safe_log_message(message, *args, **kwargs)
    logger_func(safe_message)