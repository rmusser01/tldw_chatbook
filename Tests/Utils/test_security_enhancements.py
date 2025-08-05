"""Tests for security enhancement utilities."""

import pytest
from pathlib import Path
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Utils.log_sanitizer import (
    sanitize_string, sanitize_dict, sanitize_list,
    create_safe_log_message, sanitize_log_params
)


class TestValidatePathSimple:
    """Test the simple path validation function."""
    
    def test_valid_paths(self):
        """Test that valid paths are accepted."""
        # Relative path
        result = validate_path_simple("test.txt")
        assert isinstance(result, Path)
        assert str(result) == "test.txt"
        
        # Absolute path
        result = validate_path_simple("/tmp/test.txt")
        assert isinstance(result, Path)
        assert str(result) == "/tmp/test.txt"
    
    def test_dangerous_patterns_rejected(self):
        """Test that dangerous patterns are rejected."""
        dangerous_paths = [
            "../../etc/passwd",  # Path traversal
            "../..",  # Multiple parent refs
            "test;rm -rf /",  # Command injection
            "test && cat /etc/passwd",  # Command chaining
            "test || whoami",  # Command chaining
            "test`whoami`",  # Command substitution
            "test$(whoami)",  # Command substitution
            "test${PATH}",  # Variable expansion
            "test\x00file",  # Null byte
            "test|cat",  # Pipe
            "~/sensitive",  # Home directory
        ]
        
        for path in dangerous_paths:
            with pytest.raises(ValueError, match="dangerous pattern|null byte"):
                validate_path_simple(path)
    
    def test_require_exists_option(self):
        """Test the require_exists option."""
        # Non-existent file should fail when require_exists=True
        with pytest.raises(ValueError, match="does not exist"):
            validate_path_simple("/tmp/definitely_does_not_exist_12345.txt", require_exists=True)
        
        # Should pass when require_exists=False
        result = validate_path_simple("/tmp/new_file.txt", require_exists=False)
        assert isinstance(result, Path)


class TestLogSanitizer:
    """Test the log sanitization utilities."""
    
    def test_sanitize_string_api_keys(self):
        """Test that API keys are sanitized from strings."""
        test_cases = [
            ("api_key=sk-1234567890abcdef", "api_key=***REDACTED***"),
            ("Bearer sk-abcdefghijklmnopqrstuvwxyz123456789012345678", "Bearer ***OPENAI_KEY***"),
            ("OPENAI_API_KEY=sk-test123", "OPENAI_API_KEY=***REDACTED***"),
            ('{"api_key": "secret123"}', '{"api_key": "***REDACTED***"}'),
            ("password: mypassword123", "password=***REDACTED***"),
            ("https://user:pass@example.com", "https://***:***@example.com"),
        ]
        
        for input_str, expected in test_cases:
            result = sanitize_string(input_str)
            assert result == expected
    
    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        test_dict = {
            "name": "test",
            "api_key": "sk-123456",
            "password": "secret",
            "nested": {
                "token": "bearer123",
                "safe": "value"
            },
            "config": "api_key=embedded_secret"
        }
        
        result = sanitize_dict(test_dict)
        
        assert result["name"] == "test"
        assert result["api_key"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"
        assert result["nested"]["token"] == "***REDACTED***"
        assert result["nested"]["safe"] == "value"
        assert "***REDACTED***" in result["config"]
    
    def test_sanitize_list(self):
        """Test list sanitization."""
        test_list = [
            "safe value",
            "api_key=secret",
            {"password": "hidden"},
            ["nested", "token=abc123"]
        ]
        
        result = sanitize_list(test_list)
        
        assert result[0] == "safe value"
        assert "***REDACTED***" in result[1]
        assert result[2]["password"] == "***REDACTED***"
        assert "***REDACTED***" in result[3][1]
    
    def test_create_safe_log_message(self):
        """Test safe log message creation."""
        # Test with positional args (OpenAI keys need 20+ chars after sk-)
        msg = create_safe_log_message("User {} logged in with key {}", "john", "sk-abcdefghijklmnopqrstuvwxyz123456")
        assert msg == "User john logged in with key ***OPENAI_KEY***"
        
        # Test with keyword args
        msg = create_safe_log_message("Config: {config}", config={"api_key": "secret"})
        assert "***REDACTED***" in msg
    
    def test_sanitize_log_params(self):
        """Test parameter sanitization."""
        args = ("test", {"api_key": "secret"}, "password=123")
        kwargs = {"token": "bearer123", "safe": "value"}
        
        clean_args, clean_kwargs = sanitize_log_params(*args, **kwargs)
        
        assert clean_args[0] == "test"
        assert clean_args[1]["api_key"] == "***REDACTED***"
        assert "***REDACTED***" in clean_args[2]
        assert clean_kwargs["token"] == "***REDACTED***"
        assert clean_kwargs["safe"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])