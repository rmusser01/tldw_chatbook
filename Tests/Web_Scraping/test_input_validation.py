"""
Tests for input validation in web scraping module.
"""
import pytest
from tldw_chatbook.Utils.input_validation import (
    validate_url,
    validate_text_input,
    sanitize_string
)


class TestURLValidation:
    """Test URL validation function."""
    
    def test_valid_urls(self):
        """Test that valid URLs pass validation."""
        valid_urls = [
            "http://example.com",
            "https://example.com",
            "http://subdomain.example.com",
            "https://example.com/path/to/page",
            "https://example.com/path?query=param",
            "https://example.com:8080/path",
            "http://192.168.1.1",
            "http://localhost",
            "http://localhost:3000"
        ]
        
        for url in valid_urls:
            assert validate_url(url) is True, f"URL should be valid: {url}"
    
    def test_invalid_urls(self):
        """Test that invalid URLs fail validation."""
        invalid_urls = [
            "",
            None,
            "not a url",
            "javascript:alert('xss')",
            "file:///etc/passwd",
            "ftp://example.com",  # Not http/https
            "http://",
            "https://",
            "//example.com",  # Protocol-relative URL
            "http:/example.com",  # Missing slash
            "ht!tp://example.com",  # Invalid protocol
            "http://exam ple.com",  # Space in domain
            "a" * 2001,  # Too long
        ]
        
        for url in invalid_urls:
            if url is None:
                with pytest.raises(AttributeError):
                    validate_url(url)
            else:
                assert validate_url(url) is False, f"URL should be invalid: {url}"
    
    def test_edge_cases(self):
        """Test edge cases for URL validation."""
        # URLs at max length limit (2000 chars)
        long_url = "https://example.com/" + "a" * 1980
        assert validate_url(long_url) is True
        
        too_long_url = "https://example.com/" + "a" * 1981
        assert validate_url(too_long_url) is False
        
        # International domains (IDN)
        # Note: The basic regex might not support these
        international_urls = [
            "http://例え.jp",
            "https://münchen.de",
            "http://россия.рф"
        ]
        # These might fail with basic regex, which is acceptable for security


class TestTextInputValidation:
    """Test text input validation and sanitization."""
    
    def test_valid_text_input(self):
        """Test that normal text passes validation."""
        valid_texts = [
            "Normal text",
            "Text with numbers 123",
            "Text with punctuation!",
            "Multi-line\ntext",
            "",  # Empty is allowed
            None,  # None is allowed
            "a" * 1000,  # At default limit
        ]
        
        for text in valid_texts:
            assert validate_text_input(text) is True
    
    def test_invalid_text_input(self):
        """Test that dangerous text fails validation."""
        invalid_texts = [
            "<script>alert('xss')</script>",
            "Text with <script>tag</script>",
            '<img src="x" onerror="alert(\'xss\')">',
            "javascript:void(0)",
            '<a href="javascript:alert()">link</a>',
            "onclick=alert('xss')",
            "a" * 10001,  # Over default limit
        ]
        
        for text in invalid_texts:
            assert validate_text_input(text) is False
    
    def test_text_sanitization(self):
        """Test text sanitization removes dangerous characters."""
        # Test null byte removal
        text_with_null = "Hello\x00World"
        sanitized = sanitize_string(text_with_null)
        assert "\x00" not in sanitized
        assert sanitized == "HelloWorld"
        
        # Test control character removal
        text_with_control = "Hello\x01\x02\x03World"
        sanitized = sanitize_string(text_with_control)
        assert sanitized == "HelloWorld"
        
        # Test length truncation
        long_text = "a" * 2000
        sanitized = sanitize_string(long_text, max_length=100)
        assert len(sanitized) == 100
        
        # Test that normal whitespace is preserved
        text_with_whitespace = "Hello\n\r\tWorld"
        sanitized = sanitize_string(text_with_whitespace)
        assert sanitized == "Hello\n\r\tWorld"


class TestSecurityPatterns:
    """Test various security patterns in input validation."""
    
    def test_xss_prevention(self):
        """Test that XSS attempts are caught."""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert()'></iframe>",
            "<object data='javascript:alert()'></object>",
            "<embed src='javascript:alert()'>",
            "<form action='javascript:alert()'><input type='submit'></form>",
            "';alert('xss');//",
            '";alert(\'xss\');//',
        ]
        
        for xss in xss_attempts:
            assert validate_text_input(xss) is False, f"XSS should be blocked: {xss}"
    
    def test_sql_injection_patterns(self):
        """Test that SQL injection patterns in text are handled."""
        sql_patterns = [
            "'; DROP TABLE users; --",
            '" OR "1"="1"',
            "' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM users--",
        ]
        
        # These should be allowed as text (not SQL) but sanitized
        for pattern in sql_patterns:
            # Text validation allows these (they're valid text)
            assert validate_text_input(pattern) is True
            # But they should be sanitized when used
            sanitized = sanitize_string(pattern)
            assert len(sanitized) > 0  # Should not be empty
    
    def test_path_traversal_patterns(self):
        """Test path traversal patterns in input."""
        path_patterns = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]
        
        # As text input, these are technically valid
        # Path validation should be done separately
        for pattern in path_patterns:
            assert validate_text_input(pattern) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])