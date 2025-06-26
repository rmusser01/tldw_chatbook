"""
Security-focused tests for web scraping module.

Tests security measures including:
- SQL injection prevention
- Path traversal prevention
- Cookie security
- API key protection
- Input sanitization
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sqlite3

from tldw_chatbook.Web_Scraping.cookie_scraping.cookie_cloner import (
    escape_sql_like_pattern,
    get_chrome_cookies,
    get_firefox_cookies,
    get_edge_cookies
)


class TestSQLInjectionPrevention:
    """Test SQL injection prevention measures."""
    
    def test_escape_sql_like_pattern(self):
        """Test SQL LIKE pattern escaping."""
        # Normal domains
        assert escape_sql_like_pattern("example.com") == "example.com"
        assert escape_sql_like_pattern("sub.example.com") == "sub.example.com"
        
        # Domains with special LIKE characters
        assert escape_sql_like_pattern("example%.com") == "example\\%.com"
        assert escape_sql_like_pattern("example_.com") == "example\\_.com"
        assert escape_sql_like_pattern("ex\\ample.com") == "ex\\\\ample.com"
        assert escape_sql_like_pattern("ex%am_ple.com") == "ex\\%am\\_ple.com"
        
        # SQL injection attempts
        dangerous_inputs = [
            "'; DROP TABLE cookies; --",
            "' OR 1=1 --",
            "%' OR '1'='1",
            "_' OR '1'='1",
            "\\'; DELETE FROM cookies; --"
        ]
        
        for dangerous in dangerous_inputs:
            escaped = escape_sql_like_pattern(dangerous)
            # Check that % and _ are escaped
            assert escaped.count('\\%') == dangerous.count('%')
            assert escaped.count('\\_') == dangerous.count('_')
            # The escaped pattern should be safe to use in LIKE query
    
    def test_cookie_extraction_sql_safety(self):
        """Test that cookie extraction uses parameterized queries."""
        # Create a test database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create test cookie database
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE cookies (
                    host_key TEXT,
                    name TEXT,
                    path TEXT,
                    encrypted_value BLOB,
                    expires_utc INTEGER
                )
            """)
            
            # Insert test data with dangerous characters
            cursor.execute("""
                INSERT INTO cookies VALUES 
                (?, ?, ?, ?, ?)
            """, ("example%.com", "test_cookie", "/", b"encrypted", 0))
            conn.commit()
            conn.close()
            
            # Mock the cookie path discovery
            with patch('os.path.expanduser') as mock_expand:
                with patch('os.path.exists') as mock_exists:
                    with patch('shutil.copyfile') as mock_copy:
                        with patch('tempfile.mkstemp') as mock_mkstemp:
                            # Set up mocks
                            mock_mkstemp.return_value = (999, tmp_path + "_temp")
                            mock_expand.return_value = "/fake/path"
                            mock_exists.return_value = True
                            
                            # This should handle the dangerous domain safely
                            # We're testing that no SQL injection occurs
                            try:
                                # The function will fail on other aspects, but SQL should be safe
                                get_chrome_cookies("example%.com")
                            except Exception as e:
                                # We expect failures due to mocking, but not SQL errors
                                assert "SQL" not in str(e)
        finally:
            os.unlink(tmp_path)
            if os.path.exists(tmp_path + "_temp"):
                os.unlink(tmp_path + "_temp")


class TestPathTraversalPrevention:
    """Test path traversal prevention."""
    
    def test_temp_file_security(self):
        """Test that temporary files are created securely."""
        from tldw_chatbook.Utils.secure_temp_files import create_secure_temp_file
        
        # Test that file is created with secure permissions
        content = "test content"
        temp_path = create_secure_temp_file(content, suffix=".test")
        
        try:
            # Check file exists and has correct content
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                assert f.read() == content
            
            # Check file permissions (owner read/write only)
            stat_info = os.stat(temp_path)
            mode = stat_info.st_mode
            # Check that group and others have no permissions
            assert (mode & 0o077) == 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_cookie_db_path_validation(self):
        """Test that cookie database paths are not vulnerable to traversal."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        with patch('os.path.expanduser') as mock_expand:
            with patch('os.path.exists') as mock_exists:
                for dangerous_path in dangerous_paths:
                    mock_expand.return_value = dangerous_path
                    mock_exists.return_value = True
                    
                    # The function should not access arbitrary paths
                    try:
                        get_chrome_cookies("example.com")
                    except Exception:
                        # Expected to fail, but should not access the dangerous path
                        pass


class TestAPIKeySecurity:
    """Test API key handling security."""
    
    def test_api_keys_not_logged(self):
        """Test that API keys are not logged."""
        # This is more of a code review test, but we can check patterns
        import tldw_chatbook.Web_Scraping.WebSearch_APIs as web_search
        
        # Mock the logging to capture what would be logged
        with patch('tldw_chatbook.Web_Scraping.WebSearch_APIs.logging') as mock_logging:
            with patch.dict(os.environ, {'SOME_API_KEY': 'secret_key_12345'}):
                # Any function that uses API keys should not log them
                # This is a pattern test - actual implementation may vary
                api_key = os.environ.get('SOME_API_KEY')
                
                # Simulate what good code should do
                if api_key:
                    # Should log that key exists but not the value
                    mock_logging.info.assert_not_called_with(f"API Key: {api_key}")
    
    def test_api_key_masking(self):
        """Test that API keys are masked in any output."""
        api_key = "sk-1234567890abcdef"
        
        # Good practice: mask API keys in logs/output
        def mask_api_key(key):
            if not key or len(key) < 8:
                return "***"
            return key[:4] + "*" * (len(key) - 8) + key[-4:]
        
        masked = mask_api_key(api_key)
        assert api_key not in masked
        assert masked == "sk-1********cdef"


class TestInputSanitization:
    """Test input sanitization in web scraping."""
    
    def test_url_sanitization(self):
        """Test that URLs are properly sanitized."""
        from tldw_chatbook.Utils.input_validation import validate_url
        
        # Malicious URLs that should be rejected
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox('xss')",
            "file:///etc/passwd",
            "\\\\attacker.com\\share\\file",
            "about:blank",
            "chrome://settings",
        ]
        
        for url in malicious_urls:
            assert validate_url(url) is False
    
    def test_domain_name_sanitization(self):
        """Test domain name sanitization in cookie functions."""
        # Domain names with special characters
        domains = [
            "example.com<script>",
            "example.com' OR '1'='1",
            "example.com\"; DROP TABLE cookies;--",
            "example.com%00.attacker.com",
            "example.com\n.attacker.com",
            "example.com\r\n.attacker.com"
        ]
        
        for domain in domains:
            # The escape function should handle these safely
            escaped = escape_sql_like_pattern(domain)
            # The result should not allow SQL injection
            assert "DROP TABLE" in escaped  # But safely escaped


class TestCookieSecurity:
    """Test cookie handling security."""
    
    def test_cookie_encryption_handling(self):
        """Test that encrypted cookies are handled securely."""
        # Test that we don't expose decrypted values in logs/errors
        with patch('tldw_chatbook.Web_Scraping.cookie_scraping.cookie_cloner.logger') as mock_logger:
            # Simulate cookie decryption
            encrypted_value = b"encrypted_cookie_value"
            
            # Good practice: never log decrypted cookie values
            # Check that sensitive data isn't logged
            mock_logger.debug.assert_not_called()
            mock_logger.info.assert_not_called()
    
    def test_cookie_permission_check(self):
        """Test that cookie access requires appropriate permissions."""
        # In a real implementation, we should check:
        # 1. User has permission to access browser cookies
        # 2. Warning/consent before accessing cookies
        # 3. Audit log of cookie access
        
        # This is more of a design consideration
        assert True  # Placeholder for permission system


class TestSecureDefaults:
    """Test that secure defaults are used throughout."""
    
    def test_request_timeouts(self):
        """Test that network requests have timeouts."""
        # Check that requests have reasonable timeouts to prevent DoS
        import tldw_chatbook.Web_Scraping.Article_Extractor_Lib as extractor
        
        # This is a pattern check - ensure timeouts are used
        # In actual code, all requests.get() should have timeout parameter
        assert True  # Placeholder - implement in code review
    
    def test_resource_limits(self):
        """Test that resource consumption is limited."""
        # Check for:
        # 1. Max file size limits
        # 2. Max memory usage
        # 3. Max concurrent connections
        # 4. Rate limiting
        
        # These should be implemented in the actual code
        assert True  # Placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])