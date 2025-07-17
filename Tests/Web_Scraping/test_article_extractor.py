"""
Unit tests for Article_Extractor_Lib.py

Tests core functionality including:
- URL validation
- Page title extraction
- Article scraping
- Error handling
- Security measures
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Import the module to test
from tldw_chatbook.Web_Scraping.Article_Extractor_Lib import (
    get_page_title,
    scrape_article,
    scrape_and_no_summarize_then_ingest,
    ContentMetadataHandler
)


class TestURLValidation:
    """Test URL validation in scraping functions."""
    
    @pytest.mark.asyncio
    async def test_scrape_article_invalid_url(self):
        """Test that invalid URLs are rejected."""
        invalid_urls = [
            "not-a-url",
            "javascript:alert('xss')",
            "file:///etc/passwd",
            "ftp://example.com",
            "",
            None,
            "http://",
            "https://",
        ]
        
        for url in invalid_urls:
            result = await scrape_article(url)
            assert result['extraction_successful'] is False
            assert 'Invalid URL' in result['title']
    
    @pytest.mark.asyncio
    async def test_scrape_article_valid_url(self):
        """Test that valid URLs are accepted."""
        valid_urls = [
            "http://example.com",
            "https://example.com/article",
            "https://subdomain.example.com/path/to/article?param=value",
            "http://localhost:8000/test",
            "https://192.168.1.1/page"
        ]
        
        # Mock the config
        with patch('tldw_chatbook.Web_Scraping.Article_Extractor_Lib.load_and_log_configs') as mock_config:
            mock_config.return_value = {
                'web_scraper': {
                    'web_scraper_retry_count': 3,
                    'web_scraper_request_timeout': 60,
                    'web_scraper_stealth_mode': False
                }
            }
            
            # Mock the actual scraping to avoid network calls
            with patch('tldw_chatbook.Web_Scraping.Article_Extractor_Lib.async_playwright') as mock_playwright:
                mock_browser = AsyncMock()
                mock_context = AsyncMock()
                mock_page = AsyncMock()
                
                mock_playwright.return_value.__aenter__.return_value.chromium.launch = AsyncMock(return_value=mock_browser)
                mock_browser.new_context = AsyncMock(return_value=mock_context)
                mock_context.new_page = AsyncMock(return_value=mock_page)
                mock_page.goto = AsyncMock()
                mock_page.content = AsyncMock(return_value="<html><body>Test content</body></html>")
                mock_page.wait_for_load_state = AsyncMock()
                
                for url in valid_urls:
                    result = await scrape_article(url)
                    # Should at least attempt to scrape valid URLs
                    assert 'Invalid URL' not in result.get('title', '')


class TestPageTitleExtraction:
    """Test get_page_title function."""
    
    def test_get_page_title_success(self):
        """Test successful page title extraction."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><head><title>Test Title</title></head></html>"
            mock_get.return_value = mock_response
            
            title = get_page_title("https://example.com")
            assert title == "Test Title"
    
    def test_get_page_title_no_title_tag(self):
        """Test page without title tag."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><head></head></html>"
            mock_get.return_value = mock_response
            
            title = get_page_title("https://example.com")
            assert title == "Untitled"
    
    def test_get_page_title_invalid_url(self):
        """Test get_page_title with invalid URL."""
        title = get_page_title("not-a-valid-url")
        assert title == "Untitled"
    
    def test_get_page_title_network_error(self):
        """Test get_page_title with network error."""
        with patch('requests.get') as mock_get:
            import requests
            mock_get.side_effect = requests.RequestException("Network error")
            
            title = get_page_title("https://example.com")
            assert title == "Untitled"


class TestSecurityMeasures:
    """Test security measures in the web scraping module."""
    
    def test_sql_like_pattern_escaping(self):
        """Test SQL LIKE pattern escaping."""
        # Import from cookie_cloner since that's where it's defined
        from tldw_chatbook.Web_Scraping.cookie_scraping.cookie_cloner import escape_sql_like_pattern
        
        # Test normal domain
        assert escape_sql_like_pattern("example.com") == "example.com"
        
        # Test domains with special characters
        assert escape_sql_like_pattern("example%.com") == "example\\%.com"
        assert escape_sql_like_pattern("example_.com") == "example\\_.com"
        assert escape_sql_like_pattern("example\\test.com") == "example\\\\test.com"
        
        # Test SQL injection attempts
        assert escape_sql_like_pattern("'; DROP TABLE cookies; --") == "'; DROP TABLE cookies; --"
        assert escape_sql_like_pattern("%' OR '1'='1") == "\\%' OR '1'='1"
    
    def test_secure_temp_file_usage(self):
        """Test that temporary files are created securely."""
        with patch('tldw_chatbook.Web_Scraping.Article_Extractor_Lib.get_temp_manager') as mock_manager:
            mock_temp_manager = Mock()
            mock_temp_manager.create_temp_file.return_value = "/secure/temp/file.xml"
            mock_manager.return_value = mock_temp_manager
            
            # This would be called in generate_temp_sitemap_from_links
            # We're testing the pattern is followed
            from tldw_chatbook.Web_Scraping.Article_Extractor_Lib import generate_temp_sitemap_from_links
            
            # Pass a set of links
            links = {"https://example.com/page1", "https://example.com/page2"}
            result = generate_temp_sitemap_from_links(links)
            
            # Check that temp file was created
            mock_temp_manager.create_temp_file.assert_called_once()
            assert result == "/secure/temp/file.xml"


class TestContentMetadataHandler:
    """Test ContentMetadataHandler class."""
    
    def test_metadata_handler_initialization(self):
        """Test ContentMetadataHandler initialization."""
        # ContentMetadataHandler is a static class with no instance attributes
        # Just verify it can be imported and has expected static methods
        assert hasattr(ContentMetadataHandler, 'format_content_with_metadata')
        assert hasattr(ContentMetadataHandler, 'extract_metadata')
        assert hasattr(ContentMetadataHandler, 'has_metadata')
        assert hasattr(ContentMetadataHandler, 'strip_metadata')
    


class TestErrorHandling:
    """Test error handling in scraping functions."""
    
    @pytest.mark.asyncio
    async def test_scrape_article_timeout(self):
        """Test handling of timeout errors."""
        # Mock the config
        with patch('tldw_chatbook.Web_Scraping.Article_Extractor_Lib.load_and_log_configs') as mock_config:
            mock_config.return_value = {
                'web_scraper': {
                    'web_scraper_retry_count': 3,
                    'web_scraper_request_timeout': 60,
                    'web_scraper_stealth_mode': False
                }
            }
            
            with patch('tldw_chatbook.Web_Scraping.Article_Extractor_Lib.async_playwright') as mock_playwright:
                mock_browser = AsyncMock()
                mock_context = AsyncMock()
                mock_page = AsyncMock()
                
                mock_playwright.return_value.__aenter__.return_value.chromium.launch = AsyncMock(return_value=mock_browser)
                mock_browser.new_context = AsyncMock(return_value=mock_context)
                mock_context.new_page = AsyncMock(return_value=mock_page)
                
                # Simulate timeout
                mock_page.goto = AsyncMock(side_effect=TimeoutError("Navigation timeout"))
                
                result = await scrape_article("https://example.com")
                assert result['extraction_successful'] is False
    
    def test_scrape_and_no_summarize_invalid_url(self):
        """Test scrape_and_no_summarize_then_ingest with invalid URL."""
        result = scrape_and_no_summarize_then_ingest(
            "invalid-url",
            ["test", "keywords"],
            "Test Title"
        )
        assert result == "Invalid URL provided."


class TestConcurrentOperations:
    """Test concurrent operation handling."""
    
    @pytest.mark.asyncio
    async def test_multiple_scrapes(self):
        """Test multiple concurrent scrapes."""
        urls = [
            "https://example1.com",
            "https://example2.com",
            "https://example3.com"
        ]
        
        # Mock the config
        with patch('tldw_chatbook.Web_Scraping.Article_Extractor_Lib.load_and_log_configs') as mock_config:
            mock_config.return_value = {
                'web_scraper': {
                    'web_scraper_retry_count': 3,
                    'web_scraper_request_timeout': 60,
                    'web_scraper_stealth_mode': False
                }
            }
            
            with patch('tldw_chatbook.Web_Scraping.Article_Extractor_Lib.async_playwright') as mock_playwright:
                # Mock the browser operations
                mock_browser = AsyncMock()
                mock_context = AsyncMock()
                mock_page = AsyncMock()
                
                mock_playwright.return_value.__aenter__.return_value.chromium.launch = AsyncMock(return_value=mock_browser)
                mock_browser.new_context = AsyncMock(return_value=mock_context)
                mock_context.new_page = AsyncMock(return_value=mock_page)
                mock_page.goto = AsyncMock()
                mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")
                mock_page.wait_for_load_state = AsyncMock()
                
                # Run multiple scrapes concurrently
                tasks = [scrape_article(url) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check that all completed without raising exceptions
                for result in results:
                    # Result should be a dict, not an exception
                    assert isinstance(result, dict)
                    assert not isinstance(result, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])