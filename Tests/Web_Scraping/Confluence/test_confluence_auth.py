# test_confluence_auth.py
#
# Unit tests for Confluence authentication
#
import pytest
import os
from unittest.mock import MagicMock, patch, Mock
import requests
from tldw_chatbook.Web_Scraping.Confluence.confluence_auth import (
    ConfluenceAuth,
    AuthMethod,
    create_confluence_auth
)


class TestConfluenceAuth:
    """Test suite for Confluence authentication"""
    
    def test_init_confluence_auth(self):
        """Test ConfluenceAuth initialization"""
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        assert auth.base_url == "https://example.atlassian.net/wiki"
        assert auth.auth_method == AuthMethod.API_TOKEN
        assert not auth._auth_configured
        
    def test_configure_api_token(self):
        """Test API token configuration"""
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        auth.configure_api_token("user@example.com", "test-api-token")
        
        assert auth._auth_configured
        assert auth.session.auth is not None
        assert auth.session.auth.username == "user@example.com"
        assert auth.session.auth.password == "test-api-token"
        
    def test_configure_api_token_missing_params(self):
        """Test API token configuration with missing parameters"""
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        
        with pytest.raises(ValueError, match="Username and API token are required"):
            auth.configure_api_token("", "token")
            
        with pytest.raises(ValueError, match="Username and API token are required"):
            auth.configure_api_token("user", "")
            
    def test_configure_basic_auth(self):
        """Test basic authentication configuration"""
        auth = ConfluenceAuth("https://example.com/confluence")
        auth.configure_basic_auth("testuser", "testpass")
        
        assert auth._auth_configured
        assert auth.session.auth is not None
        assert auth.session.auth.username == "testuser"
        assert auth.session.auth.password == "testpass"
        
    def test_configure_oauth(self):
        """Test OAuth configuration"""
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        auth.configure_oauth("test-oauth-token")
        
        assert auth._auth_configured
        assert auth.session.headers['Authorization'] == 'Bearer test-oauth-token'
        
    def test_configure_oauth_missing_token(self):
        """Test OAuth configuration with missing token"""
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        
        with pytest.raises(ValueError, match="OAuth token is required"):
            auth.configure_oauth("")
            
    @patch('tldw_chatbook.Web_Scraping.Confluence.confluence_auth.get_cookies')
    def test_configure_cookies(self, mock_get_cookies):
        """Test cookie-based authentication"""
        mock_get_cookies.return_value = {
            'session_cookie': 'test_value',
            'auth_cookie': 'auth_value'
        }
        
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        auth.configure_cookies(browser='chrome')
        
        assert auth._auth_configured
        mock_get_cookies.assert_called_once_with('example.atlassian.net', browser='chrome')
        
    @patch('tldw_chatbook.Web_Scraping.Confluence.confluence_auth.get_cookies')
    def test_configure_cookies_no_cookies_found(self, mock_get_cookies):
        """Test cookie configuration when no cookies are found"""
        mock_get_cookies.return_value = {}
        
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        
        with pytest.raises(ValueError, match="No cookies found"):
            auth.configure_cookies(browser='chrome')
            
    def test_get_session_not_configured(self):
        """Test getting session when auth not configured"""
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        
        with pytest.raises(RuntimeError, match="Authentication not configured"):
            auth.get_session()
            
    def test_get_session_configured(self):
        """Test getting session when auth is configured"""
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        auth.configure_api_token("user@example.com", "token")
        
        session = auth.get_session()
        assert session is not None
        assert isinstance(session, requests.Session)
        
    @patch.object(requests.Session, 'get')
    def test_test_authentication_success(self, mock_get):
        """Test successful authentication test"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'displayName': 'Test User'}
        mock_get.return_value = mock_response
        
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        auth.configure_api_token("user@example.com", "token")
        
        result = auth.test_authentication()
        assert result is True
        mock_get.assert_called_once()
        
    @patch.object(requests.Session, 'get')
    def test_test_authentication_failure(self, mock_get):
        """Test failed authentication test"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        auth.configure_api_token("user@example.com", "wrong-token")
        
        result = auth.test_authentication()
        assert result is False
        
    def test_get_auth_headers_api_token(self):
        """Test getting auth headers for API token auth"""
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        auth.configure_api_token("user@example.com", "token")
        
        headers = auth.get_auth_headers()
        assert 'Authorization' in headers
        assert headers['Authorization'].startswith('Basic ')
        
    def test_get_auth_headers_oauth(self):
        """Test getting auth headers for OAuth"""
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        auth.configure_oauth("oauth-token")
        
        headers = auth.get_auth_headers()
        assert headers['Authorization'] == 'Bearer oauth-token'
        
    @patch.object(requests.Session, 'request')
    def test_make_request(self, mock_request):
        """Test making authenticated request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response
        
        auth = ConfluenceAuth("https://example.atlassian.net/wiki")
        auth.configure_api_token("user@example.com", "token")
        
        response = auth.make_request('GET', '/rest/api/content/12345')
        
        assert response == mock_response
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == 'GET'
        assert call_args[0][1] == 'https://example.atlassian.net/wiki/rest/api/content/12345'
        
    def test_create_confluence_auth_api_token(self):
        """Test factory function with API token config"""
        config = {
            'auth_method': 'api_token',
            'username': 'user@example.com',
            'api_token': 'test-token'
        }
        
        auth = create_confluence_auth("https://example.atlassian.net/wiki", config)
        
        assert auth._auth_configured
        assert auth.auth_method == AuthMethod.API_TOKEN
        
    @patch.dict(os.environ, {'CONFLUENCE_USERNAME': 'env_user', 'CONFLUENCE_API_TOKEN': 'env_token'})
    def test_create_confluence_auth_from_env(self):
        """Test factory function using environment variables"""
        config = {'auth_method': 'api_token'}
        
        auth = create_confluence_auth("https://example.atlassian.net/wiki", config)
        
        assert auth._auth_configured
        assert auth.session.auth.username == 'env_user'
        assert auth.session.auth.password == 'env_token'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])