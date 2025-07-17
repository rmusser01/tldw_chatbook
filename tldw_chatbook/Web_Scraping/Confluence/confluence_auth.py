# confluence_auth.py
#
# Handles authentication for Confluence API access
#
# Imports
import os
import base64
from typing import Dict, Optional, Any, List
from enum import Enum
import requests
from requests.auth import HTTPBasicAuth
#
# Third-party imports
from loguru import logger
#
# Local imports
from ..cookie_scraping.cookie_cloner import get_cookies
#
#######################################################################################################################
#
# Classes and Functions

class AuthMethod(Enum):
    """Supported authentication methods for Confluence"""
    API_TOKEN = "api_token"
    OAUTH = "oauth"
    COOKIES = "cookies"
    BASIC = "basic"


class ConfluenceAuth:
    """Handles authentication for Confluence API requests"""
    
    def __init__(self, base_url: str, auth_method: AuthMethod = AuthMethod.API_TOKEN):
        """
        Initialize Confluence authentication
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://example.atlassian.net/wiki)
            auth_method: The authentication method to use
        """
        self.base_url = base_url.rstrip('/')
        self.auth_method = auth_method
        self.session = requests.Session()
        self._auth_configured = False
        
    def configure_api_token(self, username: str, api_token: str) -> None:
        """
        Configure API token authentication (recommended for Atlassian Cloud)
        
        Args:
            username: User's email address
            api_token: API token generated from Atlassian account settings
        """
        if not username or not api_token:
            raise ValueError("Username and API token are required for API token authentication")
            
        # For Atlassian Cloud, use Basic Auth with email and API token
        self.session.auth = HTTPBasicAuth(username, api_token)
        self._auth_configured = True
        logger.info(f"Configured API token authentication for user: {username}")
        
    def configure_basic_auth(self, username: str, password: str) -> None:
        """
        Configure basic authentication (for self-hosted Confluence)
        
        Args:
            username: Username
            password: Password
        """
        if not username or not password:
            raise ValueError("Username and password are required for basic authentication")
            
        self.session.auth = HTTPBasicAuth(username, password)
        self._auth_configured = True
        logger.info(f"Configured basic authentication for user: {username}")
        
    def configure_oauth(self, oauth_token: str, oauth_token_secret: Optional[str] = None) -> None:
        """
        Configure OAuth authentication
        
        Args:
            oauth_token: OAuth access token
            oauth_token_secret: OAuth token secret (for OAuth 1.0)
        """
        if not oauth_token:
            raise ValueError("OAuth token is required for OAuth authentication")
            
        # For OAuth 2.0 (Bearer token)
        self.session.headers.update({
            'Authorization': f'Bearer {oauth_token}'
        })
        self._auth_configured = True
        logger.info("Configured OAuth authentication")
        
    def configure_cookies(self, browser: str = 'all') -> None:
        """
        Configure cookie-based authentication by extracting cookies from browser
        
        Args:
            browser: Browser to extract cookies from ('chrome', 'firefox', 'edge', 'safari', or 'all')
        """
        # Extract domain from base URL
        from urllib.parse import urlparse
        domain = urlparse(self.base_url).netloc
        
        # Get cookies from browser
        cookies = get_cookies(domain, browser=browser)
        
        if not cookies:
            raise ValueError(f"No cookies found for domain {domain} in browser {browser}")
            
        # Convert cookies to requests-compatible format
        for name, value in cookies.items():
            self.session.cookies.set(name, value, domain=domain)
            
        self._auth_configured = True
        logger.info(f"Configured cookie authentication with {len(cookies)} cookies from {browser}")
        
    def configure_from_env(self) -> None:
        """Configure authentication from environment variables"""
        auth_method = os.getenv('CONFLUENCE_AUTH_METHOD', 'api_token').lower()
        
        if auth_method == 'api_token':
            username = os.getenv('CONFLUENCE_USERNAME')
            api_token = os.getenv('CONFLUENCE_API_TOKEN')
            if username and api_token:
                self.configure_api_token(username, api_token)
                logger.info("Configured authentication from environment variables")
            else:
                raise ValueError("CONFLUENCE_USERNAME and CONFLUENCE_API_TOKEN environment variables required")
                
        elif auth_method == 'basic':
            username = os.getenv('CONFLUENCE_USERNAME')
            password = os.getenv('CONFLUENCE_PASSWORD')
            if username and password:
                self.configure_basic_auth(username, password)
                logger.info("Configured basic auth from environment variables")
            else:
                raise ValueError("CONFLUENCE_USERNAME and CONFLUENCE_PASSWORD environment variables required")
                
        elif auth_method == 'oauth':
            oauth_token = os.getenv('CONFLUENCE_OAUTH_TOKEN')
            if oauth_token:
                self.configure_oauth(oauth_token)
                logger.info("Configured OAuth from environment variables")
            else:
                raise ValueError("CONFLUENCE_OAUTH_TOKEN environment variable required")
                
        elif auth_method == 'cookies':
            browser = os.getenv('CONFLUENCE_BROWSER', 'all')
            self.configure_cookies(browser)
            
        else:
            raise ValueError(f"Unsupported auth method from environment: {auth_method}")
            
    def get_session(self) -> requests.Session:
        """
        Get the configured requests session
        
        Returns:
            Configured requests.Session object
        """
        if not self._auth_configured:
            raise RuntimeError("Authentication not configured. Call one of the configure_* methods first.")
            
        return self.session
        
    def test_authentication(self) -> bool:
        """
        Test if authentication is working by making a simple API call
        
        Returns:
            True if authentication is successful, False otherwise
        """
        try:
            # Try to get current user info
            response = self.session.get(
                f"{self.base_url}/rest/api/user/current",
                timeout=10
            )
            
            if response.status_code == 200:
                user_data = response.json()
                logger.info(f"Authentication successful. Logged in as: {user_data.get('displayName', 'Unknown')}")
                return True
            else:
                logger.error(f"Authentication failed with status code: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication test failed: {str(e)}")
            return False
            
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for manual requests
        
        Returns:
            Dictionary of authentication headers
        """
        if not self._auth_configured:
            raise RuntimeError("Authentication not configured")
            
        headers = dict(self.session.headers)
        
        # Add basic auth header if using HTTPBasicAuth
        if self.session.auth and isinstance(self.session.auth, HTTPBasicAuth):
            credentials = f"{self.session.auth.username}:{self.session.auth.password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers['Authorization'] = f'Basic {encoded}'
            
        return headers
        
    def make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an authenticated request to Confluence API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base URL)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response object
        """
        if not self._auth_configured:
            raise RuntimeError("Authentication not configured")
            
        url = f"{self.base_url}{endpoint}"
        
        # Set default headers
        if 'headers' not in kwargs:
            kwargs['headers'] = {}
        kwargs['headers'].setdefault('Accept', 'application/json')
        kwargs['headers'].setdefault('Content-Type', 'application/json')
        
        # Make request
        response = self.session.request(method, url, **kwargs)
        
        # Log request details for debugging
        logger.debug(f"{method} {url} - Status: {response.status_code}")
        
        return response


def create_confluence_auth(base_url: str, config: Dict[str, Any]) -> ConfluenceAuth:
    """
    Factory function to create ConfluenceAuth instance from configuration
    
    Args:
        base_url: Confluence base URL
        config: Configuration dictionary with auth settings
        
    Returns:
        Configured ConfluenceAuth instance
    """
    auth_method = config.get('auth_method', 'api_token')
    auth = ConfluenceAuth(base_url, AuthMethod(auth_method))
    
    if auth_method == 'api_token':
        username = config.get('username') or os.getenv('CONFLUENCE_USERNAME')
        api_token = config.get('api_token') or os.getenv('CONFLUENCE_API_TOKEN')
        auth.configure_api_token(username, api_token)
        
    elif auth_method == 'basic':
        username = config.get('username') or os.getenv('CONFLUENCE_USERNAME')
        password = config.get('password') or os.getenv('CONFLUENCE_PASSWORD')
        auth.configure_basic_auth(username, password)
        
    elif auth_method == 'oauth':
        oauth_token = config.get('oauth_token') or os.getenv('CONFLUENCE_OAUTH_TOKEN')
        auth.configure_oauth(oauth_token)
        
    elif auth_method == 'cookies':
        browser = config.get('browser', 'all')
        auth.configure_cookies(browser)
        
    else:
        auth.configure_from_env()
        
    return auth

#
# End of confluence_auth.py
#######################################################################################################################