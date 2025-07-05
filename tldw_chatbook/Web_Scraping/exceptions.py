"""
Custom exceptions for the web scraping module.

This module defines specific exception types for different error scenarios
in web scraping, improving error handling and debugging.
"""


class WebScrapingError(Exception):
    """Base exception for all web scraping errors."""
    pass


class InvalidURLError(WebScrapingError):
    """Raised when an invalid URL is provided."""
    pass


class NetworkError(WebScrapingError):
    """Raised when network-related errors occur."""
    pass


class TimeoutError(WebScrapingError):
    """Raised when a scraping operation times out."""
    pass


class BrowserError(WebScrapingError):
    """Raised when browser-related errors occur."""
    pass


class ContentExtractionError(WebScrapingError):
    """Raised when content extraction fails."""
    pass


class CookieError(WebScrapingError):
    """Raised when cookie-related operations fail."""
    pass


class RateLimitError(WebScrapingError):
    """Raised when rate limits are exceeded."""
    pass


class AuthenticationError(WebScrapingError):
    """Raised when authentication is required or fails."""
    pass


class ConfigurationError(WebScrapingError):
    """Raised when configuration is invalid or missing."""
    pass


class ResourceExhaustedError(WebScrapingError):
    """Raised when system resources are exhausted."""
    pass


class MaxRetriesExceededError(WebScrapingError):
    """Raised when maximum retry attempts are exceeded."""
    def __init__(self, url: str, attempts: int, last_error: Exception = None):
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Max retries ({attempts}) exceeded for URL: {url}")


class SiteStructureError(WebScrapingError):
    """Raised when site structure prevents scraping."""
    pass