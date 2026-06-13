"""Server web-scraping management interoperability services."""

from .server_web_scraping_service import ServerWebScrapingService
from .web_scraping_scope_service import WebScrapingBackend, WebScrapingScopeService

__all__ = [
    "ServerWebScrapingService",
    "WebScrapingBackend",
    "WebScrapingScopeService",
]
