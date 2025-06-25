# __init__.py for Confluence scraping module
#
# This module provides functionality for scraping Confluence pages and spaces
# using the existing web scraping infrastructure.
#
from .confluence_auth import ConfluenceAuth
from .confluence_scraper import ConfluenceScraper
from .confluence_crawler import ConfluenceCrawler
from .confluence_main import (
    scrape_confluence_page,
    scrape_confluence_space,
    scrape_confluence_search,
    scrape_confluence_with_config
)
from .confluence_utils import (
    convert_confluence_to_markdown,
    extract_confluence_metadata,
    parse_confluence_url
)

__all__ = [
    'ConfluenceAuth',
    'ConfluenceScraper',
    'ConfluenceCrawler',
    'scrape_confluence_page',
    'scrape_confluence_space',
    'scrape_confluence_search',
    'scrape_confluence_with_config',
    'convert_confluence_to_markdown',
    'extract_confluence_metadata',
    'parse_confluence_url'
]