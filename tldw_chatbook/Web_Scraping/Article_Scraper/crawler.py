"""
article_scraper/crawler.py
==========================

Website crawling and sitemap parsing functionality.

This module provides tools for discovering URLs on websites through:
- Crawling internal links with configurable depth
- Parsing XML sitemaps
- URL filtering to focus on content pages

Functions:
----------
- crawl_site(): Discover URLs by crawling internal links
- get_urls_from_sitemap(): Extract URLs from sitemap.xml
- default_url_filter(): Filter out non-content URLs

Example:
--------
    # Crawl a website
    urls = await crawl_site(
        base_url="https://example.com",
        max_pages=100,
        max_depth=3
    )
    
    # Parse sitemap
    urls = await get_urls_from_sitemap("https://example.com/sitemap.xml")
"""
#
# Imports
import asyncio
import logging
from typing import List, Set, Callable, Optional
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET
#
# Third-Party Libraries
import aiohttp
from bs4 import BeautifulSoup
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

# Default URL filter to avoid crawling non-content pages
def default_url_filter(url: str) -> bool:
    """
    Default filter to determine if a URL should be crawled.
    Excludes common non-content pages and file types.
    """
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        # Exclude specific file extensions
        excluded_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.css', '.js')
        if path.endswith(excluded_extensions):
            return False

        # Exclude common non-article patterns
        excluded_patterns = [
            '/tag/', '/category/', '/author/', '/search/', '/page/',
            'wp-content', 'wp-includes', 'wp-json', 'wp-admin',
            'login', 'register', 'cart', 'checkout', 'account', 'tel:', 'mailto:'
        ]
        if any(pattern in url.lower() for pattern in excluded_patterns):
            return False

    except (ValueError, AttributeError):
        return False  # Ignore malformed URLs

    return True


async def crawl_site(
        base_url: str,
        max_pages: int = 100,
        max_depth: int = 5,
        url_filter: Callable[[str], bool] = default_url_filter
) -> Set[str]:
    """
    Asynchronously crawls a website to discover internal links.

    Args:
        base_url: The starting URL for the crawl.
        max_pages: The maximum number of pages to crawl.
        max_depth: The maximum depth to follow links from the base URL.
        url_filter: A function to decide if a URL should be included.

    Returns:
        A set of discovered and filtered URLs.
    """
    logging.info(f"Starting crawl of {base_url} (max_pages={max_pages}, max_depth={max_depth})")

    # Use a set for visited URLs to avoid duplicates and ensure fast lookups
    visited: Set[str] = set()
    # Use a list as a queue for URLs to visit, storing (url, depth)
    to_visit: List[tuple[str, int]] = [(base_url, 0)]

    # Keep track of the domain to stay on the same site
    base_domain = urlparse(base_url).netloc

    async with aiohttp.ClientSession() as session:
        while to_visit and len(visited) < max_pages:
            current_url, current_depth = to_visit.pop(0)

            if current_url in visited or not url_filter(current_url):
                continue

            if current_depth > max_depth:
                continue

            visited.add(current_url)
            logging.debug(f"Crawling (Depth {current_depth}): {current_url}")

            try:
                async with session.get(current_url, timeout=10) as response:
                    if response.status != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
                        continue

                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')

                    for link in soup.find_all('a', href=True):
                        href = link.get('href')
                        if not href:
                            continue

                        # Create an absolute URL from the relative link
                        full_url = urljoin(current_url, href).split('#')[0]  # Remove fragments

                        # Check if the URL is on the same domain and hasn't been seen
                        if urlparse(full_url).netloc == base_domain and full_url not in visited:
                            to_visit.append((full_url, current_depth + 1))

            except Exception as e:
                logging.warning(f"Failed to crawl {current_url}: {e}")

    logging.info(f"Crawl finished. Found {len(visited)} valid URLs.")
    return visited


async def get_urls_from_sitemap(sitemap_url: str, url_filter: Callable[[str], bool] = default_url_filter) -> List[str]:
    """
    Fetches and parses a sitemap.xml file to extract a list of URLs.

    Args:
        sitemap_url: The URL of the sitemap.xml file.
        url_filter: A function to decide if a URL should be included.

    Returns:
        A list of filtered URLs found in the sitemap.
    """
    logging.info(f"Fetching URLs from sitemap: {sitemap_url}")
    urls: List[str] = []

    # Namespace for sitemap XML
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(sitemap_url, timeout=30) as response:
                response.raise_for_status()
                xml_content = await response.text()

        root = ET.fromstring(xml_content)

        for url_element in root.findall('sm:url', ns):
            loc_element = url_element.find('sm:loc', ns)
            if loc_element is not None and loc_element.text:
                url = loc_element.text.strip()
                if url_filter(url):
                    urls.append(url)

    except aiohttp.ClientError as e:
        logging.error(f"HTTP error fetching sitemap {sitemap_url}: {e}")
    except ET.ParseError as e:
        logging.error(f"XML parse error for sitemap {sitemap_url}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing sitemap {sitemap_url}: {e}")

    logging.info(f"Found {len(urls)} URLs in sitemap.")
    return urls

#
# End of article_scraper/crawler.py
#######################################################################################################################
