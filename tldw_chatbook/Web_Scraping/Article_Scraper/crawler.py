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
import time
from typing import List, Set, Callable, Optional
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET
#
# Third-Party Libraries
import aiohttp
from bs4 import BeautifulSoup
#
# Local Imports
from ...Metrics.metrics_logger import log_counter, log_histogram
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
    start_time = time.time()
    logging.info(f"Starting crawl of {base_url} (max_pages={max_pages}, max_depth={max_depth})")
    
    # Log crawl start
    log_counter("crawler_site_crawl_start", labels={
        "max_pages": str(max_pages),
        "max_depth": str(max_depth)
    })

    # Use a set for visited URLs to avoid duplicates and ensure fast lookups
    visited: Set[str] = set()
    # Use a list as a queue for URLs to visit, storing (url, depth)
    to_visit: List[tuple[str, int]] = [(base_url, 0)]

    # Keep track of the domain to stay on the same site
    base_domain = urlparse(base_url).netloc
    
    # Track statistics
    pages_crawled = 0
    errors_count = 0
    max_depth_reached = 0

    async with aiohttp.ClientSession() as session:
        while to_visit and len(visited) < max_pages:
            current_url, current_depth = to_visit.pop(0)

            if current_url in visited or not url_filter(current_url):
                log_counter("crawler_url_filtered", labels={"reason": "visited_or_filtered"})
                continue

            if current_depth > max_depth:
                log_counter("crawler_url_filtered", labels={"reason": "max_depth_exceeded"})
                continue

            visited.add(current_url)
            pages_crawled += 1
            max_depth_reached = max(max_depth_reached, current_depth)
            logging.debug(f"Crawling (Depth {current_depth}): {current_url}")
            
            # Log page crawl attempt
            page_start = time.time()
            log_counter("crawler_page_attempt", labels={"depth": str(current_depth)})

            try:
                async with session.get(current_url, timeout=10) as response:
                    if response.status != 200:
                        log_counter("crawler_http_response", labels={"status": str(response.status)})
                        continue
                    
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' not in content_type:
                        log_counter("crawler_content_type_skip", labels={"content_type": content_type.split(';')[0]})
                        continue

                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    
                    # Count links found
                    links_found = 0
                    new_links_added = 0

                    for link in soup.find_all('a', href=True):
                        href = link.get('href')
                        if not href:
                            continue
                        
                        links_found += 1

                        # Create an absolute URL from the relative link
                        full_url = urljoin(current_url, href).split('#')[0]  # Remove fragments

                        # Check if the URL is on the same domain and hasn't been seen
                        if urlparse(full_url).netloc == base_domain and full_url not in visited:
                            to_visit.append((full_url, current_depth + 1))
                            new_links_added += 1
                    
                    # Log page crawl success
                    page_duration = time.time() - page_start
                    log_histogram("crawler_page_duration", page_duration, labels={"status": "success"})
                    log_histogram("crawler_links_per_page", links_found)
                    log_histogram("crawler_new_links_per_page", new_links_added)
                    log_counter("crawler_page_success", labels={"depth": str(current_depth)})

            except asyncio.TimeoutError:
                errors_count += 1
                page_duration = time.time() - page_start
                log_histogram("crawler_page_duration", page_duration, labels={"status": "timeout"})
                log_counter("crawler_page_error", labels={"error_type": "timeout", "depth": str(current_depth)})
                logging.warning(f"Timeout crawling {current_url}")
            except Exception as e:
                errors_count += 1
                page_duration = time.time() - page_start
                log_histogram("crawler_page_duration", page_duration, labels={"status": "error"})
                log_counter("crawler_page_error", labels={"error_type": type(e).__name__, "depth": str(current_depth)})
                logging.warning(f"Failed to crawl {current_url}: {e}")

    # Log final statistics
    duration = time.time() - start_time
    log_histogram("crawler_site_duration", duration, labels={
        "pages_found": str(len(visited)),
        "pages_crawled": str(pages_crawled)
    })
    log_counter("crawler_site_complete", labels={
        "pages_found": str(len(visited)),
        "pages_crawled": str(pages_crawled),
        "errors": str(errors_count),
        "max_depth_reached": str(max_depth_reached),
        "hit_max_pages": str(len(visited) >= max_pages)
    })
    
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
    start_time = time.time()
    logging.info(f"Fetching URLs from sitemap: {sitemap_url}")
    log_counter("crawler_sitemap_request")
    
    urls: List[str] = []
    total_urls_in_sitemap = 0
    filtered_urls = 0

    # Namespace for sitemap XML
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    try:
        async with aiohttp.ClientSession() as session:
            fetch_start = time.time()
            async with session.get(sitemap_url, timeout=30) as response:
                response.raise_for_status()
                xml_content = await response.text()
                
                # Log successful fetch
                fetch_duration = time.time() - fetch_start
                log_histogram("crawler_sitemap_fetch_duration", fetch_duration, labels={"status": "success"})
                log_counter("crawler_sitemap_fetch_success", labels={"status_code": str(response.status)})

        # Parse XML
        parse_start = time.time()
        root = ET.fromstring(xml_content)

        for url_element in root.findall('sm:url', ns):
            loc_element = url_element.find('sm:loc', ns)
            if loc_element is not None and loc_element.text:
                total_urls_in_sitemap += 1
                url = loc_element.text.strip()
                if url_filter(url):
                    urls.append(url)
                else:
                    filtered_urls += 1
        
        # Log parsing success
        parse_duration = time.time() - parse_start
        log_histogram("crawler_sitemap_parse_duration", parse_duration, labels={"status": "success"})

    except aiohttp.ClientError as e:
        # Log HTTP error
        duration = time.time() - start_time
        log_histogram("crawler_sitemap_duration", duration, labels={"status": "http_error"})
        log_counter("crawler_sitemap_error", labels={"error_type": "http_error", "error_class": type(e).__name__})
        logging.error(f"HTTP error fetching sitemap {sitemap_url}: {e}")
    except ET.ParseError as e:
        # Log XML parse error
        duration = time.time() - start_time
        log_histogram("crawler_sitemap_duration", duration, labels={"status": "parse_error"})
        log_counter("crawler_sitemap_error", labels={"error_type": "xml_parse_error"})
        logging.error(f"XML parse error for sitemap {sitemap_url}: {e}")
    except Exception as e:
        # Log unexpected error
        duration = time.time() - start_time
        log_histogram("crawler_sitemap_duration", duration, labels={"status": "error"})
        log_counter("crawler_sitemap_error", labels={"error_type": "unexpected", "error_class": type(e).__name__})
        logging.error(f"An unexpected error occurred while processing sitemap {sitemap_url}: {e}")

    # Log final results
    total_duration = time.time() - start_time
    log_histogram("crawler_sitemap_duration", total_duration, labels={
        "status": "complete",
        "urls_found": str(len(urls))
    })
    log_counter("crawler_sitemap_complete", labels={
        "total_urls": str(total_urls_in_sitemap),
        "accepted_urls": str(len(urls)),
        "filtered_urls": str(filtered_urls)
    })
    
    logging.info(f"Found {len(urls)} URLs in sitemap.")
    return urls

#
# End of article_scraper/crawler.py
#######################################################################################################################
