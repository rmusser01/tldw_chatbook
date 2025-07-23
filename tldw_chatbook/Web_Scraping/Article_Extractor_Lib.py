"""
Article_Extractor_Lib.py
========================

Article Extraction Library for web scraping and content processing.

This module provides comprehensive functionality for:
- Scraping articles from web pages using Playwright
- Extracting content with Trafilatura
- Processing multiple URLs with summarization
- Handling sitemaps and crawling websites
- Managing bookmarks from various browsers
- Content metadata handling and deduplication

Main Functions:
--------------
- get_page_title(url): Extract page title from URL
- scrape_article(url, custom_cookies): Async scrape single article
- scrape_and_summarize_multiple(): Process multiple URLs with optional summarization
- scrape_entire_site(): Crawl and scrape entire website
- collect_bookmarks(): Import bookmarks from browsers
- ContentMetadataHandler: Manage content metadata

Dependencies:
------------
- playwright: Browser automation
- trafilatura: Content extraction
- beautifulsoup4: HTML parsing
- pandas: Data manipulation
- asyncio: Asynchronous operations
"""
#
# Import necessary libraries
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
import random
import tempfile
from typing import Any, Dict, List, Union, Optional, Tuple
#
# 3rd-Party Imports
import asyncio
from urllib.parse import (
    urljoin,
    urlparse
)
from xml.dom import minidom
import xml.etree.ElementTree as xET
#
# External Libraries
# Handle optional web scraping dependencies
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from playwright.async_api import (
        TimeoutError,
        async_playwright
    )
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    TimeoutError = Exception
    async_playwright = None
    sync_playwright = None

import requests

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    trafilatura = None

from tqdm import tqdm
#
# Import Local
from tldw_chatbook.LLM_Calls.Summarization_General_Lib import analyze
from tldw_chatbook.Metrics.metrics_logger import log_histogram, log_counter
from tldw_chatbook.Logging_Config import logging
from tldw_chatbook.DB.Client_Media_DB_v2 import ingest_article_to_db_new
from tldw_chatbook.Utils.input_validation import validate_url, sanitize_string
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Utils.secure_temp_files import secure_temp_file, get_temp_manager
from tldw_chatbook.Web_Scraping.exceptions import (
    InvalidURLError, NetworkError, BrowserError, ContentExtractionError,
    MaxRetriesExceededError, TimeoutError as ScrapingTimeoutError
)

#
#######################################################################################################################
# Function Definitions
#
load_and_log_configs = lambda: {}
# FIXME - Add a config file option/check for the user agent
web_scraping_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

#################################################################
#
# Concurrency control
#
# Global semaphore to limit concurrent browser operations
# This prevents resource exhaustion and improves stability
MAX_CONCURRENT_SCRAPERS = 5  # Configurable limit
scraping_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPERS)

# Thread-local event loop storage for proper async/sync handling
import threading
_thread_local = threading.local()


def get_or_create_event_loop():
    """Get the current event loop or create one if needed."""
    try:
        # Try to get the running loop
        loop = asyncio.get_running_loop()
        return loop, False  # Return loop and indicate we're already in async context
    except RuntimeError:
        # No running loop, check thread-local storage
        if not hasattr(_thread_local, 'loop') or _thread_local.loop.is_closed():
            _thread_local.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_thread_local.loop)
        return _thread_local.loop, True  # Return loop and indicate we created it


def run_async_function(async_func, *args, **kwargs):
    """
    Properly run an async function from sync context.
    
    This avoids creating multiple event loops and handles the case where
    we're already in an async context.
    """
    loop, created = get_or_create_event_loop()
    
    if created:
        # We're in sync context, run the coroutine
        return loop.run_until_complete(async_func(*args, **kwargs))
    else:
        # We're already in async context, this shouldn't happen
        # but if it does, create a task
        raise RuntimeError("Cannot use run_async_function from within an async context. Use await directly.")


async def scrape_urls_batch(urls: List[str], progress_callback=None) -> List[Dict]:
    """
    Scrape multiple URLs with controlled concurrency.
    
    Args:
        urls: List of URLs to scrape
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List of article data dictionaries
    """
    results = []
    
    async def scrape_with_progress(url: str, index: int) -> Dict:
        try:
            result = await scrape_article(url)
            if progress_callback:
                progress_callback(index + 1, len(urls))
            return result
        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
            return {
                'url': url,
                'extraction_successful': False,
                'error': str(e)
            }
    
    # Create tasks for all URLs
    tasks = [scrape_with_progress(url, i) for i, url in enumerate(urls)]
    
    # Execute with controlled concurrency (semaphore is used inside scrape_article)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and None values
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Batch scraping error: {result}")
        elif result and isinstance(result, dict):
            valid_results.append(result)
    
    return valid_results

#################################################################
#
# Scraping-related functions:

def get_page_title(url: str) -> str:
    """
    Extract the title from a web page.
    
    Args:
        url (str): The URL of the page to extract title from
        
    Returns:
        str: The page title, or "Untitled" if extraction fails
        
    Example:
        >>> title = get_page_title("https://example.com")
        >>> print(title)
        "Example Domain"
    """
    # Validate URL before processing
    if not validate_url(url):
        logging.error(f"Invalid URL provided to get_page_title: {url}")
        return "Untitled"
    
    if not BS4_AVAILABLE:
        logging.warning("BeautifulSoup not available. Install with: pip install tldw_chatbook[websearch]")
        return "Untitled (BeautifulSoup not available)"
    
    try:
        response = requests.get(url, timeout=10)  # Add timeout
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.string.strip() if title_tag and title_tag.string else "Untitled"
            log_counter("page_title_extracted", labels={"success": "true"})
            return title
        elif response.status_code == 401:
            logging.warning(f"Authentication required for {url}")
            return "Untitled (Authentication Required)"
        elif response.status_code == 403:
            logging.warning(f"Access forbidden for {url}")
            return "Untitled (Access Forbidden)"
        elif response.status_code == 404:
            logging.warning(f"Page not found: {url}")
            return "Untitled (Not Found)"
        else:
            logging.error(f"Failed to fetch {url}, status code: {response.status_code}")
            return "Untitled"
    except requests.Timeout:
        logging.error(f"Timeout fetching page title from {url}")
        log_counter("page_title_extracted", labels={"success": "false", "error": "timeout"})
        return "Untitled (Timeout)"
    except requests.ConnectionError:
        logging.error(f"Connection error fetching page title from {url}")
        log_counter("page_title_extracted", labels={"success": "false", "error": "connection"})
        return "Untitled (Connection Error)"
    except requests.RequestException as e:
        logging.error(f"Error fetching page title: {e}")
        log_counter("page_title_extracted", labels={"success": "false", "error": "other"})
        return "Untitled"


async def scrape_article(url: str, custom_cookies: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Asynchronously scrape an article from a URL using Playwright.
    
    This function handles the complete scraping process including:
    - Browser automation with optional stealth mode
    - Cookie injection for authenticated scraping
    - Content extraction with Trafilatura
    - Metadata preservation
    
    Args:
        url (str): The URL to scrape
        custom_cookies (Optional[List[Dict[str, Any]]]): Browser cookies for authentication
            Each cookie dict should have: name, value, domain, path, etc.
            
    Returns:
        Dict[str, Any]: Article data containing:
            - title: Article title
            - author: Article author
            - content: Extracted content in markdown
            - date: Publication date
            - url: Source URL
            - extraction_successful: Success status
            
    Example:
        >>> article = await scrape_article("https://example.com/article")
        >>> if article['extraction_successful']:
        ...     print(article['title'])
    """
    # Validate URL before processing
    if not validate_url(url):
        logging.error(f"Invalid URL provided: {url}")
        return {
            'title': 'Invalid URL',
            'author': 'N/A',
            'content': 'The provided URL is invalid or malformed.',
            'date': datetime.now().isoformat(),
            'url': url,
            'extraction_successful': False
        }
    
    logging.info(f"Scraping article from URL: {url}")
    
    # Check required dependencies
    if not PLAYWRIGHT_AVAILABLE:
        logging.error("Playwright not available. Install with: pip install tldw_chatbook[websearch]")
        return {
            'title': 'Error',
            'author': 'Unknown',
            'content': 'Playwright not available for web scraping',
            'date': datetime.now().isoformat(),
            'url': url,
            'extraction_successful': False,
            'error': 'Playwright not installed'
        }
    
    if not TRAFILATURA_AVAILABLE:
        logging.error("Trafilatura not available. Install with: pip install tldw_chatbook[websearch]")
        return {
            'title': 'Error',
            'author': 'Unknown',
            'content': 'Trafilatura not available for content extraction',
            'date': datetime.now().isoformat(),
            'url': url,
            'extraction_successful': False,
            'error': 'Trafilatura not installed'
        }
    
    async def fetch_html(url: str) -> str:
            # Load and log the configuration
            loaded_config = load_and_log_configs()

            # load retry count from config
            scrape_retry_count = loaded_config['web_scraper'].get('web_scraper_retry_count', 3)
            retries = scrape_retry_count
            # Load retry timeout value from config
            web_scraper_retry_timeout = loaded_config['web_scraper'].get('web_scraper_retry_timeout', 60)
            timeout_ms = web_scraper_retry_timeout

            # Whether stealth mode is enabled
            stealth_enabled = loaded_config['web_scraper'].get('web_scraper_stealth_playwright', False)

            for attempt in range(retries):  # Introduced a retry loop to attempt fetching HTML multiple times
                browser = None
                try:
                    logging.info(f"Fetching HTML from {url} (Attempt {attempt + 1}/{retries})")

                    async with async_playwright as p:
                        browser = await p.chromium.launch(headless=True)
                        context = await browser.new_context(
                            user_agent=web_scraping_user_agent,
                            # Simulating a normal browser window size for better compatibility
                            viewport={"width": 1280, "height": 720},
                        )
                        if custom_cookies:
                            # Apply cookies if provided
                            await context.add_cookies(custom_cookies)

                        page = await context.new_page()

                        # Check if stealth mode is enabled in the config
                        if stealth_enabled:
                            from playwright_stealth import stealth_async
                            await stealth_async(page)

                        # Navigate to the URL
                        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)


                        # If stealth is enabled, give the page extra time to finish loading/spawning content
                        if stealth_enabled:
                            await page.wait_for_timeout(5000)  # 5-second delay
                        else:
                            # Alternatively, wait for network to be idle
                            await page.wait_for_load_state("networkidle", timeout=timeout_ms)

                        # Capture final HTML
                        content = await page.content()

                        logging.info(f"HTML fetched successfully from {url}")
                        log_counter("html_fetched", labels={"url": url})

                        # Return the scraped HTML
                        return content

                except TimeoutError as e:
                    logging.error(f"Timeout fetching HTML from {url} on attempt {attempt + 1}: {e}")
                    error_type = "timeout"
                    last_error = ScrapingTimeoutError(f"Page load timeout for {url}")
                    
                except Exception as e:
                    # Categorize the error
                    if "net::" in str(e) or "Network" in str(e):
                        error_type = "network"
                        last_error = NetworkError(f"Network error accessing {url}: {e}")
                    elif "browser" in str(e).lower() or "chrome" in str(e).lower():
                        error_type = "browser"
                        last_error = BrowserError(f"Browser error accessing {url}: {e}")
                    else:
                        error_type = "unknown"
                        last_error = e
                    
                    logging.error(f"Error fetching HTML from {url} on attempt {attempt + 1}: {e}")

                    if attempt < retries - 1:
                        logging.info(f"Retrying in {2 * (attempt + 1)} seconds...")
                        await asyncio.sleep(2 * (attempt + 1))  # Exponential backoff
                    else:
                        logging.error("Max retries reached, giving up on this URL.")
                        log_counter("html_fetch_error", labels={"url": url, "error": error_type})
                        raise MaxRetriesExceededError(url, retries, last_error)

                finally:
                    # Ensure the browser is closed before returning
                    if browser is not None:
                        await browser.close()

            # If for some reason you exit the loop without returning (unlikely), return empty string
            return ""

    def extract_article_data(html: str, url: str) -> dict:
        logging.info(f"Extracting article data from HTML for {url}")
        # FIXME - Add option for extracting comments/tables/images
        downloaded = trafilatura.extract(html, include_comments=False, include_tables=False, include_images=False)
        metadata = trafilatura.extract_metadata(html)

        result = {
            'title': 'N/A',
            'author': 'N/A',
            'content': '',
            'date': 'N/A',
            'url': url,
            'extraction_successful': False
        }

        if downloaded:
            logging.info(f"Content extracted successfully from {url}")
            log_counter("article_extracted", labels={"success": "true", "url": url})
            # Add metadata to content
            result['content'] = ContentMetadataHandler.format_content_with_metadata(
                url=url,
                content=downloaded,
                pipeline="Trafilatura",
                additional_metadata={
                    "extracted_date": metadata.date if metadata and metadata.date else 'N/A',
                    "author": metadata.author if metadata and metadata.author else 'N/A'
                }
            )
            result['extraction_successful'] = True

        if metadata:
            result.update({
                'title': metadata.title if metadata.title else 'N/A',
                'author': metadata.author if metadata.author else 'N/A',
                'date': metadata.date if metadata.date else 'N/A'
            })
        else:
            log_counter("article_extracted", labels={"success": "false", "url": url})
            logging.warning("Metadata extraction failed.")

        if not downloaded:
            logging.warning("Content extraction failed.")

        return result

    def convert_html_to_markdown(html: str) -> str:
        logging.info("Converting HTML to Markdown")
        if not BS4_AVAILABLE:
            logging.warning("BeautifulSoup not available for HTML to Markdown conversion")
            # Return raw HTML as fallback
            return html
        soup = BeautifulSoup(html, 'html.parser')
        for para in soup.find_all('p'):
            # Add a newline at the end of each paragraph for markdown separation
            para.append('\n')
        # Use .get_text() with separator to keep paragraph separation
        return soup.get_text(separator='\n\n')

    # Use semaphore to limit concurrent browser instances
    async with scraping_semaphore:
        try:
            html = await fetch_html(url)
            article_data = extract_article_data(html, url)
            if article_data['extraction_successful']:
                article_data['content'] = convert_html_to_markdown(article_data['content'])
                logging.info(f"Article content length: {len(article_data['content'])}")
                log_histogram("article_content_length", len(article_data['content']), labels={"url": url})
            return article_data
        except MaxRetriesExceededError as e:
            logging.error(f"Failed to scrape {url} after {e.attempts} attempts: {e.last_error}")
            return {
                'title': 'Scraping Failed',
                'author': 'N/A',
                'content': f'Failed to scrape article after {e.attempts} attempts. Last error: {e.last_error}',
                'date': datetime.now().isoformat(),
                'url': url,
                'extraction_successful': False,
                'error': str(e.last_error)
            }
        except (InvalidURLError, NetworkError, BrowserError, ScrapingTimeoutError) as e:
            logging.error(f"Scraping error for {url}: {e}")
            return {
                'title': 'Scraping Error',
                'author': 'N/A',
                'content': f'Error scraping article: {e}',
                'date': datetime.now().isoformat(),
                'url': url,
                'extraction_successful': False,
                'error': str(e)
            }


# FIXME - Add keyword integration/tagging
async def scrape_and_summarize_multiple(
    urls: str,
    custom_prompt_arg: Optional[str],
    api_name: str,
    api_key: Optional[str],
    keywords: str,
    custom_article_titles: Optional[str],
    system_message: Optional[str] = None,
    summarize_checkbox: bool = False,
    custom_cookies: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Scrape and optionally summarize multiple URLs concurrently.
    
    This function processes multiple URLs in parallel, extracting content
    and optionally generating summaries using the specified LLM API.
    
    Args:
        urls (str): Newline-separated string of URLs to process
        custom_prompt_arg (Optional[str]): Custom prompt for summarization
        api_name (str): Name of the LLM API to use (e.g., 'openai', 'anthropic')
        api_key (Optional[str]): API key for the LLM service
        keywords (str): Keywords for categorization
        custom_article_titles (Optional[str]): Newline-separated custom titles
        system_message (Optional[str]): System message for LLM
        summarize_checkbox (bool): Whether to generate summaries
        custom_cookies (Optional[List[Dict[str, Any]]]): Cookies for authentication
        temperature (float): LLM temperature setting (0.0-1.0)
        
    Returns:
        List[Dict[str, Any]]: List of processed articles with:
            - All fields from scrape_article()
            - summary: Generated summary (if enabled)
            
    Example:
        =>>> results = await scrape_and_summarize_multiple(
        ...     urls="https://example.com/1\nhttps://example.com/2",
        ...     api_name="openai",
        ...     summarize_checkbox=True
        ... )
    """
    # Check required dependencies
    if not PLAYWRIGHT_AVAILABLE or not TRAFILATURA_AVAILABLE:
        missing = []
        if not PLAYWRIGHT_AVAILABLE:
            missing.append("playwright")
        if not TRAFILATURA_AVAILABLE:
            missing.append("trafilatura")
        logging.error(f"Missing dependencies: {', '.join(missing)}. Install with: pip install tldw_chatbook[websearch]")
        return []
    
    urls_list = [url.strip() for url in urls.split('\n') if url.strip()]
    custom_titles = custom_article_titles.split('\n') if custom_article_titles else []

    results = []
    errors = []

    # Create a tqdm progress bar
    progress_bar = tqdm(total=len(urls_list), desc="Scraping and Summarizing")

    # Loop over each URL to scrape and optionally summarize
    for i, url in enumerate(urls_list):
        custom_title = custom_titles[i] if i < len(custom_titles) else None
        try:
            # Scrape the article
            article = await scrape_article(url, custom_cookies=custom_cookies)
            if article and article['extraction_successful']:
                log_counter("article_scraped", labels={"success": "true", "url": url})
                if custom_title:
                    article['title'] = custom_title

                # If summarization is requested
                if summarize_checkbox:
                    content = article.get('content', '')
                    if content:
                        # Prepare prompts
                        system_message_final = system_message or \
                                               "Act as a professional summarizer and summarize this article."
                        article_custom_prompt = custom_prompt_arg or \
                                                "Act as a professional summarizer and summarize this article."

                        # Summarize the content using the summarize function
                        summary = analyze(
                            input_data=content,
                            custom_prompt_arg=article_custom_prompt,
                            api_name=api_name,
                            api_key=api_key,
                            temp=temperature,
                            system_message=system_message_final
                        )
                        article['summary'] = summary
                        log_counter("article_summarized", labels={"success": "true", "url": url})
                        logging.info(f"Summary generated for URL {url}")
                    else:
                        article['summary'] = "No content available to summarize."
                        logging.warning(f"No content to summarize for URL {url}")
                else:
                    article['summary'] = None

                results.append(article)
            else:
                error_message = f"Extraction unsuccessful for URL {url}"
                errors.append(error_message)
                logging.error(error_message)
                log_counter("article_scraped", labels={"success": "false", "url": url})
        except Exception as e:
            log_counter("article_processing_error", labels={"url": url})
            error_message = f"Error processing URL {i + 1} ({url}): {str(e)}"
            errors.append(error_message)
            logging.error(error_message, exc_info=True)
        finally:
            # Update the progress bar
            progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    if errors:
        logging.error("\n".join(errors))

    if not results:
        logging.error("No articles were successfully scraped and summarized/analyzed.")
        return []

    log_histogram("articles_processed", len(results))
    return results


def scrape_and_no_summarize_then_ingest(url, keywords, custom_article_title):
    # Validate URL before processing
    if not validate_url(url):
        logging.error(f"Invalid URL provided: {url}")
        return "Invalid URL provided."
    
    try:
        # Step 1: Scrape the article
        article_data = run_async_function(scrape_article, url)
        print(f"Scraped Article Data: {article_data}")  # Debugging statement
        if not article_data:
            log_counter("article_scrape_failed", labels={"url": url})
            return "Failed to scrape the article."

        # Use the custom title if provided, otherwise use the scraped title
        title = custom_article_title.strip() if custom_article_title else article_data.get('title', 'Untitled')
        author = article_data.get('author', 'Unknown')
        content = article_data.get('content', '')
        ingestion_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Title: {title}, Author: {author}, Content Length: {len(content)}")  # Debugging statement

        # Step 2: Ingest the article into the database
        ingestion_result = ingest_article_to_db_new(url, title, author, content, keywords, ingestion_date, None, None)
        log_counter("article_ingested", labels={"success": str(ingestion_result).lower(), "url": url})

        # When displaying content, we might want to strip metadata
        display_content = ContentMetadataHandler.strip_metadata(content)
        return f"Title: {title}\nAuthor: {author}\nIngestion Result: {ingestion_result}\n\nArticle Contents: {display_content}"
    except Exception as e:
        log_counter("article_processing_error", labels={"url": url})
        logging.error(f"Error processing URL {url}: {str(e)}")
        return f"Failed to process URL {url}: {str(e)}"


def scrape_from_filtered_sitemap(sitemap_file: str, filter_function) -> list:
    """
    Scrape articles from a sitemap file, applying an additional filter function.

    :param sitemap_file: Path to the sitemap file
    :param filter_function: A function that takes a URL and returns True if it should be scraped
    :return: List of scraped articles
    """
    try:
        tree = xET.parse(sitemap_file)
        root = tree.getroot()

        articles = []
        for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
            if filter_function(url.text):
                article_data = scrape_article(url.text)
                if article_data:
                    articles.append(article_data)

        return articles
    except xET.ParseError as e:
        logging.error(f"Error parsing sitemap: {e}")
        return []


def is_content_page(url: str) -> bool:
    """
    Determine if a URL is likely to be a content page.
    This is a basic implementation and may need to be adjusted based on the specific website structure.

    :param url: The URL to check
    :return: True if the URL is likely a content page, False otherwise
    """
    # Add more specific checks here based on the website's structure
    # Exclude common non-content pages
    exclude_patterns = [
        '/tag/', '/category/', '/author/', '/search/', '/page/',
        'wp-content', 'wp-includes', 'wp-json', 'wp-admin',
        'login', 'register', 'cart', 'checkout', 'account',
        '.jpg', '.png', '.gif', '.pdf', '.zip'
    ]
    return not any(pattern in url.lower() for pattern in exclude_patterns)

def scrape_and_convert_with_filter(source: str, output_file: str, filter_function=is_content_page, level: int = None):
    """
    Scrape articles from a sitemap or by URL level, apply filtering, and convert to a single markdown file.

    :param source: URL of the sitemap, base URL for level-based scraping, or path to a local sitemap file
    :param output_file: Path to save the output markdown file
    :param filter_function: Function to filter URLs (default is is_content_page)
    :param level: URL level for scraping (None if using sitemap)
    """
    if level is not None:
        # Scraping by URL level
        articles = scrape_by_url_level(source, level)
        articles = [article for article in articles if filter_function(article['url'])]
    elif source.startswith('http'):
        # Scraping from online sitemap
        articles = scrape_from_sitemap(source)
        articles = [article for article in articles if filter_function(article['url'])]
    else:
        # Scraping from local sitemap file
        articles = scrape_from_filtered_sitemap(source, filter_function)

    articles = [article for article in articles if filter_function(article['url'])]
    markdown_content = convert_to_markdown(articles)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    logging.info(f"Scraped and filtered content saved to {output_file}")


async def scrape_entire_site(base_url: str) -> List[Dict]:
    """
    Scrape the entire site by generating a temporary sitemap and extracting content from each page.

    :param base_url: The base URL of the site to scrape
    :return: A list of dictionaries containing scraped article data
    """
    # Validate URL before processing
    if not validate_url(base_url):
        logging.error(f"Invalid base URL provided: {base_url}")
        return []
    
    # Step 1: Collect internal links from the site
    links = collect_internal_links(base_url)
    log_histogram("internal_links_collected", len(links), labels={"base_url": base_url})
    logging.info(f"Collected {len(links)} internal links.")

    # Step 2: Generate the temporary sitemap
    temp_sitemap_path = generate_temp_sitemap_from_links(links)

    # Step 3: Scrape each URL in the sitemap
    scraped_articles = []
    try:
        async def scrape_and_log(link):
            logging.info(f"Scraping {link} ...")
            article_data = await scrape_article(link)

            if article_data:
                logging.info(f"Title: {article_data['title']}")
                logging.info(f"Author: {article_data['author']}")
                logging.info(f"Date: {article_data['date']}")
                logging.info(f"Content: {article_data['content'][:500]}...")

                return article_data
            return None

        # Use asyncio.gather to scrape multiple articles concurrently
        scraped_articles = await asyncio.gather(*[scrape_and_log(link) for link in links])
        # Remove any None values (failed scrapes)
        scraped_articles = [article for article in scraped_articles if article is not None]
        log_histogram("articles_scraped", len(scraped_articles), labels={"base_url": base_url})

    finally:
        # Clean up the temporary sitemap file
        os.unlink(temp_sitemap_path)
        logging.info("Temporary sitemap file deleted")

    return scraped_articles


def scrape_by_url_level(base_url: str, level: int) -> list:
    """Scrape articles from URLs up to a certain level under the base URL."""

    def get_url_level(url: str) -> int:
        return len(urlparse(url).path.strip('/').split('/'))

    links = collect_internal_links(base_url)
    filtered_links = [link for link in links if get_url_level(link) <= level]

    return [article for link in filtered_links if (article := scrape_article(link))]


def scrape_from_sitemap(sitemap_url: str) -> list:
    """Scrape articles from a sitemap URL."""
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = xET.fromstring(response.content)

        return [article for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if (article := scrape_article(url.text))]
    except requests.RequestException as e:
        logging.error(f"Error fetching sitemap: {e}")
        return []

#
# End of Scraping Functions
#######################################################
#
# Sitemap/Crawling-related Functions


def collect_internal_links(base_url: str) -> set:
    """
    Crawl a website and collect all internal links.
    
    This function performs a breadth-first crawl of a website,
    discovering all internal links within the same domain.
    
    Args:
        base_url (str): The starting URL for crawling
        
    Returns:
        set: Set of discovered internal URLs
        
    Note:
        - Only follows links within the same domain
        - Handles relative URLs correctly
        - Avoids infinite loops with visited tracking
    """
    if not BS4_AVAILABLE:
        logging.error("BeautifulSoup not available for link collection. Install with: pip install tldw_chatbook[websearch]")
        return set()
        
    visited = set()
    to_visit = {base_url}

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue

        try:
            response = requests.get(current_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Collect internal links
            for link in soup.find_all('a', href=True):
                full_url = urljoin(base_url, link['href'])
                # Only process links within the same domain
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    if full_url not in visited:
                        to_visit.add(full_url)

            visited.add(current_url)
        except requests.RequestException as e:
            logging.error(f"Error visiting {current_url}: {e}")
            continue

    return visited


def generate_temp_sitemap_from_links(links: set) -> str:
    """
    Generate a temporary sitemap file from collected links and return its path.

    :param links: A set of URLs to include in the sitemap
    :return: Path to the temporary sitemap file
    """
    # Create the root element
    urlset = xET.Element("urlset")
    urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    # Add each link to the sitemap
    for link in links:
        url = xET.SubElement(urlset, "url")
        loc = xET.SubElement(url, "loc")
        loc.text = link
        lastmod = xET.SubElement(url, "lastmod")
        lastmod.text = datetime.now().strftime("%Y-%m-%d")
        changefreq = xET.SubElement(url, "changefreq")
        changefreq.text = "daily"
        priority = xET.SubElement(url, "priority")
        priority.text = "0.5"

    # Create the tree and get it as a string
    xml_string = xET.tostring(urlset, 'utf-8')

    # Pretty print the XML
    pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")

    # Create a secure temporary file
    temp_manager = get_temp_manager()
    temp_file_path = temp_manager.create_temp_file(pretty_xml, suffix=".xml", prefix="sitemap_")
    
    logging.info(f"Temporary sitemap created at: {temp_file_path}")
    return temp_file_path


def generate_sitemap_for_url(url: str) -> List[Dict[str, str]]:
    """
    Generate a sitemap for the given URL using the create_filtered_sitemap function.

    Args:
        url (str): The base URL to generate the sitemap for

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'url' and 'title' keys
    """
    with secure_temp_file(suffix=".xml", prefix="filtered_sitemap_") as temp_file:
        create_filtered_sitemap(url, temp_file.name, is_content_page)
        temp_file.seek(0)
        tree = xET.parse(temp_file.name)
        root = tree.getroot()

        sitemap = []
        for url_elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
            sitemap.append({"url": loc, "title": loc.split("/")[-1] or url})  # Use the last part of the URL as a title

    return sitemap

def create_filtered_sitemap(base_url: str, output_file: str, filter_function):
    """
    Create a sitemap from internal links and filter them based on a custom function.

    :param base_url: The base URL of the website
    :param output_file: The file to save the sitemap to
    :param filter_function: A function that takes a URL and returns True if it should be included
    """
    links = collect_internal_links(base_url)
    filtered_links = set(filter(filter_function, links))

    root = xET.Element("urlset")
    root.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    for link in filtered_links:
        url = xET.SubElement(root, "url")
        loc = xET.SubElement(url, "loc")
        loc.text = link

    tree = xET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Filtered sitemap saved to {output_file}")


#
# End of Crawling Functions
#################################################################
#
# Utility Functions

def convert_to_markdown(articles: list) -> str:
    """Convert a list of article data into a single markdown document."""
    markdown = ""
    for article in articles:
        markdown += f"# {article['title']}\n\n"
        markdown += f"Author: {article['author']}\n"
        markdown += f"Date: {article['date']}\n\n"
        markdown += f"{article['content']}\n\n"
        markdown += "---\n\n"  # Separator between articles
    return markdown

def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def load_hashes(filename: str) -> Dict[str, str]:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_hashes(hashes: Dict[str, str], filename: str):
    with open(filename, 'w') as f:
        json.dump(hashes, f)

def has_page_changed(url: str, new_hash: str, stored_hashes: Dict[str, str]) -> bool:
    old_hash = stored_hashes.get(url)
    return old_hash != new_hash


#
#
###################################################
#
# Bookmark Parsing Functions

def parse_chromium_bookmarks(json_data: dict) -> Dict[str, Union[str, List[str]]]:
    """
    Parse Chromium-based browser bookmarks from JSON data.

    :param json_data: The JSON data from the bookmarks file
    :return: A dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    """
    bookmarks = {}

    def recurse_bookmarks(nodes):
        for node in nodes:
            if node.get('type') == 'url':
                name = node.get('name')
                url = node.get('url')
                if name and url:
                    if name in bookmarks:
                        if isinstance(bookmarks[name], list):
                            bookmarks[name].append(url)
                        else:
                            bookmarks[name] = [bookmarks[name], url]
                    else:
                        bookmarks[name] = url
            elif node.get('type') == 'folder' and 'children' in node:
                recurse_bookmarks(node['children'])

    # Chromium bookmarks have a 'roots' key
    if 'roots' in json_data:
        for root in json_data['roots'].values():
            if 'children' in root:
                recurse_bookmarks(root['children'])
    else:
        recurse_bookmarks(json_data.get('children', []))

    return bookmarks


def parse_firefox_bookmarks(html_content: str) -> Dict[str, Union[str, List[str]]]:
    """
    Parse Firefox bookmarks from HTML content.

    :param html_content: The HTML content from the bookmarks file
    :return: A dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    """
    if not BS4_AVAILABLE:
        logging.error("BeautifulSoup not available for parsing bookmarks. Install with: pip install tldw_chatbook[websearch]")
        return {}
        
    bookmarks = {}
    soup = BeautifulSoup(html_content, 'html.parser')

    # Firefox stores bookmarks within <a> tags inside <dt>
    for a in soup.find_all('a'):
        name = a.get_text()
        url = a.get('href')
        if name and url:
            if name in bookmarks:
                if isinstance(bookmarks[name], list):
                    bookmarks[name].append(url)
                else:
                    bookmarks[name] = [bookmarks[name], url]
            else:
                bookmarks[name] = url

    return bookmarks


def load_bookmarks(file_path: str) -> Dict[str, Union[str, List[str]]]:
    """
    Load bookmarks from a file (JSON for Chrome/Edge or HTML for Firefox).

    :param file_path: Path to the bookmarks file
    :return: A dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    :raises ValueError: If the file format is unsupported or parsing fails
    """
    if not os.path.isfile(file_path):
        logging.error(f"File '{file_path}' does not exist.")
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.json' or ext == '':
        # Attempt to parse as JSON (Chrome/Edge)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            return parse_chromium_bookmarks(json_data)
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON. Ensure the file is a valid Chromium bookmarks JSON file.")
            raise ValueError("Invalid JSON format for Chromium bookmarks.")
    elif ext in ['.html', '.htm']:
        # Parse as HTML (Firefox)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return parse_firefox_bookmarks(html_content)
        except Exception as e:
            logging.error(f"Failed to parse HTML bookmarks: {e}")
            raise ValueError(f"Failed to parse HTML bookmarks: {e}")
    else:
        logging.error("Unsupported file format. Please provide a JSON (Chrome/Edge) or HTML (Firefox) bookmarks file.")
        raise ValueError("Unsupported file format for bookmarks.")


def collect_bookmarks(file_path: str) -> Dict[str, Union[str, List[str]]]:
    """
    Collect bookmarks from the provided bookmarks file and return a dictionary.

    :param file_path: Path to the bookmarks file
    :return: Dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    """
    try:
        bookmarks = load_bookmarks(file_path)
        logging.info(f"Successfully loaded {len(bookmarks)} bookmarks from '{file_path}'.")
        return bookmarks
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error loading bookmarks: {e}")
        return {}


def parse_csv_urls(file_path: str) -> Dict[str, Union[str, List[str]]]:
    """
    Parse URLs from a CSV file. The CSV should have at minimum a 'url' column,
    and optionally a 'title' or 'name' column.

    :param file_path: Path to the CSV file
    :return: Dictionary with titles/names as keys and URLs as values
    """
    if not PANDAS_AVAILABLE:
        logging.error("Pandas not available for CSV parsing. Install with: pip install tldw_chatbook[websearch]")
        return {}
        
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Check if required columns exist
        if 'url' not in df.columns:
            raise ValueError("CSV must contain a 'url' column")

        # Initialize result dictionary
        urls_dict = {}

        # Determine which column to use as key
        key_column = next((col for col in ['title', 'name'] if col in df.columns), None)

        for idx in range(len(df)):
            url = df.iloc[idx]['url'].strip()

            # Use title/name if available, otherwise use URL as key
            if key_column:
                key = df.iloc[idx][key_column].strip()
            else:
                key = f"Article {idx + 1}"

            # Handle duplicate keys
            if key in urls_dict:
                if isinstance(urls_dict[key], list):
                    urls_dict[key].append(url)
                else:
                    urls_dict[key] = [urls_dict[key], url]
            else:
                urls_dict[key] = url

        return urls_dict

    except Exception as e:
        if PANDAS_AVAILABLE and hasattr(pd, 'errors') and isinstance(e, pd.errors.EmptyDataError):
            logging.error("The CSV file is empty")
        else:
            logging.error(f"Error parsing CSV file: {str(e)}")
        return {}


def collect_urls_from_file(file_path: str) -> Dict[str, Union[str, List[str]]]:
    """
    Load URLs from bookmark files or CSV.
    
    Supports multiple file formats:
    - Chrome/Edge bookmarks (JSON)
    - Firefox bookmarks (HTML)
    - CSV files with 'url' column
    
    Args:
        file_path (str): Path to the bookmarks or CSV file
        
    Returns:
        Dict[str, Union[str, List[str]]]: Dictionary mapping names to URLs
            If duplicate names exist, value will be a list of URLs
            
    Supported Formats:
        - .json: Chrome/Edge bookmarks
        - .html/.htm: Firefox bookmarks  
        - .csv: Must have 'url' column, optionally 'title' or 'name'
        
    Example:
        >>> bookmarks = collect_urls_from_file("/path/to/Bookmarks")
        >>> for name, url in bookmarks.items():
        ...     print(f"{name}: {url}")

    Unified function to collect URLs from either bookmarks or CSV files.

    :param file_path: Path to the file (bookmarks or CSV)
    :return: Dictionary with names as keys and URLs as values
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.csv':
        return parse_csv_urls(file_path)
    else:
        return collect_bookmarks(file_path)

# Usage:
# from Article_Extractor_Lib import collect_bookmarks
#
# # Path to your bookmarks file
# # For Chrome or Edge (JSON format)
# chromium_bookmarks_path = "/path/to/Bookmarks"
#
# # For Firefox (HTML format)
# firefox_bookmarks_path = "/path/to/bookmarks.html"
#
# # Collect bookmarks from Chromium-based browser
# chromium_bookmarks = collect_bookmarks(chromium_bookmarks_path)
# print("Chromium Bookmarks:")
# for name, url in chromium_bookmarks.items():
#     print(f"{name}: {url}")
#
# # Collect bookmarks from Firefox
# firefox_bookmarks = collect_bookmarks(firefox_bookmarks_path)
# print("\nFirefox Bookmarks:")
# for name, url in firefox_bookmarks.items():
#     print(f"{name}: {url}")

#
# End of Bookmarking Parsing Functions
#####################################################################


#####################################################################
#
# Article Scraping Metadata Functions

class ContentMetadataHandler:
    """
    Handles metadata for scraped content.
    
    This class provides utilities for:
    - Adding metadata headers to content
    - Extracting metadata from content
    - Content deduplication via hashing
    - Change detection between versions
    
    Metadata includes:
    - Source URL
    - Ingestion date
    - Content hash
    - Scraping pipeline used
    - Custom metadata fields
    
    The metadata is stored in a structured format at the beginning
    of the content, making it easy to track content provenance.
    
    Example:
        >>> handler = ContentMetadataHandler()
        >>> content_with_meta = handler.format_content_with_metadata(
        ...     url="https://example.com",
        ...     content="Article text here...",
        ...     pipeline="playwright-trafilatura"
        ... )
    """

    METADATA_START = "[METADATA]"
    METADATA_END = "[/METADATA]"

    @staticmethod
    def format_content_with_metadata(
            url: str,
            content: str,
            pipeline: str = "Trafilatura",
            additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format content with metadata header.

        Args:
            url: The source URL
            content: The scraped content
            pipeline: The scraping pipeline used
            additional_metadata: Optional dictionary of additional metadata to include

        Returns:
            Formatted content with metadata header
        """
        metadata = {
            "url": url,
            "ingestion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content_hash": hashlib.sha256(content.encode('utf-8')).hexdigest(),
            "scraping_pipeline": pipeline
        }

        # Add any additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        formatted_content = f"""{ContentMetadataHandler.METADATA_START}
        {json.dumps(metadata, indent=2)}
        {ContentMetadataHandler.METADATA_END}
        
        {content}"""

        return formatted_content

    @staticmethod
    def extract_metadata(content: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract metadata and content separately.

        Args:
            content: The full content including metadata

        Returns:
            Tuple of (metadata dict, clean content)
        """
        try:
            metadata_start = content.index(ContentMetadataHandler.METADATA_START) + len(
                ContentMetadataHandler.METADATA_START)
            metadata_end = content.index(ContentMetadataHandler.METADATA_END)
            metadata_json = content[metadata_start:metadata_end].strip()
            metadata = json.loads(metadata_json)
            clean_content = content[metadata_end + len(ContentMetadataHandler.METADATA_END):].strip()
            return metadata, clean_content
        except (ValueError, json.JSONDecodeError) as e:
            return {}, content

    @staticmethod
    def has_metadata(content: str) -> bool:
        """
        Check if content contains metadata.

        Args:
            content: The content to check

        Returns:
            bool: True if metadata is present
        """
        return (ContentMetadataHandler.METADATA_START in content and
                ContentMetadataHandler.METADATA_END in content)

    @staticmethod
    def strip_metadata(content: str) -> str:
        """
        Remove metadata from content if present.

        Args:
            content: The content to strip metadata from

        Returns:
            Content without metadata
        """
        try:
            metadata_end = content.index(ContentMetadataHandler.METADATA_END)
            return content[metadata_end + len(ContentMetadataHandler.METADATA_END):].strip()
        except ValueError:
            return content

    @staticmethod
    def get_content_hash(content: str) -> str:
        """
        Get hash of content without metadata.

        Args:
            content: The content to hash

        Returns:
            SHA-256 hash of the clean content
        """
        clean_content = ContentMetadataHandler.strip_metadata(content)
        return hashlib.sha256(clean_content.encode('utf-8')).hexdigest()

    @staticmethod
    def content_changed(old_content: str, new_content: str) -> bool:
        """
        Check if content has changed by comparing hashes.

        Args:
            old_content: Previous version of content
            new_content: New version of content

        Returns:
            bool: True if content has changed
        """
        old_hash = ContentMetadataHandler.get_content_hash(old_content)
        new_hash = ContentMetadataHandler.get_content_hash(new_content)
        return old_hash != new_hash


##############################################################
#
# Scraping Functions

def get_url_depth(url: str) -> int:
    return len(urlparse(url).path.strip('/').split('/'))

def sync_recursive_scrape(url_input, max_pages, max_depth, delay=1.0, custom_cookies=None):
    """
    Synchronous wrapper for recursive_scrape function.
    
    Uses proper event loop handling to avoid conflicts.
    """
    return run_async_function(
        recursive_scrape,
        url_input, 
        max_pages, 
        max_depth, 
        delay=delay, 
        custom_cookies=custom_cookies
    )

async def recursive_scrape(
        base_url: str,
        max_pages: int,
        max_depth: int,
        delay: float = 1.0,
        resume_file: str = 'scrape_progress.json',
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        custom_cookies: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[callable] = None
) -> List[Dict]:
    async def save_progress():
        temp_file = resume_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump({
                'visited': list(visited),
                'to_visit': to_visit,
                'scraped_articles': scraped_articles,
                'pages_scraped': pages_scraped
            }, f)
        os.replace(temp_file, resume_file)  # Atomic replace

    def is_valid_url(url: str) -> bool:
        return url.startswith("http") and len(url) > 0

    # Load progress if resume file exists
    if os.path.exists(resume_file):
        with open(resume_file, 'r') as f:
            progress_data = json.load(f)
            visited = set(progress_data['visited'])
            to_visit = progress_data['to_visit']
            scraped_articles = progress_data['scraped_articles']
            pages_scraped = progress_data['pages_scraped']
    else:
        visited = set()
        to_visit = [(base_url, 0)]  # (url, depth)
        scraped_articles = []
        pages_scraped = 0

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=user_agent)

            # Set custom cookies if provided
            if custom_cookies:
                await context.add_cookies(custom_cookies)

            try:
                while to_visit and pages_scraped < max_pages:
                    current_url, current_depth = to_visit.pop(0)

                    if current_url in visited or current_depth > max_depth:
                        continue

                    visited.add(current_url)

                    # Update progress if callback provided
                    if progress_callback:
                        progress_callback(f"Scraping page {pages_scraped + 1}/{max_pages}: {current_url}")

                    try:
                        await asyncio.sleep(random.uniform(delay * 0.8, delay * 1.2))

                        article_data = await scrape_article_async(context, current_url)

                        if article_data and article_data['extraction_successful']:
                            scraped_articles.append(article_data)
                            pages_scraped += 1

                        # If we haven't reached max depth, add child links to to_visit
                        if current_depth < max_depth:
                            page = await context.new_page()
                            await page.goto(current_url)
                            await page.wait_for_load_state("networkidle")

                            links = await page.eval_on_selector_all('a[href]',
                                                                    "(elements) => elements.map(el => el.href)")
                            for link in links:
                                child_url = urljoin(base_url, link)
                                if is_valid_url(child_url) and child_url.startswith(
                                        base_url) and child_url not in visited and should_scrape_url(child_url):
                                    to_visit.append((child_url, current_depth + 1))

                            await page.close()

                    except Exception as e:
                        logging.error(f"Error scraping {current_url}: {str(e)}")

                    # Save progress periodically (e.g., every 10 pages)
                    if pages_scraped % 10 == 0:
                        await save_progress()

            finally:
                await browser.close()

    finally:
        # These statements are now guaranteed to be reached after the scraping is done
        await save_progress()

        # Remove the progress file when scraping is completed successfully
        if os.path.exists(resume_file):
            os.remove(resume_file)

        # Final progress update
        if progress_callback:
            progress_callback(f"Scraping completed. Total pages scraped: {pages_scraped}")

        return scraped_articles

async def scrape_article_async(context, url: str) -> Dict[str, Any]:
    page = await context.new_page()
    try:
        await page.goto(url)
        await page.wait_for_load_state("networkidle")

        title = await page.title()
        content = await page.content()

        return {
            'url': url,
            'title': title,
            'content': content,
            'extraction_successful': True
        }
    except Exception as e:
        logging.error(f"Error scraping article {url}: {str(e)}")
        return {
            'url': url,
            'extraction_successful': False,
            'error': str(e)
        }
    finally:
        await page.close()

def scrape_article_sync(url: str, custom_cookies: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Synchronous version of scrape_article using the async implementation.
    
    This ensures consistency between sync and async versions and properly
    handles the event loop.
    
    Args:
        url: The URL to scrape
        custom_cookies: Optional browser cookies for authentication
        
    Returns:
        Article data dictionary with the same structure as async version
    """
    return run_async_function(scrape_article, url, custom_cookies=custom_cookies)

def should_scrape_url(url: str) -> bool:
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()

    # List of patterns to exclude
    exclude_patterns = [
        '/tag/', '/category/', '/author/', '/search/', '/page/',
        'wp-content', 'wp-includes', 'wp-json', 'wp-admin',
        'login', 'register', 'cart', 'checkout', 'account',
        '.jpg', '.png', '.gif', '.pdf', '.zip'
    ]

    # Check if the URL contains any exclude patterns
    if any(pattern in path for pattern in exclude_patterns):
        return False

    # Add more sophisticated checks here
    # For example, you might want to only include URLs with certain patterns
    include_patterns = ['/article/', '/post/', '/blog/']
    if any(pattern in path for pattern in include_patterns):
        return True

    # By default, return True if no exclusion or inclusion rules matched
    return True

async def scrape_with_retry(url: str, max_retries: int = 3, retry_delay: float = 5.0):
    for attempt in range(max_retries):
        try:
            return await scrape_article(url)
        except TimeoutError:
            if attempt < max_retries - 1:
                logging.warning(f"Timeout error scraping {url}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logging.error(f"Failed to scrape {url} after {max_retries} attempts.")
                return None
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return None

def convert_json_to_markdown(json_str: str) -> str:
    """
    Converts the JSON output from the scraping process into a markdown format.

    Args:
        json_str (str): JSON-formatted string containing the website collection data

    Returns:
        str: Markdown-formatted string of the website collection data
    """
    try:
        # Parse the JSON string
        data = json.loads(json_str)

        # Check if there's an error in the JSON
        if "error" in data:
            return f"# Error\n\n{data['error']}"

        # Start building the markdown string
        markdown = f"# Website Collection: {data['base_url']}\n\n"

        # Add metadata
        markdown += "## Metadata\n\n"
        markdown += f"- **Scrape Method:** {data['scrape_method']}\n"
        markdown += f"- **API Used:** {data['api_used']}\n"
        markdown += f"- **Keywords:** {data['keywords']}\n"
        if data.get('url_level') is not None:
            markdown += f"- **URL Level:** {data['url_level']}\n"
        if data.get('max_pages') is not None:
            markdown += f"- **Maximum Pages:** {data['max_pages']}\n"
        if data.get('max_depth') is not None:
            markdown += f"- **Maximum Depth:** {data['max_depth']}\n"
        markdown += f"- **Total Articles Scraped:** {data['total_articles_scraped']}\n\n"

        # Add URLs Scraped
        markdown += "## URLs Scraped\n\n"
        for url in data['urls_scraped']:
            markdown += f"- {url}\n"
        markdown += "\n"

        # Add the content
        markdown += "## Content\n\n"
        markdown += data['content']

        return markdown

    except json.JSONDecodeError:
        return "# Error\n\nInvalid JSON string provided."
    except KeyError as e:
        return f"# Error\n\nMissing key in JSON data: {str(e)}"
    except Exception as e:
        return f"# Error\n\nAn unexpected error occurred: {str(e)}"

#
# End of Scraping functions
##################################################################

#
# End of Article_Extractor_Lib.py
#######################################################################################################################
