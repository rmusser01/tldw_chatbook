"""
article_scraper/scraper.py
==========================

Core scraping functionality using Playwright browser automation.

This module provides the Scraper class which manages browser instances
for efficient web scraping with features like:
- Context manager for proper resource cleanup
- Stealth mode to avoid detection
- Cookie injection for authenticated scraping
- Automatic retries on failure
- Content extraction with Trafilatura

Classes:
--------
Scraper: Main scraping class with async context manager support

Example:
--------
    async with Scraper(config=ScraperConfig(stealth=True)) as scraper:
        result = await scraper.scrape("https://example.com")
        if result['extraction_successful']:
            print(result['content'])
"""
#
# Imports
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
#
# Third-Party Libraries
from playwright.async_api import async_playwright, Browser, BrowserContext
import trafilatura
#
# Local Imports
from .config import ScraperConfig
from .utils import convert_html_to_markdown, ContentMetadataHandler
from ...Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
#
# Functions:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Scraper:
    """
    Manages a Playwright browser instance for efficient web scraping.
    
    This class provides a high-level interface for web scraping using
    Playwright browser automation. It handles browser lifecycle management,
    page creation, and content extraction.
    
    Features:
        - Async context manager for automatic cleanup
        - Configurable retry logic
        - Stealth mode support
        - Cookie injection for authentication
        - Batch scraping of multiple URLs
        
    Attributes:
        config (ScraperConfig): Configuration for scraping behavior
        custom_cookies (List[Dict]): Optional cookies for authentication
        
    Example:
        >>> config = ScraperConfig(stealth=True, retries=3)
        >>> async with Scraper(config=config) as scraper:
        ...     result = await scraper.scrape("https://example.com")
    """

    def __init__(self, config: Optional[ScraperConfig] = None, custom_cookies: Optional[List[Dict]] = None):
        self.config = config or ScraperConfig()
        self.custom_cookies = custom_cookies
        self._playwright = None
        self._browser: Optional[Browser] = None

    async def __aenter__(self):
        """
        Initialize browser when entering async context.
        
        Starts Playwright and launches a Chromium browser instance.
        This method is called automatically when using 'async with'.
        
        Returns:
            Scraper: Self for use in context manager
            
        Raises:
            PlaywrightError: If browser fails to start
        """
        start_time = time.time()
        logging.info("Starting Playwright browser...")
        log_counter("article_scraper_browser_init", labels={"stealth": str(self.config.stealth)})
        
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            
            # Log successful initialization
            duration = time.time() - start_time
            log_histogram("article_scraper_browser_init_duration", duration, labels={"status": "success"})
            log_counter("article_scraper_browser_init_success")
            
            return self
        except Exception as e:
            # Log initialization failure
            duration = time.time() - start_time
            log_histogram("article_scraper_browser_init_duration", duration, labels={"status": "error"})
            log_counter("article_scraper_browser_init_error", labels={"error_type": type(e).__name__})
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the browser and stops Playwright when exiting the block."""
        logging.info("Closing Playwright browser...")
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _fetch_html(self, context: BrowserContext, url: str) -> str:
        """
        Fetch HTML content from a URL with retry logic.
        
        Handles page navigation, stealth mode, and automatic retries
        on failure. Waits for content to load based on configuration.
        
        Args:
            context (BrowserContext): Playwright browser context
            url (str): URL to fetch
            
        Returns:
            str: HTML content of the page, empty string on failure
            
        Note:
            - Applies stealth mode if configured
            - Retries on timeout or navigation errors
            - Returns empty string after all retries exhausted
        """
        start_time = time.time()
        
        for attempt in range(self.config.retries):
            attempt_start = time.time()
            log_counter("article_scraper_fetch_attempt", labels={
                "attempt": str(attempt + 1),
                "stealth": str(self.config.stealth)
            })
            
            try:
                page = await context.new_page()
                if self.config.stealth:
                    try:
                        from playwright_stealth import stealth_async
                        await stealth_async(page)
                    except ImportError:
                        logging.warning("playwright-stealth not installed. Running without stealth.")

                await page.goto(url, wait_until="domcontentloaded", timeout=self.config.request_timeout_ms)

                if self.config.stealth:
                    await page.wait_for_timeout(self.config.stealth_wait_ms)
                else:
                    await page.wait_for_load_state("networkidle", timeout=self.config.request_timeout_ms)

                content = await page.content()
                await page.close()
                
                # Log successful fetch
                attempt_duration = time.time() - attempt_start
                total_duration = time.time() - start_time
                log_histogram("article_scraper_fetch_duration", total_duration, labels={
                    "status": "success",
                    "attempts": str(attempt + 1)
                })
                log_counter("article_scraper_fetch_success", labels={"attempts": str(attempt + 1)})
                
                return content
            except Exception as e:
                logging.error(f"Error fetching {url} on attempt {attempt + 1}: {e}")
                log_counter("article_scraper_fetch_retry", labels={
                    "attempt": str(attempt + 1),
                    "error_type": type(e).__name__
                })
                
                if attempt >= self.config.retries - 1:
                    # Log final failure
                    total_duration = time.time() - start_time
                    log_histogram("article_scraper_fetch_duration", total_duration, labels={
                        "status": "error",
                        "attempts": str(self.config.retries)
                    })
                    log_counter("article_scraper_fetch_error", labels={
                        "error_type": type(e).__name__,
                        "final_attempt": "true"
                    })
                    return ""  # Return empty on final failure
                await asyncio.sleep(2)
        return ""

    def _extract_data(self, html: str, url: str) -> Dict[str, Any]:
        """
        Extract structured article data from HTML.
        
        Uses Trafilatura to extract content and metadata, then
        converts to clean Markdown format. Adds metadata wrapper
        for tracking content provenance.
        
        Args:
            html (str): Raw HTML content
            url (str): Source URL for metadata
            
        Returns:
            Dict[str, Any]: Extracted data containing:
                - url: Source URL
                - title: Article title
                - author: Article author
                - date: Publication date
                - content: Clean markdown content
                - content_with_meta: Content with metadata header
                - extraction_successful: Success status
                - error: Error message if failed
        """
        start_time = time.time()
        log_counter("article_scraper_extraction_attempt")
        
        if not html:
            log_counter("article_scraper_extraction_error", labels={"error_type": "empty_html"})
            return {'extraction_successful': False, 'error': 'HTML content was empty.'}

        try:
            main_content_html = trafilatura.extract(
                html,
                include_comments=self.config.include_comments,
                include_tables=self.config.include_tables,
                include_images=self.config.include_images
            )
            metadata = trafilatura.extract_metadata(html)

            if not main_content_html or not metadata:
                log_counter("article_scraper_extraction_error", labels={"error_type": "trafilatura_failure"})
                return {'extraction_successful': False, 'error': 'Trafilatura failed to extract content or metadata.'}

            # Convert the extracted HTML content to clean Markdown
            main_content_md = convert_html_to_markdown(main_content_html)

            article_data = {
                'url': url,
                'title': metadata.title or 'N/A',
                'author': metadata.author or 'N/A',
                'date': metadata.date or 'N/A',
                'content': main_content_md,
                'extraction_successful': True
            }

            # Add our own metadata wrapper
            article_data['content_with_meta'] = ContentMetadataHandler.format_content_with_metadata(
                url=url,
                content=main_content_md,
                pipeline="trafilatura-playwright",
                additional_metadata={'author': article_data['author'], 'extracted_date': article_data['date']}
            )
            
            # Log successful extraction
            duration = time.time() - start_time
            content_length = len(main_content_md) if main_content_md else 0
            log_histogram("article_scraper_extraction_duration", duration, labels={"status": "success"})
            log_histogram("article_scraper_content_length", content_length)
            log_counter("article_scraper_extraction_success", labels={
                "has_title": str(bool(metadata.title)),
                "has_author": str(bool(metadata.author)),
                "has_date": str(bool(metadata.date))
            })
            
            return article_data
        except Exception as e:
            # Log extraction error
            duration = time.time() - start_time
            log_histogram("article_scraper_extraction_duration", duration, labels={"status": "error"})
            log_counter("article_scraper_extraction_error", labels={"error_type": type(e).__name__})
            logging.error(f"Extraction error for {url}: {e}")
            return {'extraction_successful': False, 'error': f'Extraction error: {str(e)}'}

    async def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape a single article from a URL.
        
        Complete scraping pipeline: creates browser context,
        fetches HTML, extracts content, and returns structured data.
        
        Args:
            url (str): URL to scrape
            
        Returns:
            Dict[str, Any]: Article data (see _extract_data for structure)
            
        Raises:
            RuntimeError: If called outside async context manager
            
        Example:
            >>> async with Scraper() as scraper:
            ...     article = await scraper.scrape("https://example.com/article")
            ...     if article['extraction_successful']:
            ...         print(f"Title: {article['title']}")
        """
        if not self._browser:
            raise RuntimeError("Scraper must be used within an `async with` block.")

        start_time = time.time()
        logging.info(f"Scraping article from: {url}")
        log_counter("article_scraper_scrape_request", labels={
            "has_cookies": str(bool(self.custom_cookies))
        })

        try:
            context = await self._browser.new_context(
                user_agent=self.config.user_agent,
                viewport={"width": 1280, "height": 720},
            )
            if self.custom_cookies:
                await context.add_cookies(self.custom_cookies)

            html = await self._fetch_html(context, url)
            await context.close()

            result = self._extract_data(html, url)
            
            # Log scraping result
            duration = time.time() - start_time
            if result['extraction_successful']:
                logging.info(f"Successfully extracted article: '{result.get('title', 'N/A')}'")
                log_histogram("article_scraper_scrape_duration", duration, labels={"status": "success"})
                log_counter("article_scraper_scrape_success")
            else:
                logging.warning(f"Failed to extract article from {url}. Reason: {result.get('error')}")
                log_histogram("article_scraper_scrape_duration", duration, labels={"status": "error"})
                log_counter("article_scraper_scrape_error", labels={
                    "error_reason": result.get('error', 'unknown').replace(' ', '_').lower()[:50]
                })

            return result
        except Exception as e:
            # Log scraping error
            duration = time.time() - start_time
            log_histogram("article_scraper_scrape_duration", duration, labels={"status": "error"})
            log_counter("article_scraper_scrape_error", labels={"error_type": type(e).__name__})
            logging.error(f"Scraping error for {url}: {e}")
            return {'extraction_successful': False, 'error': f'Scraping error: {str(e)}'}

    async def scrape_many(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs concurrently.
        
        Processes all URLs in parallel for maximum performance.
        Each URL is scraped independently, and failures don't
        affect other URLs.
        
        Args:
            urls (List[str]): List of URLs to scrape
            
        Returns:
            List[Dict[str, Any]]: List of article data in same order as input
            
        Note:
            - Results maintain the same order as input URLs
            - Failed extractions are included with error details
            - All URLs are processed even if some fail
            
        Example:
            >>> urls = ["https://example.com/1", "https://example.com/2"]
            >>> async with Scraper() as scraper:
            ...     results = await scraper.scrape_many(urls)
            ...     for result in results:
            ...         if result['extraction_successful']:
            ...             print(result['title'])
        """
        start_time = time.time()
        url_count = len(urls)
        log_counter("article_scraper_batch_request", labels={"url_count": str(url_count)})
        
        tasks = [self.scrape(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        # Log batch results
        duration = time.time() - start_time
        success_count = sum(1 for r in results if r.get('extraction_successful', False))
        failure_count = url_count - success_count
        
        log_histogram("article_scraper_batch_duration", duration, labels={
            "url_count": str(url_count),
            "success_count": str(success_count)
        })
        log_counter("article_scraper_batch_complete", labels={
            "total_urls": str(url_count),
            "success_count": str(success_count),
            "failure_count": str(failure_count)
        })
        
        return results

#
# End of article_scraper/scraper.py
#######################################################################################################################
