# generic_scraper.py
# Description: Generic web scraping pipeline using BeautifulSoup
#
# This scraper handles general websites using CSS selectors to extract content.
# It supports customizable selectors for different page elements.
#
# Imports
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
import asyncio
#
# Third-Party Imports
import httpx
from bs4 import BeautifulSoup
from loguru import logger
#
# Local Imports
from ..web_scraping_pipelines import (
    BaseScrapingPipeline, ScrapedItem, ScrapingConfig, ContentExtractor
)
from ...Metrics.metrics_logger import log_counter
# Integration with existing web scraping module
try:
    from ...Web_Scraping.Article_Extractor_Lib import scrape_article
    from ...Web_Scraping.Article_Scraper import Scraper, ScraperConfig
    ARTICLE_EXTRACTOR_AVAILABLE = True
except ImportError:
    ARTICLE_EXTRACTOR_AVAILABLE = False
    scrape_article = None
    Scraper = None
    ScraperConfig = None
#
########################################################################################################################
#
# Generic Web Scraping Pipeline
#
########################################################################################################################

class GenericWebScrapingPipeline(BaseScrapingPipeline):
    """
    General-purpose web scraper using BeautifulSoup and CSS selectors.
    
    This scraper can extract content from most websites using configurable
    CSS selectors for different elements like title, content, author, etc.
    """
    
    DEFAULT_SELECTORS = {
        'title': 'h1, h2, title',
        'content': 'article, main, .content, #content, .post-content',
        'author': '.author, .by-line, .byline, [rel="author"]',
        'date': 'time, .date, .published, .post-date',
        'categories': '.category, .tag, .tags a'
    }
    
    def __init__(self, config: ScrapingConfig):
        """Initialize generic scraper with configuration."""
        super().__init__(config)
        
        # Merge default selectors with custom ones
        self.selectors = self.DEFAULT_SELECTORS.copy()
        if config.selectors:
            self.selectors.update(config.selectors)
        
        # Additional options
        self.extract_links = config.options.get('extract_links', True)
        self.extract_images = config.options.get('extract_images', False)
        self.follow_pagination = config.options.get('follow_pagination', False)
        self.max_pages = config.options.get('max_pages', 1)
        self.clean_content = config.options.get('clean_content', True)
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate generic scraper configuration."""
        # Check URL is valid
        try:
            parsed = urlparse(self.config.url)
            if not parsed.scheme or not parsed.netloc:
                return False, "Invalid URL format"
            
            if parsed.scheme not in ['http', 'https']:
                return False, "URL must use HTTP or HTTPS protocol"
            
        except Exception as e:
            return False, f"URL parsing error: {str(e)}"
        
        # Validate selectors
        if not isinstance(self.selectors, dict):
            return False, "Selectors must be a dictionary"
        
        # Check max_pages is reasonable
        if self.max_pages > 50:
            return False, "max_pages cannot exceed 50"
        
        return True, None
    
    async def fetch_content(self, url: str) -> str:
        """Fetch web page content."""
        logger.debug(f"Fetching content from: {url}")
        
        # Try to use existing Article_Extractor_Lib if available and JS not required
        if ARTICLE_EXTRACTOR_AVAILABLE and not self.config.javascript_enabled:
            try:
                # Use the existing scrape_article function
                article_data = await scrape_article(
                    url, 
                    custom_cookies=self.config.options.get('cookies')
                )
                
                if article_data and article_data.get('extraction_successful'):
                    log_counter("generic_scraper_fetch", labels={
                        "status": "success",
                        "method": "article_extractor",
                        "domain": self.get_domain()
                    })
                    # Return the raw HTML for further processing
                    return article_data.get('html', article_data.get('content', ''))
                
            except Exception as e:
                logger.warning(f"Article extractor failed, falling back to httpx: {str(e)}")
        
        # Fallback to httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = self.get_headers()
            
            try:
                response = await client.get(
                    url, 
                    headers=headers, 
                    follow_redirects=True
                )
                response.raise_for_status()
                
                log_counter("generic_scraper_fetch", labels={
                    "status": "success",
                    "method": "httpx",
                    "domain": self.get_domain()
                })
                
                return response.text
                
            except httpx.HTTPStatusError as e:
                log_counter("generic_scraper_fetch", labels={
                    "status": "http_error",
                    "status_code": str(e.response.status_code),
                    "domain": self.get_domain()
                })
                raise
            
            except Exception as e:
                log_counter("generic_scraper_fetch", labels={
                    "status": "error",
                    "error_type": type(e).__name__,
                    "domain": self.get_domain()
                })
                raise
    
    def parse_content(self, raw_content: str, url: str) -> List[ScrapedItem]:
        """Parse web page into structured items."""
        items = []
        
        try:
            soup = BeautifulSoup(raw_content, 'html.parser')
            
            # Check if this is a listing page or single article
            if self._is_listing_page(soup):
                items.extend(self._parse_listing_page(soup, url))
            else:
                item = self._parse_single_page(soup, url)
                if item:
                    items.append(item)
            
            # Handle pagination if enabled
            if self.follow_pagination and len(items) < self.max_pages * 10:
                next_url = self._find_next_page_url(soup, url)
                if next_url:
                    logger.debug(f"Following pagination to: {next_url}")
                    # Note: This would need to be handled by the caller
                    # to avoid infinite recursion
            
            logger.info(f"Parsed {len(items)} items from {url}")
            log_counter("generic_scraper_items", labels={
                "domain": self.get_domain(),
                "item_count": str(len(items))
            })
            
        except Exception as e:
            logger.error(f"Error parsing content: {str(e)}")
            log_counter("generic_scraper_errors", labels={
                "error_type": "parse_error",
                "domain": self.get_domain()
            })
            raise
        
        return items
    
    def _is_listing_page(self, soup: BeautifulSoup) -> bool:
        """Detect if page is a listing of multiple items."""
        # Look for common listing indicators
        listing_indicators = [
            'article', 'post', 'entry', 'item',
            'card', 'teaser', 'summary'
        ]
        
        for indicator in listing_indicators:
            # Check class names
            elements = soup.find_all(class_=lambda x: x and indicator in x.lower())
            if len(elements) > 2:  # Multiple items suggest a listing
                return True
            
            # Check tag names
            if len(soup.find_all(indicator)) > 2:
                return True
        
        return False
    
    def _parse_listing_page(self, soup: BeautifulSoup, base_url: str) -> List[ScrapedItem]:
        """Parse a listing page with multiple items."""
        items = []
        
        # Find article containers
        article_selectors = [
            'article', '.article', '.post', '.entry',
            '[class*="article"]', '[class*="post"]'
        ]
        
        articles = []
        for selector in article_selectors:
            articles = soup.select(selector)
            if articles:
                break
        
        for article in articles:
            item = self._extract_item_from_element(article, base_url)
            if item:
                items.append(item)
        
        return items
    
    def _parse_single_page(self, soup: BeautifulSoup, url: str) -> Optional[ScrapedItem]:
        """Parse a single article page."""
        return self._extract_item_from_element(soup, url)
    
    def _extract_item_from_element(self, element, base_url: str) -> Optional[ScrapedItem]:
        """Extract item data from a BeautifulSoup element."""
        try:
            # Extract title
            title = self._extract_with_selector(element, self.selectors['title'])
            if not title:
                return None
            
            # Extract content
            content = self._extract_with_selector(element, self.selectors['content'])
            if self.clean_content and content:
                content = ContentExtractor.extract_text_content(str(element))
            
            # Extract author
            author = self._extract_with_selector(element, self.selectors['author'])
            
            # Extract date
            date_str = self._extract_with_selector(element, self.selectors['date'])
            published_date = self._parse_date(date_str) if date_str else None
            
            # Extract categories/tags
            categories = self._extract_categories(element)
            
            # Extract URL (for listing pages)
            item_url = self._extract_item_url(element, base_url)
            
            # Extract additional metadata
            metadata = self._extract_metadata(element, base_url)
            
            return ScrapedItem(
                url=item_url,
                title=title,
                content=content,
                author=author,
                published_date=published_date,
                categories=categories,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error extracting item: {str(e)}")
            return None
    
    def _extract_with_selector(self, element, selector: str) -> Optional[str]:
        """Extract text using CSS selector."""
        if not selector:
            return None
        
        # Try multiple selectors if comma-separated
        selectors = [s.strip() for s in selector.split(',')]
        
        for sel in selectors:
            try:
                found = element.select_one(sel)
                if found:
                    # Try to get clean text
                    text = found.get_text(strip=True)
                    if text:
                        return text
            except Exception:
                continue
        
        return None
    
    def _extract_categories(self, element) -> List[str]:
        """Extract categories/tags from element."""
        categories = []
        
        if not self.selectors.get('categories'):
            return categories
        
        try:
            elements = element.select(self.selectors['categories'])
            for elem in elements:
                text = elem.get_text(strip=True)
                if text and text not in categories:
                    categories.append(text)
        except Exception:
            pass
        
        return categories
    
    def _extract_item_url(self, element, base_url: str) -> str:
        """Extract item URL from listing element."""
        # Look for links in common patterns
        link_selectors = [
            'h1 a', 'h2 a', 'h3 a',
            '.title a', '.headline a',
            'a[href*="article"]', 'a[href*="post"]'
        ]
        
        for selector in link_selectors:
            link = element.select_one(selector)
            if link and link.get('href'):
                return urljoin(base_url, link['href'])
        
        # Fallback to any link
        link = element.find('a', href=True)
        if link:
            return urljoin(base_url, link['href'])
        
        return base_url
    
    def _extract_metadata(self, element, base_url: str) -> Dict[str, Any]:
        """Extract additional metadata from element."""
        metadata = {
            'source': 'generic_web',
            'domain': self.get_domain()
        }
        
        # Extract meta description if available
        meta_desc = element.select_one('meta[name="description"]')
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        # Extract images if requested
        if self.extract_images:
            images = []
            for img in element.select('img[src]')[:5]:  # Limit to 5 images
                img_url = urljoin(base_url, img['src'])
                images.append({
                    'url': img_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })
            if images:
                metadata['images'] = images
        
        # Extract links if requested
        if self.extract_links:
            links = []
            for link in element.select('a[href]')[:10]:  # Limit to 10 links
                href = link.get('href', '')
                if href and not href.startswith('#'):
                    links.append({
                        'url': urljoin(base_url, href),
                        'text': link.get_text(strip=True)[:100]
                    })
            if links:
                metadata['links'] = links
        
        return metadata
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object."""
        if not date_str:
            return None
        
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y',
            '%m/%d/%Y',
            '%d/%m/%Y'
        ]
        
        for fmt in date_formats:
            try:
                # Try parsing with timezone awareness
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        
        # Try parsing ISO format
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            pass
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def _find_next_page_url(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Find URL for next page of results."""
        # Common pagination patterns
        next_selectors = [
            'a.next', 'a[rel="next"]', '.pagination .next',
            'a:contains("Next")', 'a:contains("→")',
            '.pager-next a', '.nav-next a'
        ]
        
        for selector in next_selectors:
            try:
                # BeautifulSoup doesn't support :contains, use find instead
                if ':contains' in selector:
                    text = selector.split('"')[1]
                    next_link = soup.find('a', string=lambda s: s and text in s)
                else:
                    next_link = soup.select_one(selector)
                
                if next_link and next_link.get('href'):
                    return urljoin(current_url, next_link['href'])
            except:
                continue
        
        return None


# End of generic_scraper.py