# monitoring_engine.py
# Description: Core monitoring engine for RSS/Atom feeds and URL change detection
#
# This module provides secure feed parsing and URL monitoring capabilities with:
# - XXE protection for XML parsing
# - Change detection for URLs
# - Rate limiting
# - Circuit breaker pattern
# - Content extraction
#
# Imports
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
import re
from difflib import SequenceMatcher
#
# Third-Party Imports
import httpx
from loguru import logger

try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
    logger.warning("defusedxml not available, using standard xml.etree. Install defusedxml for better security.")
from bs4 import BeautifulSoup
from loguru import logger
#
# Local Imports
from ..DB.Subscriptions_DB import SubscriptionsDB, RateLimitError, AuthenticationError
from ..Metrics.metrics_logger import log_histogram, log_counter
from .security import SecurityValidator, SSRFProtector
#
########################################################################################################################
#
# Core Classes
#
########################################################################################################################

class RateLimiter:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, tokens_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            tokens_per_minute: Maximum requests per minute
        """
        self.rate = tokens_per_minute / 60.0  # Tokens per second
        self.max_tokens = tokens_per_minute
        self.tokens = float(self.max_tokens)
        self.last_update = time.time()
        self.domain_buckets = {}  # Per-domain rate limiting
        
    async def acquire_token(self, domain: str) -> bool:
        """
        Try to acquire a token for the given domain.
        
        Args:
            domain: The domain to rate limit
            
        Returns:
            True if token acquired, False if rate limited
        """
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        
        # Refill tokens
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
        
        # Check if we have tokens
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        
        return False
    
    def get_retry_after(self) -> float:
        """Get seconds until a token will be available."""
        if self.tokens >= 1.0:
            return 0.0
        tokens_needed = 1.0 - self.tokens
        return tokens_needed / self.rate


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "closed"
        
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_attempt(self) -> bool:
        """Check if we can attempt an operation."""
        if self.state == "closed":
            return True
            
        if self.state == "open":
            # Check if we should try recovery
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker entering half-open state")
                    return True
            return False
            
        # half_open state
        return True


class ContentExtractor:
    """Extract and process content from various sources."""
    
    @staticmethod
    def extract_text_from_html(html: str, ignore_selectors: List[str] = None) -> str:
        """
        Extract clean text from HTML.
        
        Args:
            html: HTML content
            ignore_selectors: CSS selectors to ignore
            
        Returns:
            Extracted text
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Remove elements matching ignore selectors
        if ignore_selectors:
            for selector in ignore_selectors:
                for element in soup.select(selector):
                    element.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    @staticmethod
    def calculate_content_hash(content: str) -> str:
        """Calculate SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def calculate_change_percentage(old_content: str, new_content: str) -> float:
        """
        Calculate percentage of change between two texts.
        
        Args:
            old_content: Previous content
            new_content: New content
            
        Returns:
            Change percentage (0.0 to 1.0)
        """
        if not old_content and not new_content:
            return 0.0
        if not old_content or not new_content:
            return 1.0
            
        matcher = SequenceMatcher(None, old_content, new_content)
        similarity = matcher.ratio()
        return 1.0 - similarity


class FeedMonitor:
    """Monitor RSS/Atom feeds with security and performance features."""
    
    def __init__(self, rate_limiter: RateLimiter = None, security_validator: SecurityValidator = None):
        """
        Initialize feed monitor.
        
        Args:
            rate_limiter: Rate limiter instance
            security_validator: Security validator instance
        """
        self.rate_limiter = rate_limiter or RateLimiter()
        self.security_validator = security_validator
        self.circuit_breakers = {}  # Per-subscription circuit breakers
        
    async def check_feed(self, subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check a feed for new items.
        
        Args:
            subscription: Subscription dictionary from database
            
        Returns:
            List of new/updated items
        """
        start_time = time.time()
        subscription_id = subscription['id']
        feed_url = subscription['source']
        
        # Check circuit breaker
        if subscription_id not in self.circuit_breakers:
            self.circuit_breakers[subscription_id] = CircuitBreaker()
        
        breaker = self.circuit_breakers[subscription_id]
        if not breaker.can_attempt():
            raise RateLimitError(f"Circuit breaker open for subscription {subscription_id}")
        
        try:
            # Parse URL for rate limiting
            parsed = urlparse(feed_url)
            domain = parsed.netloc
            
            # Check rate limit
            if not await self.rate_limiter.acquire_token(domain):
                retry_after = self.rate_limiter.get_retry_after()
                raise RateLimitError(f"Rate limited. Retry after {retry_after:.1f} seconds")
            
            # Fetch feed
            items = await self._fetch_and_parse_feed(subscription)
            
            # Record success
            breaker.record_success()
            
            # Log metrics
            duration = time.time() - start_time
            log_histogram("subscription_check_duration", duration, labels={
                "type": subscription['type'],
                "status": "success"
            })
            log_counter("subscription_checks", labels={
                "type": subscription['type'],
                "status": "success"
            })
            
            return items
            
        except Exception as e:
            # Record failure
            breaker.record_failure()
            
            # Log metrics
            duration = time.time() - start_time
            log_histogram("subscription_check_duration", duration, labels={
                "type": subscription['type'],
                "status": "error"
            })
            log_counter("subscription_checks", labels={
                "type": subscription['type'],
                "status": "error",
                "error_type": type(e).__name__
            })
            
            raise
    
    async def _fetch_and_parse_feed(self, subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch and parse a feed.
        
        Args:
            subscription: Subscription dictionary
            
        Returns:
            List of feed items
        """
        feed_url = subscription['source']
        
        # Build headers
        headers = {
            'User-Agent': 'tldw-chatbook/1.0 (+https://github.com/tldw/chatbook)',
            'Accept': 'application/rss+xml, application/atom+xml, application/xml, text/xml',
            'Accept-Encoding': 'gzip, deflate'
        }
        
        # Add ETag/Last-Modified if available
        if subscription.get('etag'):
            headers['If-None-Match'] = subscription['etag']
        if subscription.get('last_modified'):
            headers['If-Modified-Since'] = subscription['last_modified']
        
        # Add custom headers
        if subscription.get('custom_headers'):
            try:
                custom = json.loads(subscription['custom_headers'])
                headers.update(custom)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Add authentication if configured
        auth = None
        if subscription.get('auth_config'):
            try:
                auth_config = json.loads(subscription['auth_config'])
                auth_type = auth_config.get('type')
                
                if auth_type == 'basic':
                    auth = httpx.BasicAuth(
                        auth_config.get('username', ''),
                        auth_config.get('password', '')
                    )
                elif auth_type == 'bearer':
                    headers['Authorization'] = f"Bearer {auth_config.get('token', '')}"
                elif auth_type == 'api_key':
                    key_header = auth_config.get('header', 'X-API-Key')
                    headers[key_header] = auth_config.get('key', '')
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid auth config for subscription {subscription['id']}")
        
        # Fetch feed
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            verify=subscription.get('ssl_verify', True) != 0
        ) as client:
            response = await client.get(feed_url, headers=headers, auth=auth)
            
        # Handle response
        if response.status_code == 304:
            # Not modified
            logger.info(f"Feed not modified: {feed_url}")
            return []
            
        if response.status_code == 401:
            raise AuthenticationError("Authentication failed")
            
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After', '60')
            raise RateLimitError(f"Rate limited by server. Retry after {retry_after} seconds")
            
        response.raise_for_status()
        
        # Parse feed based on type
        content_type = response.headers.get('content-type', '').lower()
        
        if 'json' in content_type or subscription['type'] == 'json_feed':
            return self._parse_json_feed(response.text)
        else:
            return self._parse_xml_feed(response.text, subscription['type'])
    
    def _parse_xml_feed(self, content: str, feed_type: str) -> List[Dict[str, Any]]:
        """
        Parse RSS/Atom XML feed with XXE protection.
        
        Args:
            content: Feed XML content
            feed_type: Type of feed (rss, atom)
            
        Returns:
            List of parsed items
        """
        try:
            # Parse XML (with defusedxml if available for XXE protection)
            root = ET.fromstring(content)
            
            items = []
            
            if feed_type == 'atom' or root.tag.endswith('feed'):
                # Atom feed
                entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                for entry in entries:
                    item = self._parse_atom_entry(entry)
                    if item:
                        items.append(item)
            else:
                # RSS feed
                channel = root.find('.//channel')
                if channel is not None:
                    for item_elem in channel.findall('item'):
                        item = self._parse_rss_item(item_elem)
                        if item:
                            items.append(item)
            
            return items
            
        except (ET.ParseError, Exception) as e:
            logger.error(f"XML parse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing feed: {e}")
            raise
    
    def _parse_rss_item(self, item_elem) -> Optional[Dict[str, Any]]:
        """Parse a single RSS item."""
        try:
            item = {
                'title': self._get_text(item_elem, 'title'),
                'url': self._get_text(item_elem, 'link'),
                'content': self._get_text(item_elem, 'description'),
                'author': self._get_text(item_elem, 'author') or self._get_text(item_elem, 'dc:creator'),
                'published_date': self._parse_date(self._get_text(item_elem, 'pubDate')),
                'categories': [cat.text for cat in item_elem.findall('category') if cat.text],
                'enclosures': []
            }
            
            # Get enclosures
            for enclosure in item_elem.findall('enclosure'):
                enc = {
                    'url': enclosure.get('url'),
                    'type': enclosure.get('type'),
                    'length': enclosure.get('length')
                }
                if enc['url']:
                    item['enclosures'].append(enc)
            
            # Calculate content hash
            content_for_hash = f"{item['title']}{item['content']}"
            item['content_hash'] = ContentExtractor.calculate_content_hash(content_for_hash)
            
            return item
            
        except Exception as e:
            logger.error(f"Error parsing RSS item: {e}")
            return None
    
    def _parse_atom_entry(self, entry) -> Optional[Dict[str, Any]]:
        """Parse a single Atom entry."""
        try:
            # Define Atom namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            item = {
                'title': self._get_text(entry, 'atom:title', ns),
                'url': None,
                'content': self._get_text(entry, 'atom:content', ns) or self._get_text(entry, 'atom:summary', ns),
                'author': None,
                'published_date': self._parse_date(self._get_text(entry, 'atom:published', ns)),
                'categories': [],
                'enclosures': []
            }
            
            # Get link
            link = entry.find('atom:link[@rel="alternate"]', ns)
            if link is None:
                link = entry.find('atom:link', ns)
            if link is not None:
                item['url'] = link.get('href')
            
            # Get author
            author = entry.find('atom:author', ns)
            if author is not None:
                item['author'] = self._get_text(author, 'atom:name', ns)
            
            # Get categories
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term')
                if term:
                    item['categories'].append(term)
            
            # Calculate content hash
            content_for_hash = f"{item['title']}{item['content']}"
            item['content_hash'] = ContentExtractor.calculate_content_hash(content_for_hash)
            
            return item
            
        except Exception as e:
            logger.error(f"Error parsing Atom entry: {e}")
            return None
    
    def _parse_json_feed(self, content: str) -> List[Dict[str, Any]]:
        """Parse JSON Feed format."""
        try:
            feed = json.loads(content)
            items = []
            
            for feed_item in feed.get('items', []):
                item = {
                    'title': feed_item.get('title', 'Untitled'),
                    'url': feed_item.get('url') or feed_item.get('external_url'),
                    'content': feed_item.get('content_html') or feed_item.get('content_text', ''),
                    'author': None,
                    'published_date': self._parse_date(feed_item.get('date_published')),
                    'categories': feed_item.get('tags', []),
                    'enclosures': []
                }
                
                # Get author
                if 'author' in feed_item:
                    item['author'] = feed_item['author'].get('name')
                elif 'authors' in feed_item and feed_item['authors']:
                    item['author'] = feed_item['authors'][0].get('name')
                
                # Get attachments
                for attachment in feed_item.get('attachments', []):
                    enc = {
                        'url': attachment.get('url'),
                        'type': attachment.get('mime_type'),
                        'length': attachment.get('size_in_bytes')
                    }
                    if enc['url']:
                        item['enclosures'].append(enc)
                
                # Calculate content hash
                content_for_hash = f"{item['title']}{item['content']}"
                item['content_hash'] = ContentExtractor.calculate_content_hash(content_for_hash)
                
                items.append(item)
            
            return items
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing JSON feed: {e}")
            raise
    
    def _get_text(self, elem, tag: str, namespaces: Dict[str, str] = None) -> Optional[str]:
        """Safely get text from XML element."""
        if elem is None:
            return None
        
        child = elem.find(tag, namespaces)
        if child is not None and child.text:
            return child.text.strip()
        return None
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse various date formats to ISO format."""
        if not date_str:
            return None
            
        # Common date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',  # RFC 822
            '%a, %d %b %Y %H:%M:%S %Z',  # RFC 822 with timezone name
            '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
            '%Y-%m-%dT%H:%M:%SZ',        # ISO 8601 UTC
            '%Y-%m-%d %H:%M:%S',         # Simple format
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.isoformat()
            except ValueError:
                continue
        
        # If no format matches, return the original string
        logger.warning(f"Could not parse date: {date_str}")
        return date_str


class URLMonitor:
    """Monitor URLs for changes."""
    
    def __init__(self, db: SubscriptionsDB, rate_limiter: RateLimiter = None):
        """
        Initialize URL monitor.
        
        Args:
            db: Subscriptions database instance
            rate_limiter: Rate limiter instance
        """
        self.db = db
        self.rate_limiter = rate_limiter or RateLimiter()
        self.circuit_breakers = {}
        
    async def check_url(self, subscription: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check a URL for changes.
        
        Args:
            subscription: Subscription dictionary
            
        Returns:
            Change information if changed, None otherwise
        """
        subscription_id = subscription['id']
        url = subscription['source']
        
        # Check circuit breaker
        if subscription_id not in self.circuit_breakers:
            self.circuit_breakers[subscription_id] = CircuitBreaker()
        
        breaker = self.circuit_breakers[subscription_id]
        if not breaker.can_attempt():
            raise RateLimitError(f"Circuit breaker open for subscription {subscription_id}")
        
        try:
            # Fetch current content
            current_content = await self._fetch_url_content(subscription)
            
            # Get previous snapshot
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT content_hash, extracted_content
                FROM url_snapshots
                WHERE subscription_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (subscription_id,))
            
            previous = cursor.fetchone()
            
            if not previous:
                # First check - store baseline
                await self._store_snapshot(subscription_id, url, current_content)
                breaker.record_success()
                return None
            
            # Calculate change
            current_hash = ContentExtractor.calculate_content_hash(current_content['text'])
            
            if current_hash == previous['content_hash']:
                # No change
                breaker.record_success()
                return None
            
            # Calculate change details
            change_percentage = ContentExtractor.calculate_change_percentage(
                previous['extracted_content'] or '',
                current_content['text']
            )
            
            # Check if change exceeds threshold
            threshold = subscription.get('change_threshold', 0.1)
            if change_percentage < threshold:
                # Change too small
                breaker.record_success()
                return None
            
            # Significant change detected
            change_info = {
                'type': 'url_change',
                'url': url,
                'title': f"Change detected: {subscription['name']}",
                'content': current_content['text'],
                'content_hash': current_hash,
                'previous_hash': previous['content_hash'],
                'change_percentage': change_percentage,
                'change_type': 'content',
                'published_date': datetime.now(timezone.utc).isoformat()
            }
            
            # Store new snapshot
            await self._store_snapshot(subscription_id, url, current_content, current_hash)
            
            breaker.record_success()
            return change_info
            
        except Exception as e:
            breaker.record_failure()
            raise
    
    async def _fetch_url_content(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch content from a URL.
        
        Args:
            subscription: Subscription dictionary
            
        Returns:
            Dictionary with content and metadata
        """
        url = subscription['source']
        
        # Parse URL for rate limiting
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Check rate limit
        if not await self.rate_limiter.acquire_token(domain):
            retry_after = self.rate_limiter.get_retry_after()
            raise RateLimitError(f"Rate limited. Retry after {retry_after:.1f} seconds")
        
        # Build headers
        headers = {
            'User-Agent': 'tldw-chatbook/1.0 (+https://github.com/tldw/chatbook)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate'
        }
        
        # Add custom headers
        if subscription.get('custom_headers'):
            try:
                custom = json.loads(subscription['custom_headers'])
                headers.update(custom)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Fetch content
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            verify=subscription.get('ssl_verify', True) != 0
        ) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
        
        # Extract content based on extraction method
        extraction_method = subscription.get('extraction_method', 'auto')
        ignore_selectors = None
        
        if subscription.get('ignore_selectors'):
            ignore_selectors = [s.strip() for s in subscription['ignore_selectors'].split('\n') if s.strip()]
        
        if extraction_method == 'full' or extraction_method == 'auto':
            # Extract text from HTML
            text = ContentExtractor.extract_text_from_html(response.text, ignore_selectors)
        else:
            # Raw content
            text = response.text
        
        return {
            'text': text,
            'html': response.text,
            'headers': dict(response.headers),
            'status_code': response.status_code
        }
    
    async def _store_snapshot(self, subscription_id: int, url: str, 
                            content: Dict[str, Any], content_hash: str = None) -> None:
        """Store a URL snapshot."""
        if not content_hash:
            content_hash = ContentExtractor.calculate_content_hash(content['text'])
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO url_snapshots
                (subscription_id, url, content_hash, extracted_content, raw_html, headers)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                subscription_id,
                url,
                content_hash,
                content['text'],
                content['html'],
                json.dumps(content['headers'])
            ))


# End of monitoring_engine.py