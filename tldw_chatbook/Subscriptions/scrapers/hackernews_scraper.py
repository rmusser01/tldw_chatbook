# hackernews_scraper.py
# Description: Hacker News scraping pipeline using RSS and API
#
# This scraper handles Hacker News content via:
# - RSS feeds for stories
# - Algolia API for search and comments
# - Firebase API for real-time data
#
# Imports
import re
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, quote, urlencode
#
# Third-Party Imports
import httpx
import defusedxml.ElementTree as ET
from loguru import logger
#
# Local Imports
from ..web_scraping_pipelines import BaseScrapingPipeline, ScrapedItem, ScrapingConfig
from ...Metrics.metrics_logger import log_counter
#
########################################################################################################################
#
# Hacker News Scraping Pipeline
#
########################################################################################################################

class HackerNewsScrapingPipeline(BaseScrapingPipeline):
    """
    Scraping pipeline for Hacker News content.
    
    Supports multiple HN sources:
    - Front page (top stories)
    - New stories
    - Best stories  
    - Ask HN
    - Show HN
    - Jobs
    - User submissions
    - Search results
    """
    
    HN_BASE_URL = "https://news.ycombinator.com"
    HN_RSS_URL = "https://hnrss.org/frontpage"
    HN_ALGOLIA_API = "https://hn.algolia.com/api/v1"
    HN_FIREBASE_API = "https://hacker-news.firebaseio.com/v0"
    
    def __init__(self, config: ScrapingConfig):
        """Initialize HN scraper with configuration."""
        super().__init__(config)
        
        # HN-specific options
        self.feed_type = config.options.get('feed_type', 'frontpage')  # frontpage, newest, best, ask, show, jobs
        self.points_threshold = config.options.get('points_threshold', 0)
        self.comments_threshold = config.options.get('comments_threshold', 0)
        self.include_comments = config.options.get('include_comments', False)
        self.search_query = config.options.get('search_query')
        self.search_tags = config.options.get('search_tags', [])  # story, comment, poll, job
        self.time_range = config.options.get('time_range')  # last24h, pastWeek, pastMonth, pastYear
        self.author_filter = config.options.get('author_filter', [])
        self.domain_filter = config.options.get('domain_filter', [])
        self.exclude_domains = config.options.get('exclude_domains', [])
        self.max_items = config.options.get('max_items', 30)
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate HN-specific configuration."""
        # Validate feed type
        valid_types = ['frontpage', 'newest', 'best', 'ask', 'show', 'jobs', 'user', 'search']
        if self.feed_type not in valid_types:
            return False, f"Invalid feed type. Must be one of: {', '.join(valid_types)}"
        
        # Check search query for search type
        if self.feed_type == 'search' and not self.search_query:
            return False, "Search query required for search feed type"
        
        # Extract username if user feed
        if self.feed_type == 'user':
            # URL should contain username
            url_match = re.search(r'user[=/](\w+)', self.config.url)
            if url_match:
                self.username = url_match.group(1)
            elif self.author_filter:
                self.username = self.author_filter[0]
            else:
                return False, "Username required for user feed type"
        
        return True, None
    
    def get_rate_limit(self) -> int:
        """Get rate limit for HN."""
        # HN is generally lenient with rate limits
        return 600  # 10 requests per minute
    
    async def fetch_content(self, url: str) -> str:
        """Fetch HN content."""
        if self.feed_type == 'search':
            return await self._fetch_search_results()
        elif self.feed_type == 'user':
            return await self._fetch_user_content()
        elif self.include_comments:
            return await self._fetch_with_comments()
        else:
            return await self._fetch_rss_feed()
    
    async def _fetch_rss_feed(self) -> str:
        """Fetch HN RSS feed from hnrss.org."""
        # Build RSS URL based on feed type
        feed_urls = {
            'frontpage': 'https://hnrss.org/frontpage',
            'newest': 'https://hnrss.org/newest',
            'best': 'https://hnrss.org/best',
            'ask': 'https://hnrss.org/ask',
            'show': 'https://hnrss.org/show',
            'jobs': 'https://hnrss.org/jobs'
        }
        
        base_url = feed_urls.get(self.feed_type, 'https://hnrss.org/frontpage')
        
        # Add parameters
        params = []
        if self.points_threshold > 0:
            params.append(f'points={self.points_threshold}')
        if self.comments_threshold > 0:
            params.append(f'comments={self.comments_threshold}')
        if self.max_items != 30:
            params.append(f'count={self.max_items}')
        
        if params:
            feed_url = f"{base_url}?{'&'.join(params)}"
        else:
            feed_url = base_url
        
        logger.debug(f"Fetching HN RSS from: {feed_url}")
        
        # Fetch feed
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = self.get_headers()
            headers['Accept'] = 'application/rss+xml, application/xml'
            
            response = await client.get(feed_url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            log_counter("hn_scraper_fetch", labels={
                "status": "success",
                "feed_type": self.feed_type,
                "method": "rss"
            })
            
            return response.text
    
    async def _fetch_search_results(self) -> str:
        """Fetch search results from Algolia API."""
        search_url = f"{self.HN_ALGOLIA_API}/search"
        
        # Build query parameters
        params = {
            'query': self.search_query,
            'hitsPerPage': self.max_items
        }
        
        # Add tags filter
        if self.search_tags:
            params['tags'] = ','.join(f'({tag})' for tag in self.search_tags)
        else:
            params['tags'] = 'story'  # Default to stories only
        
        # Add time range
        if self.time_range:
            time_ranges = {
                'last24h': 'created_at_i>86400',
                'pastWeek': 'created_at_i>604800',
                'pastMonth': 'created_at_i>2592000',
                'pastYear': 'created_at_i>31536000'
            }
            if self.time_range in time_ranges:
                params['numericFilters'] = time_ranges[self.time_range]
        
        logger.debug(f"Fetching HN search from: {search_url}")
        
        # Fetch results
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url, params=params, headers=self.get_headers())
            response.raise_for_status()
            
            log_counter("hn_scraper_fetch", labels={
                "status": "success",
                "feed_type": "search",
                "method": "api"
            })
            
            return response.text
    
    async def _fetch_user_content(self) -> str:
        """Fetch user submissions via API."""
        # Get user data
        user_url = f"{self.HN_FIREBASE_API}/user/{self.username}.json"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get user profile
            response = await client.get(user_url, headers=self.get_headers())
            response.raise_for_status()
            
            user_data = response.json()
            submitted_ids = user_data.get('submitted', [])[:self.max_items]
            
            # Fetch each submission
            items = []
            for item_id in submitted_ids:
                item_url = f"{self.HN_FIREBASE_API}/item/{item_id}.json"
                item_response = await client.get(item_url, headers=self.get_headers())
                
                if item_response.status_code == 200:
                    item_data = item_response.json()
                    if item_data and item_data.get('type') in ['story', 'job', 'poll']:
                        items.append(item_data)
            
            log_counter("hn_scraper_fetch", labels={
                "status": "success",
                "feed_type": "user",
                "method": "api"
            })
            
            # Convert to consistent format
            return json.dumps({'items': items})
    
    async def _fetch_with_comments(self) -> str:
        """Fetch stories with their comments."""
        # First get stories via RSS
        rss_content = await self._fetch_rss_feed()
        
        # Parse to get story IDs
        items = self._parse_rss_feed(rss_content)
        
        # Fetch comments for each story
        enriched_items = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for item in items[:10]:  # Limit to avoid too many requests
                # Extract HN ID from URL
                hn_id = self._extract_hn_id(item.url)
                if hn_id:
                    # Fetch item details including comments
                    item_url = f"{self.HN_ALGOLIA_API}/items/{hn_id}"
                    response = await client.get(item_url, headers=self.get_headers())
                    
                    if response.status_code == 200:
                        item_data = response.json()
                        item.metadata['comments_data'] = item_data.get('children', [])
                
                enriched_items.append(item)
        
        # Convert back to JSON for consistent parsing
        return json.dumps({'items': [item.__dict__ for item in enriched_items]})
    
    def parse_content(self, raw_content: str, url: str) -> List[ScrapedItem]:
        """Parse HN content into items."""
        # Check if JSON (API response) or XML (RSS)
        if raw_content.strip().startswith('{'):
            return self._parse_api_response(raw_content)
        else:
            return self._parse_rss_feed(raw_content)
    
    def _parse_rss_feed(self, raw_content: str) -> List[ScrapedItem]:
        """Parse HN RSS feed."""
        items = []
        
        try:
            # Parse RSS XML safely
            root = ET.fromstring(raw_content)
            
            # Find all items
            for item in root.findall('.//item'):
                scraped_item = self._parse_rss_item(item)
                
                if scraped_item and self._should_include_item(scraped_item):
                    items.append(scraped_item)
            
            logger.info(f"Parsed {len(items)} HN items from RSS")
            log_counter("hn_scraper_items", labels={
                "feed_type": self.feed_type,
                "item_count": str(len(items))
            })
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse HN RSS: {str(e)}")
            raise
        
        return items
    
    def _parse_rss_item(self, item_element) -> Optional[ScrapedItem]:
        """Parse a single RSS item."""
        try:
            # Extract fields
            title = self._get_text(item_element, 'title')
            link = self._get_text(item_element, 'link')
            pub_date = self._get_text(item_element, 'pubDate')
            description = self._get_text(item_element, 'description')
            comments_link = self._get_text(item_element, 'comments')
            
            if not title:
                return None
            
            # Parse date
            published_date = None
            if pub_date:
                try:
                    published_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                except ValueError:
                    logger.warning(f"Could not parse date: {pub_date}")
            
            # Extract metadata from description
            metadata = self._extract_hn_metadata(description, comments_link)
            
            # Determine actual URL (story link or HN discussion)
            if link and not link.startswith(self.HN_BASE_URL):
                # External link
                story_url = link
                metadata['hn_url'] = comments_link
            else:
                # Self post (Ask HN, etc)
                story_url = comments_link or link
                metadata['is_self_post'] = True
            
            return ScrapedItem(
                title=title,
                url=story_url,
                content=description or title,
                published_date=published_date,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing HN RSS item: {str(e)}")
            return None
    
    def _parse_api_response(self, raw_content: str) -> List[ScrapedItem]:
        """Parse HN API response."""
        items = []
        
        try:
            data = json.loads(raw_content)
            
            # Handle different response formats
            if 'hits' in data:  # Algolia search response
                items_data = data['hits']
            elif 'items' in data:  # Our custom format
                items_data = data['items']
            else:
                items_data = [data]  # Single item
            
            for item_data in items_data:
                # Handle pre-parsed ScrapedItem dicts
                if 'metadata' in item_data and 'url' in item_data:
                    scraped_item = ScrapedItem(**item_data)
                else:
                    scraped_item = self._parse_api_item(item_data)
                
                if scraped_item and self._should_include_item(scraped_item):
                    items.append(scraped_item)
            
            logger.info(f"Parsed {len(items)} HN items from API")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse HN API response: {str(e)}")
            raise
        
        return items
    
    def _parse_api_item(self, item_data: dict) -> Optional[ScrapedItem]:
        """Parse a single API item."""
        try:
            # Algolia format
            if 'objectID' in item_data:
                title = item_data.get('title', '')
                url = item_data.get('url')
                story_text = item_data.get('story_text', '')
                points = item_data.get('points', 0)
                num_comments = item_data.get('num_comments', 0)
                author = item_data.get('author', '')
                created_at = item_data.get('created_at_i')
                
                if not url:
                    url = f"{self.HN_BASE_URL}/item?id={item_data['objectID']}"
                
                metadata = {
                    'hn_id': item_data['objectID'],
                    'points': points,
                    'comments': num_comments,
                    'author': author,
                    'tags': item_data.get('_tags', [])
                }
                
                if created_at:
                    published_date = datetime.fromtimestamp(created_at, tz=timezone.utc)
                else:
                    published_date = None
            
            # Firebase format
            else:
                title = item_data.get('title', '')
                url = item_data.get('url')
                text = item_data.get('text', '')
                score = item_data.get('score', 0)
                descendants = item_data.get('descendants', 0)
                by = item_data.get('by', '')
                time_val = item_data.get('time')
                
                if not url:
                    url = f"{self.HN_BASE_URL}/item?id={item_data.get('id')}"
                
                metadata = {
                    'hn_id': str(item_data.get('id')),
                    'points': score,
                    'comments': descendants,
                    'author': by,
                    'type': item_data.get('type', 'story')
                }
                
                if time_val:
                    published_date = datetime.fromtimestamp(time_val, tz=timezone.utc)
                else:
                    published_date = None
                
                story_text = text
            
            if not title:
                return None
            
            # Extract domain from URL
            if url and not url.startswith(self.HN_BASE_URL):
                domain = urlparse(url).netloc
                metadata['domain'] = domain
            
            return ScrapedItem(
                title=title,
                url=url,
                content=story_text or title,
                published_date=published_date,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing HN API item: {str(e)}")
            return None
    
    def _extract_hn_metadata(self, description: str, comments_link: str) -> Dict[str, Any]:
        """Extract metadata from RSS description."""
        metadata = {}
        
        # Extract points
        points_match = re.search(r'Points:\s*(\d+)', description)
        if points_match:
            metadata['points'] = int(points_match.group(1))
        
        # Extract comments count
        comments_match = re.search(r'Comments:\s*(\d+)', description)
        if comments_match:
            metadata['comments'] = int(comments_match.group(1))
        
        # Extract HN ID from comments link
        if comments_link:
            id_match = re.search(r'id=(\d+)', comments_link)
            if id_match:
                metadata['hn_id'] = id_match.group(1)
        
        return metadata
    
    def _extract_hn_id(self, url: str) -> Optional[str]:
        """Extract HN story ID from URL."""
        if url and self.HN_BASE_URL in url:
            id_match = re.search(r'id=(\d+)', url)
            if id_match:
                return id_match.group(1)
        return None
    
    def _should_include_item(self, item: ScrapedItem) -> bool:
        """Check if item passes configured filters."""
        metadata = item.metadata
        
        # Check points threshold
        points = metadata.get('points', 0)
        if points < self.points_threshold:
            return False
        
        # Check comments threshold
        comments = metadata.get('comments', 0)
        if comments < self.comments_threshold:
            return False
        
        # Check author filter
        author = metadata.get('author', '')
        if self.author_filter and author:
            if author not in self.author_filter:
                return False
        
        # Check domain filter
        domain = metadata.get('domain', '')
        if self.domain_filter and domain:
            if not any(d in domain for d in self.domain_filter):
                return False
        
        # Check domain exclusion
        if self.exclude_domains and domain:
            if any(d in domain for d in self.exclude_domains):
                return False
        
        return True
    
    def _get_text(self, element, tag: str) -> Optional[str]:
        """Safely extract text from XML element."""
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return None


# End of hackernews_scraper.py