# reddit_scraper.py
# Description: Reddit-specific scraping pipeline using RSS feeds
#
# This scraper handles Reddit content via RSS feeds, supporting:
# - Subreddit feeds
# - User feeds  
# - Search feeds
# - Multi-reddit feeds
#
# Imports
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, quote
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
# Reddit Scraping Pipeline
#
########################################################################################################################

class RedditScrapingPipeline(BaseScrapingPipeline):
    """
    Scraping pipeline for Reddit content via RSS feeds.
    
    Supports multiple Reddit RSS feed types:
    - Subreddit: /r/subreddit/.rss
    - User: /user/username/.rss
    - Search: /r/subreddit/search.rss?q=query
    - Domain: /domain/example.com/.rss
    """
    
    RSS_BASE_URL = "https://www.reddit.com"
    
    def __init__(self, config: ScrapingConfig):
        """Initialize Reddit scraper with configuration."""
        super().__init__(config)
        
        # Reddit-specific options
        self.score_threshold = config.options.get('score_threshold', 0)
        self.post_limit = config.options.get('post_limit', 25)
        self.comment_threshold = config.options.get('comment_threshold', 0)
        self.flair_filter = config.options.get('flair_filter', [])
        self.time_filter = config.options.get('time_filter', None)  # hour, day, week, month, year, all
        self.author_filter = config.options.get('author_filter', [])
        self.exclude_authors = config.options.get('exclude_authors', [])
        self.include_comments = config.options.get('include_comments', False)
        self.multi_subreddits = config.options.get('multi_subreddits', [])
        self.include_nsfw = config.options.get('include_nsfw', False)
        self.sort_by = config.options.get('sort_by', 'hot')  # hot, new, top, rising
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate Reddit-specific configuration."""
        # Check URL format
        url = self.config.url
        
        # Extract Reddit-specific part from URL
        reddit_pattern = r'reddit\.com/(r/[\w]+|user/[\w-]+|domain/[\w.]+)'
        match = re.search(reddit_pattern, url)
        
        if not match:
            return False, "Invalid Reddit URL format. Expected /r/subreddit, /user/username, or /domain/example.com"
        
        # Validate sort option
        valid_sorts = ['hot', 'new', 'top', 'rising', 'controversial']
        if self.sort_by not in valid_sorts:
            return False, f"Invalid sort option: {self.sort_by}. Must be one of {valid_sorts}"
        
        # Validate time filter
        valid_times = ['hour', 'day', 'week', 'month', 'year', 'all']
        if self.time_filter not in valid_times:
            return False, f"Invalid time filter: {self.time_filter}. Must be one of {valid_times}"
        
        return True, None
    
    def _build_rss_url(self, base_path: str) -> str:
        """Build complete RSS URL with parameters."""
        # Start with base RSS URL
        rss_url = f"{self.RSS_BASE_URL}{base_path}.rss"
        
        # Add parameters for certain sort types
        params = []
        if self.sort_by in ['top', 'controversial']:
            params.append(f"t={self.time_filter}")
        
        if self.post_limit != 25:
            params.append(f"limit={self.post_limit}")
        
        if params:
            rss_url += "?" + "&".join(params)
        
        return rss_url
    
    async def fetch_content(self, url: str) -> str:
        """Fetch Reddit RSS feed content."""
        # Check if we need to handle multi-subreddit
        if self.multi_subreddits:
            return await self._fetch_multi_subreddit_content(url)
        
        # Parse Reddit URL to get the path
        parsed = urlparse(url)
        reddit_path = parsed.path.rstrip('/')
        
        # Handle different feed types
        if reddit_path.startswith('/r/') and self.sort_by != 'hot':
            # Add sort to subreddit URL
            reddit_path = f"{reddit_path}/{self.sort_by}"
        
        # Build RSS URL
        rss_url = self._build_rss_url(reddit_path)
        
        # Add search parameters if present
        if parsed.query:
            separator = "&" if "?" in rss_url else "?"
            rss_url += separator + parsed.query
        
        logger.debug(f"Fetching Reddit RSS from: {rss_url}")
        
        # Fetch RSS content
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = self.get_headers()
            headers['Accept'] = 'application/rss+xml, application/xml'
            
            response = await client.get(rss_url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            log_counter("reddit_scraper_fetch", labels={
                "status": "success",
                "feed_type": self._get_feed_type(reddit_path)
            })
            
            return response.text
    
    async def _fetch_multi_subreddit_content(self, base_url: str) -> str:
        """Fetch and combine multiple subreddit feeds."""
        # Parse base URL to get the primary subreddit
        parsed = urlparse(base_url)
        primary_sub = re.search(r'/r/([\w]+)', parsed.path)
        
        subreddits = [primary_sub.group(1)] if primary_sub else []
        subreddits.extend(self.multi_subreddits)
        
        # Remove duplicates while preserving order
        subreddits = list(dict.fromkeys(subreddits))
        
        # Create multi-reddit URL (r/sub1+sub2+sub3)
        multi_path = f"/r/{'+'.join(subreddits)}"
        
        if self.sort_by != 'hot':
            multi_path += f"/{self.sort_by}"
        
        # Build RSS URL
        rss_url = self._build_rss_url(multi_path)
        
        logger.debug(f"Fetching multi-subreddit RSS from: {rss_url}")
        
        # Fetch RSS content
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = self.get_headers()
            headers['Accept'] = 'application/rss+xml, application/xml'
            
            response = await client.get(rss_url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            log_counter("reddit_scraper_fetch", labels={
                "status": "success",
                "feed_type": "multi_subreddit",
                "subreddit_count": str(len(subreddits))
            })
            
            return response.text
    
    def parse_content(self, raw_content: str, url: str) -> List[ScrapedItem]:
        """Parse Reddit RSS feed into items."""
        items = []
        
        try:
            # Parse RSS XML safely
            root = ET.fromstring(raw_content)
            
            # Find all items in the feed
            for item in root.findall('.//item'):
                scraped_item = self._parse_reddit_item(item)
                
                if scraped_item:
                    # Apply filters
                    if self._should_include_item(scraped_item):
                        items.append(scraped_item)
            
            logger.info(f"Parsed {len(items)} Reddit items from feed")
            log_counter("reddit_scraper_items", labels={
                "feed_type": self._get_feed_type(url),
                "item_count": str(len(items))
            })
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse Reddit RSS: {str(e)}")
            log_counter("reddit_scraper_errors", labels={
                "error_type": "parse_error"
            })
            raise
        
        return items
    
    def _parse_reddit_item(self, item_element) -> Optional[ScrapedItem]:
        """Parse a single Reddit RSS item."""
        try:
            # Extract basic fields
            title = self._get_text(item_element, 'title')
            link = self._get_text(item_element, 'link')
            pub_date = self._get_text(item_element, 'pubDate')
            description = self._get_text(item_element, 'description')
            
            if not title or not link:
                return None
            
            # Parse publication date
            published = None
            if pub_date:
                try:
                    # Reddit uses RFC 822 format
                    published = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                except ValueError:
                    logger.warning(f"Could not parse date: {pub_date}")
            
            # Extract Reddit-specific metadata
            metadata = self._extract_reddit_metadata(item_element, description)
            
            # Extract author from title (Reddit format: "Title by /u/username")
            author = None
            author_match = re.search(r'by /u/([\w-]+)', title)
            if author_match:
                author = f"/u/{author_match.group(1)}"
                # Clean up title
                title = title.replace(f" by {author}", "").strip()
            
            # Extract categories (subreddit)
            categories = []
            category = self._get_text(item_element, 'category')
            if category:
                categories.append(category)
            
            # Create scraped item
            return ScrapedItem(
                url=link,
                title=title,
                content=self._clean_reddit_content(description),
                author=author,
                published_date=published,
                categories=categories,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing Reddit item: {str(e)}")
            return None
    
    def _extract_reddit_metadata(self, item_element, description: str) -> Dict[str, Any]:
        """Extract Reddit-specific metadata from item."""
        metadata = {
            'source': 'reddit',
            'feed_type': self._get_feed_type(self.config.url)
        }
        
        # Try to extract score/comments from description
        # Reddit includes these in the description HTML
        if description:
            # Look for score
            score_match = re.search(r'\[(\d+) points?\]', description)
            if score_match:
                metadata['score'] = int(score_match.group(1))
            
            # Look for comment count
            comments_match = re.search(r'\[(\d+) comments?\]', description)
            if comments_match:
                metadata['comment_count'] = int(comments_match.group(1))
            
            # Check for NSFW
            if '[NSFW]' in description or 'nsfw:yes' in description.lower():
                metadata['nsfw'] = True
        
        # Extract subreddit from category or URL
        subreddit_match = re.search(r'/r/([\w]+)', self.config.url)
        if subreddit_match:
            metadata['subreddit'] = subreddit_match.group(1)
        
        return metadata
    
    def _clean_reddit_content(self, html_content: str) -> str:
        """Clean Reddit HTML content."""
        if not html_content:
            return ""
        
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Reddit includes metadata in brackets, remove those
            text = soup.get_text()
            
            # Remove metadata patterns
            patterns_to_remove = [
                r'\[\d+ points?\]',
                r'\[\d+ comments?\]',
                r'\[link\]',
                r'\[comments\]',
                r'submitted by /u/[\w-]+',
                r'to /r/[\w]+',
                r'\[NSFW\]'
            ]
            
            for pattern in patterns_to_remove:
                text = re.sub(pattern, '', text)
            
            # Clean up extra whitespace
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning Reddit content: {str(e)}")
            return html_content
    
    def _should_include_item(self, item: ScrapedItem) -> bool:
        """Check if item passes configured filters."""
        metadata = item.metadata
        
        # Check score threshold
        score = metadata.get('score', 0)
        if score < self.score_threshold:
            return False
        
        # Check comment threshold
        comments = metadata.get('comments', 0)
        if comments < self.comment_threshold:
            return False
        
        # Check NSFW filter
        if not self.include_nsfw and metadata.get('nsfw', False):
            return False
        
        # Check flair filter (inclusion list)
        if self.flair_filter:
            flair = metadata.get('flair', '').lower()
            if not any(f.lower() in flair for f in self.flair_filter):
                return False
        
        # Check author filter (inclusion list)
        author = metadata.get('author', '')
        if self.author_filter and author:
            if author not in self.author_filter:
                return False
        
        # Check author exclusion list
        if self.exclude_authors and author:
            if author in self.exclude_authors:
                return False
        
        # Check time filter
        if self.time_filter and item.published_date:
            now = datetime.now(timezone.utc)
            age = now - item.published_date
            
            time_limits = {
                'hour': timedelta(hours=1),
                'day': timedelta(days=1),
                'week': timedelta(weeks=1),
                'month': timedelta(days=30),
                'year': timedelta(days=365)
            }
            
            if self.time_filter in time_limits:
                if age > time_limits[self.time_filter]:
                    return False
        
        return True
    
    def _get_feed_type(self, url: str) -> str:
        """Determine Reddit feed type from URL."""
        if '/r/' in url:
            if '/search' in url:
                return 'search'
            return 'subreddit'
        elif '/user/' in url:
            return 'user'
        elif '/domain/' in url:
            return 'domain'
        else:
            return 'unknown'
    
    def _get_text(self, element, tag: str) -> Optional[str]:
        """Safely extract text from XML element."""
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return None


# End of reddit_scraper.py