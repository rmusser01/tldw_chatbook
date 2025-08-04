# youtube_scraper.py
# Description: YouTube channel scraping pipeline using RSS feeds
#
# This scraper handles YouTube channel and playlist RSS feeds.
# YouTube provides RSS feeds for channels and playlists without requiring API keys.
#
# Channel RSS format: https://www.youtube.com/feeds/videos.xml?channel_id=CHANNEL_ID
# Playlist RSS format: https://www.youtube.com/feeds/videos.xml?playlist_id=PLAYLIST_ID
#
# Imports
import re
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qs, quote
#
# Third-Party Imports
import httpx
from loguru import logger
try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
    logger.warning("defusedxml not available, using standard xml.etree. Install defusedxml for better security.")
#
# Local Imports
from ..web_scraping_pipelines import BaseScrapingPipeline, ScrapedItem, ScrapingConfig
from ...Metrics.metrics_logger import log_counter
#
########################################################################################################################
#
# YouTube Scraping Pipeline
#
########################################################################################################################

class YouTubeScrapingPipeline(BaseScrapingPipeline):
    """
    Scraping pipeline for YouTube channels and playlists via RSS.
    
    Supports:
    - Channel feeds by channel ID
    - Channel feeds by username/handle
    - Playlist feeds
    - Video metadata extraction
    - Filtering by duration, views, age
    """
    
    YOUTUBE_BASE_URL = "https://www.youtube.com"
    YOUTUBE_RSS_BASE = "https://www.youtube.com/feeds/videos.xml"
    
    # XML namespaces used in YouTube RSS
    NAMESPACES = {
        'atom': 'http://www.w3.org/2005/Atom',
        'yt': 'http://www.youtube.com/xml/schemas/2015',
        'media': 'http://search.yahoo.com/mrss/'
    }
    
    def __init__(self, config: ScrapingConfig):
        """Initialize YouTube scraper with configuration."""
        super().__init__(config)
        
        # YouTube-specific options
        self.min_duration = config.options.get('min_duration', 0)  # seconds
        self.max_duration = config.options.get('max_duration', float('inf'))  # seconds
        self.min_views = config.options.get('min_views', 0)
        self.max_age_days = config.options.get('max_age_days', 30)
        self.include_shorts = config.options.get('include_shorts', True)
        self.include_live = config.options.get('include_live', True)
        self.keyword_filter = config.options.get('keyword_filter', [])
        self.exclude_keywords = config.options.get('exclude_keywords', [])
        self.max_videos = config.options.get('max_videos', 50)
        
        # Extract channel/playlist info from URL
        self._parse_youtube_url()
    
    def _parse_youtube_url(self):
        """Parse YouTube URL to extract channel or playlist ID."""
        url = self.config.url
        parsed = urlparse(url)
        
        # Reset IDs
        self.channel_id = None
        self.playlist_id = None
        self.username = None
        self.handle = None
        
        # Check for direct RSS feed URL
        if 'feeds/videos.xml' in url:
            query_params = parse_qs(parsed.query)
            if 'channel_id' in query_params:
                self.channel_id = query_params['channel_id'][0]
            elif 'playlist_id' in query_params:
                self.playlist_id = query_params['playlist_id'][0]
            return
        
        # Parse YouTube URLs
        path_parts = parsed.path.strip('/').split('/')
        
        # Channel by ID: youtube.com/channel/UC...
        if 'channel' in path_parts:
            idx = path_parts.index('channel')
            if idx + 1 < len(path_parts):
                self.channel_id = path_parts[idx + 1]
        
        # Channel by username: youtube.com/user/username or youtube.com/c/username
        elif 'user' in path_parts:
            idx = path_parts.index('user')
            if idx + 1 < len(path_parts):
                self.username = path_parts[idx + 1]
        elif 'c' in path_parts:
            idx = path_parts.index('c')
            if idx + 1 < len(path_parts):
                self.username = path_parts[idx + 1]
        
        # Channel by handle: youtube.com/@handle
        elif path_parts and path_parts[0].startswith('@'):
            self.handle = path_parts[0][1:]  # Remove @
        
        # Playlist: youtube.com/playlist?list=PL...
        elif 'playlist' in parsed.path:
            query_params = parse_qs(parsed.query)
            if 'list' in query_params:
                self.playlist_id = query_params['list'][0]
        
        # Watch URL with playlist
        elif 'watch' in parsed.path:
            query_params = parse_qs(parsed.query)
            if 'list' in query_params:
                self.playlist_id = query_params['list'][0]
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate YouTube-specific configuration."""
        # Check if we have valid YouTube identifiers
        if not any([self.channel_id, self.playlist_id, self.username, self.handle]):
            return False, "No valid YouTube channel or playlist identifier found in URL"
        
        # Validate duration settings
        if self.min_duration < 0:
            return False, "Minimum duration cannot be negative"
        
        if self.max_duration < self.min_duration:
            return False, "Maximum duration must be greater than minimum duration"
        
        return True, None
    
    def get_rate_limit(self) -> int:
        """Get rate limit for YouTube RSS."""
        # YouTube RSS feeds are generally lenient
        return 60  # 1 request per second
    
    async def fetch_content(self, url: str) -> str:
        """Fetch YouTube RSS feed."""
        # Build RSS URL based on what we have
        if self.channel_id:
            feed_url = f"{self.YOUTUBE_RSS_BASE}?channel_id={self.channel_id}"
        elif self.playlist_id:
            feed_url = f"{self.YOUTUBE_RSS_BASE}?playlist_id={self.playlist_id}"
        elif self.username or self.handle:
            # For username/handle, we need to fetch the channel page first
            # to get the channel ID
            channel_id = await self._resolve_channel_id()
            if not channel_id:
                raise ValueError(f"Could not resolve channel ID for {self.username or self.handle}")
            self.channel_id = channel_id
            feed_url = f"{self.YOUTUBE_RSS_BASE}?channel_id={self.channel_id}"
        else:
            raise ValueError("No valid YouTube identifier available")
        
        logger.debug(f"Fetching YouTube RSS from: {feed_url}")
        
        # Fetch RSS feed
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = self.get_headers()
            headers['Accept'] = 'application/rss+xml, application/xml'
            
            response = await client.get(feed_url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            log_counter("youtube_scraper_fetch", labels={
                "status": "success",
                "type": "playlist" if self.playlist_id else "channel"
            })
            
            return response.text
    
    async def _resolve_channel_id(self) -> Optional[str]:
        """Resolve channel ID from username or handle."""
        try:
            # Build channel URL
            if self.handle:
                channel_url = f"{self.YOUTUBE_BASE_URL}/@{self.handle}"
            else:
                channel_url = f"{self.YOUTUBE_BASE_URL}/c/{self.username}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(channel_url, headers=self.get_headers())
                response.raise_for_status()
                
                # Extract channel ID from page content
                # Look for canonical URL or channel ID in page source
                content = response.text
                
                # Try to find channel ID in various places
                patterns = [
                    r'"channelId":"(UC[a-zA-Z0-9_-]+)"',
                    r'<meta itemprop="channelId" content="(UC[a-zA-Z0-9_-]+)"',
                    r'"externalId":"(UC[a-zA-Z0-9_-]+)"',
                    r'/channel/(UC[a-zA-Z0-9_-]+)'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, content)
                    if match:
                        return match.group(1)
                
                return None
                
        except Exception as e:
            logger.error(f"Error resolving channel ID: {str(e)}")
            return None
    
    def parse_content(self, raw_content: str, url: str) -> List[ScrapedItem]:
        """Parse YouTube RSS feed into items."""
        items = []
        
        try:
            # Parse RSS XML safely
            root = ET.fromstring(raw_content)
            
            # Find all video entries
            entries = root.findall('.//atom:entry', self.NAMESPACES)
            
            for entry in entries[:self.max_videos]:
                item = self._parse_entry(entry)
                
                if item and self._should_include_video(item):
                    items.append(item)
            
            logger.info(f"Parsed {len(items)} YouTube videos from RSS")
            log_counter("youtube_scraper_items", labels={
                "type": "playlist" if self.playlist_id else "channel",
                "item_count": str(len(items))
            })
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse YouTube RSS: {str(e)}")
            raise
        
        return items
    
    def _parse_entry(self, entry) -> Optional[ScrapedItem]:
        """Parse a single RSS entry."""
        try:
            # Extract basic info
            video_id = self._get_text(entry, 'yt:videoId', self.NAMESPACES)
            title = self._get_text(entry, 'atom:title', self.NAMESPACES)
            published = self._get_text(entry, 'atom:published', self.NAMESPACES)
            
            if not video_id or not title:
                return None
            
            # Build video URL
            video_url = f"{self.YOUTUBE_BASE_URL}/watch?v={video_id}"
            
            # Parse published date
            published_date = None
            if published:
                try:
                    published_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                except ValueError:
                    logger.warning(f"Could not parse date: {published}")
            
            # Extract media info
            media_group = entry.find('media:group', self.NAMESPACES)
            description = ""
            thumbnail_url = None
            duration = None
            
            if media_group is not None:
                # Description
                desc_elem = media_group.find('media:description', self.NAMESPACES)
                if desc_elem is not None and desc_elem.text:
                    description = desc_elem.text
                
                # Thumbnail
                thumbnail_elem = media_group.find('media:thumbnail', self.NAMESPACES)
                if thumbnail_elem is not None:
                    thumbnail_url = thumbnail_elem.get('url')
                
                # Duration (in seconds)
                content_elem = media_group.find('media:content', self.NAMESPACES)
                if content_elem is not None:
                    duration_str = content_elem.get('duration')
                    if duration_str:
                        duration = int(duration_str)
            
            # Extract statistics
            stats_elem = media_group.find('media:community', self.NAMESPACES) if media_group else None
            views = 0
            
            if stats_elem is not None:
                stats_elem = stats_elem.find('media:statistics', self.NAMESPACES)
                if stats_elem is not None:
                    views_str = stats_elem.get('views', '0')
                    views = int(views_str)
            
            # Channel info
            author_elem = entry.find('atom:author/atom:name', self.NAMESPACES)
            channel_name = author_elem.text if author_elem is not None else ""
            
            # Build metadata
            metadata = {
                'video_id': video_id,
                'channel_name': channel_name,
                'channel_id': self.channel_id,
                'duration': duration,
                'views': views,
                'thumbnail_url': thumbnail_url,
                'is_short': self._is_short(title, duration),
                'is_live': self._is_live(title)
            }
            
            return ScrapedItem(
                title=title,
                url=video_url,
                content=description or title,
                author=channel_name,
                published_date=published_date,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing YouTube entry: {str(e)}")
            return None
    
    def _should_include_video(self, item: ScrapedItem) -> bool:
        """Check if video passes configured filters."""
        metadata = item.metadata
        
        # Check if it's a short and we're excluding them
        if not self.include_shorts and metadata.get('is_short'):
            return False
        
        # Check if it's live and we're excluding them
        if not self.include_live and metadata.get('is_live'):
            return False
        
        # Check duration filters
        duration = metadata.get('duration', 0)
        if duration and (duration < self.min_duration or duration > self.max_duration):
            return False
        
        # Check view count
        views = metadata.get('views', 0)
        if views < self.min_views:
            return False
        
        # Check age
        if item.published_date and self.max_age_days:
            age_days = (datetime.now(timezone.utc) - item.published_date).days
            if age_days > self.max_age_days:
                return False
        
        # Check keyword filters
        text_to_check = f"{item.title} {item.content}".lower()
        
        # Include keywords (if specified, at least one must match)
        if self.keyword_filter:
            if not any(keyword.lower() in text_to_check for keyword in self.keyword_filter):
                return False
        
        # Exclude keywords (none should match)
        if self.exclude_keywords:
            if any(keyword.lower() in text_to_check for keyword in self.exclude_keywords):
                return False
        
        return True
    
    def _is_short(self, title: str, duration: Optional[int]) -> bool:
        """Detect if video is a YouTube Short."""
        # Shorts are typically under 60 seconds
        if duration and duration <= 60:
            return True
        
        # Check title for #Shorts hashtag
        if '#shorts' in title.lower():
            return True
        
        return False
    
    def _is_live(self, title: str) -> bool:
        """Detect if video is a live stream."""
        live_indicators = ['🔴', '[live]', '(live)', 'live:', 'live stream', 'streaming']
        title_lower = title.lower()
        
        return any(indicator in title_lower for indicator in live_indicators)
    
    def _get_text(self, element, path: str, namespaces: dict) -> Optional[str]:
        """Safely extract text from XML element."""
        child = element.find(path, namespaces)
        if child is not None and child.text:
            return child.text.strip()
        return None
    
    def format_duration(self, seconds: Optional[int]) -> str:
        """Format duration in seconds to human-readable string."""
        if not seconds:
            return "Unknown"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"


# End of youtube_scraper.py