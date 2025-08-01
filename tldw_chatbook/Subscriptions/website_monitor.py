# website_monitor.py
# Description: Website monitoring integration for subscriptions
#
# This module integrates web scraping with RSS feed generation to enable
# monitoring of any website as if it were an RSS feed.
#
# Imports
import asyncio
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
#
# Third-Party Imports
from loguru import logger
#
# Local Imports
from .web_scraping_pipelines import ScrapingPipelineFactory, ScrapingConfig
from .rss_feed_generator import WebsiteToFeedConverter
from .baseline_manager import BaselineManager
from .monitoring_engine import FeedMonitor, RateLimiter
from .security import SecurityValidator
from ..DB.Subscriptions_DB import SubscriptionsDB
from ..Metrics.metrics_logger import log_histogram, log_counter
# Try to import existing web scraping functionality
try:
    from ..Web_Scraping.Article_Extractor_Lib import scrape_article, scrape_and_summarize_multiple
    from ..Web_Scraping.Article_Scraper.main import scrape_and_process_urls
    from ..Web_Scraping.Article_Scraper.config import ScraperConfig as ArticleScraperConfig
    ARTICLE_SCRAPER_AVAILABLE = True
except ImportError:
    ARTICLE_SCRAPER_AVAILABLE = False
    logger.warning("Article scraper module not available, using basic scrapers only")
#
########################################################################################################################
#
# Website Monitor
#
########################################################################################################################

class WebsiteMonitor:
    """
    Monitor websites for changes and convert to RSS feeds.
    
    This class orchestrates:
    - Web scraping using appropriate pipelines
    - Change detection with baselines
    - RSS feed generation from scraped content
    - Integration with subscription system
    """
    
    def __init__(self, db: SubscriptionsDB):
        """
        Initialize website monitor.
        
        Args:
            db: Subscriptions database instance
        """
        self.db = db
        self.baseline_manager = BaselineManager(db)
        self.feed_monitor = FeedMonitor(RateLimiter(), SecurityValidator())
        self.feed_cache_dir = Path.home() / '.config' / 'tldw_cli' / 'feed_cache'
        self.feed_cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def monitor_website(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor a website subscription for changes.
        
        Args:
            subscription: Subscription configuration
            
        Returns:
            Monitoring result with items and change report
        """
        start_time = datetime.now(timezone.utc)
        result = {
            'subscription_id': subscription['id'],
            'url': subscription['source'],
            'items': [],
            'change_report': None,
            'feed_generated': False,
            'error': None
        }
        
        try:
            # Determine monitoring approach based on type
            sub_type = subscription['type']
            
            if sub_type in ['rss', 'atom', 'json_feed', 'podcast']:
                # These are already feeds, use existing monitoring
                return await self._monitor_feed(subscription)
            
            elif sub_type == 'url':
                # Single URL monitoring
                items = await self._monitor_single_url(subscription)
                
            elif sub_type == 'url_list':
                # Multiple URLs monitoring
                items = await self._monitor_url_list(subscription)
                
            elif sub_type == 'sitemap':
                # Sitemap monitoring
                items = await self._monitor_sitemap(subscription)
                
            else:
                # Use generic web scraping
                items = await self._scrape_website(subscription)
            
            # Store items
            result['items'] = items
            
            # Generate RSS feed if items found
            if items:
                feed_path = await self._generate_and_cache_feed(subscription, items)
                result['feed_generated'] = True
                result['feed_path'] = str(feed_path)
            
            # Record success
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            log_histogram("website_monitor_duration", duration, labels={
                "type": sub_type,
                "success": "true"
            })
            log_counter("website_monitor_checks", labels={
                "type": sub_type,
                "status": "success",
                "item_count": str(len(items))
            })
            
        except Exception as e:
            logger.error(f"Error monitoring website {subscription['source']}: {str(e)}")
            result['error'] = str(e)
            
            log_counter("website_monitor_checks", labels={
                "type": subscription.get('type', 'unknown'),
                "status": "error",
                "error_type": type(e).__name__
            })
        
        return result
    
    async def _monitor_single_url(self, subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Monitor a single URL for changes."""
        url = subscription['source']
        
        # Check for changes using baseline
        new_content = await self._fetch_url_content(url, subscription)
        
        if not new_content:
            return []
        
        # Check against baseline
        ignore_selectors = subscription.get('ignore_selectors', '').split(',') if subscription.get('ignore_selectors') else []
        change_report = await self.baseline_manager.check_for_changes(
            subscription['id'],
            url,
            new_content,
            ignore_selectors
        )
        
        # Only return item if significant changes detected
        if change_report.has_changed and change_report.change_percentage >= subscription.get('change_threshold', 0.1):
            # Extract content using appropriate pipeline
            items = await self._extract_items_from_content(url, new_content, subscription)
            
            # Add change metadata
            for item in items:
                item['change_percentage'] = change_report.change_percentage
                item['change_type'] = change_report.change_type
                item['change_summary'] = change_report.summary
            
            return items
        
        return []
    
    async def _monitor_url_list(self, subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Monitor multiple URLs from a list."""
        # Parse URL list from subscription config
        url_list = subscription.get('extraction_rules', {}).get('urls', [])
        if isinstance(url_list, str):
            url_list = [u.strip() for u in url_list.split('\n') if u.strip()]
        
        all_items = []
        
        # Process each URL
        for url in url_list[:50]:  # Limit to 50 URLs
            try:
                items = await self._scrape_and_extract_url(url, subscription)
                all_items.extend(items)
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                continue
        
        return all_items
    
    async def _monitor_sitemap(self, subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Monitor website via sitemap."""
        sitemap_url = subscription['source']
        
        # Use existing article scraper if available
        if ARTICLE_SCRAPER_AVAILABLE:
            try:
                from ..Web_Scraping.Article_Scraper.crawler import get_urls_from_sitemap
                urls = await get_urls_from_sitemap(sitemap_url)
                
                # Limit URLs based on config
                max_urls = subscription.get('processing_options', {}).get('max_urls', 50)
                urls = urls[:max_urls]
                
                # Process URLs
                all_items = []
                for url in urls:
                    items = await self._scrape_and_extract_url(url, subscription)
                    all_items.extend(items)
                
                return all_items
                
            except Exception as e:
                logger.error(f"Sitemap processing failed: {str(e)}")
        
        return []
    
    async def _scrape_website(self, subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape website using configured pipeline."""
        # Create scraping config
        config = ScrapingConfig(
            url=subscription['source'],
            pipeline_type=subscription.get('extraction_method', 'generic'),
            selectors=subscription.get('extraction_rules', {}).get('selectors', {}),
            rate_limit=subscription.get('rate_limit_config', {}),
            javascript_enabled=subscription.get('processing_options', {}).get('javascript', False),
            options=subscription.get('processing_options', {})
        )
        
        # Create and run pipeline
        try:
            pipeline = ScrapingPipelineFactory.create_pipeline(config)
            items = await pipeline.scrape()
            
            # Convert to standard format
            return [item.to_dict() for item in items]
            
        except Exception as e:
            logger.error(f"Pipeline scraping failed: {str(e)}")
            # Fallback to article extractor if available
            if ARTICLE_SCRAPER_AVAILABLE:
                return await self._fallback_article_scraper(subscription)
            raise
    
    async def _fallback_article_scraper(self, subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use article scraper as fallback."""
        try:
            # Use scrape_article for single URL
            article_data = await scrape_article(subscription['source'])
            
            if article_data and article_data.get('extraction_successful'):
                return [{
                    'url': subscription['source'],
                    'title': article_data.get('title', 'Untitled'),
                    'content': article_data.get('content', ''),
                    'author': article_data.get('author'),
                    'published_date': article_data.get('published_date'),
                    'content_hash': article_data.get('content_hash')
                }]
        except Exception as e:
            logger.error(f"Article scraper fallback failed: {str(e)}")
        
        return []
    
    async def _fetch_url_content(self, url: str, subscription: Dict[str, Any]) -> Optional[str]:
        """Fetch content from URL."""
        try:
            # Try article extractor first
            if ARTICLE_SCRAPER_AVAILABLE:
                article_data = await scrape_article(url)
                if article_data and article_data.get('extraction_successful'):
                    return article_data.get('html', article_data.get('content', ''))
            
            # Fallback to basic fetch
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
                
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {str(e)}")
            return None
    
    async def _extract_items_from_content(self, url: str, content: str, 
                                        subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract items from HTML content."""
        # Use configured extraction method
        method = subscription.get('extraction_method', 'auto')
        
        if method == 'custom' and subscription.get('extraction_rules'):
            # Use custom extraction rules
            config = ScrapingConfig(
                url=url,
                pipeline_type='custom',
                options={
                    'rules': subscription['extraction_rules']
                }
            )
            pipeline = ScrapingPipelineFactory.create_pipeline(config)
            items = pipeline.parse_content(content, url)
            return [item.to_dict() for item in items]
        
        else:
            # Use generic extraction
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract basic info
            title = soup.find('title')
            title_text = title.get_text() if title else 'Untitled'
            
            # Try to extract publish date
            pub_date = None
            time_elem = soup.find('time')
            if time_elem and time_elem.get('datetime'):
                try:
                    pub_date = datetime.fromisoformat(time_elem['datetime'].replace('Z', '+00:00'))
                except:
                    pass
            
            return [{
                'url': url,
                'title': title_text,
                'content': content,
                'published_date': pub_date or datetime.now(timezone.utc),
                'content_hash': hashlib.sha256(content.encode()).hexdigest()
            }]
    
    async def _scrape_and_extract_url(self, url: str, subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape and extract content from a single URL."""
        try:
            content = await self._fetch_url_content(url, subscription)
            if content:
                return await self._extract_items_from_content(url, content, subscription)
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {str(e)}")
        
        return []
    
    async def _generate_and_cache_feed(self, subscription: Dict[str, Any], 
                                     items: List[Dict[str, Any]]) -> Path:
        """Generate RSS feed and cache it."""
        # Create feed converter
        converter = WebsiteToFeedConverter(
            website_url=subscription['source'],
            feed_type='rss',
            feed_title=subscription['name'],
            feed_description=subscription.get('description', '')
        )
        
        # Generate feed
        feed_content = converter.convert_items_to_feed(items)
        
        # Save to cache
        safe_name = subscription['name'].replace('/', '_').replace('\\', '_')
        feed_path = self.feed_cache_dir / f"{safe_name}_{subscription['id']}.xml"
        
        with open(feed_path, 'w', encoding='utf-8') as f:
            f.write(feed_content)
        
        logger.info(f"Generated RSS feed for {subscription['name']}: {feed_path}")
        
        return feed_path
    
    async def _monitor_feed(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor existing RSS/Atom feed using FeedMonitor."""
        start_time = datetime.now(timezone.utc)
        result = {
            'subscription_id': subscription['id'],
            'url': subscription['source'],
            'items': [],
            'change_report': None,
            'feed_generated': False,
            'error': None
        }
        
        try:
            # Use FeedMonitor to check the feed
            items = await self.feed_monitor.check_feed(subscription)
            
            # Convert items to standard format if needed
            formatted_items = []
            for item in items:
                # Ensure each item has the expected structure
                formatted_item = {
                    'url': item.get('link', item.get('url', '')),
                    'title': item.get('title', ''),
                    'content': item.get('description', item.get('content', '')),
                    'author': item.get('author'),
                    'published_date': item.get('published_date'),
                    'categories': item.get('categories', []),
                    'guid': item.get('guid', item.get('id')),
                    'metadata': item.get('metadata', {})
                }
                formatted_items.append(formatted_item)
            
            result['items'] = formatted_items
            
            # Log metrics
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            log_histogram("feed_monitor_duration", duration, labels={
                "type": subscription['type'],
                "success": "true"
            })
            log_counter("feed_monitor_checks", labels={
                "type": subscription['type'],
                "status": "success",
                "item_count": str(len(formatted_items))
            })
            
            logger.info(f"Successfully monitored {subscription['type']} feed: {subscription['name']} - {len(formatted_items)} items")
            
        except Exception as e:
            logger.error(f"Error monitoring feed {subscription['source']}: {str(e)}")
            result['error'] = str(e)
            
            log_counter("feed_monitor_checks", labels={
                "type": subscription.get('type', 'unknown'),
                "status": "error",
                "error_type": type(e).__name__
            })
        
        return result
    
    def get_cached_feed_path(self, subscription_id: int) -> Optional[Path]:
        """Get path to cached feed for subscription."""
        # Find cached feed file
        for feed_file in self.feed_cache_dir.glob(f"*_{subscription_id}.xml"):
            if feed_file.exists():
                return feed_file
        return None


# End of website_monitor.py