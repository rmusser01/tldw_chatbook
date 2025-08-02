# web_scraping_pipelines.py
# Description: Base architecture for web scraping pipelines
#
# This module provides the abstract base class and common functionality
# for all web scraping pipelines in the subscription system.
#
# Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import urlparse
import hashlib
import json
#
# Third-Party Imports
from loguru import logger
#
# Local Imports
from ..Metrics.metrics_logger import log_histogram, log_counter
from .site_config_manager import get_site_config_manager
#
########################################################################################################################
#
# Data Classes
#
########################################################################################################################

@dataclass
class ScrapedItem:
    """Represents a single scraped content item."""
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    categories: List[str] = None
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.categories is None:
            self.categories = []
        if self.metadata is None:
            self.metadata = {}
        if self.content_hash is None and self.content:
            self.content_hash = self._calculate_hash(self.content)
    
    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'author': self.author,
            'published_date': self.published_date.isoformat() if self.published_date else None,
            'categories': self.categories,
            'content_hash': self.content_hash,
            'extracted_data': self.metadata
        }


@dataclass
class ScrapingConfig:
    """Configuration for a scraping pipeline."""
    url: str
    pipeline_type: str
    selectors: Dict[str, str] = None
    rate_limit: Dict[str, Any] = None
    authentication: Dict[str, Any] = None
    javascript_enabled: bool = False
    custom_headers: Dict[str, str] = None
    options: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize optional fields."""
        if self.selectors is None:
            self.selectors = {}
        if self.rate_limit is None:
            self.rate_limit = {'requests_per_minute': 60}
        if self.custom_headers is None:
            self.custom_headers = {}
        if self.options is None:
            self.options = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapingConfig':
        """Create config from dictionary."""
        return cls(**data)


########################################################################################################################
#
# Base Scraping Pipeline
#
########################################################################################################################

class BaseScrapingPipeline(ABC):
    """
    Abstract base class for all web scraping pipelines.
    
    This class defines the interface that all scrapers must implement
    and provides common functionality for rate limiting, error handling,
    and metrics collection.
    """
    
    def __init__(self, config: ScrapingConfig):
        """
        Initialize the scraping pipeline.
        
        Args:
            config: Configuration for this pipeline
        """
        self.config = config
        self.pipeline_type = self.__class__.__name__
        self.site_config_manager = get_site_config_manager()
        logger.info(f"Initialized {self.pipeline_type} for {config.url}")
    
    @abstractmethod
    async def fetch_content(self, url: str) -> str:
        """
        Fetch raw content from the given URL.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Raw content as string
            
        Raises:
            Various exceptions based on implementation
        """
        pass
    
    @abstractmethod
    def parse_content(self, raw_content: str, url: str) -> List[ScrapedItem]:
        """
        Parse raw content into structured items.
        
        Args:
            raw_content: Raw content to parse
            url: Source URL for context
            
        Returns:
            List of parsed items
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the pipeline configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    async def scrape(self, url: Optional[str] = None) -> List[ScrapedItem]:
        """
        Main scraping method that combines fetch and parse.
        
        Args:
            url: Optional URL override (uses config URL if not provided)
            
        Returns:
            List of scraped items
        """
        start_time = datetime.now(timezone.utc)
        target_url = url or self.config.url
        items = []
        
        try:
            # Validate configuration
            is_valid, error_msg = self.validate_config()
            if not is_valid:
                raise ValueError(f"Invalid configuration: {error_msg}")
            
            # Check rate limiting
            can_request, wait_time = self.site_config_manager.check_rate_limit(target_url)
            if not can_request:
                raise ValueError(f"Rate limit exceeded. Wait {wait_time:.1f} seconds.")
            
            # Get site configuration
            site_config = self.site_config_manager.get_config(target_url)
            
            # Fetch content
            logger.debug(f"Fetching content from {target_url}")
            raw_content = await self.fetch_content(target_url)
            
            # Parse content
            logger.debug(f"Parsing content from {target_url}")
            items = self.parse_content(raw_content, target_url)
            
            # Record success
            site_config.record_success()
            self.site_config_manager.save_config(site_config)
            
            # Log success metrics
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            log_histogram("scraping_pipeline_duration", duration, labels={
                "pipeline_type": self.pipeline_type,
                "success": "true"
            })
            log_counter("scraping_pipeline_items", labels={
                "pipeline_type": self.pipeline_type,
                "item_count": str(len(items))
            })
            
            logger.info(f"Successfully scraped {len(items)} items from {target_url}")
            return items
            
        except Exception as e:
            # Record error
            site_config = self.site_config_manager.get_config(target_url)
            site_config.record_error(str(e))
            self.site_config_manager.save_config(site_config)
            
            # Log error metrics
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            log_histogram("scraping_pipeline_duration", duration, labels={
                "pipeline_type": self.pipeline_type,
                "success": "false"
            })
            log_counter("scraping_pipeline_errors", labels={
                "pipeline_type": self.pipeline_type,
                "error_type": type(e).__name__
            })
            
            logger.error(f"Error scraping {target_url}: {str(e)}")
            raise
    
    def get_rate_limit(self) -> Dict[str, Any]:
        """
        Get rate limit configuration for this pipeline.
        
        Returns:
            Rate limit configuration dict
        """
        return self.config.rate_limit
    
    def get_domain(self) -> str:
        """
        Extract domain from configured URL.
        
        Returns:
            Domain name
        """
        parsed = urlparse(self.config.url)
        return parsed.netloc
    
    def should_use_javascript(self) -> bool:
        """
        Check if JavaScript rendering is required.
        
        Returns:
            True if JavaScript should be used
        """
        # Check site-specific config first
        site_config = self.site_config_manager.get_config(self.config.url)
        if site_config.requires_javascript:
            return True
        
        # Fall back to pipeline config
        return self.config.javascript_enabled
    
    def get_auth(self) -> Optional[Tuple[str, str]]:
        """
        Get authentication credentials if configured.
        
        Returns:
            Tuple of (username, password) for basic auth, or None
        """
        site_config = self.site_config_manager.get_config(self.config.url)
        return site_config.get_auth()
    
    def get_site_config(self):
        """
        Get the site-specific configuration.
        
        Returns:
            SiteConfig instance
        """
        return self.site_config_manager.get_config(self.config.url)
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for requests, merging with site-specific config.
        
        Returns:
            Headers dictionary
        """
        default_headers = {
            'User-Agent': 'tldw-chatbook/1.0 (Subscription Monitor)',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Cache-Control': 'no-cache',
            'DNT': '1'
        }
        
        # Get site-specific configuration
        site_config = self.site_config_manager.get_config(self.config.url)
        
        # Merge headers in order: default -> config -> site-specific
        if self.config.custom_headers:
            default_headers.update(self.config.custom_headers)
        
        # Apply site-specific headers with auth
        site_headers = site_config.get_headers(default_headers)
        
        return site_headers


########################################################################################################################
#
# Pipeline Factory
#
########################################################################################################################

class ScrapingPipelineFactory:
    """Factory for creating scraping pipeline instances."""
    
    _pipeline_registry: Dict[str, type] = {}
    
    @classmethod
    def register_pipeline(cls, pipeline_type: str, pipeline_class: type):
        """
        Register a pipeline class with the factory.
        
        Args:
            pipeline_type: Type identifier for the pipeline
            pipeline_class: Pipeline class to register
        """
        if not issubclass(pipeline_class, BaseScrapingPipeline):
            raise ValueError(f"{pipeline_class} must inherit from BaseScrapingPipeline")
        
        cls._pipeline_registry[pipeline_type] = pipeline_class
        logger.info(f"Registered pipeline type: {pipeline_type}")
    
    @classmethod
    def create_pipeline(cls, config: ScrapingConfig) -> BaseScrapingPipeline:
        """
        Create a pipeline instance based on configuration.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Pipeline instance
            
        Raises:
            ValueError: If pipeline type is not registered
        """
        pipeline_type = config.pipeline_type
        
        if pipeline_type not in cls._pipeline_registry:
            available = ", ".join(cls._pipeline_registry.keys())
            raise ValueError(
                f"Unknown pipeline type: {pipeline_type}. "
                f"Available types: {available}"
            )
        
        pipeline_class = cls._pipeline_registry[pipeline_type]
        return pipeline_class(config)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """
        Get list of available pipeline types.
        
        Returns:
            List of registered pipeline type names
        """
        return list(cls._pipeline_registry.keys())


########################################################################################################################
#
# Common Utilities
#
########################################################################################################################

class ContentExtractor:
    """Common content extraction utilities."""
    
    @staticmethod
    def extract_text_content(html: str, remove_scripts: bool = True, 
                           remove_styles: bool = True) -> str:
        """
        Extract clean text from HTML.
        
        Args:
            html: HTML content
            remove_scripts: Remove script tags
            remove_styles: Remove style tags
            
        Returns:
            Cleaned text content
        """
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted tags
            if remove_scripts:
                for script in soup.find_all('script'):
                    script.decompose()
            
            if remove_styles:
                for style in soup.find_all('style'):
                    style.decompose()
            
            # Get text and clean up whitespace
            text = soup.get_text()
            lines = [line.strip() for line in text.splitlines()]
            text = '\n'.join(line for line in lines if line)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text content: {str(e)}")
            return ""
    
    @staticmethod
    def extract_metadata(html: str) -> Dict[str, Any]:
        """
        Extract metadata from HTML.
        
        Args:
            html: HTML content
            
        Returns:
            Metadata dictionary
        """
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            metadata = {}
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Extract meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', meta.get('property', ''))
                content = meta.get('content', '')
                
                if name and content:
                    metadata[name] = content
            
            # Extract OpenGraph data
            og_data = {}
            for meta in soup.find_all('meta', property=lambda x: x and x.startswith('og:')):
                prop = meta.get('property', '').replace('og:', '')
                content = meta.get('content', '')
                if prop and content:
                    og_data[prop] = content
            
            if og_data:
                metadata['opengraph'] = og_data
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {}


# End of web_scraping_pipelines.py