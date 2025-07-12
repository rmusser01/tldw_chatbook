"""
article_scraper/config.py
=========================

Configuration dataclasses for the article scraper.

This module defines configuration objects used throughout the
scraping pipeline to control behavior and processing options.

Classes:
--------
- ScraperConfig: Browser and extraction settings
- ProcessorConfig: Content processing and API settings

Example:
--------
    # Configure scraper with stealth mode
    scraper_config = ScraperConfig(
        stealth=True,
        retries=3,
        request_timeout_ms=30000
    )
    
    # Configure processor for summarization
    processor_config = ProcessorConfig(
        api_name="openai",
        api_key="sk-...",
        summarize=True,
        custom_prompt="Extract key insights"
    )
"""
#
# Imports
from dataclasses import dataclass, field
from typing import List, Dict, Any
#
# Third-Party Imports
#
# Imports
#
#######################################################################################################################
#
# Functions:

@dataclass
class ScraperConfig:
    """
    Configuration for web scraping behavior.
    
    Controls browser settings, retry logic, and content extraction options.
    All parameters have sensible defaults for typical web scraping.
    
    Attributes:
        user_agent: Browser user agent string for requests
        request_timeout_ms: Maximum time to wait for page load (milliseconds)
        retries: Number of retry attempts on failure
        stealth: Enable stealth mode to avoid bot detection
        stealth_wait_ms: Additional wait time in stealth mode
        include_comments: Extract comment sections
        include_tables: Extract table data
        include_images: Extract image information
    """
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    request_timeout_ms: int = 60000  # 60 seconds
    retries: int = 3
    stealth: bool = True
    # Time to wait after page load if stealth is enabled
    stealth_wait_ms: int = 5000

    # Trafilatura settings
    include_comments: bool = False
    include_tables: bool = False
    include_images: bool = False


@dataclass
class ProcessorConfig:
    """
    Configuration for content processing and LLM integration.
    
    Defines how scraped content should be processed, including
    summarization settings and API credentials.
    
    Attributes:
        api_name: Name of LLM API service (e.g., 'openai', 'anthropic')
        api_key: Authentication key for the API
        summarize: Whether to generate summaries
        custom_prompt: Prompt template for summarization
        system_message: System message for LLM context
        temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)
        keywords: Keywords for content categorization
    """
    api_name: str
    api_key: str
    summarize: bool = False
    custom_prompt: str = "Please provide a concise summary of the following article."
    system_message: str = "You are an expert summarization assistant."
    temperature: float = 0.7
    keywords: List[str] = field(default_factory=list)

#
# End of article_scraper/config.py
#######################################################################################################################
