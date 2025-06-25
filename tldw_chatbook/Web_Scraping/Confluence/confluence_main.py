# confluence_main.py
#
# High-level API functions for Confluence scraping
#
# Imports
import asyncio
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
#
# Third-party imports
from loguru import logger
#
# Local imports
from .confluence_auth import ConfluenceAuth, create_confluence_auth
from .confluence_scraper import ConfluenceScraper
from .confluence_crawler import ConfluenceCrawler, create_page_filter
from .confluence_utils import parse_confluence_url, format_page_hierarchy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tldw_chatbook.DB.Client_Media_DB_v2 import ingest_article_to_db_new
from tldw_chatbook.config import load_cli_config_and_ensure_existence
#
#######################################################################################################################
#
# High-level API Functions

async def scrape_confluence_page(
    url_or_id: str,
    auth: Optional[ConfluenceAuth] = None,
    config: Optional[Dict[str, Any]] = None,
    include_attachments: bool = False,
    ingest_to_db: bool = False,
    keywords: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scrape a single Confluence page
    
    Args:
        url_or_id: Confluence page URL or page ID
        auth: Optional ConfluenceAuth instance (will create from config if not provided)
        config: Optional configuration dict (will load from file if not provided)
        include_attachments: Whether to fetch attachments
        ingest_to_db: Whether to ingest the scraped content to database
        keywords: Keywords for database ingestion
        
    Returns:
        Scraped page data
    """
    # Setup authentication
    if not auth:
        if not config:
            config = load_confluence_config()
        auth = create_confluence_auth(config['base_url'], config)
        
    scraper = ConfluenceScraper(auth)
    
    # Determine if input is URL or ID
    if url_or_id.startswith('http'):
        if include_attachments:
            # Extract page ID first
            page_data = await scraper.scrape_page_by_url(url_or_id)
            if page_data['extraction_successful']:
                page_id = page_data['metadata']['page_id']
                result = await scraper.scrape_with_attachments(page_id)
            else:
                result = page_data
        else:
            result = await scraper.scrape_page_by_url(url_or_id)
    else:
        # It's a page ID
        if include_attachments:
            result = await scraper.scrape_with_attachments(url_or_id)
        else:
            result = await scraper.scrape_page_by_id(url_or_id)
            
    # Ingest to database if requested
    if ingest_to_db and result['extraction_successful']:
        ingestion_result = ingest_confluence_to_db(result, keywords)
        result['database_ingestion'] = ingestion_result
        
    return result


async def scrape_confluence_space(
    space_key: str,
    auth: Optional[ConfluenceAuth] = None,
    config: Optional[Dict[str, Any]] = None,
    max_pages: int = 100,
    max_depth: Optional[int] = None,
    include_attachments: bool = False,
    page_filter: Optional[Dict[str, Any]] = None,
    ingest_to_db: bool = False,
    keywords: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Scrape all pages in a Confluence space
    
    Args:
        space_key: The Confluence space key
        auth: Optional ConfluenceAuth instance
        config: Optional configuration dict
        max_pages: Maximum number of pages to scrape
        max_depth: Maximum depth to traverse
        include_attachments: Whether to fetch attachments
        page_filter: Filter criteria dict (include_patterns, exclude_patterns, labels, modified_after)
        ingest_to_db: Whether to ingest to database
        keywords: Keywords for database ingestion
        
    Returns:
        List of scraped pages
    """
    # Setup authentication
    if not auth:
        if not config:
            config = load_confluence_config()
        auth = create_confluence_auth(config['base_url'], config)
        
    scraper = ConfluenceScraper(auth)
    crawler = ConfluenceCrawler(auth, scraper)
    
    # Create page filter if criteria provided
    filter_func = None
    if page_filter:
        filter_func = create_page_filter(**page_filter)
        
    # Crawl the space
    results = await crawler.crawl_space(
        space_key=space_key,
        max_pages=max_pages,
        max_depth=max_depth,
        page_filter=filter_func,
        include_attachments=include_attachments
    )
    
    # Ingest to database if requested
    if ingest_to_db:
        for page in results:
            if page['extraction_successful']:
                ingestion_result = ingest_confluence_to_db(page, keywords)
                page['database_ingestion'] = ingestion_result
                
    # Log statistics
    stats = crawler.get_crawl_stats()
    logger.info(f"Space crawl complete: {stats}")
    
    return results


async def scrape_confluence_search(
    query: str,
    auth: Optional[ConfluenceAuth] = None,
    config: Optional[Dict[str, Any]] = None,
    max_pages: int = 50,
    crawl_children: bool = False,
    max_child_depth: int = 1,
    ingest_to_db: bool = False,
    keywords: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search and scrape Confluence pages using CQL
    
    Args:
        query: CQL query string (e.g., "text ~ 'python' AND space = 'DEV'")
        auth: Optional ConfluenceAuth instance
        config: Optional configuration dict
        max_pages: Maximum number of pages to return
        crawl_children: Whether to also crawl child pages
        max_child_depth: Maximum depth for child crawling
        ingest_to_db: Whether to ingest to database
        keywords: Keywords for database ingestion
        
    Returns:
        List of scraped pages matching the search
    """
    # Setup authentication
    if not auth:
        if not config:
            config = load_confluence_config()
        auth = create_confluence_auth(config['base_url'], config)
        
    scraper = ConfluenceScraper(auth)
    crawler = ConfluenceCrawler(auth, scraper)
    
    # Search and crawl
    results = await crawler.search_and_crawl(
        cql_query=query,
        max_pages=max_pages,
        crawl_children=crawl_children,
        max_child_depth=max_child_depth
    )
    
    # Ingest to database if requested
    if ingest_to_db:
        for page in results:
            if page['extraction_successful']:
                ingestion_result = ingest_confluence_to_db(page, keywords)
                page['database_ingestion'] = ingestion_result
                
    return results


async def scrape_confluence_with_config(
    operation: str,
    target: str,
    config_file: Optional[str] = None,
    **kwargs
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Scrape Confluence using configuration file
    
    Args:
        operation: Type of operation ('page', 'space', 'search')
        target: Target identifier (URL/ID for page, key for space, query for search)
        config_file: Path to configuration file (uses default if not provided)
        **kwargs: Additional arguments passed to specific scrape functions
        
    Returns:
        Scraped content (single page or list of pages)
    """
    # Load configuration
    if config_file:
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = load_confluence_config()
        
    # Create auth from config
    auth = create_confluence_auth(config['base_url'], config)
    
    # Test authentication
    if not auth.test_authentication():
        raise RuntimeError("Confluence authentication failed. Please check your credentials.")
        
    # Execute appropriate operation
    if operation == 'page':
        return await scrape_confluence_page(target, auth=auth, config=config, **kwargs)
    elif operation == 'space':
        return await scrape_confluence_space(target, auth=auth, config=config, **kwargs)
    elif operation == 'search':
        return await scrape_confluence_search(target, auth=auth, config=config, **kwargs)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def load_confluence_config() -> Dict[str, Any]:
    """
    Load Confluence configuration from the main config file
    
    Returns:
        Confluence configuration dictionary
    """
    try:
        # Load main config
        main_config = load_cli_config_and_ensure_existence()
        
        # Extract Confluence config or use defaults
        confluence_config = main_config.get('confluence', {})
        
        # Set defaults
        defaults = {
            'base_url': os.getenv('CONFLUENCE_BASE_URL', ''),
            'auth_method': os.getenv('CONFLUENCE_AUTH_METHOD', 'api_token'),
            'username': os.getenv('CONFLUENCE_USERNAME', ''),
            'api_token': os.getenv('CONFLUENCE_API_TOKEN', ''),
            'space_keys': []
        }
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in confluence_config:
                confluence_config[key] = value
                
        return confluence_config
        
    except Exception as e:
        logger.error(f"Error loading Confluence config: {str(e)}")
        # Return minimal config from environment
        return {
            'base_url': os.getenv('CONFLUENCE_BASE_URL', ''),
            'auth_method': os.getenv('CONFLUENCE_AUTH_METHOD', 'api_token'),
            'username': os.getenv('CONFLUENCE_USERNAME', ''),
            'api_token': os.getenv('CONFLUENCE_API_TOKEN', '')
        }


def ingest_confluence_to_db(page_data: Dict[str, Any], keywords: Optional[str] = None) -> str:
    """
    Ingest Confluence page data to the media database
    
    Args:
        page_data: Scraped Confluence page data
        keywords: Optional keywords for categorization
        
    Returns:
        Ingestion result message
    """
    try:
        # Extract required fields
        url = page_data.get('url', '')
        title = page_data.get('title', 'Untitled')
        
        # Build author string from metadata
        metadata = page_data.get('metadata', {})
        author = metadata.get('created_by', 'Unknown')
        if metadata.get('modified_by') and metadata['modified_by'] != author:
            author += f", {metadata['modified_by']}"
            
        # Use content with metadata
        content = page_data.get('content_with_meta', page_data.get('content', ''))
        
        # Add Confluence-specific keywords
        confluence_keywords = []
        if keywords:
            confluence_keywords.append(keywords)
            
        # Add space key as keyword
        if metadata.get('space_key'):
            confluence_keywords.append(f"space:{metadata['space_key']}")
            
        # Add labels as keywords
        if metadata.get('labels'):
            confluence_keywords.extend([f"label:{label}" for label in metadata['labels']])
            
        # Add hierarchy as keyword
        if metadata.get('ancestors'):
            hierarchy = format_page_hierarchy(metadata['ancestors'], title)
            confluence_keywords.append(f"path:{hierarchy}")
            
        combined_keywords = ', '.join(confluence_keywords)
        
        # Use modification date or current date
        ingestion_date = metadata.get('last_modified', datetime.now().strftime('%Y-%m-%d'))
        
        # Ingest to database
        result = ingest_article_to_db_new(
            url=url,
            title=title,
            author=author,
            content=content,
            keywords=combined_keywords,
            ingestion_date=ingestion_date,
            custom_prompt=None,
            system_prompt=None
        )
        
        logger.info(f"Ingested Confluence page '{title}' to database: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Failed to ingest Confluence page to database: {str(e)}"
        logger.error(error_msg)
        return error_msg


# Synchronous wrappers for CLI usage
def sync_scrape_confluence_page(url_or_id: str, **kwargs) -> Dict[str, Any]:
    """Synchronous wrapper for scrape_confluence_page"""
    return asyncio.run(scrape_confluence_page(url_or_id, **kwargs))


def sync_scrape_confluence_space(space_key: str, **kwargs) -> List[Dict[str, Any]]:
    """Synchronous wrapper for scrape_confluence_space"""
    return asyncio.run(scrape_confluence_space(space_key, **kwargs))


def sync_scrape_confluence_search(query: str, **kwargs) -> List[Dict[str, Any]]:
    """Synchronous wrapper for scrape_confluence_search"""
    return asyncio.run(scrape_confluence_search(query, **kwargs))

#
# End of confluence_main.py
#######################################################################################################################