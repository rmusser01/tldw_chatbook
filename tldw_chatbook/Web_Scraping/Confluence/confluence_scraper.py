# confluence_scraper.py
#
# Scraper for Confluence pages extending the existing Article Scraper
#
# Imports
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import re
#
# Third-party imports
from loguru import logger
import requests
from bs4 import BeautifulSoup
#
# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Article_Scraper.scraper import Scraper
from Article_Scraper.config import ScraperConfig
from Article_Scraper.utils import ContentMetadataHandler
from .confluence_auth import ConfluenceAuth
from .confluence_utils import convert_confluence_to_markdown, extract_confluence_metadata
#
#######################################################################################################################
#
# Classes and Functions

class ConfluenceScraper(Scraper):
    """
    Confluence-specific scraper that extends the base Scraper class
    """
    
    def __init__(self, auth: ConfluenceAuth, config: Optional[ScraperConfig] = None):
        """
        Initialize Confluence scraper
        
        Args:
            auth: Configured ConfluenceAuth instance
            config: Optional scraper configuration
        """
        super().__init__(config)
        self.auth = auth
        self.base_url = auth.base_url
        self.api_base = f"{self.base_url}/rest/api"
        
    async def scrape_page_by_id(self, page_id: str) -> Dict[str, Any]:
        """
        Scrape a Confluence page by its ID
        
        Args:
            page_id: The Confluence page ID
            
        Returns:
            Dictionary containing scraped page data
        """
        try:
            # Fetch page content via API
            response = self.auth.make_request(
                'GET',
                f'/rest/api/content/{page_id}',
                params={
                    'expand': 'body.storage,version,space,history,ancestors,metadata.labels'
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch page {page_id}: {response.status_code}")
                return {
                    'extraction_successful': False,
                    'error': f'API returned status code {response.status_code}'
                }
                
            page_data = response.json()
            
            # Extract content and metadata
            result = self._process_page_data(page_data)
            
            logger.info(f"Successfully scraped Confluence page: {result.get('title', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error scraping Confluence page {page_id}: {str(e)}")
            return {
                'extraction_successful': False,
                'error': str(e)
            }
            
    async def scrape_page_by_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape a Confluence page by its URL
        
        Args:
            url: The Confluence page URL
            
        Returns:
            Dictionary containing scraped page data
        """
        # Extract page ID from URL
        page_id = self._extract_page_id_from_url(url)
        if not page_id:
            return {
                'extraction_successful': False,
                'error': 'Could not extract page ID from URL'
            }
            
        return await self.scrape_page_by_id(page_id)
        
    async def scrape_space(self, space_key: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Scrape all pages in a Confluence space
        
        Args:
            space_key: The Confluence space key
            limit: Maximum number of pages to scrape
            
        Returns:
            List of scraped page data
        """
        pages = []
        start = 0
        
        while len(pages) < limit:
            try:
                # Fetch pages in space
                response = self.auth.make_request(
                    'GET',
                    f'/rest/api/content',
                    params={
                        'spaceKey': space_key,
                        'type': 'page',
                        'start': start,
                        'limit': min(25, limit - len(pages)),  # API limit is typically 25
                        'expand': 'body.storage,version,space,history,ancestors,metadata.labels'
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch pages from space {space_key}: {response.status_code}")
                    break
                    
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    break
                    
                # Process each page
                for page_data in results:
                    page_result = self._process_page_data(page_data)
                    if page_result['extraction_successful']:
                        pages.append(page_result)
                        
                # Check if there are more pages
                if 'next' not in data.get('_links', {}):
                    break
                    
                start += len(results)
                
            except Exception as e:
                logger.error(f"Error scraping space {space_key}: {str(e)}")
                break
                
        logger.info(f"Scraped {len(pages)} pages from space {space_key}")
        return pages
        
    async def scrape_by_cql(self, cql_query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Scrape pages using Confluence Query Language (CQL)
        
        Args:
            cql_query: CQL query string
            limit: Maximum number of pages to scrape
            
        Returns:
            List of scraped page data
        """
        pages = []
        start = 0
        
        while len(pages) < limit:
            try:
                response = self.auth.make_request(
                    'GET',
                    '/rest/api/content/search',
                    params={
                        'cql': cql_query,
                        'start': start,
                        'limit': min(25, limit - len(pages)),
                        'expand': 'body.storage,version,space,history,ancestors,metadata.labels'
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"CQL search failed: {response.status_code}")
                    break
                    
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    break
                    
                for page_data in results:
                    page_result = self._process_page_data(page_data)
                    if page_result['extraction_successful']:
                        pages.append(page_result)
                        
                if 'next' not in data.get('_links', {}):
                    break
                    
                start += len(results)
                
            except Exception as e:
                logger.error(f"Error with CQL search: {str(e)}")
                break
                
        logger.info(f"CQL search returned {len(pages)} pages")
        return pages
        
    def _process_page_data(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw Confluence API page data into standardized format
        
        Args:
            page_data: Raw page data from Confluence API
            
        Returns:
            Processed page data
        """
        try:
            # Extract basic metadata
            page_id = page_data.get('id', '')
            title = page_data.get('title', 'Untitled')
            
            # Extract content
            body = page_data.get('body', {}).get('storage', {})
            content_html = body.get('value', '')
            
            # Convert to markdown
            content_markdown = convert_confluence_to_markdown(content_html)
            
            # Extract additional metadata
            metadata = extract_confluence_metadata(page_data)
            
            # Build page URL
            page_url = f"{self.base_url}/spaces/{metadata['space_key']}/pages/{page_id}"
            
            # Format content with metadata
            formatted_content = ContentMetadataHandler.format_content_with_metadata(
                url=page_url,
                content=content_markdown,
                pipeline="Confluence API",
                additional_metadata={
                    'page_id': page_id,
                    'space_key': metadata['space_key'],
                    'space_name': metadata['space_name'],
                    'version': metadata['version'],
                    'last_modified': metadata['last_modified'],
                    'created_by': metadata['created_by'],
                    'modified_by': metadata['modified_by'],
                    'labels': metadata['labels'],
                    'ancestors': metadata['ancestors']
                }
            )
            
            return {
                'url': page_url,
                'title': title,
                'author': metadata['created_by'],
                'date': metadata['created_date'],
                'content': content_markdown,
                'content_with_meta': formatted_content,
                'extraction_successful': True,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing page data: {str(e)}")
            return {
                'extraction_successful': False,
                'error': str(e)
            }
            
    def _extract_page_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract page ID from Confluence URL
        
        Args:
            url: Confluence page URL
            
        Returns:
            Page ID or None if not found
        """
        # Try different URL patterns
        patterns = [
            r'/pages/(\d+)',  # New URL format
            r'pageId=(\d+)',  # Query parameter format
            r'/display/[^/]+/.*\?.*pageId=(\d+)',  # Legacy format with pageId
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
                
        # If no ID found, try to fetch the page and get ID from response
        try:
            # For legacy URLs, we need to resolve them
            if '/display/' in url:
                response = self.auth.session.get(url, allow_redirects=True)
                if response.status_code == 200:
                    # Look for page ID in the HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    meta_tag = soup.find('meta', {'name': 'ajs-page-id'})
                    if meta_tag:
                        return meta_tag.get('content')
        except Exception as e:
            logger.error(f"Error extracting page ID from URL: {str(e)}")
            
        return None
        
    async def scrape_with_attachments(self, page_id: str) -> Dict[str, Any]:
        """
        Scrape a page including its attachments
        
        Args:
            page_id: Confluence page ID
            
        Returns:
            Page data with attachments information
        """
        # First scrape the page
        page_data = await self.scrape_page_by_id(page_id)
        
        if not page_data['extraction_successful']:
            return page_data
            
        # Fetch attachments
        try:
            response = self.auth.make_request(
                'GET',
                f'/rest/api/content/{page_id}/child/attachment',
                params={'expand': 'version,container'}
            )
            
            if response.status_code == 200:
                attachments_data = response.json()
                attachments = []
                
                for attachment in attachments_data.get('results', []):
                    att_info = {
                        'title': attachment.get('title'),
                        'id': attachment.get('id'),
                        'type': attachment.get('type'),
                        'size': attachment.get('extensions', {}).get('fileSize'),
                        'download_url': f"{self.base_url}{attachment.get('_links', {}).get('download', '')}"
                    }
                    attachments.append(att_info)
                    
                page_data['attachments'] = attachments
                logger.info(f"Found {len(attachments)} attachments for page {page_id}")
                
        except Exception as e:
            logger.error(f"Error fetching attachments: {str(e)}")
            page_data['attachments'] = []
            
        return page_data
        
    async def scrape_many(self, page_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple pages concurrently
        
        Args:
            page_ids: List of page IDs to scrape
            
        Returns:
            List of scraped page data
        """
        tasks = [self.scrape_page_by_id(page_id) for page_id in page_ids]
        results = await asyncio.gather(*tasks)
        return results

#
# End of confluence_scraper.py
#######################################################################################################################