# confluence_crawler.py
#
# Crawler for traversing Confluence spaces and page hierarchies
#
# Imports
import asyncio
from typing import Set, List, Dict, Any, Optional, Callable
from urllib.parse import urljoin
import time
#
# Third-party imports
from loguru import logger
#
# Local imports
from .confluence_auth import ConfluenceAuth
from .confluence_scraper import ConfluenceScraper
#
#######################################################################################################################
#
# Classes and Functions

class ConfluenceCrawler:
    """
    Crawler for systematically traversing Confluence spaces and pages
    """
    
    def __init__(self, auth: ConfluenceAuth, scraper: Optional[ConfluenceScraper] = None):
        """
        Initialize Confluence crawler
        
        Args:
            auth: Configured ConfluenceAuth instance
            scraper: Optional ConfluenceScraper instance (will create one if not provided)
        """
        self.auth = auth
        self.scraper = scraper or ConfluenceScraper(auth)
        self.visited_pages: Set[str] = set()
        self.failed_pages: Set[str] = set()
        
    async def crawl_space(
        self,
        space_key: str,
        max_pages: int = 100,
        max_depth: Optional[int] = None,
        page_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        include_attachments: bool = False,
        follow_links: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Crawl all pages in a Confluence space
        
        Args:
            space_key: The Confluence space key
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth to traverse (None for unlimited)
            page_filter: Optional function to filter pages
            include_attachments: Whether to fetch attachments for each page
            follow_links: Whether to follow links to pages in other spaces
            
        Returns:
            List of scraped page data
        """
        logger.info(f"Starting crawl of space {space_key} (max_pages={max_pages}, max_depth={max_depth})")
        
        scraped_pages = []
        pages_to_process = []
        
        # Get root pages in space
        root_pages = await self._get_space_root_pages(space_key)
        
        # Add root pages to processing queue with depth 0
        for page in root_pages:
            pages_to_process.append((page['id'], 0))
            
        while pages_to_process and len(scraped_pages) < max_pages:
            page_id, depth = pages_to_process.pop(0)
            
            # Skip if already visited
            if page_id in self.visited_pages:
                continue
                
            # Skip if max depth exceeded
            if max_depth is not None and depth > max_depth:
                continue
                
            self.visited_pages.add(page_id)
            
            try:
                # Scrape the page
                if include_attachments:
                    page_data = await self.scraper.scrape_with_attachments(page_id)
                else:
                    page_data = await self.scraper.scrape_page_by_id(page_id)
                    
                if page_data['extraction_successful']:
                    # Apply filter if provided
                    if page_filter is None or page_filter(page_data):
                        scraped_pages.append(page_data)
                        logger.info(f"Scraped page: {page_data['title']} (depth={depth})")
                        
                    # Get child pages
                    child_pages = await self._get_child_pages(page_id)
                    for child in child_pages:
                        if child['id'] not in self.visited_pages:
                            pages_to_process.append((child['id'], depth + 1))
                            
                    # Handle links if requested
                    if follow_links:
                        linked_pages = await self._extract_linked_pages(page_data)
                        for linked_id in linked_pages:
                            if linked_id not in self.visited_pages:
                                pages_to_process.append((linked_id, depth + 1))
                else:
                    self.failed_pages.add(page_id)
                    logger.warning(f"Failed to scrape page {page_id}")
                    
            except Exception as e:
                logger.error(f"Error crawling page {page_id}: {str(e)}")
                self.failed_pages.add(page_id)
                
            # Add a small delay to avoid overwhelming the server
            await asyncio.sleep(0.5)
            
        logger.info(f"Crawl complete. Scraped {len(scraped_pages)} pages, failed {len(self.failed_pages)}")
        return scraped_pages
        
    async def crawl_page_tree(
        self,
        root_page_id: str,
        max_depth: Optional[int] = None,
        include_siblings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Crawl a page and all its descendants
        
        Args:
            root_page_id: ID of the root page to start from
            max_depth: Maximum depth to traverse
            include_siblings: Whether to include sibling pages
            
        Returns:
            List of scraped page data
        """
        scraped_pages = []
        
        async def crawl_recursive(page_id: str, current_depth: int):
            if max_depth is not None and current_depth > max_depth:
                return
                
            if page_id in self.visited_pages:
                return
                
            self.visited_pages.add(page_id)
            
            # Scrape current page
            page_data = await self.scraper.scrape_page_by_id(page_id)
            if page_data['extraction_successful']:
                scraped_pages.append(page_data)
                
                # Get child pages
                child_pages = await self._get_child_pages(page_id)
                
                # Crawl children recursively
                for child in child_pages:
                    await crawl_recursive(child['id'], current_depth + 1)
                    
        # Start crawling from root
        await crawl_recursive(root_page_id, 0)
        
        # Include siblings if requested
        if include_siblings:
            parent_id = await self._get_parent_page_id(root_page_id)
            if parent_id:
                siblings = await self._get_child_pages(parent_id)
                for sibling in siblings:
                    if sibling['id'] != root_page_id and sibling['id'] not in self.visited_pages:
                        await crawl_recursive(sibling['id'], 1)
                        
        return scraped_pages
        
    async def search_and_crawl(
        self,
        cql_query: str,
        max_pages: int = 100,
        crawl_children: bool = True,
        max_child_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Search for pages using CQL and optionally crawl their children
        
        Args:
            cql_query: Confluence Query Language query
            max_pages: Maximum number of pages to return
            crawl_children: Whether to also crawl child pages
            max_child_depth: Maximum depth for child crawling
            
        Returns:
            List of scraped page data
        """
        # First, get pages matching the search
        search_results = await self.scraper.scrape_by_cql(cql_query, max_pages)
        
        all_pages = search_results.copy()
        
        if crawl_children:
            for page in search_results:
                if page['extraction_successful']:
                    # Crawl children of each search result
                    child_pages = await self.crawl_page_tree(
                        page['metadata']['page_id'],
                        max_depth=max_child_depth,
                        include_siblings=False
                    )
                    
                    # Add children that aren't already in results
                    existing_ids = {p['metadata']['page_id'] for p in all_pages}
                    for child in child_pages:
                        if child['metadata']['page_id'] not in existing_ids:
                            all_pages.append(child)
                            
        return all_pages[:max_pages]
        
    async def _get_space_root_pages(self, space_key: str) -> List[Dict[str, str]]:
        """Get root-level pages in a space"""
        try:
            response = self.auth.make_request(
                'GET',
                '/rest/api/content',
                params={
                    'spaceKey': space_key,
                    'type': 'page',
                    'depth': 'root',
                    'limit': 100,
                    'expand': 'ancestors'
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                pages = []
                for result in data.get('results', []):
                    # Root pages have no ancestors
                    if not result.get('ancestors', []):
                        pages.append({
                            'id': result['id'],
                            'title': result['title']
                        })
                return pages
            else:
                logger.error(f"Failed to get root pages for space {space_key}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting root pages: {str(e)}")
            return []
            
    async def _get_child_pages(self, page_id: str) -> List[Dict[str, str]]:
        """Get child pages of a given page"""
        try:
            response = self.auth.make_request(
                'GET',
                f'/rest/api/content/{page_id}/child/page',
                params={'limit': 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                return [
                    {'id': result['id'], 'title': result['title']}
                    for result in data.get('results', [])
                ]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting child pages for {page_id}: {str(e)}")
            return []
            
    async def _get_parent_page_id(self, page_id: str) -> Optional[str]:
        """Get parent page ID"""
        try:
            response = self.auth.make_request(
                'GET',
                f'/rest/api/content/{page_id}',
                params={'expand': 'ancestors'}
            )
            
            if response.status_code == 200:
                data = response.json()
                ancestors = data.get('ancestors', [])
                if ancestors:
                    # The immediate parent is the last ancestor
                    return ancestors[-1]['id']
            return None
            
        except Exception as e:
            logger.error(f"Error getting parent page: {str(e)}")
            return None
            
    async def _extract_linked_pages(self, page_data: Dict[str, Any]) -> Set[str]:
        """Extract linked page IDs from page content"""
        linked_pages = set()
        
        # This would require parsing the content HTML to find links
        # For now, return empty set - this could be enhanced later
        # to parse ac:link elements and extract page references
        
        return linked_pages
        
    def get_crawl_stats(self) -> Dict[str, Any]:
        """Get statistics about the crawl"""
        return {
            'pages_visited': len(self.visited_pages),
            'pages_failed': len(self.failed_pages),
            'success_rate': len(self.visited_pages) / (len(self.visited_pages) + len(self.failed_pages))
            if (len(self.visited_pages) + len(self.failed_pages)) > 0 else 0
        }
        
    def reset(self):
        """Reset crawler state"""
        self.visited_pages.clear()
        self.failed_pages.clear()


def create_page_filter(
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    modified_after: Optional[str] = None
) -> Callable[[Dict[str, Any]], bool]:
    """
    Create a page filter function based on various criteria
    
    Args:
        include_patterns: List of patterns that page title must match (any)
        exclude_patterns: List of patterns that page title must not match (any)
        labels: List of labels that page must have (any)
        modified_after: ISO date string - only include pages modified after this date
        
    Returns:
        Filter function for use with crawler
    """
    def page_filter(page_data: Dict[str, Any]) -> bool:
        # Check title patterns
        title = page_data.get('title', '')
        
        if include_patterns:
            if not any(pattern.lower() in title.lower() for pattern in include_patterns):
                return False
                
        if exclude_patterns:
            if any(pattern.lower() in title.lower() for pattern in exclude_patterns):
                return False
                
        # Check labels
        if labels:
            page_labels = page_data.get('metadata', {}).get('labels', [])
            if not any(label in page_labels for label in labels):
                return False
                
        # Check modification date
        if modified_after:
            last_modified = page_data.get('metadata', {}).get('last_modified', '')
            if last_modified and last_modified < modified_after:
                return False
                
        return True
        
    return page_filter

#
# End of confluence_crawler.py
#######################################################################################################################