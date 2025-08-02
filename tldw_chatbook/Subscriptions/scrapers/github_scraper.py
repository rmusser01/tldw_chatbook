# github_scraper.py
# Description: GitHub-specific scraping pipeline using Atom feeds
#
# This scraper handles GitHub content via Atom feeds, supporting:
# - Repository releases
# - Repository commits
# - User activity
# - Repository issues (via API)
# - Pull requests (via API)
#
# Imports
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, quote
import json
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
# GitHub Scraping Pipeline
#
########################################################################################################################

class GitHubScrapingPipeline(BaseScrapingPipeline):
    """
    Scraping pipeline for GitHub content via Atom feeds and API.
    
    Supports multiple GitHub feed types:
    - Releases: /owner/repo/releases.atom
    - Commits: /owner/repo/commits.atom or /owner/repo/commits/branch.atom
    - Tags: /owner/repo/tags.atom
    - User activity: /username.atom
    - Issues/PRs: via API with optional authentication
    """
    
    GITHUB_BASE_URL = "https://github.com"
    GITHUB_API_BASE = "https://api.github.com"
    
    def __init__(self, config: ScrapingConfig):
        """Initialize GitHub scraper with configuration."""
        super().__init__(config)
        
        # GitHub-specific options
        self.feed_type = config.options.get('feed_type', 'releases')  # releases, commits, tags, activity, issues, pulls
        self.branch = config.options.get('branch', 'main')
        self.include_pre_releases = config.options.get('include_pre_releases', False)
        self.include_drafts = config.options.get('include_drafts', False)
        self.labels_filter = config.options.get('labels_filter', [])  # For issues/PRs
        self.state_filter = config.options.get('state', 'open')  # open, closed, all
        self.author_filter = config.options.get('author_filter', [])
        self.exclude_authors = config.options.get('exclude_authors', [])
        self.github_token = config.options.get('github_token')  # For API access
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate GitHub-specific configuration."""
        # Check URL format
        url = self.config.url
        
        # Extract owner/repo from URL
        match = re.match(r'https?://github\.com/([^/]+)/([^/]+)', url)
        if not match:
            return False, "Invalid GitHub URL format. Expected: https://github.com/owner/repo"
        
        self.owner = match.group(1)
        self.repo = match.group(2).rstrip('.git')
        
        # Validate feed type
        valid_types = ['releases', 'commits', 'tags', 'activity', 'issues', 'pulls']
        if self.feed_type not in valid_types:
            return False, f"Invalid feed type. Must be one of: {', '.join(valid_types)}"
        
        # Check if API token is required
        if self.feed_type in ['issues', 'pulls'] and not self.github_token:
            logger.warning("GitHub API token recommended for issues/pulls to avoid rate limits")
        
        return True, None
    
    def get_rate_limit(self) -> int:
        """Get rate limit for GitHub."""
        # GitHub API rate limits
        if self.github_token:
            return 5000  # 5000 requests per hour with auth
        else:
            return 60    # 60 requests per hour without auth
    
    async def fetch_content(self, url: str) -> str:
        """Fetch GitHub feed content."""
        if self.feed_type in ['issues', 'pulls']:
            return await self._fetch_via_api()
        else:
            return await self._fetch_atom_feed()
    
    async def _fetch_atom_feed(self) -> str:
        """Fetch GitHub Atom feed."""
        # Build feed URL based on type
        if self.feed_type == 'releases':
            feed_url = f"{self.GITHUB_BASE_URL}/{self.owner}/{self.repo}/releases.atom"
        elif self.feed_type == 'commits':
            if self.branch and self.branch != 'main':
                feed_url = f"{self.GITHUB_BASE_URL}/{self.owner}/{self.repo}/commits/{self.branch}.atom"
            else:
                feed_url = f"{self.GITHUB_BASE_URL}/{self.owner}/{self.repo}/commits.atom"
        elif self.feed_type == 'tags':
            feed_url = f"{self.GITHUB_BASE_URL}/{self.owner}/{self.repo}/tags.atom"
        elif self.feed_type == 'activity':
            # For user activity, owner is the username
            feed_url = f"{self.GITHUB_BASE_URL}/{self.owner}.atom"
        else:
            raise ValueError(f"Unsupported feed type for Atom: {self.feed_type}")
        
        logger.debug(f"Fetching GitHub Atom feed from: {feed_url}")
        
        # Fetch feed
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = self.get_headers()
            headers['Accept'] = 'application/atom+xml, application/xml'
            
            response = await client.get(feed_url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            log_counter("github_scraper_fetch", labels={
                "status": "success",
                "feed_type": self.feed_type,
                "method": "atom"
            })
            
            return response.text
    
    async def _fetch_via_api(self) -> str:
        """Fetch GitHub data via API."""
        # Build API URL
        if self.feed_type == 'issues':
            api_url = f"{self.GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/issues"
        elif self.feed_type == 'pulls':
            api_url = f"{self.GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/pulls"
        else:
            raise ValueError(f"Unsupported feed type for API: {self.feed_type}")
        
        # Build query parameters
        params = {
            'state': self.state_filter,
            'sort': 'created',
            'direction': 'desc',
            'per_page': 30
        }
        
        if self.labels_filter:
            params['labels'] = ','.join(self.labels_filter)
        
        logger.debug(f"Fetching GitHub API from: {api_url}")
        
        # Fetch data
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = self.get_headers()
            headers['Accept'] = 'application/vnd.github.v3+json'
            
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            
            response = await client.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            
            log_counter("github_scraper_fetch", labels={
                "status": "success",
                "feed_type": self.feed_type,
                "method": "api"
            })
            
            # Convert to JSON for consistent handling
            return json.dumps(response.json())
    
    def parse_content(self, raw_content: str, url: str) -> List[ScrapedItem]:
        """Parse GitHub content into items."""
        if self.feed_type in ['issues', 'pulls']:
            return self._parse_api_response(raw_content)
        else:
            return self._parse_atom_feed(raw_content)
    
    def _parse_atom_feed(self, raw_content: str) -> List[ScrapedItem]:
        """Parse GitHub Atom feed."""
        items = []
        
        try:
            # Parse Atom XML safely
            root = ET.fromstring(raw_content)
            
            # Define Atom namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            # Find all entries
            for entry in root.findall('.//atom:entry', ns):
                scraped_item = self._parse_atom_entry(entry, ns)
                
                if scraped_item and self._should_include_item(scraped_item):
                    items.append(scraped_item)
            
            logger.info(f"Parsed {len(items)} GitHub items from Atom feed")
            log_counter("github_scraper_items", labels={
                "feed_type": self.feed_type,
                "item_count": str(len(items))
            })
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse GitHub Atom feed: {str(e)}")
            log_counter("github_scraper_errors", labels={
                "error_type": "parse_error"
            })
            raise
        
        return items
    
    def _parse_atom_entry(self, entry, ns: dict) -> Optional[ScrapedItem]:
        """Parse a single Atom entry."""
        try:
            # Extract basic fields
            title = self._get_atom_text(entry, 'atom:title', ns)
            link_elem = entry.find('atom:link[@rel="alternate"]', ns)
            link = link_elem.get('href') if link_elem is not None else None
            
            published = self._get_atom_text(entry, 'atom:published', ns)
            updated = self._get_atom_text(entry, 'atom:updated', ns)
            content = self._get_atom_text(entry, 'atom:content', ns)
            
            if not title or not link:
                return None
            
            # Parse dates
            published_date = None
            if published:
                try:
                    published_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                except ValueError:
                    logger.warning(f"Could not parse date: {published}")
            
            # Extract author
            author = None
            author_elem = entry.find('.//atom:author/atom:name', ns)
            if author_elem is not None:
                author = author_elem.text
            
            # Extract metadata based on feed type
            metadata = self._extract_github_metadata(entry, ns)
            metadata['author'] = author
            metadata['feed_type'] = self.feed_type
            
            # Clean content
            clean_content = self._clean_github_content(content)
            
            return ScrapedItem(
                title=title,
                url=link,
                content=clean_content,
                published_date=published_date,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing GitHub Atom entry: {str(e)}")
            return None
    
    def _parse_api_response(self, raw_content: str) -> List[ScrapedItem]:
        """Parse GitHub API JSON response."""
        items = []
        
        try:
            data = json.loads(raw_content)
            
            for item_data in data:
                scraped_item = self._parse_api_item(item_data)
                
                if scraped_item and self._should_include_item(scraped_item):
                    items.append(scraped_item)
            
            logger.info(f"Parsed {len(items)} GitHub items from API")
            log_counter("github_scraper_items", labels={
                "feed_type": self.feed_type,
                "item_count": str(len(items))
            })
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GitHub API response: {str(e)}")
            raise
        
        return items
    
    def _parse_api_item(self, item_data: dict) -> Optional[ScrapedItem]:
        """Parse a single API item (issue or PR)."""
        try:
            title = item_data.get('title', '')
            url = item_data.get('html_url', '')
            body = item_data.get('body', '')
            
            if not title or not url:
                return None
            
            # Parse dates
            created_at = item_data.get('created_at')
            published_date = None
            if created_at:
                published_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            
            # Extract metadata
            metadata = {
                'author': item_data.get('user', {}).get('login'),
                'state': item_data.get('state'),
                'number': item_data.get('number'),
                'comments': item_data.get('comments', 0),
                'labels': [label['name'] for label in item_data.get('labels', [])],
                'feed_type': self.feed_type
            }
            
            # Add PR-specific metadata
            if self.feed_type == 'pulls':
                metadata['merged'] = item_data.get('merged', False)
                metadata['draft'] = item_data.get('draft', False)
            
            return ScrapedItem(
                title=f"#{metadata['number']}: {title}",
                url=url,
                content=body or f"No description provided for this {self.feed_type[:-1]}",
                published_date=published_date,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing GitHub API item: {str(e)}")
            return None
    
    def _extract_github_metadata(self, entry, ns: dict) -> Dict[str, Any]:
        """Extract GitHub-specific metadata from Atom entry."""
        metadata = {}
        
        # Try to extract version/tag info for releases
        if self.feed_type == 'releases':
            # Look for version in title or content
            title = self._get_atom_text(entry, 'atom:title', ns)
            if title:
                version_match = re.search(r'v?(\d+\.\d+\.\d+)', title)
                if version_match:
                    metadata['version'] = version_match.group(1)
        
        # Extract commit SHA for commits
        elif self.feed_type == 'commits':
            id_elem = entry.find('atom:id', ns)
            if id_elem is not None and id_elem.text:
                # GitHub uses tag:github.com,2008:Grit::Commit/SHA format
                sha_match = re.search(r'Commit/([a-f0-9]+)$', id_elem.text)
                if sha_match:
                    metadata['commit_sha'] = sha_match.group(1)
        
        return metadata
    
    def _clean_github_content(self, content: str) -> str:
        """Clean GitHub content."""
        if not content:
            return ""
        
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Convert to text
            text = soup.get_text(separator='\n', strip=True)
            
            # Limit length for commits (they can be very long)
            if self.feed_type == 'commits' and len(text) > 1000:
                text = text[:1000] + '...'
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning GitHub content: {str(e)}")
            return content
    
    def _should_include_item(self, item: ScrapedItem) -> bool:
        """Check if item passes configured filters."""
        metadata = item.metadata
        
        # Check author filters
        author = metadata.get('author', '')
        if self.author_filter and author:
            if author not in self.author_filter:
                return False
        
        if self.exclude_authors and author:
            if author in self.exclude_authors:
                return False
        
        # Check release filters
        if self.feed_type == 'releases':
            # Check for pre-release (simple heuristic)
            if not self.include_pre_releases:
                title_lower = item.title.lower()
                if any(term in title_lower for term in ['alpha', 'beta', 'rc', 'pre-release']):
                    return False
        
        # Check PR filters
        elif self.feed_type == 'pulls':
            if not self.include_drafts and metadata.get('draft', False):
                return False
        
        # Check label filters for issues/PRs
        if self.feed_type in ['issues', 'pulls'] and self.labels_filter:
            item_labels = metadata.get('labels', [])
            if not any(label in item_labels for label in self.labels_filter):
                return False
        
        return True
    
    def _get_atom_text(self, element, path: str, ns: dict) -> Optional[str]:
        """Safely extract text from Atom XML element."""
        elem = element.find(path, ns)
        if elem is not None:
            # Handle both text content and type="html" content
            if elem.text:
                return elem.text.strip()
            elif elem.get('type') == 'html':
                # Get all text including tags
                return ''.join(elem.itertext()).strip()
        return None


# End of github_scraper.py