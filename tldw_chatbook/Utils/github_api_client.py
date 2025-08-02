# tldw_chatbook/Utils/github_api_client.py
# Description: GitHub API client for fetching repository structure and content
#
# This module handles all GitHub API interactions for the repository file selector.

from __future__ import annotations
from typing import Optional, List, Dict, Any
import base64
import re
import os
import time
import hashlib
import sys
from urllib.parse import urlparse

import httpx
from loguru import logger

# Import config utilities
from ..config import get_cli_setting

logger = logger.bind(module="github_api_client")


class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors."""
    pass


class GitHubAPIClient:
    """Client for interacting with GitHub API."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the GitHub API client.
        
        Args:
            token: Optional GitHub personal access token for private repos
        """
        # Try to get token from: 1) parameter, 2) env var, 3) config file
        if token:
            self.token = token
        else:
            # Check environment variable first
            env_var = get_cli_setting("github", "api_token_env_var", "GITHUB_API_TOKEN")
            self.token = os.getenv(env_var)
            
            # If not in env, check config file
            if not self.token:
                config_token = get_cli_setting("github", "api_token", "")
                if config_token and not config_token.startswith("<") and not config_token.endswith(">"):
                    self.token = config_token
        
        self.base_url = "https://api.github.com"
        self._client: Optional[httpx.AsyncClient] = None
        
        # Load config settings
        self.enable_rate_limit_handling = get_cli_setting("github", "enable_rate_limit_handling", True)
        self.cache_ttl = get_cli_setting("github", "cache_ttl_seconds", 300)
        self.max_retries = get_cli_setting("github", "max_retries", 3)
        self.max_concurrent_requests = get_cli_setting("github", "max_concurrent_requests", 5)
        
        # Initialize cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_order: List[str] = []  # Track insertion order for LRU
        self.max_cache_size = get_cli_setting("github", "max_cache_entries", 100)
        self.cache_size_mb = 0
        self.max_cache_size_mb = get_cli_setting("github", "max_cache_size_mb", 50)
        
        if self.token:
            logger.info("GitHub API client initialized with authentication token")
        else:
            logger.info("GitHub API client initialized without authentication (public repos only)")
        
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "tldw-chatbook-repo-selector"
            }
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=30.0
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()
        self._cache_order.clear()
        self.cache_size_mb = 0
        logger.info("GitHub API cache cleared")
    
    def parse_github_url(self, url: str) -> tuple[str, str]:
        """Parse GitHub URL to extract owner and repo.
        
        Args:
            url: GitHub repository URL
            
        Returns:
            Tuple of (owner, repo)
            
        Raises:
            ValueError: If URL is not a valid GitHub repository URL
        """
        # Handle various GitHub URL formats
        patterns = [
            r'github\.com[/:]([^/]+)/([^/\.]+)',  # HTTPS and SSH
            r'([^/]+)/([^/]+)$'  # Simple owner/repo format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                owner, repo = match.groups()
                # Remove .git extension if present
                repo = repo.replace('.git', '')
                return owner, repo
        
        raise ValueError(f"Invalid GitHub repository URL: {url}")
    
    async def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get basic repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository information dict
        """
        url = f"{self.base_url}/repos/{owner}/{repo}"
        
        # Check cache first
        cached_data = self._get_from_cache(url)
        if cached_data is not None:
            return cached_data
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self._save_to_cache(url, data)
            return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise GitHubAPIError(f"Repository not found: {owner}/{repo}")
            elif e.response.status_code == 403:
                # Check if it's rate limit or permission issue
                remaining = e.response.headers.get('X-RateLimit-Remaining', '0')
                if remaining == '0':
                    reset_time = e.response.headers.get('X-RateLimit-Reset', '0')
                    raise GitHubAPIError(f"API rate limit exceeded. Resets at {reset_time}")
                else:
                    raise GitHubAPIError("Access denied. Private repository requires authentication.")
            else:
                raise GitHubAPIError(f"GitHub API error: {e}")
        except Exception as e:
            raise GitHubAPIError(f"Failed to fetch repository info: {e}")
    
    async def get_branches(self, owner: str, repo: str) -> List[str]:
        """Get list of branches for a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            List of branch names
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/branches"
        
        # Check cache first
        cached_data = self._get_from_cache(url)
        if cached_data is not None:
            return cached_data
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            branches = response.json()
            branch_names = [branch['name'] for branch in branches]
            
            # Cache the response
            self._save_to_cache(url, branch_names)
            return branch_names
        except Exception as e:
            logger.error(f"Failed to fetch branches: {e}")
            return ["main", "master"]  # Fallback to common branch names
    
    def _get_cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate a cache key for the given URL and parameters."""
        cache_str = url
        if params:
            cache_str += str(sorted(params.items()))
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is still valid."""
        if not cache_entry:
            return False
        timestamp = cache_entry.get('timestamp', 0)
        return (time.time() - timestamp) < self.cache_ttl
    
    def _get_from_cache(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get data from cache if valid."""
        cache_key = self._get_cache_key(url, params)
        cache_entry = self._cache.get(cache_key)
        
        if cache_entry and self._is_cache_valid(cache_entry):
            logger.debug(f"Cache hit for {url}")
            return cache_entry['data']
        
        return None
    
    def _save_to_cache(self, url: str, data: Any, params: Optional[Dict] = None) -> None:
        """Save data to cache with size limits."""
        cache_key = self._get_cache_key(url, params)
        
        # Estimate size of data (rough approximation)
        import sys
        data_size = sys.getsizeof(str(data)) / (1024 * 1024)  # Convert to MB
        
        # Check if we need to evict entries
        while (len(self._cache) >= self.max_cache_size or 
               self.cache_size_mb + data_size > self.max_cache_size_mb) and self._cache_order:
            # Remove oldest entry (LRU)
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._cache:
                old_data = self._cache.pop(oldest_key)
                old_size = sys.getsizeof(str(old_data.get('data', ''))) / (1024 * 1024)
                self.cache_size_mb -= old_size
                logger.debug(f"Evicted cache entry: {oldest_key}")
        
        # Add new entry
        self._cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        self._cache_order.append(cache_key)
        self.cache_size_mb += data_size
        
        logger.debug(f"Cached response for {url} (cache size: {len(self._cache)} entries, {self.cache_size_mb:.1f} MB)")
    
    async def get_repository_tree(
        self,
        owner: str,
        repo: str,
        branch: str = "main",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """Get repository tree structure.
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name
            recursive: Whether to fetch recursively
            
        Returns:
            List of tree items with structure:
            {
                'path': 'src/index.js',
                'type': 'blob' or 'tree',
                'size': 1234,  # for files
                'name': 'index.js'
            }
        """
        # Check cache for tree data
        tree_cache_key = f"{owner}/{repo}/tree/{branch}/recursive={recursive}"
        
        # For non-recursive calls, we'll use a different approach
        if not recursive:
            # Use the contents API for non-recursive directory listing
            return await self.get_directory_contents(owner, repo, "", branch)
        
        tree_url = f"{self.base_url}/repos/{owner}/{repo}/git/trees/{branch}?recursive={int(recursive)}"
        
        cached_tree = self._get_from_cache(tree_url)
        if cached_tree is not None:
            return cached_tree
        
        # First try to get the branch SHA
        branch_url = f"{self.base_url}/repos/{owner}/{repo}/branches/{branch}"
        
        try:
            response = await self.client.get(branch_url)
            if response.status_code == 404:
                # Try 'master' if 'main' doesn't exist
                if branch == "main":
                    return await self.get_repository_tree(owner, repo, "master", recursive)
                else:
                    raise GitHubAPIError(f"Branch not found: {branch}")
            
            response.raise_for_status()
            branch_data = response.json()
            tree_sha = branch_data['commit']['sha']
            
            # Get the tree
            tree_url = f"{self.base_url}/repos/{owner}/{repo}/git/trees/{tree_sha}"
            if recursive:
                tree_url += "?recursive=1"
            
            response = await self.client.get(tree_url)
            response.raise_for_status()
            tree_data = response.json()
            
            # Process tree items to add name field
            items = []
            for item in tree_data.get('tree', []):
                item['name'] = item['path'].split('/')[-1]
                items.append(item)
            
            # Cache the processed tree
            self._save_to_cache(tree_url, items)
            
            return items
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                # Check if it's rate limit or permission issue
                remaining = e.response.headers.get('X-RateLimit-Remaining', '0')
                if remaining == '0':
                    reset_time = e.response.headers.get('X-RateLimit-Reset', '0')
                    raise GitHubAPIError(f"API rate limit exceeded. Resets at {reset_time}")
                else:
                    raise GitHubAPIError("Access denied. Private repository requires authentication.")
            else:
                raise GitHubAPIError(f"Failed to fetch repository tree: {e}")
        except Exception as e:
            raise GitHubAPIError(f"Failed to fetch repository tree: {e}")
    
    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        branch: str = "main"
    ) -> str:
        """Get file content from repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path in repository
            branch: Branch name
            
        Returns:
            File content as string
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": branch}
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if it's a file
            if data.get('type') != 'file':
                raise GitHubAPIError(f"Path is not a file: {path}")
            
            # Decode base64 content
            content = data.get('content', '')
            decoded = base64.b64decode(content).decode('utf-8')
            
            return decoded
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise GitHubAPIError(f"File not found: {path}")
            else:
                raise GitHubAPIError(f"Failed to fetch file content: {e}")
        except Exception as e:
            raise GitHubAPIError(f"Failed to fetch file content: {e}")
    
    async def get_rate_limit(self) -> Dict[str, Any]:
        """Get current API rate limit status.
        
        Returns:
            Rate limit information
        """
        url = f"{self.base_url}/rate_limit"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch rate limit: {e}")
            return {}
    
    async def get_directory_contents(
        self,
        owner: str,
        repo: str,
        path: str = "",
        branch: str = "main"
    ) -> List[Dict[str, Any]]:
        """Get contents of a specific directory (non-recursive).
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: Directory path (empty string for root)
            branch: Branch name
            
        Returns:
            List of items in the directory
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        if branch:
            url += f"?ref={branch}"
        
        # Check cache
        cached_data = self._get_from_cache(url)
        if cached_data is not None:
            return cached_data
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            contents = response.json()
            
            # Transform to match tree API format
            items = []
            for item in contents:
                items.append({
                    'path': item['path'],
                    'type': 'tree' if item['type'] == 'dir' else 'blob',
                    'size': item.get('size', 0),
                    'name': item['name']
                })
            
            # Cache the response
            self._save_to_cache(url, items)
            
            return items
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Try with 'master' branch if 'main' fails
                if branch == "main":
                    return await self.get_directory_contents(owner, repo, path, "master")
                raise GitHubAPIError(f"Directory not found: {path}")
            elif e.response.status_code == 403:
                # Check if it's rate limit or permission issue
                remaining = e.response.headers.get('X-RateLimit-Remaining', '0')
                if remaining == '0':
                    reset_time = e.response.headers.get('X-RateLimit-Reset', '0')
                    raise GitHubAPIError(f"API rate limit exceeded. Resets at {reset_time}")
                else:
                    raise GitHubAPIError("Access denied. Private repository requires authentication.")
            else:
                raise GitHubAPIError(f"Failed to fetch directory contents: {e}")
        except Exception as e:
            raise GitHubAPIError(f"Failed to fetch directory contents: {e}")
    
    async def get_files_content_batch(
        self,
        owner: str,
        repo: str,
        file_paths: List[str],
        branch: str = "main",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, str]:
        """Fetch multiple files concurrently.
        
        Args:
            owner: Repository owner
            repo: Repository name
            file_paths: List of file paths to fetch
            branch: Branch name
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping file paths to content
        """
        import asyncio
        
        results = {}
        errors = {}
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def fetch_with_semaphore(path: str) -> tuple[str, Optional[str], Optional[str]]:
            async with semaphore:
                try:
                    content = await self.get_file_content(owner, repo, path, branch)
                    return (path, content, None)
                except Exception as e:
                    logger.error(f"Failed to fetch {path}: {e}")
                    return (path, None, str(e))
        
        # Create tasks for all files
        tasks = [fetch_with_semaphore(path) for path in file_paths]
        
        # Process results as they complete
        completed = 0
        for coro in asyncio.as_completed(tasks):
            path, content, error = await coro
            completed += 1
            
            if content is not None:
                results[path] = content
            else:
                errors[path] = error
            
            # Call progress callback
            if progress_callback:
                progress_callback(completed, len(file_paths), path)
        
        # Log summary
        logger.info(f"Batch fetch complete: {len(results)} successful, {len(errors)} failed")
        
        return results
    
    def build_tree_hierarchy(self, flat_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert flat tree structure to hierarchical structure.
        
        Args:
            flat_items: Flat list of tree items from API
            
        Returns:
            Hierarchical tree structure
        """
        # Create a map of paths to items
        path_map = {}
        root_items = []
        
        # First pass: create all nodes
        for item in flat_items:
            path = item['path']
            path_map[path] = {
                'path': path,
                'name': item['name'],
                'type': item['type'],
                'size': item.get('size'),
                'children': []
            }
        
        # Second pass: build hierarchy
        for path, node in path_map.items():
            parts = path.split('/')
            
            if len(parts) == 1:
                # Root level item
                root_items.append(node)
            else:
                # Find parent
                parent_path = '/'.join(parts[:-1])
                if parent_path in path_map:
                    path_map[parent_path]['children'].append(node)
        
        # Sort items at each level
        def sort_tree(items):
            for item in items:
                if item['children']:
                    sort_tree(item['children'])
            items.sort(key=lambda x: (x['type'] != 'tree', x['name'].lower()))
        
        sort_tree(root_items)
        
        return root_items