# tldw_chatbook/Utils/github_api_client.py
# Description: GitHub API client for fetching repository structure and content
#
# This module handles all GitHub API interactions for the repository file selector.

from __future__ import annotations
from typing import Optional, List, Dict, Any
import base64
import re
from urllib.parse import urlparse

import httpx
from loguru import logger

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
        self.token = token
        self.base_url = "https://api.github.com"
        self._client: Optional[httpx.AsyncClient] = None
        
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
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise GitHubAPIError(f"Repository not found: {owner}/{repo}")
            elif e.response.status_code == 403:
                raise GitHubAPIError("API rate limit exceeded or access denied")
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
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            branches = response.json()
            return [branch['name'] for branch in branches]
        except Exception as e:
            logger.error(f"Failed to fetch branches: {e}")
            return ["main", "master"]  # Fallback to common branch names
    
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
            
            return items
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                raise GitHubAPIError("API rate limit exceeded")
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