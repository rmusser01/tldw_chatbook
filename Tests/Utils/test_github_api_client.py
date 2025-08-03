"""
Unit tests for GitHub API client.

Tests the GitHubAPIClient in isolation with mocked HTTP responses.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import base64
import json
import httpx

from tldw_chatbook.Utils.github_api_client import GitHubAPIClient, GitHubAPIError


class TestGitHubAPIClient:
    """Test suite for GitHubAPIClient."""
    
    @pytest.fixture
    def api_client(self):
        """Create a GitHub API client instance."""
        return GitHubAPIClient(token="test_token")
    
    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        client = AsyncMock()
        return client
    
    def test_client_initialization_with_token(self):
        """Test client initialization with token."""
        client = GitHubAPIClient(token="test_token")
        
        assert client.token == "test_token"
        assert client.base_url == "https://api.github.com"
        assert client._client is None
    
    def test_client_initialization_without_token(self):
        """Test client initialization without token."""
        client = GitHubAPIClient()
        
        assert client.token is None
        assert client.base_url == "https://api.github.com"
        assert client._client is None
    
    def test_client_property_creates_client(self, api_client):
        """Test that accessing client property creates HTTP client."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value = mock_instance
            
            # Access client property
            client = api_client.client
            
            # Check client was created with correct headers
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args
            
            headers = call_args.kwargs['headers']
            assert headers['Accept'] == "application/vnd.github.v3+json"
            assert headers['User-Agent'] == "tldw-chatbook-repo-selector"
            assert headers['Authorization'] == "token test_token"
            
            # Check same instance is returned
            assert api_client.client is client
    
    def test_client_property_without_token(self):
        """Test client creation without token."""
        client = GitHubAPIClient()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            http_client = client.client
            
            call_args = mock_client_class.call_args
            headers = call_args.kwargs['headers']
            
            # Should not have Authorization header
            assert 'Authorization' not in headers
    
    @pytest.mark.parametrize("url,expected_owner,expected_repo", [
        ("https://github.com/owner/repo", "owner", "repo"),
        ("https://github.com/owner/repo.git", "owner", "repo"),
        ("git@github.com:owner/repo.git", "owner", "repo"),
        ("github.com/owner/repo", "owner", "repo"),
        ("owner/repo", "owner", "repo"),
        ("https://github.com/some-org/some-repo", "some-org", "some-repo"),
        ("https://github.com/user123/project_name", "user123", "project_name"),
    ])
    def test_parse_github_url_valid(self, api_client, url, expected_owner, expected_repo):
        """Test parsing valid GitHub URLs."""
        owner, repo = api_client.parse_github_url(url)
        
        assert owner == expected_owner
        assert repo == expected_repo
    
    @pytest.mark.parametrize("invalid_url", [
        "not a url",
        "https://github.com/",
        "owner",
        "",
    ])
    def test_parse_github_url_invalid(self, api_client, invalid_url):
        """Test parsing invalid GitHub URLs raises ValueError."""
        with pytest.raises(ValueError, match="Invalid GitHub repository URL"):
            api_client.parse_github_url(invalid_url)
    
    @pytest.mark.asyncio
    async def test_get_repository_info_success(self, api_client, mock_http_client):
        """Test successful repository info retrieval."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "description": "Test repository",
            "default_branch": "main"
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response
        
        # Patch client
        api_client._client = mock_http_client
        
        # Get repo info
        info = await api_client.get_repository_info("owner", "test-repo")
        
        # Verify
        assert info["name"] == "test-repo"
        assert info["full_name"] == "owner/test-repo"
        mock_http_client.get.assert_called_once_with(
            "https://api.github.com/repos/owner/test-repo"
        )
    
    @pytest.mark.asyncio
    async def test_get_repository_info_not_found(self, api_client, mock_http_client):
        """Test repository not found error."""
        # Mock 404 response - use MagicMock not AsyncMock
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=MagicMock(), response=mock_response
        )
        mock_http_client.get.return_value = mock_response
        
        api_client._client = mock_http_client
        
        # Should raise GitHubAPIError
        with pytest.raises(GitHubAPIError, match="Repository not found"):
            await api_client.get_repository_info("owner", "nonexistent")
    
    @pytest.mark.asyncio
    async def test_get_repository_info_rate_limit(self, api_client, mock_http_client):
        """Test rate limit error."""
        # Mock 403 response - use MagicMock not AsyncMock
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited", request=MagicMock(), response=mock_response
        )
        mock_http_client.get.return_value = mock_response
        
        api_client._client = mock_http_client
        
        # Should raise GitHubAPIError
        with pytest.raises(GitHubAPIError, match="API rate limit exceeded"):
            await api_client.get_repository_info("owner", "repo")
    
    @pytest.mark.asyncio
    async def test_get_branches_success(self, api_client, mock_http_client):
        """Test successful branch listing."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "main"},
            {"name": "develop"},
            {"name": "feature/test"}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response
        
        api_client._client = mock_http_client
        
        # Get branches
        branches = await api_client.get_branches("owner", "repo")
        
        # Verify
        assert branches == ["main", "develop", "feature/test"]
        mock_http_client.get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/branches"
        )
    
    @pytest.mark.asyncio
    async def test_get_branches_error_fallback(self, api_client, mock_http_client):
        """Test branch listing falls back to common names on error."""
        # Mock error
        mock_http_client.get.side_effect = Exception("Network error")
        api_client._client = mock_http_client
        
        # Should return fallback branches
        branches = await api_client.get_branches("owner", "repo")
        
        assert branches == ["main", "master"]
    
    @pytest.mark.asyncio
    async def test_get_repository_tree_success(self, api_client, mock_http_client):
        """Test successful repository tree retrieval."""
        # Mock branch response
        branch_response = MagicMock()
        branch_response.status_code = 200
        branch_response.json.return_value = {
            "commit": {"sha": "abc123"}
        }
        branch_response.raise_for_status = MagicMock()
        
        # Mock tree response
        tree_response = MagicMock()
        tree_response.status_code = 200
        tree_response.json.return_value = {
            "tree": [
                {
                    "path": "README.md",
                    "type": "blob",
                    "size": 1234
                },
                {
                    "path": "src",
                    "type": "tree"
                },
                {
                    "path": "src/main.py",
                    "type": "blob",
                    "size": 2048
                }
            ]
        }
        tree_response.raise_for_status = MagicMock()
        
        # Set up mock to return different responses
        mock_http_client.get.side_effect = [branch_response, tree_response]
        api_client._client = mock_http_client
        
        # Get tree
        items = await api_client.get_repository_tree("owner", "repo", "main", recursive=True)
        
        # Verify
        assert len(items) == 3
        assert items[0]["path"] == "README.md"
        assert items[0]["name"] == "README.md"
        assert items[1]["path"] == "src"
        assert items[1]["name"] == "src"
        assert items[2]["path"] == "src/main.py"
        assert items[2]["name"] == "main.py"
        
        # Check API calls
        assert mock_http_client.get.call_count == 2
        calls = mock_http_client.get.call_args_list
        assert calls[0][0][0] == "https://api.github.com/repos/owner/repo/branches/main"
        assert calls[1][0][0] == "https://api.github.com/repos/owner/repo/git/trees/abc123?recursive=1"
    
    @pytest.mark.asyncio
    async def test_get_repository_tree_main_fallback_to_master(self, api_client, mock_http_client):
        """Test fallback from main to master branch."""
        # Mock 404 for main branch
        main_response = MagicMock()
        main_response.status_code = 404
        
        # Mock success for master branch
        master_response = MagicMock()
        master_response.status_code = 200
        master_response.json.return_value = {
            "commit": {"sha": "def456"}
        }
        master_response.raise_for_status = MagicMock()
        
        # Mock tree response
        tree_response = MagicMock()
        tree_response.status_code = 200
        tree_response.json.return_value = {"tree": []}
        tree_response.raise_for_status = MagicMock()
        
        mock_http_client.get.side_effect = [main_response, master_response, tree_response]
        api_client._client = mock_http_client
        
        # Should succeed with master branch
        items = await api_client.get_repository_tree("owner", "repo", "main")
        
        # Verify fallback happened
        assert mock_http_client.get.call_count == 3
        calls = mock_http_client.get.call_args_list
        assert "branches/main" in calls[0][0][0]
        assert "branches/master" in calls[1][0][0]
    
    @pytest.mark.asyncio
    async def test_get_file_content_success(self, api_client, mock_http_client):
        """Test successful file content retrieval."""
        # Create base64 encoded content
        content = "Hello, World!"
        encoded_content = base64.b64encode(content.encode()).decode()
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "type": "file",
            "content": encoded_content
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response
        
        api_client._client = mock_http_client
        
        # Get file content
        result = await api_client.get_file_content("owner", "repo", "test.txt", "main")
        
        # Verify
        assert result == content
        mock_http_client.get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/test.txt",
            params={"ref": "main"}
        )
    
    @pytest.mark.asyncio
    async def test_get_file_content_not_file(self, api_client, mock_http_client):
        """Test error when path is not a file."""
        # Mock response for directory
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "type": "dir"
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response
        
        api_client._client = mock_http_client
        
        # Should raise error
        with pytest.raises(GitHubAPIError, match="Path is not a file"):
            await api_client.get_file_content("owner", "repo", "src", "main")
    
    @pytest.mark.asyncio
    async def test_get_rate_limit_success(self, api_client, mock_http_client):
        """Test rate limit info retrieval."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "rate": {
                "limit": 5000,
                "remaining": 4999,
                "reset": 1234567890
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response
        
        api_client._client = mock_http_client
        
        # Get rate limit
        rate_limit = await api_client.get_rate_limit()
        
        # Verify
        assert rate_limit["rate"]["limit"] == 5000
        assert rate_limit["rate"]["remaining"] == 4999
    
    @pytest.mark.asyncio
    async def test_get_rate_limit_error(self, api_client, mock_http_client):
        """Test rate limit retrieval error handling."""
        # Mock error
        mock_http_client.get.side_effect = Exception("Network error")
        api_client._client = mock_http_client
        
        # Should return empty dict
        rate_limit = await api_client.get_rate_limit()
        
        assert rate_limit == {}
    
    def test_build_tree_hierarchy_empty(self, api_client):
        """Test building hierarchy from empty list."""
        result = api_client.build_tree_hierarchy([])
        assert result == []
    
    def test_build_tree_hierarchy_flat(self, api_client):
        """Test building hierarchy from flat structure."""
        flat_items = [
            {"path": "README.md", "name": "README.md", "type": "blob", "size": 100},
            {"path": "LICENSE", "name": "LICENSE", "type": "blob", "size": 200},
            {"path": ".gitignore", "name": ".gitignore", "type": "blob", "size": 50}
        ]
        
        result = api_client.build_tree_hierarchy(flat_items)
        
        # All items should be at root level
        assert len(result) == 3
        assert all(item["children"] == [] for item in result)
        assert result[0]["path"] == ".gitignore"  # Sorted alphabetically
        assert result[1]["path"] == "LICENSE"
        assert result[2]["path"] == "README.md"
    
    def test_build_tree_hierarchy_nested(self, api_client):
        """Test building hierarchy from nested structure."""
        flat_items = [
            {"path": "README.md", "name": "README.md", "type": "blob"},
            {"path": "src", "name": "src", "type": "tree"},
            {"path": "src/main.py", "name": "main.py", "type": "blob"},
            {"path": "src/utils", "name": "utils", "type": "tree"},
            {"path": "src/utils/helpers.py", "name": "helpers.py", "type": "blob"},
            {"path": "tests", "name": "tests", "type": "tree"},
            {"path": "tests/test_main.py", "name": "test_main.py", "type": "blob"}
        ]
        
        result = api_client.build_tree_hierarchy(flat_items)
        
        # Check root level
        assert len(result) == 3
        root_names = [item["name"] for item in result]
        assert root_names == ["src", "tests", "README.md"]  # Directories first
        
        # Check src directory
        src_item = result[0]
        assert len(src_item["children"]) == 2
        src_children_names = [child["name"] for child in src_item["children"]]
        assert src_children_names == ["utils", "main.py"]  # Directory first
        
        # Check nested utils directory
        utils_item = src_item["children"][0]
        assert len(utils_item["children"]) == 1
        assert utils_item["children"][0]["name"] == "helpers.py"
        
        # Check tests directory
        tests_item = result[1]
        assert len(tests_item["children"]) == 1
        assert tests_item["children"][0]["name"] == "test_main.py"
    
    def test_build_tree_hierarchy_sorting(self, api_client):
        """Test that hierarchy is properly sorted."""
        flat_items = [
            {"path": "z_file.txt", "name": "z_file.txt", "type": "blob"},
            {"path": "a_dir", "name": "a_dir", "type": "tree"},
            {"path": "m_file.txt", "name": "m_file.txt", "type": "blob"},
            {"path": "b_dir", "name": "b_dir", "type": "tree"},
            {"path": "A_file.txt", "name": "A_file.txt", "type": "blob"}  # Capital letter
        ]
        
        result = api_client.build_tree_hierarchy(flat_items)
        
        # Check sorting: directories first, then case-insensitive alphabetical
        names = [item["name"] for item in result]
        assert names == ["a_dir", "b_dir", "A_file.txt", "m_file.txt", "z_file.txt"]
    
    @pytest.mark.asyncio
    async def test_close_client(self, api_client):
        """Test closing the HTTP client."""
        # Create mock client
        mock_client = AsyncMock()
        api_client._client = mock_client
        
        # Close
        await api_client.close()
        
        # Verify
        mock_client.aclose.assert_called_once()
        assert api_client._client is None
    
    @pytest.mark.asyncio
    async def test_close_client_no_client(self, api_client):
        """Test closing when no client exists."""
        # Should not raise error
        await api_client.close()
        assert api_client._client is None