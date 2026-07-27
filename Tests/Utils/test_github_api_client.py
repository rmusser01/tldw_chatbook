"""
Unit tests for GitHub API client.

Tests the GitHubAPIClient in isolation with mocked HTTP responses.
"""

import asyncio
import threading
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import base64
import httpx

from tldw_chatbook.Utils.github_api_client import GitHubAPIClient, GitHubAPIError
from tldw_chatbook.Utils.egress import MAX_FETCH_BYTES_GITHUB_FILE


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
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value = mock_instance

            # Access client property
            client = api_client.client

            # Check client was created with correct headers
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args

            headers = call_args.kwargs["headers"]
            assert headers["Accept"] == "application/vnd.github.v3+json"
            assert headers["User-Agent"] == "tldw-chatbook-repo-selector"
            assert headers["Authorization"] == "token test_token"

            # Check same instance is returned
            assert api_client.client is client

    def test_client_property_without_token(self):
        """Test client creation without token."""
        with (
            patch(
                "tldw_chatbook.Utils.github_api_client.get_cli_setting",
                side_effect=lambda _section, _key, default: default,
            ),
            patch("tldw_chatbook.Utils.github_api_client.os.getenv", return_value=None),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            api_client = GitHubAPIClient()
            api_client.client

            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args
            headers = call_args.kwargs["headers"]

            assert headers["Accept"] == "application/vnd.github.v3+json"
            assert headers["User-Agent"] == "tldw-chatbook-repo-selector"
            assert "Authorization" not in headers

    @pytest.mark.parametrize(
        "url,expected_owner,expected_repo",
        [
            ("https://github.com/owner/repo", "owner", "repo"),
            ("https://github.com/owner/repo.git", "owner", "repo"),
            ("git@github.com:owner/repo.git", "owner", "repo"),
            ("github.com/owner/repo", "owner", "repo"),
            ("owner/repo", "owner", "repo"),
            ("https://github.com/some-org/some-repo", "some-org", "some-repo"),
            ("https://github.com/user123/project_name", "user123", "project_name"),
        ],
    )
    def test_parse_github_url_valid(
        self, api_client, url, expected_owner, expected_repo
    ):
        """Test parsing valid GitHub URLs."""
        owner, repo = api_client.parse_github_url(url)

        assert owner == expected_owner
        assert repo == expected_repo

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "not a url",
            "https://github.com/",
            "owner",
            "",
        ],
    )
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
            "default_branch": "main",
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
            {"name": "feature/test"},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        api_client._client = mock_http_client

        # Get branches
        branches = await api_client.get_branches("owner", "repo")

        # Verify (per_page=100: remote-skill-fetch slash-ref disambiguation
        # needs the full first page — GitHub defaults to 30/page).
        assert branches == ["main", "develop", "feature/test"]
        mock_http_client.get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/branches",
            params={"per_page": 100},
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
    async def test_get_repository_tree_success(
        self, api_client, mock_http_client, monkeypatch
    ):
        """Test successful repository tree retrieval."""
        # Mock branch response
        branch_response = MagicMock()
        branch_response.status_code = 200
        branch_response.json.return_value = {"commit": {"sha": "abc123"}}
        branch_response.raise_for_status = MagicMock()

        # Mock tree response
        tree_response = MagicMock()
        tree_response.status_code = 200
        tree_response.json.return_value = {
            "tree": [
                {"path": "README.md", "type": "blob", "size": 1234},
                {"path": "src", "type": "tree"},
                {"path": "src/main.py", "type": "blob", "size": 2048},
            ]
        }
        tree_response.raise_for_status = MagicMock()

        # get_repository_tree now streams through the egress-guarded fetch
        # helper (real byte cap) instead of calling self.client.get directly.
        mock_fetch = AsyncMock(side_effect=[branch_response, tree_response])
        monkeypatch.setattr(
            "tldw_chatbook.Utils.egress.guarded_fetch_httpx_async", mock_fetch
        )
        api_client._client = mock_http_client

        # Get tree
        items = await api_client.get_repository_tree(
            "owner", "repo", "main", recursive=True
        )

        # Verify
        assert len(items) == 3
        assert items[0]["path"] == "README.md"
        assert items[0]["name"] == "README.md"
        assert items[1]["path"] == "src"
        assert items[1]["name"] == "src"
        assert items[2]["path"] == "src/main.py"
        assert items[2]["name"] == "main.py"

        # Check API calls
        assert mock_fetch.call_count == 2
        calls = mock_fetch.call_args_list
        assert calls[0][0][0] == "https://api.github.com/repos/owner/repo/branches/main"
        assert calls[0].kwargs["client"] is mock_http_client
        assert calls[0].kwargs["max_bytes"] == MAX_FETCH_BYTES_GITHUB_FILE
        assert (
            calls[1][0][0]
            == "https://api.github.com/repos/owner/repo/git/trees/abc123?recursive=1"
        )

    @pytest.mark.asyncio
    async def test_get_repository_tree_main_fallback_to_master(
        self, api_client, mock_http_client, monkeypatch
    ):
        """Test fallback from main to master branch."""
        # Mock 404 for main branch
        main_response = MagicMock()
        main_response.status_code = 404

        # Mock success for master branch
        master_response = MagicMock()
        master_response.status_code = 200
        master_response.json.return_value = {"commit": {"sha": "def456"}}
        master_response.raise_for_status = MagicMock()

        # Mock tree response
        tree_response = MagicMock()
        tree_response.status_code = 200
        tree_response.json.return_value = {"tree": []}
        tree_response.raise_for_status = MagicMock()

        mock_fetch = AsyncMock(
            side_effect=[main_response, master_response, tree_response]
        )
        monkeypatch.setattr(
            "tldw_chatbook.Utils.egress.guarded_fetch_httpx_async", mock_fetch
        )
        api_client._client = mock_http_client

        # Should succeed with master branch
        await api_client.get_repository_tree("owner", "repo", "main")

        # Verify fallback happened
        assert mock_fetch.call_count == 3
        calls = mock_fetch.call_args_list
        assert "branches/main" in calls[0][0][0]
        assert "branches/master" in calls[1][0][0]

    @pytest.mark.asyncio
    async def test_get_file_content_success(
        self, api_client, mock_http_client, monkeypatch
    ):
        """Test successful file content retrieval."""
        # Create base64 encoded content
        content = "Hello, World!"
        encoded_content = base64.b64encode(content.encode()).decode()

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"type": "file", "content": encoded_content}
        mock_response.raise_for_status = MagicMock()

        # get_file_content now streams through the egress-guarded fetch
        # helper (real byte cap) instead of calling self.client.get directly.
        mock_fetch = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(
            "tldw_chatbook.Utils.egress.guarded_fetch_httpx_async", mock_fetch
        )
        api_client._client = mock_http_client

        # Get file content
        result = await api_client.get_file_content("owner", "repo", "test.txt", "main")

        # Verify
        assert result == content
        mock_fetch.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/test.txt",
            client=mock_http_client,
            max_bytes=MAX_FETCH_BYTES_GITHUB_FILE,
            params={"ref": "main"},
        )

    @pytest.mark.asyncio
    async def test_get_file_content_not_file(
        self, api_client, mock_http_client, monkeypatch
    ):
        """Test error when path is not a file."""
        # Mock response for directory
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"type": "dir"}
        mock_response.raise_for_status = MagicMock()

        mock_fetch = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(
            "tldw_chatbook.Utils.egress.guarded_fetch_httpx_async", mock_fetch
        )
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
            "rate": {"limit": 5000, "remaining": 4999, "reset": 1234567890}
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
            {"path": ".gitignore", "name": ".gitignore", "type": "blob", "size": 50},
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
            {"path": "tests/test_main.py", "name": "test_main.py", "type": "blob"},
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
            {
                "path": "A_file.txt",
                "name": "A_file.txt",
                "type": "blob",
            },  # Capital letter
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


class TestGitHubAPIClientCrossEventLoop:
    """Regression tests for TASK-981: ``GitHubAPIClient.client`` is shared by
    handlers on the app's long-lived event loop AND by ``@work(thread=True)``
    workers decorated with ``async def`` (``CodeRepoCopyPasteWindow``'s
    ``_export_to_zip_worker`` / ``load_node_children``). Those workers run
    on a brand-new event loop created by Textual's ``Worker._run_threaded``
    via ``asyncio.run()``, which closes that loop when the worker returns.
    Caching a single ``httpx.AsyncClient`` across those loops is a real
    cross-loop hazard: a client built on one loop must never be handed back
    to a caller running on a different loop, since httpx pins connection
    pool/transport state to the loop active at construction time.
    """

    @staticmethod
    async def _touch_client(api_client: GitHubAPIClient) -> httpx.AsyncClient:
        """Access the ``client`` property from whatever loop is running."""
        return api_client.client

    def test_client_is_not_reused_across_two_asyncio_run_loops(self):
        """Two separate ``asyncio.run()`` calls == two separate throwaway
        loops, exactly like two invocations of the same ``@work(thread=True)``
        async worker. The client built for the first loop must not be handed
        back once that loop is closed and a second, unrelated loop is
        running -- if it were, awaiting anything on it would risk
        ``RuntimeError: Event loop is closed``.
        """
        api_client = GitHubAPIClient(token="t")

        client_first = asyncio.run(self._touch_client(api_client))
        loop_first = api_client._client_loop
        assert loop_first is not None
        assert loop_first.is_closed()

        client_second = asyncio.run(self._touch_client(api_client))
        loop_second = api_client._client_loop

        # The defect this guards against: handing back `client_first`, which
        # is bound to `loop_first` -- a loop that is now closed. (Both loops
        # are closed by the time we inspect them here -- `asyncio.run()`
        # always closes its loop before returning -- so the meaningful proof
        # is that the *instances* differ, not their closed-ness.)
        assert client_second is not client_first
        assert loop_second is not loop_first
        assert api_client._client is client_second

    def test_second_call_after_owning_loop_closed_still_works(self):
        """Directly exercises the acceptance-criteria phrasing: a second
        call to the accessor, made after the loop that built the first
        client has closed, must still succeed (not raise, not hang) and
        must not return the dead-loop-bound instance.
        """
        api_client = GitHubAPIClient(token="t")

        stale_client = asyncio.run(self._touch_client(api_client))
        assert api_client._client_loop.is_closed()

        # This call happens on a *third* loop, entirely unrelated to the
        # one `stale_client` was built on. Before the fix, this returned
        # `stale_client` unconditionally (single-slot unconditional cache).
        fresh_client = asyncio.run(self._touch_client(api_client))

        assert fresh_client is not stale_client
        assert fresh_client.is_closed is False

    def test_stale_client_on_still_running_loop_is_closed_via_handoff(self):
        """Simulates the realistic ordering in ``CodeRepoCopyPasteWindow``:
        the app loop (long-lived, never closed by us) builds the client
        first via a handler like ``load_repository``; later a worker thread
        (``_export_to_zip_worker`` / ``load_node_children``) running its own
        throwaway loop touches the same ``GitHubAPIClient`` instance. The
        app-loop client must not be silently leaked -- it should be handed
        off for a graceful ``aclose()`` via ``run_coroutine_threadsafe``,
        which the app loop (still alive) can complete on its next iteration.
        """
        api_client = GitHubAPIClient(token="t")
        app_loop = asyncio.new_event_loop()
        try:
            app_client = app_loop.run_until_complete(self._touch_client(api_client))
            assert app_client.is_closed is False
            assert api_client._client_loop is app_loop

            # Simulate a `@work(thread=True)` async worker touching the same
            # shared api_client from its own asyncio.run() loop.
            worker_client = asyncio.run(self._touch_client(api_client))
            assert worker_client is not app_client
            assert api_client._client_loop is not app_loop

            # The app loop is still alive (unlike a worker's loop) -- pump it
            # once so the scheduled `aclose()` hand-off actually runs.
            app_loop.run_until_complete(asyncio.sleep(0))
            assert app_client.is_closed is True, (
                "stale app-loop client was not closed via the "
                "run_coroutine_threadsafe hand-off -- it would otherwise leak"
            )
        finally:
            app_loop.close()

    @pytest.mark.asyncio
    async def test_close_from_different_loop_does_not_raise(self):
        """``close()`` itself must not attempt to await a client bound to a
        different loop than the one ``close()`` is running on -- that is
        the same cross-loop hazard, just triggered from the cleanup path
        (``CodeRepoCopyPasteWindow.__aexit__``) instead of the accessor.
        """
        api_client = GitHubAPIClient(token="t")

        # Build a client on an unrelated, already-closed loop. This test is
        # itself async (running on pytest-asyncio's loop), so this can't be
        # nested via `asyncio.run()` directly -- do what the real worker
        # does: run it on a separate OS thread with its own throwaway loop.
        worker_thread = threading.Thread(
            target=lambda: asyncio.run(self._touch_client(api_client))
        )
        worker_thread.start()
        worker_thread.join()

        assert api_client._client is not None
        assert api_client._client_loop.is_closed()

        # close() now runs on pytest-asyncio's loop, which is a different,
        # live loop. This must not raise or hang.
        await api_client.close()
        assert api_client._client is None
