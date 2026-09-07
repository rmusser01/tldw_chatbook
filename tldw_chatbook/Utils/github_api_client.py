# tldw_chatbook/Utils/github_api_client.py
# Description: GitHub API client for fetching repository structure and content
#
# This module handles all GitHub API interactions for the repository file selector.

from __future__ import annotations
from typing import Optional, List, Dict, Any
import asyncio
import base64
import re
import os
import time
import hashlib
import sys
import weakref

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
                if (
                    config_token
                    and not config_token.startswith("<")
                    and not config_token.endswith(">")
                ):
                    self.token = config_token

        self.base_url = "https://api.github.com"
        self._client: Optional[httpx.AsyncClient] = None
        # The event loop that ``self._client`` was constructed on, if known.
        # Mirrors whichever entry in ``_loop_clients`` was most recently
        # resolved -- kept for the "unknown loop" escape hatch (see the
        # ``client`` property) and for introspection.
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None
        # Per-loop client cache: every event loop that is *currently alive*
        # and has touched ``client`` gets its own ``httpx.AsyncClient``, so
        # that two loops alive at the same time (the app's long-lived loop
        # and a `@work(thread=True)` worker's throwaway loop) never fight
        # over -- or close -- each other's client. Keyed by the loop object
        # itself via a ``WeakKeyDictionary`` so an entry can be reclaimed as
        # soon as nothing else references that loop; pruned proactively in
        # ``_prune_closed_loops`` (below) so a long-running process that
        # spawns many short-lived worker loops over time doesn't accumulate
        # dead entries waiting on GC.
        self._loop_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
            weakref.WeakKeyDictionary()
        )

        # Load config settings
        self.enable_rate_limit_handling = get_cli_setting(
            "github", "enable_rate_limit_handling", True
        )
        self.cache_ttl = get_cli_setting("github", "cache_ttl_seconds", 300)
        self.max_retries = get_cli_setting("github", "max_retries", 3)
        self.max_concurrent_requests = get_cli_setting(
            "github", "max_concurrent_requests", 5
        )

        # Initialize cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_order: List[str] = []  # Track insertion order for LRU
        self.max_cache_size = get_cli_setting("github", "max_cache_entries", 100)
        self.cache_size_mb = 0
        self.max_cache_size_mb = get_cli_setting("github", "max_cache_size_mb", 50)

        if self.token:
            logger.info("GitHub API client initialized with authentication token")
        else:
            logger.info(
                "GitHub API client initialized without authentication (public repos only)"
            )

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client, scoped to the current event loop.

        ``httpx.AsyncClient`` pins its connection pool/transport to whichever
        asyncio event loop is running when it is constructed. This client is
        shared by a single ``GitHubAPIClient`` instance that is used both
        from the app's long-lived event loop and from the short-lived,
        throwaway loops that Textual creates for ``@work(thread=True)``
        workers decorated with ``async def`` (``Worker._run_threaded`` routes
        those through ``asyncio.run()``, which closes that loop when the
        worker returns). Reusing a single cached instance across those loops
        produces ``RuntimeError: Event loop is closed`` / "attached to a
        different loop" errors, or hangs.

        Guard against that with a per-loop cache (``_loop_clients``): every
        currently running loop that touches this property gets its own
        client, so an app-loop request in flight and a worker-loop request
        running concurrently never fight over, or close, each other's
        client -- unlike invalidating a single cached slot, which would
        require discarding (and scheduling a close of) whichever client the
        *other*, still-live loop was using. Entries for loops that have
        since closed are pruned so the cache cannot grow unboundedly across
        many worker invocations. If ``_client_loop`` is unknown (e.g. a test
        injected ``_client`` directly without going through this property)
        we trust the cached client as-is rather than second-guessing it.

        Returns:
            The ``httpx.AsyncClient`` bound to the caller's current event
            loop (or the legacy single-slot ``_client`` when no loop is
            running, or when it was injected without a known owning loop).
        """
        try:
            loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            # No running loop to key by -- trust whatever is cached (sync
            # context, or a test that injected ``_client`` directly).
            if self._client is None:
                self._client = self._build_client()
            return self._client

        self._prune_closed_loops()

        cached = self._loop_clients.get(loop)
        if cached is not None:
            self._client, self._client_loop = cached, loop
            return cached

        if self._client is not None and self._client_loop is None:
            # Unknown-loop escape hatch: a test (or caller) injected
            # ``_client`` directly without going through this property, so
            # we don't know which loop it "belongs" to. Trust it once, and
            # adopt it into the per-loop cache for the loop that is
            # actually running now so later calls on this same loop reuse
            # it instead of rebuilding.
            self._loop_clients[loop] = self._client
            self._client_loop = loop
            return self._client

        new_client = self._build_client()
        self._loop_clients[loop] = new_client
        self._client, self._client_loop = new_client, loop
        return new_client

    def _build_client(self) -> httpx.AsyncClient:
        """Construct a fresh ``httpx.AsyncClient`` with this instance's headers."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "tldw-chatbook-repo-selector",
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return httpx.AsyncClient(headers=headers, timeout=30.0)

    def _prune_closed_loops(self) -> None:
        """Drop ``_loop_clients`` entries whose owning loop has closed.

        ``WeakKeyDictionary`` alone would eventually reclaim these once the
        loop object itself is garbage collected, but asyncio loops can take
        a while to be collected (reference cycles via internal callbacks),
        and nothing here needs to keep an already-closed loop's entry
        around -- there is nothing left to gracefully close on it (see
        ``_schedule_close``). Pruning proactively on every access bounds
        the cache's size even under many short-lived worker loops.
        """
        for stale_loop in [lp for lp in self._loop_clients if lp.is_closed()]:
            del self._loop_clients[stale_loop]

    def _schedule_close(
        self, client: httpx.AsyncClient, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Best-effort close ``client`` on ``loop`` without blocking the caller.

        If ``loop`` is already closed there is nothing left to gracefully
        close -- the reference is simply dropped and the client's own
        finalizer will release its sockets. Otherwise the close is
        scheduled via ``run_coroutine_threadsafe``; the returned
        ``Future`` is kept and given a done-callback that logs any
        exception, so a failed close on another loop is never silent.
        """
        if loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(client.aclose(), loop)
        except RuntimeError:
            logger.debug(
                "Could not schedule close of stale GitHub HTTP client; "
                "owning loop is unavailable"
            )
            return

        def _log_close_failure(fut) -> None:
            try:
                exc = fut.exception()
            except Exception:
                # Cancelled or otherwise unable to retrieve the exception --
                # nothing more we can do here.
                return
            if exc is not None:
                logger.opt(exception=exc).warning(
                    "Failed to close a stale GitHub HTTP client on its owning loop: {}",
                    exc,
                )

        future.add_done_callback(_log_close_failure)

    async def close(self) -> None:
        """Close the HTTP client(s) owned by this instance.

        The client bound to the caller's current running loop is closed
        directly (safe, since we are already running on that loop). Every
        other cached per-loop client -- e.g. one built earlier by the app's
        long-lived loop, or by a different worker's throwaway loop that is
        still alive -- is closed best-effort via ``run_coroutine_threadsafe``
        on its own loop; we never await, and never close, a client bound to
        a loop we are not currently running on.
        """
        try:
            loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        self._prune_closed_loops()

        current_client: Optional[httpx.AsyncClient] = None
        if loop is not None:
            current_client = self._loop_clients.pop(loop, None)
            if current_client is None and self._client is not None and self._client_loop is None:
                # Unknown-loop escape hatch, mirroring the ``client``
                # property: an injected ``_client`` with no known owning
                # loop is treated as belonging to whichever loop is calling
                # ``close()`` now.
                current_client = self._client

        for other_loop, other_client in list(self._loop_clients.items()):
            if other_client is current_client:
                continue
            self._schedule_close(other_client, other_loop)
        self._loop_clients.clear()

        if current_client is not None and loop is not None:
            await current_client.aclose()

        self._client = None
        self._client_loop = None

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
            r"github\.com[/:]([^/]+)/([^/\.]+)",  # HTTPS and SSH
            r"([^/]+)/([^/]+)$",  # Simple owner/repo format
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                owner, repo = match.groups()
                # Remove .git extension if present
                repo = repo.replace(".git", "")
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
                remaining = self._header_value(e.response, "X-RateLimit-Remaining")
                if (
                    remaining == "0"
                    or "rate limit" in str(e).lower()
                    or "rate limited" in str(e).lower()
                ):
                    reset_time = (
                        self._header_value(e.response, "X-RateLimit-Reset") or "0"
                    )
                    raise GitHubAPIError(
                        f"API rate limit exceeded. Resets at {reset_time}"
                    )
                else:
                    raise GitHubAPIError(
                        "Access denied. Private repository requires authentication."
                    )
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
        params = {"per_page": 100}

        # Check cache first
        cached_data = self._get_from_cache(url, params)
        if cached_data is not None:
            return cached_data

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            branches = response.json()
            branch_names = [branch["name"] for branch in branches]

            # Cache the response
            self._save_to_cache(url, branch_names, params)
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

    @staticmethod
    def _header_value(response: httpx.Response, key: str) -> str | None:
        headers = getattr(response, "headers", None)
        getter = getattr(headers, "get", None)
        if not callable(getter):
            return None
        value = getter(key)
        return value if isinstance(value, str) else None

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is still valid."""
        if not cache_entry:
            return False
        timestamp = cache_entry.get("timestamp", 0)
        return (time.time() - timestamp) < self.cache_ttl

    def _get_from_cache(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get data from cache if valid."""
        cache_key = self._get_cache_key(url, params)
        cache_entry = self._cache.get(cache_key)

        if cache_entry and self._is_cache_valid(cache_entry):
            logger.debug(f"Cache hit for {url}")
            return cache_entry["data"]

        return None

    def _save_to_cache(
        self, url: str, data: Any, params: Optional[Dict] = None
    ) -> None:
        """Save data to cache with size limits."""
        cache_key = self._get_cache_key(url, params)

        # Estimate size of data (rough approximation)
        data_size = sys.getsizeof(str(data)) / (1024 * 1024)  # Convert to MB

        # Check if we need to evict entries
        while (
            len(self._cache) >= self.max_cache_size
            or self.cache_size_mb + data_size > self.max_cache_size_mb
        ) and self._cache_order:
            # Remove oldest entry (LRU)
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._cache:
                old_data = self._cache.pop(oldest_key)
                old_size = sys.getsizeof(str(old_data.get("data", ""))) / (1024 * 1024)
                self.cache_size_mb -= old_size
                logger.debug(f"Evicted cache entry: {oldest_key}")

        # Add new entry
        self._cache[cache_key] = {"data": data, "timestamp": time.time()}
        self._cache_order.append(cache_key)
        self.cache_size_mb += data_size

        logger.debug(
            f"Cached response for {url} (cache size: {len(self._cache)} entries, {self.cache_size_mb:.1f} MB)"
        )

    async def get_repository_tree(
        self, owner: str, repo: str, branch: str = "main", recursive: bool = True
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

        # For non-recursive calls, we'll use a different approach
        if not recursive:
            # Use the contents API for non-recursive directory listing
            return await self.get_directory_contents(owner, repo, "", branch)

        tree_url = f"{self.base_url}/repos/{owner}/{repo}/git/trees/{branch}?recursive={int(recursive)}"

        cached_tree = self._get_from_cache(tree_url)
        if cached_tree is not None:
            return cached_tree

        from ..Utils.egress import MAX_FETCH_BYTES_GITHUB_FILE, guarded_fetch_httpx_async

        # First try to get the branch SHA
        branch_url = f"{self.base_url}/repos/{owner}/{repo}/branches/{branch}"

        try:
            response = await guarded_fetch_httpx_async(
                branch_url,
                client=self.client,
                max_bytes=MAX_FETCH_BYTES_GITHUB_FILE,
            )
            if response.status_code == 404:
                # Try 'master' if 'main' doesn't exist
                if branch == "main":
                    return await self.get_repository_tree(
                        owner, repo, "master", recursive
                    )
                else:
                    raise GitHubAPIError(f"Branch not found: {branch}")

            response.raise_for_status()
            branch_data = response.json()
            tree_sha = branch_data["commit"]["sha"]

            # Get the tree
            tree_url = f"{self.base_url}/repos/{owner}/{repo}/git/trees/{tree_sha}"
            if recursive:
                tree_url += "?recursive=1"

            response = await guarded_fetch_httpx_async(
                tree_url,
                client=self.client,
                max_bytes=MAX_FETCH_BYTES_GITHUB_FILE,
            )
            response.raise_for_status()
            tree_data = response.json()

            # Process tree items to add name field
            items = []
            for item in tree_data.get("tree", []):
                item["name"] = item["path"].split("/")[-1]
                items.append(item)

            # Cache the processed tree
            self._save_to_cache(tree_url, items)

            return items

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                # Check if it's rate limit or permission issue
                remaining = e.response.headers.get("X-RateLimit-Remaining", "0")
                if remaining == "0":
                    reset_time = e.response.headers.get("X-RateLimit-Reset", "0")
                    raise GitHubAPIError(
                        f"API rate limit exceeded. Resets at {reset_time}"
                    )
                else:
                    raise GitHubAPIError(
                        "Access denied. Private repository requires authentication."
                    )
            else:
                raise GitHubAPIError(f"Failed to fetch repository tree: {e}")
        except Exception as e:
            raise GitHubAPIError(f"Failed to fetch repository tree: {e}")

    async def get_file_content(
        self, owner: str, repo: str, path: str, branch: str = "main"
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
        from ..Utils.egress import MAX_FETCH_BYTES_GITHUB_FILE, guarded_fetch_httpx_async

        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": branch}

        try:
            response = await guarded_fetch_httpx_async(
                url,
                client=self.client,
                max_bytes=MAX_FETCH_BYTES_GITHUB_FILE,
                params=params,
            )
            response.raise_for_status()

            data = response.json()

            # Check if it's a file
            if data.get("type") != "file":
                raise GitHubAPIError(f"Path is not a file: {path}")

            # Decode base64 content
            content = data.get("content", "")
            decoded = base64.b64decode(content).decode("utf-8")

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
        self, owner: str, repo: str, path: str = "", branch: str = "main"
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

        from ..Utils.egress import MAX_FETCH_BYTES_GITHUB_FILE, guarded_fetch_httpx_async

        # Check cache
        cached_data = self._get_from_cache(url)
        if cached_data is not None:
            return cached_data

        try:
            response = await guarded_fetch_httpx_async(
                url,
                client=self.client,
                max_bytes=MAX_FETCH_BYTES_GITHUB_FILE,
            )
            response.raise_for_status()

            contents = response.json()

            # Transform to match tree API format
            items = []
            for item in contents:
                items.append(
                    {
                        "path": item["path"],
                        "type": "tree" if item["type"] == "dir" else "blob",
                        "size": item.get("size", 0),
                        "name": item["name"],
                    }
                )

            # Cache the response
            self._save_to_cache(url, items)

            return items

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Try with 'master' branch if 'main' fails
                if branch == "main":
                    return await self.get_directory_contents(
                        owner, repo, path, "master"
                    )
                raise GitHubAPIError(f"Directory not found: {path}")
            elif e.response.status_code == 403:
                # Check if it's rate limit or permission issue
                remaining = e.response.headers.get("X-RateLimit-Remaining", "0")
                if remaining == "0":
                    reset_time = e.response.headers.get("X-RateLimit-Reset", "0")
                    raise GitHubAPIError(
                        f"API rate limit exceeded. Resets at {reset_time}"
                    )
                else:
                    raise GitHubAPIError(
                        "Access denied. Private repository requires authentication."
                    )
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
        progress_callback: Optional[callable] = None,
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
        results = {}
        errors = {}
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def fetch_with_semaphore(
            path: str,
        ) -> tuple[str, Optional[str], Optional[str]]:
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
        logger.info(
            f"Batch fetch complete: {len(results)} successful, {len(errors)} failed"
        )

        return results

    def build_tree_hierarchy(
        self, flat_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
            path = item["path"]
            path_map[path] = {
                "path": path,
                "name": item["name"],
                "type": item["type"],
                "size": item.get("size"),
                "children": [],
            }

        # Second pass: build hierarchy
        for path, node in path_map.items():
            parts = path.split("/")

            if len(parts) == 1:
                # Root level item
                root_items.append(node)
            else:
                # Find parent
                parent_path = "/".join(parts[:-1])
                if parent_path in path_map:
                    path_map[parent_path]["children"].append(node)

        # Sort items at each level
        def sort_tree(items):
            for item in items:
                if item["children"]:
                    sort_tree(item["children"])
            items.sort(key=lambda x: (x["type"] != "tree", x["name"].lower()))

        sort_tree(root_items)

        return root_items
