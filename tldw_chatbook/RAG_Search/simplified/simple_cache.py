"""
Simple cache implementation for the RAG service.

This provides a lightweight caching solution for search results,
replacing the complex cache service from the old implementation.
"""

import hashlib
import json
import time
import threading
from collections import abc
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from dataclasses import dataclass, field
from collections import OrderedDict
from loguru import logger
import sys
import asyncio

from tldw_chatbook.Metrics.metrics_logger import (
    log_counter,
    log_histogram,
    log_gauge,
    timeit,
)


# The keyword leg's pre-TASK-15400 MATCH construction. A key built with
# this value renders byte-identically to every key built before the
# construction existed, which is the invariant
# `test_the_and_construction_keeps_the_hybrid_key_byte_identical` pins.
# Spelled out here rather than imported from `rag_service` (which imports
# THIS module) -- the two are kept honest by that test, not by an import.
LEGACY_FTS_MATCH_CONSTRUCTION = "and"


def _canonicalize_metadata_allowlist(
    metadata_allowlist: Optional[Any],
) -> Optional[Tuple[frozenset, ...]]:
    """Canonicalize a metadata allowlist into a stable, hashable form.

    ``None`` (or an empty mapping/sequence) stays ``None`` so cache keys
    built without an allowlist are byte-identical to keys built before this
    parameter existed -- no behavior change for existing callers.

    Two shapes are accepted, matching ``RAGService.search``: ONE mapping
    (every key AND-ed), or a SEQUENCE of mappings (a union of AND-groups --
    what ``rag_scope.build_semantic_allowlists`` returns, one entry per
    source type, because a flat dict cannot express "media in A OR note in
    B"). A single mapping canonicalizes to a one-element tuple, so the bare
    mapping and the one-element list are the same request and share a key.

    Each entry becomes a ``frozenset`` of ``(key, sorted_values_tuple)``
    pairs so dict key order and value iteration order (values are commonly
    passed as ``set``, whose iteration order is not guaranteed) do not
    affect equality; the entries are then sorted so two orderings of the
    same union agree. Entry BOUNDARIES survive, because ``[{a}, {b}]`` (a
    union) and ``{a, b}`` (an intersection) are different searches.
    """
    if not metadata_allowlist:
        return None
    if isinstance(metadata_allowlist, abc.Mapping):
        entries: List[Any] = [metadata_allowlist]
    else:
        # Empty entries are NOT filtered out: an entry that restricts nothing
        # makes a different request from one that is absent, and dropping it
        # here would let `[{a}, {}]` share a key with `[{a}]`. The engine
        # rejects that shape outright (`_allowlist_entries`), so this is the
        # key function agreeing with the guard rather than second-guessing it.
        entries = list(metadata_allowlist)
        if not entries:
            return None
    canonical = [
        frozenset((k, tuple(sorted(str(x) for x in v))) for k, v in entry.items())
        for entry in entries
    ]
    return tuple(sorted(canonical, key=sorted))


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class SimpleRAGCache:
    """
    Simple LRU cache for RAG search results with async-safe operations.

    Features:
    - LRU eviction policy
    - TTL support
    - Size limits
    - Basic metrics
    - Async-safe for single-user Textual app
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 3600,  # 1 hour default
        enabled: bool = True,
        ttl_by_search_type: Optional[Dict[str, float]] = None,
        max_memory_mb: float = 100.0,
    ):  # Default 100MB max memory
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Default time-to-live for cache entries in seconds
            enabled: Whether caching is enabled
            ttl_by_search_type: Optional dict mapping search types to specific TTLs
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self.ttl_by_search_type = ttl_by_search_type or {}
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Use threading.RLock for thread-safe operations
        # RLock allows the same thread to acquire the lock multiple times
        self._lock = threading.RLock()

        # Keep asyncio.Lock for async methods compatibility
        self._async_lock = asyncio.Lock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_memory_bytes = 0
        self._total_requests = 0
        self._last_prune_time = time.time()
        self._prune_interval = min(
            ttl_seconds / 2, 1800
        )  # Prune every half TTL or 30 minutes, whichever is less

        # Log initialization
        logger.info(
            f"Cache initialized: max_size={max_size}, ttl={ttl_seconds}s, enabled={enabled}"
        )
        log_gauge("cache_max_size", max_size)
        log_gauge("cache_ttl_seconds", ttl_seconds)
        log_counter("cache_initialized", labels={"enabled": str(enabled)})

    def _make_key(
        self,
        query: str,
        search_type: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        metadata_allowlist: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = None,
        keyword_source_types: Optional[Collection[str]] = None,
        hybrid_fusion: Optional[Tuple[float, int, int]] = None,
        fts_match_construction: Optional[str] = None,
    ) -> str:
        """
        Create a cache key from search parameters.

        Uses xxhash for better performance than MD5.

        Args:
            query: The search query
            search_type: Type of search (semantic, hybrid, keyword)
            top_k: Number of results
            filters: Optional metadata filters
            metadata_allowlist: Optional metadata key -> allowed-values
                scoping filter -- ONE mapping (keys AND-ed) or a SEQUENCE
                of mappings (a union of AND-groups, the shape
                ``rag_scope.build_semantic_allowlists`` returns). Included
                in the key so two searches that are identical except for
                their scope never share a cached entry: a shared key is how
                a scoped search silently serves an unscoped one's rows.
                Defaults to ``None``, which produces a key identical to
                what this method returned before this parameter existed --
                and a ONE-entry allowlist keeps its pre-union rendering, so
                no existing caller's key moved either.
            keyword_source_types: Optional keyword-leg source-type selection
                (TASK-14751). Two searches identical except for which
                source types the FTS leg budgets for return DIFFERENT rows,
                so they must never share an entry -- a shared key would
                silently serve a media-only search's rows to a notes-only
                one. ``None`` is omitted from the key, so keys built without
                a selection stay byte-identical to before.
            hybrid_fusion: The RESOLVED ``(alpha, rrf_k, pool_multiplier)``
                actually used for a HYBRID search (TASK-4110 review,
                important 1). These change the fused row set/ordering just
                as much as ``top_k`` does, so without this, two hybrid
                searches identical except for `rrf_k` (or alpha, or the
                pool multiplier) would share one entry -- the SECOND
                request silently served the FIRST's stale results forever.
                Caught live: this is what would have made Task 4's
                strategy sweep report every k as "+0.000, k doesn't
                matter" on a single cached service. Callers pass ``None``
                for semantic/keyword search (those legs never depend on
                these three), and the caller (``RAGService.search``) always
                passes the RESOLVED values -- not the raw config
                attributes -- for hybrid, so two configs that happen to
                resolve to the same effective value (e.g. an out-of-range
                alpha and 0.7) correctly SHARE an entry rather than
                needlessly splitting one. A default-config hybrid key's
                exact bytes change as a result (a new key part is always
                present for `search_type == "hybrid"`); nothing pins the
                literal hash, and the cache is in-process/ephemeral only.
            fts_match_construction: The keyword leg's MATCH construction
                (TASK-15400) -- the companion to ``hybrid_fusion`` for the
                OTHER leg, and part of the key for the same reason: it
                changes which rows the FTS leg returns at all, so two
                searches identical except for it must never share an entry.
                The construction is not a user knob, but it IS mutable at
                runtime (the arc's sweep varies it on a live
                ``SearchConfig`` against a per-service cache), which is
                precisely the shape that made TASK-4110's sweep report "k
                doesn't matter" before the fusion params entered the key.
                Passed for hybrid AND keyword searches (both read the FTS
                leg), ``None`` for semantic. The pre-arc construction
                (``"and"``) contributes NO key part, so every key built
                before this parameter existed stays byte-identical. Since
                TASK-15400 shipped ``and_stopword_trim`` as the default
                (2026-08-11) that is no longer the key a DEFAULT search
                renders, and TASK-15700 moved it again (2026-08-13) to
                ``and_then_prefix``, so today a default search carries
                ``fts:and_then_prefix``. That is the point: this key is
                VALUE-keyed on the construction, so entries cached under any
                previous one are keyed apart rather than served to the new
                one. The cost is a one-time run of cold misses after each
                such flip, which is the correct trade and not new to either.

        Returns:
            A unique cache key
        """
        # Create a stable representation of the parameters
        key_parts = [
            query.lower().strip(),
            search_type,
            str(top_k),
            json.dumps(filters or {}, sort_keys=True),
        ]

        canonical_allowlist = _canonicalize_metadata_allowlist(metadata_allowlist)
        if canonical_allowlist is not None:
            # Sort for a deterministic string representation: frozenset
            # iteration order is not guaranteed (and is hash-seed
            # dependent for strings), but sorting the (key, values) tuples
            # is stable.
            if len(canonical_allowlist) == 1:
                # One AND-group: rendered exactly as it was before the union
                # shape existed, so every pre-B1 key stays byte-identical.
                key_parts.append(json.dumps(sorted(canonical_allowlist[0])))
            else:
                # A union of AND-groups. The extra nesting level is what
                # keeps `[{a}, {b}]` from colliding with `{a, b}` -- two
                # different searches over the same values.
                key_parts.append(
                    "allowlists:"
                    + json.dumps([sorted(entry) for entry in canonical_allowlist])
                )

        if keyword_source_types is not None:
            # Prefixed so this part can never be confused with the
            # allowlist part above, and sorted so set iteration order (which
            # is hash-seed dependent for strings) does not affect the key.
            # An EMPTY selection is a real, distinct request ("no keyword
            # leg") and must not collapse onto `None`, which is why the
            # presence of the part -- not its truthiness -- is the test.
            key_parts.append(
                "kst:"
                + json.dumps(sorted(str(x) for x in keyword_source_types))
            )

        if hybrid_fusion is not None:
            alpha, rrf_k, pool_multiplier = hybrid_fusion
            key_parts.append(
                "fusion:"
                + json.dumps(
                    {"alpha": alpha, "rrf_k": rrf_k, "pool_multiplier": pool_multiplier},
                    sort_keys=True,
                )
            )

        if (
            fts_match_construction is not None
            and fts_match_construction != LEGACY_FTS_MATCH_CONSTRUCTION
        ):
            # Prefixed like the parts above so it can never be confused with
            # another one. Omitted for the LEGACY (pre-TASK-15400)
            # construction (see the parameter's docstring) -- this is that
            # value's byte-identity guarantee, not a "falsy means absent"
            # test, and it is deliberately NOT re-pointed at the new default.
            key_parts.append("fts:" + fts_match_construction)

        # Use a faster hash function - fallback to md5 if xxhash not available
        key_str = "|".join(key_parts)
        try:
            import xxhash

            return xxhash.xxh64(key_str.encode()).hexdigest()
        except ImportError:
            # Fallback to MD5 for stable hashing across processes
            # MD5 is fine for cache keys - we don't need cryptographic security
            return hashlib.md5(key_str.encode()).hexdigest()

    async def get_async(
        self,
        query: str,
        search_type: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        metadata_allowlist: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = None,
        *,
        keyword_source_types: Optional[Collection[str]] = None,
        hybrid_fusion: Optional[Tuple[float, int, int]] = None,
        fts_match_construction: Optional[str] = None,
    ) -> Optional[Tuple[List[Any], str]]:
        """
        Async-safe get cached search results.

        Args:
            query: The search query
            search_type: Type of search
            top_k: Number of results
            filters: Optional metadata filters
            metadata_allowlist: Optional metadata scoping filter; included
                in the cache key so differently-scoped searches never share
                a cache entry. Defaults to ``None`` (no change for existing
                callers).
            keyword_source_types: Optional keyword-leg source-type selection;
                also part of the key, for the same reason.
            hybrid_fusion: The resolved ``(alpha, rrf_k, pool_multiplier)``
                for a hybrid search; see ``_make_key`` for why this must be
                part of the key.
            fts_match_construction: The keyword leg's MATCH construction
                (hybrid/keyword searches); see ``_make_key``.

        Returns:
            Tuple of (results, context) if found and valid, None otherwise
        """
        if not self.enabled:
            return None

        async with self._async_lock:
            self._total_requests += 1

            # Check if we need to prune expired entries
            current_time = time.time()
            if current_time - self._last_prune_time > self._prune_interval:
                await self._prune_expired_async()
                self._last_prune_time = current_time

            key = self._make_key(
                query,
                search_type,
                top_k,
                filters,
                metadata_allowlist,
                keyword_source_types,
                hybrid_fusion,
                fts_match_construction,
            )
            log_counter("cache_request", labels={"type": search_type})

            if key not in self._cache:
                self._misses += 1
                log_counter("cache_miss", labels={"type": search_type})
                logger.debug(f"Cache miss for query: '{query[:50]}...'")
                return None

            entry = self._cache[key]

            # Check TTL - use search type specific TTL if available
            ttl = self.ttl_by_search_type.get(search_type, self.ttl_seconds)
            age = time.time() - entry.timestamp
            if ttl is not None and age > ttl:
                # Expired
                del self._cache[key]
                self._misses += 1
                log_counter("cache_expired", labels={"type": search_type})
                log_histogram("cache_entry_expired_age_seconds", age)
                logger.debug(
                    f"Cache entry expired for query: '{query[:50]}...' (age: {age:.1f}s, ttl: {ttl}s)"
                )
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access()

            self._hits += 1
            log_counter("cache_hit", labels={"type": search_type})
            log_histogram("cache_entry_age_seconds", age)
            log_histogram("cache_entry_access_count", entry.access_count)

            # Update hit rate metric
            hit_rate = (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0
            )
            log_gauge("cache_hit_rate", hit_rate)

            logger.debug(
                f"Cache hit for query: '{query[:50]}...' (age: {age:.1f}s, accesses: {entry.access_count})"
            )

            return entry.value

    def get(
        self,
        query: str,
        search_type: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        metadata_allowlist: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = None,
        keyword_source_types: Optional[Collection[str]] = None,
        hybrid_fusion: Optional[Tuple[float, int, int]] = None,
        fts_match_construction: Optional[str] = None,
    ) -> Optional[Tuple[List[Any], str]]:
        """
        Thread-safe synchronous cache get.

        This method is safe to call from any context and will not cause deadlocks.
        For better performance in async contexts, use get_async() directly.

        TASK-15701: the three search-defining dimensions below are accepted
        and forwarded so this path renders the SAME key as `get_async`. They
        were previously absent from the signature entirely, which made the
        sync key a legacy, construction-less one -- harmless while `and` was
        both the legacy value and the shipped default, and a WRONG-HIT risk
        once the default moved (twice). The API was kept rather than removed
        because it is the cache's ergonomic test surface (58 call sites);
        production reads and writes go through the async path only, re-verified
        at fix time.

        Args:
            query: The search query.
            search_type: Which retrieval mode ran.
            top_k: The result window.
            filters: Optional metadata filters.
            metadata_allowlist: Optional per-source-type allowlist.
            keyword_source_types: Which sub-legs the keyword leg queried.
            hybrid_fusion: (alpha, rrf_k, pool) for the fusion step.
            fts_match_construction: How the FTS MATCH expression was built.

        Returns:
            The cached (results, context) pair, or None on a miss.
        """
        if not self.enabled:
            return None

        # Use a separate thread to avoid event loop conflicts
        import concurrent.futures

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._sync_get_impl,
                    query,
                    search_type,
                    top_k,
                    filters,
                    metadata_allowlist,
                    keyword_source_types,
                    hybrid_fusion,
                    fts_match_construction,
                )
                return future.result(
                    timeout=1.0
                )  # 1 second timeout for cache operations
        except RuntimeError:
            # No running loop, safe to run directly
            return self._sync_get_impl(
                query,
                search_type,
                top_k,
                filters,
                metadata_allowlist,
                keyword_source_types,
                hybrid_fusion,
                fts_match_construction,
            )

    def _sync_get_impl(
        self,
        query: str,
        search_type: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        metadata_allowlist: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = None,
        keyword_source_types: Optional[Collection[str]] = None,
        hybrid_fusion: Optional[Tuple[float, int, int]] = None,
        fts_match_construction: Optional[str] = None,
    ) -> Optional[Tuple[List[Any], str]]:
        """Internal synchronous implementation using threading lock."""
        with self._lock:
            self._total_requests += 1

            # Check if we need to prune expired entries
            current_time = time.time()
            if current_time - self._last_prune_time > self._prune_interval:
                self._prune_expired_sync()
                self._last_prune_time = current_time

            key = self._make_key(
                query,
                search_type,
                top_k,
                filters,
                metadata_allowlist,
                keyword_source_types,
                hybrid_fusion,
                fts_match_construction,
            )
            log_counter("cache_request", labels={"type": search_type})

            if key not in self._cache:
                self._misses += 1
                log_counter("cache_miss", labels={"type": search_type})
                logger.debug(f"Cache miss for query: '{query[:50]}...'")
                return None

            entry = self._cache[key]

            # Get TTL for this search type
            ttl = self.ttl_by_search_type.get(search_type, self.ttl_seconds)

            # Check if entry has expired
            if time.time() - entry.timestamp > ttl:
                self._misses += 1
                log_counter("cache_expired", labels={"type": search_type})
                logger.debug(f"Cache expired for query: '{query[:50]}...'")
                del self._cache[key]
                self._update_memory_sync()
                return None

            # Update access stats
            entry.access()

            # Move to end for LRU
            self._cache.move_to_end(key)

            self._hits += 1
            log_counter("cache_hit", labels={"type": search_type})
            logger.debug(f"Cache hit for query: '{query[:50]}...'")

            # Check for corrupted entry (None value)
            if entry.value is None:
                logger.warning(f"Corrupted cache entry detected for key: {key}")
                # Remove corrupted entry
                del self._cache[key]
                self._hits -= 1  # Correct the hit count
                self._misses += 1  # Count as miss
                return None

            return entry.value

    async def put_async(
        self,
        query: str,
        search_type: str,
        top_k: int,
        results: List[Any],
        context: str,
        filters: Optional[Dict[str, Any]] = None,
        metadata_allowlist: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = None,
        *,
        keyword_source_types: Optional[Collection[str]] = None,
        hybrid_fusion: Optional[Tuple[float, int, int]] = None,
        fts_match_construction: Optional[str] = None,
    ) -> None:
        """
        Async-safe cache search results.

        Args:
            query: The search query
            search_type: Type of search
            top_k: Number of results
            results: Search results to cache
            context: Context string to cache
            filters: Optional metadata filters
            metadata_allowlist: Optional metadata scoping filter; included
                in the cache key so differently-scoped searches never share
                a cache entry. Defaults to ``None`` (no change for existing
                callers).
            keyword_source_types: Optional keyword-leg source-type selection;
                also part of the key. **TASK-15701 closed the sync-twin gap
                this argument used to describe**: ``get``/``put`` now accept
                and forward this dimension too, so both paths render the same
                key and two sync searches differing only in it can no longer
                collide. (The historical hazard, for anyone reading a git
                blame: the sync twins could not take it, so two sync searches
                differing only in an omitted dimension rendered the SAME key
                and the second was served the first's rows -- a wrong hit,
                latent only because no production code called the sync API.)
            hybrid_fusion: The resolved ``(alpha, rrf_k, pool_multiplier)``
                for a hybrid search; see ``_make_key`` for why this must be
                part of the key. Carried by both paths since TASK-15701.
            fts_match_construction: The keyword leg's MATCH construction
                (hybrid/keyword searches); see ``_make_key``. Carried by both
                paths since TASK-15701.
        """
        if not self.enabled:
            return

        async with self._async_lock:
            key = self._make_key(
                query,
                search_type,
                top_k,
                filters,
                metadata_allowlist,
                keyword_source_types,
                hybrid_fusion,
                fts_match_construction,
            )

            # Calculate memory for new entry
            entry = CacheEntry(key=key, value=(results, context), timestamp=time.time())
            entry_memory = self._deep_getsizeof(entry)

            # Evict entries if needed (by size or memory)
            while (len(self._cache) >= self.max_size and key not in self._cache) or (
                self._current_memory_bytes + entry_memory > self.max_memory_bytes
            ):
                if not self._cache:
                    # Cache is empty but we still exceed memory - entry is too large
                    logger.warning(
                        f"Entry too large for cache: {entry_memory / 1024 / 1024:.1f}MB"
                    )
                    return

                # Evict least recently used
                oldest_key = next(iter(self._cache))
                evicted_entry = self._cache[oldest_key]
                evicted_memory = self._deep_getsizeof(evicted_entry)

                del self._cache[oldest_key]
                self._evictions += 1
                self._current_memory_bytes -= evicted_memory

                # Log eviction details
                log_counter(
                    "cache_eviction",
                    labels={
                        "type": search_type,
                        "reason": "memory"
                        if self._current_memory_bytes + entry_memory
                        > self.max_memory_bytes
                        else "size",
                    },
                )
                eviction_age = time.time() - evicted_entry.timestamp
                log_histogram("cache_evicted_entry_age_seconds", eviction_age)
                log_histogram(
                    "cache_evicted_entry_access_count", evicted_entry.access_count
                )
                log_histogram(
                    "cache_evicted_entry_memory_mb", evicted_memory / 1024 / 1024
                )
                logger.debug(
                    f"Evicted cache entry (age: {eviction_age:.1f}s, accesses: {evicted_entry.access_count}, memory: {evicted_memory / 1024 / 1024:.1f}MB)"
                )

            # Remove old entry if updating
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory_bytes -= self._deep_getsizeof(old_entry)

            # Store the entry
            self._cache[key] = entry
            self._current_memory_bytes += entry_memory

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Log cache statistics
            log_counter("cache_put", labels={"type": search_type})
            log_histogram("cache_result_count", len(results))
            log_histogram("cache_context_size", len(context))
            log_gauge("cache_current_size", len(self._cache))
            log_gauge("cache_eviction_count", self._evictions)
            log_gauge("cache_memory_usage_mb", self._current_memory_bytes / 1024 / 1024)
            log_histogram("cache_entry_size_mb", entry_memory / 1024 / 1024)

            logger.debug(
                f"Cached results for query: '{query[:50]}...' ({len(results)} results, {entry_memory / 1024 / 1024:.1f}MB)"
            )

    def put(
        self,
        query: str,
        search_type: str,
        top_k: int,
        results: List[Any],
        context: str,
        filters: Optional[Dict[str, Any]] = None,
        metadata_allowlist: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = None,
        keyword_source_types: Optional[Collection[str]] = None,
        hybrid_fusion: Optional[Tuple[float, int, int]] = None,
        fts_match_construction: Optional[str] = None,
    ) -> None:
        """
        Thread-safe synchronous cache put.

        This method is safe to call from any context and will not cause deadlocks.
        For better performance in async contexts, use put_async() directly.

        TASK-15701: forwards the same three search-defining dimensions as
        `put_async`, so an entry cannot be STORED under a key asserting a
        construction that did not produce it. See `get` for why the sync API
        was kept rather than removed.

        Args:
            query: The search query.
            search_type: Which retrieval mode ran.
            top_k: The result window.
            results: The rows to cache.
            context: The rendered context string.
            filters: Optional metadata filters.
            metadata_allowlist: Optional per-source-type allowlist.
            keyword_source_types: Which sub-legs the keyword leg queried.
            hybrid_fusion: (alpha, rrf_k, pool) for the fusion step.
            fts_match_construction: How the FTS MATCH expression was built.
        """
        if not self.enabled:
            return

        # Use a separate thread to avoid event loop conflicts
        import concurrent.futures

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._sync_put_impl,
                    query,
                    search_type,
                    top_k,
                    results,
                    context,
                    filters,
                    metadata_allowlist,
                    keyword_source_types,
                    hybrid_fusion,
                    fts_match_construction,
                )
                future.result(timeout=1.0)  # 1 second timeout for cache operations
        except RuntimeError:
            # No running loop, safe to run directly
            self._sync_put_impl(
                query,
                search_type,
                top_k,
                results,
                context,
                filters,
                metadata_allowlist,
                keyword_source_types,
                hybrid_fusion,
                fts_match_construction,
            )

    def _sync_put_impl(
        self,
        query: str,
        search_type: str,
        top_k: int,
        results: List[Any],
        context: str,
        filters: Optional[Dict[str, Any]],
        metadata_allowlist: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = None,
        keyword_source_types: Optional[Collection[str]] = None,
        hybrid_fusion: Optional[Tuple[float, int, int]] = None,
        fts_match_construction: Optional[str] = None,
    ) -> None:
        """Internal synchronous implementation using threading lock."""
        with self._lock:
            key = self._make_key(
                query,
                search_type,
                top_k,
                filters,
                metadata_allowlist,
                keyword_source_types,
                hybrid_fusion,
                fts_match_construction,
            )

            # Create cache entry
            entry = CacheEntry(key=key, value=(results, context), timestamp=time.time())

            # Check memory pressure before adding
            try:
                entry_memory = self._estimate_entry_size(entry)
            except Exception as e:
                logger.warning(f"Error estimating entry size, using fallback: {e}")
                # Fallback to a reasonable default size (1KB)
                entry_memory = 1024

            # Evict if necessary to make room
            while (
                self._current_memory_bytes + entry_memory > self.max_memory_bytes
                or len(self._cache) >= self.max_size
            ) and self._cache:
                self._evict_lru_sync()

            # Add new entry
            self._cache[key] = entry
            self._current_memory_bytes += entry_memory

            # Update metrics
            log_counter("cache_put", labels={"type": search_type})
            log_gauge("cache_size", len(self._cache))
            log_gauge("cache_memory_mb", self._current_memory_bytes / 1024 / 1024)

            logger.debug(
                f"Cached results for query: '{query[:50]}...' ({len(results)} results, {entry_memory / 1024 / 1024:.1f}MB)"
            )

    async def clear_async(self) -> None:
        """Async-safe clear all cache entries."""
        async with self._async_lock:
            size_before = len(self._cache)
            memory_before = self._current_memory_bytes / 1024 / 1024
            self._cache.clear()
            self._current_memory_bytes = 0
            log_counter("cache_cleared")
            log_gauge("cache_current_size", 0)
            log_gauge("cache_memory_usage_mb", 0)
            logger.info(
                f"Cache cleared ({size_before} entries, {memory_before:.1f}MB removed)"
            )

    def clear(self) -> None:
        """Thread-safe synchronous cache clear."""
        import concurrent.futures

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._sync_clear_impl)
                future.result(timeout=1.0)  # 1 second timeout
        except RuntimeError:
            # No running loop, safe to run directly
            self._sync_clear_impl()

    def _sync_clear_impl(self) -> None:
        """Internal synchronous implementation using threading lock."""
        with self._lock:
            size_before = len(self._cache)
            memory_before = self._current_memory_bytes / 1024 / 1024
            self._cache.clear()
            self._current_memory_bytes = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            log_counter("cache_cleared")
            log_gauge("cache_size", 0)
            log_gauge("cache_memory_mb", 0)
            logger.info(
                f"Cache cleared ({size_before} entries, {memory_before:.1f}MB removed)"
            )

    def _deep_getsizeof(self, obj, seen=None, max_depth=3, current_depth=0):
        """
        Calculate the deep size of an object, including all referenced objects.

        Args:
            obj: The object to measure
            seen: Set of already-seen object ids to avoid infinite recursion
            max_depth: Maximum recursion depth to prevent O(n²) complexity
            current_depth: Current recursion depth

        Returns:
            Size in bytes
        """
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0

        # Mark this object as seen first
        seen.add(obj_id)

        # Stop recursion at max depth to prevent O(n²) complexity
        if current_depth >= max_depth:
            # Return a rough estimate based on string representation
            return size + len(str(obj)) if hasattr(obj, "__str__") else size

        if isinstance(obj, dict):
            size += sum(
                self._deep_getsizeof(k, seen, max_depth, current_depth + 1)
                + self._deep_getsizeof(v, seen, max_depth, current_depth + 1)
                for k, v in obj.items()
            )
        elif hasattr(obj, "__dict__"):
            size += self._deep_getsizeof(
                obj.__dict__, seen, max_depth, current_depth + 1
            )
        elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
            try:
                size += sum(
                    self._deep_getsizeof(i, seen, max_depth, current_depth + 1)
                    for i in obj
                )
            except TypeError:
                # Some iterables don't support iteration in all contexts
                pass

        return size

    def _estimate_entry_size(self, entry: CacheEntry) -> int:
        """
        Estimate the memory size of a cache entry.
        Uses a fast estimation approach instead of deep recursion.
        """
        # Base size of the CacheEntry object
        size = sys.getsizeof(entry)

        # Add key size
        size += sys.getsizeof(entry.key)

        # Estimate value size (tuple of results and context)
        if isinstance(entry.value, tuple) and len(entry.value) == 2:
            results, context = entry.value

            # Estimate results size (list of SearchResult objects)
            if isinstance(results, list):
                # Base list size
                size += sys.getsizeof(results)
                # Estimate 1KB per result (reasonable for SearchResult objects)
                size += len(results) * 1024

            # Add context string size
            if isinstance(context, str):
                size += sys.getsizeof(context)
        else:
            # Fallback: use string representation length as rough estimate
            size += len(str(entry.value)) * 2  # 2 bytes per character estimate

        # Add overhead for metadata (timestamp, access_count, etc)
        size += 100  # Fixed overhead estimate

        return size

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache metrics.

        Returns:
            Dictionary of cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        # Use tracked memory size for efficiency
        size_bytes = self._current_memory_bytes

        metrics = {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
            "size_bytes": size_bytes,
            "max_memory_bytes": self.max_memory_bytes,
            "memory_usage_percent": (size_bytes / self.max_memory_bytes * 100)
            if self.max_memory_bytes > 0
            else 0,
            "total_requests": self._total_requests,
        }

        # Log key metrics
        log_gauge("cache_hit_rate", hit_rate)
        log_gauge("cache_memory_estimate_mb", size_bytes / (1024 * 1024))
        log_gauge(
            "cache_fill_ratio",
            len(self._cache) / self.max_size if self.max_size > 0 else 0,
        )

        return metrics

    def log_cache_efficiency(self):
        """Log cache efficiency metrics - should be called periodically."""
        metrics = self.get_metrics()

        logger.info(
            f"Cache efficiency: hit_rate={metrics['hit_rate']:.2%}, "
            f"size={metrics['size']}/{metrics['max_size']}, "
            f"memory={metrics['size_bytes'] / (1024 * 1024):.1f}MB, "
            f"evictions={metrics['evictions']}"
        )

    async def _prune_expired_async(self) -> int:
        """
        Internal async method to prune expired entries.
        Called automatically during cache operations.
        """
        if not self.enabled:
            return 0

        current_time = time.time()
        expired_keys = []
        total_age = 0
        total_accesses = 0

        for key, entry in self._cache.items():
            age = current_time - entry.timestamp
            if age > self.ttl_seconds:
                expired_keys.append(key)
                total_age += age
                total_accesses += entry.access_count

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            avg_age = total_age / len(expired_keys)
            avg_accesses = total_accesses / len(expired_keys)
            log_counter("cache_entries_expired", value=len(expired_keys))
            log_histogram("cache_pruned_avg_age_seconds", avg_age)
            log_histogram("cache_pruned_avg_access_count", avg_accesses)
            log_gauge("cache_current_size", len(self._cache))
            logger.debug(f"Auto-pruned {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    @timeit("cache_prune_expired")
    def prune_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        if not self.enabled:
            return 0

        current_time = time.time()
        expired_keys = []
        total_age = 0
        total_accesses = 0

        for key, entry in self._cache.items():
            age = current_time - entry.timestamp
            if age > self.ttl_seconds:
                expired_keys.append(key)
                total_age += age
                total_accesses += entry.access_count

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            avg_age = total_age / len(expired_keys)
            avg_accesses = total_accesses / len(expired_keys)
            log_counter("cache_entries_expired", value=len(expired_keys))
            log_histogram("cache_pruned_avg_age_seconds", avg_age)
            log_histogram("cache_pruned_avg_access_count", avg_accesses)
            log_gauge("cache_current_size", len(self._cache))
            logger.info(
                f"Pruned {len(expired_keys)} expired cache entries (avg age: {avg_age:.1f}s)"
            )

        return len(expired_keys)

    def _prune_expired_sync(self) -> int:
        """
        Internal synchronous method to prune expired entries.
        Assumes lock is already held.
        """
        current_time = time.time()
        expired_keys = []

        for key, entry in self._cache.items():
            # Get TTL for the search type (stored in entry if available)
            ttl = self.ttl_seconds
            age = current_time - entry.timestamp
            if age > ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            self._update_memory_sync()
            log_counter("cache_entries_expired", value=len(expired_keys))

        return len(expired_keys)

    def _evict_lru_sync(self) -> None:
        """
        Evict least recently used entry.
        Assumes lock is already held.
        """
        if not self._cache:
            return

        # Get the first item (least recently used)
        key, entry = next(iter(self._cache.items()))
        del self._cache[key]

        # Update memory tracking
        entry_size = self._estimate_entry_size(entry)
        self._current_memory_bytes = max(0, self._current_memory_bytes - entry_size)
        self._evictions += 1

        log_counter("cache_eviction")
        logger.debug(f"Evicted LRU entry: {key[:16]}...")

    def _update_memory_sync(self) -> None:
        """
        Update memory usage tracking.
        Assumes lock is already held.
        """
        total_memory = 0
        for entry in self._cache.values():
            total_memory += self._estimate_entry_size(entry)
        self._current_memory_bytes = total_memory
        log_gauge("cache_memory_mb", self._current_memory_bytes / 1024 / 1024)

    def __len__(self) -> int:
        """Get the number of cached entries."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache


# Global cache instance (can be replaced with per-service instance if needed)
_global_cache: Optional[SimpleRAGCache] = None


def get_rag_cache(
    max_size: int = 100,
    ttl_seconds: float = 3600,
    enabled: bool = True,
    ttl_by_search_type: Optional[Dict[str, float]] = None,
) -> SimpleRAGCache:
    """
    Get or create the global RAG cache instance.

    Args:
        max_size: Maximum number of entries
        ttl_seconds: Time-to-live in seconds
        enabled: Whether caching is enabled
        ttl_by_search_type: Optional dict mapping search types to specific TTLs

    Returns:
        The cache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = SimpleRAGCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            enabled=enabled,
            ttl_by_search_type=ttl_by_search_type,
        )

    return _global_cache
