"""Pure, immutable state for a reader's cached item pages."""

from dataclasses import dataclass
from copy import deepcopy
from collections.abc import Hashable
from typing import Any

from ...Subscriptions.watchlist_item_page import WatchlistItemCursor, WatchlistItemPage


@dataclass(frozen=True)
class ReaderItemQuery:
    """Committed reader context and deterministic service keyword arguments.

    Attributes:
        context_key: Caller-defined identity for the reader context.
        kwargs: Sorted immutable service keyword argument pairs.
    """

    context_key: Any
    kwargs: tuple[tuple[str, Any], ...]

    @classmethod
    def freeze(cls, context_key: Any, kwargs: dict[str, Any]) -> "ReaderItemQuery":
        """Return a query detached from mutable caller-owned arguments.

        Args:
            context_key: Identity for the reader context.
            kwargs: Service arguments to freeze.

        Returns:
            An immutable query value.
        """
        def scalar(value: Any) -> Any:
            if type(value) not in (str, int, bool) and value is not None:
                raise TypeError("reader query values must be scalar")
            return value

        if isinstance(context_key, tuple):
            context_key = tuple(scalar(value) for value in context_key)
        else:
            context_key = scalar(context_key)
        frozen = []
        for key, value in sorted(kwargs.items()):
            if key == "statuses":
                if not isinstance(value, (list, tuple)):
                    raise TypeError("statuses must be a list or tuple")
                value = tuple(scalar(status) for status in value)
                if any(not isinstance(status, str) for status in value):
                    raise TypeError("statuses must contain strings")
            else:
                value = scalar(value)
            frozen.append((key, value))
        return cls(context_key, tuple(frozen))

    def as_kwargs(self) -> dict[str, Any]:
        """Return fresh keyword arguments suitable for a service call.

        Returns:
            A detached dictionary, including a fresh statuses list when present.
        """
        return {
            key: list(value) if key == "statuses" and isinstance(value, tuple) else value
            for key, value in self.kwargs
        }


@dataclass(frozen=True)
class ReaderItemSnapshot:
    """Committed visible pages plus separately staged traversal state.

    Attributes:
        query: Immutable committed reader query.
        watermark: Snapshot high-water item ID.
        snapshot_count: First-page total matching count.
        pages: Cached non-empty continuation pages, plus page zero.
        seen_ids: Stable identities already admitted to visible pages.
        cursor: Cursor for the next traversal request.
        has_more: Whether traversal has another candidate page.
        pending_arrivals: Count of arrivals held outside visible pages.
    """

    query: ReaderItemQuery
    watermark: int
    snapshot_count: int
    pages: tuple[tuple[dict[str, Any], ...], ...]
    seen_ids: frozenset[Any]
    cursor: WatchlistItemCursor | None
    has_more: bool
    pending_arrivals: int = 0

    @classmethod
    def start(cls, query: ReaderItemQuery, page: WatchlistItemPage) -> "ReaderItemSnapshot":
        """Create a snapshot from its required first page.

        Args:
            query: Immutable query associated with the page.
            page: First service response page.

        Returns:
            A new reader snapshot with page zero cached.

        Raises:
            ValueError: If the first page omits its snapshot count.
        """
        if page.snapshot_count is None:
            raise ValueError("first page must provide snapshot_count")
        items, seen = cls._unique_items(page.items, frozenset())
        return cls(query, page.snapshot_max_item_id, page.snapshot_count, (deepcopy(items),), seen, page.next_cursor, page.has_more)

    def with_continuation(self, page: WatchlistItemPage) -> tuple["ReaderItemSnapshot", bool]:
        """Stage a continuation page without mutating this committed snapshot.

        Args:
            page: Candidate continuation response.

        Returns:
            A candidate snapshot and whether a visible page was appended.

        Raises:
            ValueError: If the continuation watermark differs.
        """
        if page.snapshot_max_item_id != self.watermark:
            raise ValueError("continuation watermark differs from snapshot")
        items, seen = self._unique_items(page.items, self.seen_ids)
        pages = tuple(deepcopy(cached) for cached in self.pages) + ((deepcopy(items),) if items else ())
        candidate = ReaderItemSnapshot(
            self.query, self.watermark, self.snapshot_count, pages, seen,
            page.next_cursor, page.has_more, self.pending_arrivals,
        )
        return candidate, bool(items)

    @staticmethod
    def _item_id(item: dict[str, Any]) -> Hashable | None:
        """Normalize an item's explicit or fallback identity."""
        value = item.get("item_id")
        explicit = value is not None and not (isinstance(value, str) and not value.strip())
        if not explicit:
            value = item.get("id")
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        value = value.strip() if isinstance(value, str) else value
        return value if isinstance(value, Hashable) else None

    @classmethod
    def _unique_items(
        cls, items: tuple[dict[str, Any], ...], seen: frozenset[Any]
    ) -> tuple[tuple[dict[str, Any], ...], frozenset[Any]]:
        """Filter malformed and already-seen rows, preserving service order."""
        visible = []
        updated = set(seen)
        for item in items:
            identity = cls._item_id(item)
            if identity is None or identity in updated:
                continue
            updated.add(identity)
            visible.append(item)
        return tuple(visible), frozenset(updated)

    @property
    def page_count(self) -> int:
        """Return the number of visible pages currently cached."""
        return len(self.pages)

    def page(self, index: int) -> tuple[dict[str, Any], ...]:
        """Return a cached page.

        Args:
            index: Zero-based cached page index.

        Returns:
            The requested immutable page tuple.

        Raises:
            IndexError: If index is negative or outside the cache.
        """
        if index < 0 or index >= self.page_count:
            raise IndexError(index)
        return self.pages[index]

    def has_next(self, index: int) -> bool:
        """Return whether a page has a cached or service-backed successor.

        Args:
            index: Zero-based cached page index.

        Returns:
            True when a cached or traversable successor exists.

        Raises:
            IndexError: If index is outside the cache.
        """
        if index < 0 or index >= self.page_count:
            raise IndexError(index)
        if index < self.page_count - 1:
            return True
        return self.has_more
