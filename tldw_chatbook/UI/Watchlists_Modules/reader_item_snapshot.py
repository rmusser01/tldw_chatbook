"""Pure, immutable state for a reader's cached item pages."""

from dataclasses import dataclass
from typing import Any

from ...Subscriptions.watchlist_item_page import WatchlistItemCursor, WatchlistItemPage


@dataclass(frozen=True)
class ReaderItemQuery:
    """Committed reader context and deterministic service keyword arguments."""

    context_key: Any
    kwargs: tuple[tuple[str, Any], ...]

    @classmethod
    def freeze(cls, context_key: Any, kwargs: dict[str, Any]) -> "ReaderItemQuery":
        """Return a query detached from mutable caller-owned arguments."""
        frozen = []
        for key, value in sorted(kwargs.items()):
            if key == "statuses" and isinstance(value, list):
                value = tuple(value)
            frozen.append((key, value))
        return cls(context_key, tuple(frozen))

    def as_kwargs(self) -> dict[str, Any]:
        """Return fresh keyword arguments suitable for a service call."""
        return {
            key: list(value) if key == "statuses" and isinstance(value, tuple) else value
            for key, value in self.kwargs
        }


@dataclass(frozen=True)
class ReaderItemSnapshot:
    """Committed visible pages plus separately staged traversal state."""

    query: ReaderItemQuery
    watermark: int
    snapshot_count: int
    pages: tuple[tuple[dict[str, Any], ...], ...]
    seen_ids: frozenset[int]
    cursor: WatchlistItemCursor | None
    has_more: bool
    pending_arrivals: tuple[dict[str, Any], ...] = ()

    @classmethod
    def start(cls, query: ReaderItemQuery, page: WatchlistItemPage) -> "ReaderItemSnapshot":
        """Create a snapshot from its required first page."""
        if page.snapshot_count is None:
            raise ValueError("first page must provide snapshot_count")
        items, seen = cls._unique_items(page.items, frozenset())
        return cls(query, page.snapshot_max_item_id, page.snapshot_count, (items,), seen, page.next_cursor, page.has_more)

    def with_continuation(self, page: WatchlistItemPage) -> tuple["ReaderItemSnapshot", bool]:
        """Stage a continuation page without mutating this committed snapshot."""
        if page.snapshot_max_item_id != self.watermark:
            raise ValueError("continuation watermark differs from snapshot")
        items, seen = self._unique_items(page.items, self.seen_ids)
        pages = self.pages + ((items,) if items else ())
        candidate = ReaderItemSnapshot(
            self.query, self.watermark, self.snapshot_count, pages, seen,
            page.next_cursor, page.has_more, self.pending_arrivals,
        )
        return candidate, bool(items)

    @staticmethod
    def _item_id(item: dict[str, Any]) -> int | None:
        """Normalize an item's explicit or fallback identity."""
        value = item.get("item_id", item.get("id"))
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _unique_items(
        cls, items: tuple[dict[str, Any], ...], seen: frozenset[int]
    ) -> tuple[tuple[dict[str, Any], ...], frozenset[int]]:
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
        """Return a cached page, raising IndexError for an invalid index."""
        return self.pages[index]

    def has_next(self, index: int) -> bool:
        """Return whether a page has a cached or service-backed successor."""
        if index < 0 or index >= self.page_count:
            raise IndexError(index)
        if index < self.page_count - 1:
            return True
        return self.has_more
