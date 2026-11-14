"""Pure, immutable state for a reader's cached item pages."""

from collections.abc import Hashable, Iterable, Mapping, Callable
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from ...Subscriptions.watchlist_item_page import WatchlistItemCursor, WatchlistItemPage


def _frozen_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Detach one caller-owned row and freeze it read-only.

    Plain dicts (service responses, displaced rows) are shallow-copied so
    the snapshot never aliases a dict its caller can still reach, then
    wrapped in ``MappingProxyType``. Rows that are already frozen proxies
    are shared as-is: their underlying dicts are snapshot-private and
    unreachable except through the proxy, so re-wrapping would only add a
    layer per continuation turn.

    Args:
        row: A row admitted to (or staged behind) the snapshot's pages.

    Returns:
        The immutable row value to store.
    """
    if isinstance(row, MappingProxyType):
        return row
    return MappingProxyType(dict(row))


def _frozen_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """Freeze a sequence of rows once, sharing already-frozen rows."""
    return tuple(_frozen_row(row) for row in rows)


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

        Raises:
            TypeError: If context or arguments contain unsupported mutable values.
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
            key: list(value)
            if key == "statuses" and isinstance(value, tuple)
            else value
            for key, value in self.kwargs
        }


@dataclass(frozen=True)
class ReaderItemSnapshot:
    """Committed visible pages plus separately staged traversal state.

    Pages are structurally shared: ``with_continuation`` and friends append
    new page tuples and reuse the cached page objects verbatim, so paging to
    depth N does O(new page) work instead of copying every cached row. The
    immutability the copies used to buy is enforced at the construction
    boundary instead — every stored row is a ``MappingProxyType`` over a
    snapshot-private dict.

    Attributes:
        query: Immutable committed reader query.
        watermark: Snapshot high-water item ID.
        snapshot_count: First-page total matching count.
        pages: Cached non-empty continuation pages, plus page zero.
        seen_ids: Stable identities already admitted to visible pages.
        cursor: Cursor for the next traversal request.
        has_more: Whether traversal has another candidate page.
        pending_items: Backend-ordered rows displaced from a full visible page.
        pending_arrivals: Count of arrivals held outside visible pages.
    """

    query: ReaderItemQuery
    watermark: int
    snapshot_count: int
    pages: tuple[tuple[Mapping[str, Any], ...], ...]
    seen_ids: frozenset[Any]
    cursor: WatchlistItemCursor | None
    has_more: bool
    pending_items: tuple[Mapping[str, Any], ...] = ()
    pending_arrivals: int = 0

    @classmethod
    def start(
        cls, query: ReaderItemQuery, page: WatchlistItemPage
    ) -> "ReaderItemSnapshot":
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
        return cls(
            query=query,
            watermark=page.snapshot_max_item_id,
            snapshot_count=page.snapshot_count,
            pages=(_frozen_rows(items),),
            seen_ids=seen,
            cursor=page.next_cursor,
            has_more=page.has_more,
        )

    def with_pending_items(
        self, items: tuple[Mapping[str, Any], ...]
    ) -> "ReaderItemSnapshot":
        """Stage displaced backend rows for the next service continuation."""
        pending, _ = self._unique_items((*self.pending_items, *items), self.seen_ids)
        return replace(self, pending_items=_frozen_rows(pending))

    def with_continuation(
        self, page: WatchlistItemPage, *, page_size: int | None = None
    ) -> tuple["ReaderItemSnapshot", bool]:
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
        items, _ = self._unique_items((*self.pending_items, *page.items), self.seen_ids)
        visible = items if page_size is None else items[:page_size]
        pending = () if page_size is None else items[page_size:]
        visible, seen = self._unique_items(visible, self.seen_ids)
        pages = self.pages + ((_frozen_rows(visible),) if visible else ())
        candidate = replace(
            self,
            pages=pages,
            seen_ids=seen,
            cursor=page.next_cursor,
            has_more=page.has_more,
            pending_items=_frozen_rows(pending),
        )
        return candidate, bool(visible)

    def with_pending_page(self, page_size: int) -> tuple["ReaderItemSnapshot", bool]:
        """Publish one final page from staged rows after service exhaustion."""
        visible = self.pending_items[:page_size]
        if not visible:
            return self, False
        visible, seen = self._unique_items(visible, self.seen_ids)
        pending = self.pending_items[page_size:]
        return (
            replace(
                self,
                pages=self.pages + (_frozen_rows(visible),),
                seen_ids=seen,
                pending_items=_frozen_rows(pending),
            ),
            True,
        )

    def close_to_cached_pages(self) -> "ReaderItemSnapshot":
        """Drop an unreachable service tail and advertise only cached rows."""
        return replace(
            self,
            snapshot_count=sum(len(page) for page in self.pages),
            cursor=None,
            has_more=False,
            pending_items=(),
        )

    def patch_cached_rows(
        self,
        matches: Callable[[Mapping[str, Any]], bool],
        changes: Mapping[str, Any],
    ) -> Any:
        """Rewrite matching cached rows in place, keeping this object's identity.

        Cached rows are frozen mappings, so a committed-row patch (status,
        star, queue flag) cannot mutate them; each matching row is rebuilt
        once and the ``pages`` tuple is swapped on THIS snapshot object
        rather than returning a new snapshot. Object identity is
        load-bearing in the Watchlists screen: in-flight page loads capture
        ``self._items_snapshot is snapshot`` to detect supersession, and
        replacing the object here would abort them mid-flight and strand
        ``_items_page_loading``. Swapping ``pages`` in place keeps every
        such guard true while old page tuples remain immutable values
        shared with whatever already holds them.

        Args:
            matches: Predicate selecting the cached rows to rebuild.
            changes: Field updates applied on top of each matching row.

        Returns:
            The ``id`` of the first matching row, or ``None`` when no
            cached row matched.
        """
        row_key: Any = None
        rebuilt_pages: list[tuple[Mapping[str, Any], ...]] = []
        for page in self.pages:
            if not any(matches(row) for row in page):
                rebuilt_pages.append(page)
                continue
            page_key: Any = None
            rebuilt_rows: list[Mapping[str, Any]] = []
            for row in page:
                if matches(row):
                    if page_key is None:
                        page_key = row.get("id")
                    rebuilt_rows.append(_frozen_row({**row, **changes}))
                else:
                    rebuilt_rows.append(row)
            if row_key is None:
                row_key = page_key
            rebuilt_pages.append(tuple(rebuilt_rows))
        if row_key is not None:
            object.__setattr__(self, "pages", tuple(rebuilt_pages))
        return row_key

    @staticmethod
    def _item_id(item: Mapping[str, Any]) -> Hashable | None:
        """Normalize an item's explicit or fallback identity."""
        value = item.get("item_id")
        explicit = value is not None and not (
            isinstance(value, str) and not value.strip()
        )
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
        if not isinstance(value, Hashable):
            return None
        try:
            hash(value)
        except TypeError:
            return None
        return value

    @classmethod
    def _unique_items(
        cls, items: tuple[Mapping[str, Any], ...], seen: frozenset[Any]
    ) -> tuple[tuple[Mapping[str, Any], ...], frozenset[Any]]:
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

    def page(self, index: int) -> tuple[Mapping[str, Any], ...]:
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
        return bool(self.pending_items) or self.has_more
