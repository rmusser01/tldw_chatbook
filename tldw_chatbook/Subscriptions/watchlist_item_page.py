"""Immutable values exchanged by the watchlist item reader."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WatchlistItemCursor:
    """Keyset cursor for one watchlist item page.

    Attributes:
        effective_date: Nullable date component of the ordering key.
        item_id: Item-ID component of the ordering key.
    """

    effective_date: str | None
    item_id: int


@dataclass(frozen=True)
class WatchlistItemPage:
    """One service response page in a stable reader snapshot.

    Attributes:
        items: Item dictionaries returned in reader order.
        has_more: Whether traversal can request another page.
        snapshot_max_item_id: High-water mark for this snapshot.
        snapshot_count: Total matching count, when supplied.
        next_cursor: Cursor for the next traversal request.
    """

    items: tuple[dict[str, Any], ...]
    has_more: bool
    snapshot_max_item_id: int
    snapshot_count: int | None
    next_cursor: WatchlistItemCursor | None
