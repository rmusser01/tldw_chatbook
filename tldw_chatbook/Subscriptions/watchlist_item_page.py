"""Immutable values exchanged by the watchlist item reader."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WatchlistItemCursor:
    """Keyset cursor for one watchlist item page."""

    effective_date: str | None
    item_id: int


@dataclass(frozen=True)
class WatchlistItemPage:
    """One service response page in a stable reader snapshot."""

    items: tuple[dict[str, Any], ...]
    has_more: bool
    snapshot_max_item_id: int
    snapshot_count: int | None
    next_cursor: WatchlistItemCursor | None
