"""Pure cursor/progress logic for Library review sets (task-28240).

A review set is a snapshot: an ordered list of pinned media items plus an
absolute cursor position. Items are never renumbered, so a deleted item leaves
a *tombstone* at its position. Which items are still live is decided by an
injected ``is_live`` predicate (a resolve against the Media DB, wired at a
higher layer) -- this module stays pure and DB-free so the navigation model is
unit-testable in isolation.

Everything the user sees -- progress, completion, the walk -- is computed over
LIVE items only; the cursor is an absolute position that survives deletions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

IsLive = Callable[[int], bool]
"""Predicate: does this backing media id still resolve to a live item?"""


@dataclass(frozen=True)
class ReviewSetItem:
    """One pinned member of a review set.

    Attributes:
        position: Absolute 0-based order, pinned at creation and never changed.
        backing_media_id: The local ``Media(id)``; the canonical id is
            ``local:media:{backing_media_id}``.
        title_snapshot: Title captured at pin time. Load-bearing: it is the
            only title available once the item is deleted (there is no
            cross-database join, see the design doc).
        done: Whether the user has marked this item reviewed.
        done_at: ISO timestamp of the done mark, or ``None``.
    """

    position: int
    backing_media_id: int
    title_snapshot: str
    done: bool
    done_at: str | None = None


@dataclass(frozen=True)
class ReviewSet:
    """A whole review set as loaded from persistence.

    Attributes:
        set_id: Opaque unique id.
        name: Human label (e.g. a tag or "8 selected items").
        origin: Provenance -- ``'browse'`` | ``'selection'`` | ``'read_later'``.
        cursor: Absolute position of the current item.
        active: Whether this is the one set the Reader is currently walking.
        completed_at: ISO timestamp once every live item is done, else ``None``.
        items: The pinned members, in ascending position order.
        created_at: ISO creation timestamp.
        updated_at: ISO timestamp of the last mutation.
    """

    set_id: str
    name: str
    origin: str
    cursor: int
    active: bool
    completed_at: str | None
    items: tuple[ReviewSetItem, ...]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ReviewProgress:
    """The set's live progress, as shown in the Reader chrome.

    Attributes:
        index: 1-based ordinal of the cursor's item among LIVE items (0 when
            the set has no live items).
        total: Number of live items.
        reviewed: Number of live items marked done.
    """

    index: int
    total: int
    reviewed: int


def _live_items(
    items: tuple[ReviewSetItem, ...], is_live: IsLive
) -> list[ReviewSetItem]:
    """Return the live items in ascending position order.

    ``is_live`` is evaluated exactly once per item here; callers that need both
    the cursor resolved and the cursor advanced work off this single snapshot
    (via :func:`_resolve_within`) so a liveness value that changes mid-call can
    never desync the two.
    """
    return [
        item
        for item in sorted(items, key=lambda entry: entry.position)
        if is_live(item.backing_media_id)
    ]


def _resolve_within(live_positions: list[int], cursor: int) -> int:
    """Resolve ``cursor`` against an already-computed live-position snapshot.

    Pure and snapshot-local (no ``is_live`` re-evaluation): a live cursor is
    kept, a tombstoned cursor resolves to the next live position ahead, else
    the nearest live position behind; an empty snapshot returns ``cursor``.
    """
    if not live_positions:
        return cursor
    if cursor in live_positions:
        return cursor
    ahead = [position for position in live_positions if position > cursor]
    return ahead[0] if ahead else live_positions[-1]


def resolve_cursor(
    items: tuple[ReviewSetItem, ...], cursor: int, is_live: IsLive
) -> int:
    """Return the nearest live position for ``cursor``.

    A live cursor is returned unchanged. A cursor on a tombstone resolves
    forward to the next live position, or -- when nothing live is ahead --
    back to the nearest live position. An empty (all-tombstoned) set returns
    ``cursor`` unchanged, since there is nowhere live to land.

    Args:
        items: The set's pinned items.
        cursor: The current absolute position.
        is_live: Liveness predicate.

    Returns:
        A live position, or ``cursor`` when the set has no live items.
    """
    return _resolve_within(
        [item.position for item in _live_items(items, is_live)], cursor
    )


def advance_cursor(
    items: tuple[ReviewSetItem, ...], cursor: int, step: int, is_live: IsLive
) -> int:
    """Move the cursor one live item forward (``step=1``) or back (``step=-1``).

    Tombstones between the current and the next live item are skipped. The
    cursor clamps at the first and last live items (advancing past an end stays
    put). The starting cursor is resolved first, so advancing off a tombstone
    behaves like advancing from the nearest live item.

    The live snapshot is taken once and both the resolve and the step run
    against it, so ``current`` is always present in it -- a liveness value that
    flips mid-call cannot raise (task-28241 review).

    Args:
        items: The set's pinned items.
        cursor: The current absolute position.
        step: ``+1`` for Next, ``-1`` for Prev.
        is_live: Liveness predicate.

    Returns:
        The new absolute (live) position.
    """
    live_positions = [item.position for item in _live_items(items, is_live)]
    if not live_positions:
        return cursor
    current = _resolve_within(live_positions, cursor)
    current_index = live_positions.index(current)
    new_index = current_index + (1 if step > 0 else -1 if step < 0 else 0)
    new_index = max(0, min(new_index, len(live_positions) - 1))
    return live_positions[new_index]


def review_progress(
    items: tuple[ReviewSetItem, ...], cursor: int, is_live: IsLive
) -> ReviewProgress:
    """Compute the live progress readout for ``cursor``.

    The live snapshot is taken once and the cursor is resolved within it, so
    the ordinal and the totals always agree (task-28241 review).

    Args:
        items: The set's pinned items.
        cursor: The current absolute position.
        is_live: Liveness predicate.

    Returns:
        A :class:`ReviewProgress` over the live items.
    """
    live = _live_items(items, is_live)
    total = len(live)
    reviewed = sum(1 for item in live if item.done)
    if total == 0:
        return ReviewProgress(index=0, total=0, reviewed=0)
    positions = [item.position for item in live]
    resolved = _resolve_within(positions, cursor)
    index = positions.index(resolved) + 1 if resolved in positions else 1
    return ReviewProgress(index=index, total=total, reviewed=reviewed)


def is_empty(items: tuple[ReviewSetItem, ...], is_live: IsLive) -> bool:
    """True when the set has no live items (every pinned item is a tombstone).

    Args:
        items: The set's pinned items.
        is_live: Liveness predicate.

    Returns:
        ``True`` when every pinned item is a tombstone.
    """
    return not _live_items(items, is_live)


def is_complete(items: tuple[ReviewSetItem, ...], is_live: IsLive) -> bool:
    """True when the set has at least one live item and every live item is done.

    An empty (all-tombstoned) set is NOT complete -- it is empty (see
    :func:`is_empty`).

    Args:
        items: The set's pinned items.
        is_live: Liveness predicate.

    Returns:
        ``True`` when at least one item is live and all live items are done.
    """
    live = _live_items(items, is_live)
    return bool(live) and all(item.done for item in live)
