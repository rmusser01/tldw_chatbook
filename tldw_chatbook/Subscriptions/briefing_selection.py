"""Which items go into a watchlist briefing (spec #2 phase 1).

This module is the coverage semantics, and nothing else: given a watchlist
and a mode it answers "what does the next briefing cover?". It is
deliberately **read-only** -- it writes no `briefings` row and no
`briefing_items` junction row. The generation service owns every write, so
selection can be called speculatively (a preview, a dry run, a test) without
leaving a trace that would change the answer next time.

Three rules carry the whole design:

**1. The window is an item-id watermark, not a timestamp.** Each `complete`
(or `empty`) briefing records the max item id it considered; the next window
is `id >` that watermark. The upsert key for items is
`(subscription_id, url, content_hash)` -- *new content is a new row with a
new id*, and identical re-seen content updates the existing row in place --
so ids are precise, monotonic, and immune to the one-second resolution of
`created_at` (the TASK-1361 lesson: a burst written in the same second is a
genuine tie a timestamp window cannot break). It also solves the new-source
flood for free: a source added to the watchlist later has historical items
with *low* ids, so they fall below the watermark and never flood briefing 2,
no matter how recent their timestamps are. Only the first briefing, which
has no watermark to stand on, falls back to a 7-day `created_at` window --
measured from the injected `now`, never `datetime.now()` read inline.

**2. `failed` never advances anything.** That exclusion lives in
`SubscriptionsDB.latest_completed_watermark`; this module simply asks it, so
a failed attempt re-covers the same window on retry. Failure never loses
items.

**3. The queue flag is global and generation never clears it** (ADR-018, and
see the spec's "The queue flag is global, and never auto-cleared"). A source
can sit in several watchlists, so clearing the flag when *one* watchlist
briefs the item would silently destroy another watchlist's pending curation.
The exclusion is therefore per-watchlist and comes from the junction:
"queued AND NOT already in a briefing **of this watchlist**". The same
queued item still selects for a different watchlist -- that is the rule, not
a leak.

Consequences worth stating out loud rather than leaving to be discovered:

- `covers_through_item_id` is the max id **considered**, which includes items
  the cap dropped. Those items are not silently lost: they are counted in
  `overflow_count` and the briefing body states the overflow ("12 more items
  arrived in this window and are not covered"). Having been reported, they
  are covered; re-selecting them next time would duplicate coverage.
- **`curated` never moves the window.** The watermark is the *window's* line,
  and curated mode is defined as selecting "regardless of the window" -- it
  never reads the window, so it must not move it. Curated selection echoes
  the prior watermark back unchanged (an echo rather than `None`, so a future
  consumer reading the latest row rather than a `MAX` still sees the right
  line). Three things follow, all of them honest:
  - Switching a watchlist from `curated` to `auto`/`auto_featured` delivers
    the accumulated backlog, capped like any other window, with the overflow
    stated in the body -- never a silent hole.
  - An item briefed while curated can appear once more in the auto leg after
    the switch: the junction only excludes the *curated* leg, so this is
    redundant, never lossy.
  - A watchlist curated since inception has no watermark at all, so its first
    non-curated briefing takes the ordinary 7-day first-window rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Sequence

from .watchlist_normalizers import normalize_watchlist_item

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..DB.Subscriptions_DB import SubscriptionsDB

#: Window items only.
MODE_AUTO = "auto"
#: Queued-and-not-covered-by-this-watchlist only, regardless of the window.
MODE_CURATED = "curated"
#: The union: window items plus window-exempt queued items, the latter featured.
MODE_AUTO_FEATURED = "auto_featured"

VALID_MODES: tuple[str, ...] = (MODE_AUTO, MODE_CURATED, MODE_AUTO_FEATURED)

#: The first briefing has no watermark; it covers this many days back.
FIRST_WINDOW_DAYS = 7

#: Total items handed to one LLM call (spec: "~40").
DEFAULT_ITEM_CAP = 40

# `SELECT i.*` mirrors `get_new_items`, so rows arrive with every column
# `normalize_watchlist_item` reads (including `queued_for_briefing`); the two
# joined columns are the source name/type it also reports.
#
# `created_at_utc` is `created_at` normalized by SQLite for COMPARISON only:
# `created_at` is genuinely mixed-format in this table (a
# `DEFAULT CURRENT_TIMESTAMP` row reads `'2026-07-25 09:00:00'`,
# `persist_subscription_item` writes `'2026-07-25T08:00:00+00:00'`), and a raw
# string `min()` over the two sorts on the space versus the `T` -- so it can
# pick the LATER instant as the earliest. The extra column is dropped by
# `normalize_watchlist_item`, which builds an explicit dict.
_ITEM_COLUMNS = (
    "i.*, s.name AS subscription_name, s.type AS subscription_type, "
    "datetime(i.created_at) AS created_at_utc"
)

# Membership: an item belongs to a watchlist if its source is in that
# watchlist's CURRENT sources. `watchlist_sources.added_at` is deliberately
# not consulted -- the id watermark already excludes a late-added source's
# backlog, which is why the spec calls that fix free.
_FROM_WATCHLIST_ITEMS = """
    FROM subscription_items i
    JOIN subscriptions s ON s.id = i.subscription_id
    JOIN watchlist_sources ws ON ws.subscription_id = i.subscription_id
    WHERE ws.watchlist_id = ?
"""

# Whether an item is already covered by a `complete`/`empty` briefing OF
# THIS WATCHLIST. Literal SQL text shared between `_curated_rows` (which
# selects the rows where this is false) and the window queries'
# featured-exclusion (which excludes rows where the FULL curated predicate
# -- `queued_for_briefing = 1 AND` this -- is true), so the two can never
# quietly drift onto different definitions of "already curated". The
# status allowlist is load-bearing: only a briefing that actually reached
# the user excludes an item, so a `failed` (or zombie `generating`) row
# must not bury a queued item forever. See `_curated_rows`.
_NOT_COVERED_BY_THIS_WATCHLIST = (
    "NOT EXISTS (\n"
    "          SELECT 1 FROM briefing_items bi\n"
    "          JOIN briefings b ON b.id = bi.briefing_id\n"
    "          WHERE bi.item_id = i.id AND b.watchlist_id = ?\n"
    "            AND b.status IN ('complete', 'empty')\n"
    "      )"
)


@dataclass(frozen=True)
class BriefingSelection:
    """What one briefing will cover.

    Attributes:
        items: Normalized item dicts in selection order -- featured first,
            then window items, each group newest-first by id.
        featured_ids: Raw `subscription_items.id` values of the returned
            items that are featured (given top billing in the prompt and
            flagged in the junction). A subset of the ids in `items`.
        overflow_count: How many considered items the cap dropped. Stated in
            the briefing body; never a silent truncation.
        covers_through_item_id: The new watermark -- the max item id
            considered, including dropped ones. In `curated` mode it is the
            prior watermark, echoed: that mode never reads the window, so it
            never moves it. `None` means there is no line to record, so the
            caller must NOT advance coverage.
        covers_from_ts: Timestamp this briefing's coverage starts at: the
            7-day floor for a first briefing, otherwise the oldest considered
            item's `created_at` normalized to UTC (or `now` when nothing was
            considered -- an empty window spans no time). Display only.
    """

    items: list[dict[str, Any]]
    featured_ids: set[int]
    overflow_count: int
    covers_through_item_id: int | None
    covers_from_ts: str


def _window_predicate(
    watermark: int | None, floor_ts: str | None
) -> tuple[str, tuple[Any, ...]]:
    """The window's WHERE fragment and its bound value, shared by every
    query that reads the window (materialising rows, counting them, or
    bounding them) so the three can never quietly disagree about what "in
    the window" means.

    With a watermark, the window is `id > watermark`. Without one (the first
    briefing), it is the last `FIRST_WINDOW_DAYS` days by `created_at`,
    compared through SQLite's `datetime()` so that rows written with an
    offset-bearing ISO timestamp and rows written by `CURRENT_TIMESTAMP`
    normalize to the same UTC form before comparison -- a raw string
    comparison between `'...T09:00:00+00:00'` and `'... 09:00:00'` sorts on
    the `T` versus the space and is simply wrong.
    """
    if watermark is not None:
        return "i.id > ?", (watermark,)
    return "datetime(i.created_at) >= datetime(?)", (floor_ts,)


def _window_rows(
    db: "SubscriptionsDB",
    watchlist_id: int,
    watermark: int | None,
    floor_ts: str | None,
    limit: int,
    exclude_featured: bool = False,
) -> list[dict[str, Any]]:
    """Up to `limit` window rows, newest first -- NOT the whole window.

    Whole-branch review fix 2: a watchlist that has gone unbriefed for a
    while can have a window backlog far larger than the item cap, and the
    cap was previously applied in Python *after* every row was materialised
    -- pulling the whole backlog into memory just to keep the newest `cap`
    of it. The `LIMIT` here bounds materialisation to what the caller can
    actually use; `_window_count` and `_window_bounds` answer "how many /
    which line" over the FULL window without ever fetching a row.

    `exclude_featured`, when true (auto_featured mode), removes rows
    already claimed by the featured side directly in SQL -- so the rows
    returned are exactly the deduplicated "auto" bucket the caller needs,
    not a superset it would otherwise have to de-duplicate in Python (and
    risk under-fetching if it didn't overfetch by enough). Qodo round 1,
    FIX B: this used to be an explicit `exclude_ids` list bound one
    placeholder per id -- in auto_featured mode that is the ENTIRE queued
    set, which can exceed SQLite's host-parameter limit for a heavy user's
    backlog. `featured` IS `_curated_rows(db, watchlist_id)` in that mode
    (see `select_briefing_items`), so "id is in featured" and "row matches
    the curated predicate" are the same set over this same watchlist's
    items -- the predicate is reused here instead of the enumerated ids, at
    the cost of one bound `watchlist_id`, independent of queue size.

    Ordering is by id descending. Ids are the tiebreaker the one-second
    `created_at` resolution cannot provide (first-briefing window only).
    """
    predicate, extra = _window_predicate(watermark, floor_ts)
    params: list[Any] = [watchlist_id, *extra]

    # The interpolated fragments are module-level literals chosen above --
    # no caller value reaches the SQL text; every caller value is bound.
    sql = f"SELECT {_ITEM_COLUMNS}{_FROM_WATCHLIST_ITEMS}      AND {predicate}"
    if exclude_featured:
        sql += (
            "\n      AND NOT (i.queued_for_briefing = 1 AND "
            f"{_NOT_COVERED_BY_THIS_WATCHLIST})"
        )
        params.append(watchlist_id)
    sql += "\n    ORDER BY i.id DESC LIMIT ?"
    params.append(limit)

    # dict(), not the raw `sqlite3.Row`: `normalize_watchlist_item` reads
    # optional columns with `.get`, which a Row does not have. `get_new_items`
    # converts for the same reason.
    return [dict(row) for row in db.conn.execute(sql, params).fetchall()]


def _window_count(
    db: "SubscriptionsDB",
    watchlist_id: int,
    watermark: int | None,
    floor_ts: str | None,
    exclude_featured: bool = False,
) -> int:
    """Exact `COUNT(*)` of the FULL window, without materialising a row.

    `exclude_featured`, when true, excludes rows already claimed by the
    featured side -- so this is exactly `len(auto)` from the pre-fix4
    implementation, computed in SQL instead of over a fully materialised
    Python list, and it stays exact regardless of how small the row-fetch
    `limit` is. See `_window_rows` for why this is a predicate rather than
    an enumerated id list (Qodo round 1, FIX B).
    """
    predicate, extra = _window_predicate(watermark, floor_ts)
    params: list[Any] = [watchlist_id, *extra]
    sql = f"SELECT COUNT(*){_FROM_WATCHLIST_ITEMS}      AND {predicate}"
    if exclude_featured:
        sql += (
            "\n      AND NOT (i.queued_for_briefing = 1 AND "
            f"{_NOT_COVERED_BY_THIS_WATCHLIST})"
        )
        params.append(watchlist_id)
    return int(db.conn.execute(sql, params).fetchone()[0])


def _window_bounds(
    db: "SubscriptionsDB",
    watchlist_id: int,
    watermark: int | None,
    floor_ts: str | None,
) -> tuple[int | None, str | None]:
    """`(MAX(id), MIN(created_at))` over the FULL window, unfiltered.

    Deliberately NOT excluding the featured overlap the way `_window_count`
    does: `covers_through_item_id` and `covers_from_ts` are properties of
    everything the window considers, whether or not a row is ALSO featured
    -- excluding it here would be wrong in a way excluding it from the
    auto-side *count* is not (that count exists specifically to avoid
    double-counting the same row on both sides of `overflow_count`).
    Returns `(None, None)` when the window is empty.
    """
    predicate, extra = _window_predicate(watermark, floor_ts)
    params: list[Any] = [watchlist_id, *extra]
    sql = (
        "SELECT MAX(i.id) AS max_id, MIN(datetime(i.created_at)) AS min_created"
        f"{_FROM_WATCHLIST_ITEMS}      AND {predicate}"
    )
    row = db.conn.execute(sql, params).fetchone()
    return row["max_id"], row["min_created"]


def _curated_rows(db: "SubscriptionsDB", watchlist_id: int) -> list[dict[str, Any]]:
    """Queued items of this watchlist's sources not yet covered BY IT, newest first.

    Two predicates, both load-bearing:

    - The `NOT EXISTS` is scoped through `briefings.watchlist_id`: only this
      watchlist's own briefings exclude an item. Dropping that scope would
      make one watchlist's briefing silently consume another's curation --
      exactly what keeping the flag global was meant to prevent.
    - Only a briefing that actually *reached the user* excludes an item, hence
      the positive status allowlist. A junction row belonging to a `failed`
      briefing must not bury a queued item forever -- writing junction rows
      before the LLM call is a perfectly natural way for the service to
      implement generation, so this is a live hazard rather than a
      hypothetical. Stated as an allowlist rather than `!= 'failed'` so a
      zombie `generating` row (a crashed worker, TASK-1090's shape) is
      covered by the same rule instead of needing a second one.
    """
    sql = (
        f"SELECT {_ITEM_COLUMNS}{_FROM_WATCHLIST_ITEMS}"
        "      AND i.queued_for_briefing = 1\n"
        f"      AND {_NOT_COVERED_BY_THIS_WATCHLIST}\n"
        "    ORDER BY i.id DESC"
    )
    rows = db.conn.execute(sql, (watchlist_id, watchlist_id)).fetchall()
    return [dict(row) for row in rows]


def _covers_from(rows: Sequence[dict[str, Any]], now: datetime) -> str:
    """Oldest considered timestamp, or `now` when nothing was considered.

    Compares `created_at_utc` -- SQLite's normalization of `created_at` -- not
    the raw column, because the raw column holds two different string formats
    for the same instant (see `_ITEM_COLUMNS`).
    """
    stamps = [row["created_at_utc"] for row in rows if row["created_at_utc"]]
    return min(stamps) if stamps else now.isoformat()


def select_briefing_items(
    db: "SubscriptionsDB",
    watchlist_id: int,
    *,
    mode: str,
    item_cap: int = DEFAULT_ITEM_CAP,
    now: datetime | None = None,
) -> BriefingSelection:
    """Choose the items for a watchlist's next briefing. Writes nothing.

    Args:
        db: An open `SubscriptionsDB`.
        watchlist_id: The watchlist being briefed.
        mode: One of `VALID_MODES`. `auto` takes window items only;
            `curated` takes queued-and-not-covered-by-this-watchlist items
            only, ignoring the window entirely -- and therefore leaving the
            watermark exactly where it found it; `auto_featured` (the default
            a watchlist is created with) takes the union and features the
            queued ones.
        item_cap: Total items to hand the LLM call. Newest win; featured
            items are never dropped while an auto item remains to drop.
        now: Injected clock, used only for the first briefing's 7-day floor
            and for `covers_from_ts` when nothing was considered. Defaults
            to the current UTC time.

    Returns:
        A `BriefingSelection`.

    Raises:
        ValueError: If `mode` is not one of `VALID_MODES`, or `item_cap` < 1.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"unknown briefing selection mode {mode!r}; valid modes: {list(VALID_MODES)}"
        )
    if item_cap < 1:
        raise ValueError(f"item_cap must be at least 1, got {item_cap}")

    now = now or datetime.now(timezone.utc)
    watermark = db.latest_completed_watermark(watchlist_id)
    floor_ts = (
        (now - timedelta(days=FIRST_WINDOW_DAYS)).isoformat()
        if watermark is None
        else None
    )

    curated = [] if mode == MODE_AUTO else _curated_rows(db, watchlist_id)

    if mode == MODE_AUTO_FEATURED:
        featured = curated
    else:  # MODE_AUTO and MODE_CURATED both feature nothing.
        featured = []
    featured_ids = tuple(row["id"] for row in featured)

    # The cap squeezes the auto side first. Only when featured items alone
    # exceed the cap do featured items themselves overflow -- newest kept.
    kept_featured = featured[:item_cap]
    remaining_cap = item_cap - len(kept_featured) if len(kept_featured) < item_cap else 0

    if mode == MODE_CURATED:
        # Curated never reads the coverage window at all -- `curated` IS the
        # "auto" bucket here (nothing is "featured": every item in the
        # briefing is curated, so there is nothing to give top billing
        # over). Small by construction (a user's queue, not a window
        # backlog), so whole-branch review fix 2's bound does not apply to
        # it -- it is already fully materialised above.
        auto_full = curated
        auto_full_count = len(auto_full)
        kept_auto = auto_full[:remaining_cap]
        # Curated never moves the coverage window: echo the prior line back
        # (`None` when there is none yet, which the caller already reads as
        # "do not advance"). Advancing to the newest queued id would walk
        # the line past window items no briefing ever covered -- with no
        # overflow count, no body text, and no status to show for it. See
        # the module docstring.
        covers_through = watermark
        covers_from_ts = (
            floor_ts if floor_ts is not None else _covers_from(auto_full, now)
        )
    else:
        # Whole-branch review fix 2: the window is bounded in SQL rather
        # than materialised in full and then sliced in Python. A watchlist
        # left unbriefed for a while can have a window backlog far larger
        # than the item cap; `_window_count`/`_window_bounds` answer the
        # "how many / which line" questions over the FULL window without
        # ever fetching a row, and `_window_rows` fetches only the rows this
        # call can actually use (deduplicated against `featured` in SQL, via
        # `exclude_featured`, so no Python-side dedup or overfetch is
        # needed).
        #
        # Qodo round 1, FIX B: `featured` is exactly `curated` in
        # `auto_featured` mode (line above) and empty otherwise, so
        # "exclude featured" is a boolean, not the enumerated `featured_ids`
        # -- see `_window_rows`/`_window_count` for why a per-id `NOT IN`
        # does not scale to a heavy user's queued backlog.
        exclude_featured = mode == MODE_AUTO_FEATURED
        auto_full_count = _window_count(
            db, watchlist_id, watermark, floor_ts, exclude_featured=exclude_featured
        )
        window_max_id, window_min_created = _window_bounds(
            db, watchlist_id, watermark, floor_ts
        )
        kept_auto = (
            _window_rows(
                db,
                watchlist_id,
                watermark,
                floor_ts,
                limit=remaining_cap,
                exclude_featured=exclude_featured,
            )
            if remaining_cap > 0
            else []
        )

        id_candidates = [window_max_id, *featured_ids]
        covers_through = max(
            (value for value in id_candidates if value is not None), default=None
        )

        if floor_ts is not None:
            covers_from_ts = floor_ts
        else:
            # `_covers_from` also folds in `featured`'s own timestamps: a
            # featured item can be OLDER than the window entirely (a queued
            # item below the watermark), so it may hold the true minimum
            # even though it never appears in `_window_bounds`.
            pseudo_rows = (
                [{"created_at_utc": window_min_created}] if window_min_created else []
            ) + featured
            covers_from_ts = _covers_from(pseudo_rows, now)

    overflow_count = (len(featured) - len(kept_featured)) + (
        auto_full_count - len(kept_auto)
    )

    kept = kept_featured + kept_auto
    return BriefingSelection(
        items=[normalize_watchlist_item("local", row) for row in kept],
        featured_ids={row["id"] for row in kept_featured},
        overflow_count=overflow_count,
        covers_through_item_id=covers_through,
        covers_from_ts=covers_from_ts,
    )
