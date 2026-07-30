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


def _window_rows(
    db: "SubscriptionsDB",
    watchlist_id: int,
    watermark: int | None,
    floor_ts: str | None,
) -> list[dict[str, Any]]:
    """Items of the watchlist's sources inside the coverage window, newest first.

    With a watermark, the window is `id > watermark`. Without one (the first
    briefing), it is the last `FIRST_WINDOW_DAYS` days by `created_at`,
    compared through SQLite's `datetime()` so that rows written with an
    offset-bearing ISO timestamp and rows written by `CURRENT_TIMESTAMP`
    normalize to the same UTC form before comparison -- a raw string
    comparison between `'...T09:00:00+00:00'` and `'... 09:00:00'` sorts on
    the `T` versus the space and is simply wrong. A NULL `created_at`
    normalizes to NULL and is excluded: an item with no timestamp cannot be
    placed inside a time window.

    Ordering is by id descending in both cases. Ids are the tiebreaker the
    one-second `created_at` resolution cannot provide.
    """
    if watermark is not None:
        predicate = "i.id > ?"
        params: tuple[Any, ...] = (watchlist_id, watermark)
    else:
        predicate = "datetime(i.created_at) >= datetime(?)"
        params = (watchlist_id, floor_ts)

    # The interpolated fragments are module-level literals chosen by the
    # branch above -- no caller value reaches the SQL text; every caller
    # value is bound.
    sql = (
        f"SELECT {_ITEM_COLUMNS}{_FROM_WATCHLIST_ITEMS}"
        f"      AND {predicate}\n"
        "    ORDER BY i.id DESC"
    )
    # dict(), not the raw `sqlite3.Row`: `normalize_watchlist_item` reads
    # optional columns with `.get`, which a Row does not have. `get_new_items`
    # converts for the same reason.
    return [dict(row) for row in db.conn.execute(sql, params).fetchall()]


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
        "      AND NOT EXISTS (\n"
        "          SELECT 1 FROM briefing_items bi\n"
        "          JOIN briefings b ON b.id = bi.briefing_id\n"
        "          WHERE bi.item_id = i.id AND b.watchlist_id = ?\n"
        "            AND b.status IN ('complete', 'empty')\n"
        "      )\n"
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

    window = (
        []
        if mode == MODE_CURATED
        else _window_rows(db, watchlist_id, watermark, floor_ts)
    )
    curated = [] if mode == MODE_AUTO else _curated_rows(db, watchlist_id)

    if mode == MODE_AUTO_FEATURED:
        featured = curated
        featured_ids = {row["id"] for row in featured}
        # A queued item that also falls inside the window appears once, on
        # the featured side.
        auto = [row for row in window if row["id"] not in featured_ids]
    elif mode == MODE_AUTO:
        featured, auto = [], window
    else:  # MODE_CURATED -- every item is curated, so none is "featured".
        featured, auto = [], curated

    considered = featured + auto
    if mode == MODE_CURATED:
        # Curated never reads the coverage window, so it must not move it:
        # echo the prior line back (None when there is none yet, which the
        # caller already reads as "do not advance"). Advancing to the newest
        # queued id would walk the line past window items no briefing ever
        # covered -- with no overflow count, no body text, and no status to
        # show for it. See the module docstring.
        covers_through = watermark
    else:
        covers_through = max((row["id"] for row in considered), default=None)

    # The cap squeezes the auto side first. Only when featured items alone
    # exceed the cap do featured items themselves overflow -- newest kept.
    kept_featured = featured[:item_cap]
    kept_auto = auto[: item_cap - len(kept_featured)] if len(kept_featured) < item_cap else []
    overflow_count = (len(featured) - len(kept_featured)) + (len(auto) - len(kept_auto))

    kept = kept_featured + kept_auto
    return BriefingSelection(
        items=[normalize_watchlist_item("local", row) for row in kept],
        featured_ids={row["id"] for row in kept_featured},
        overflow_count=overflow_count,
        covers_through_item_id=covers_through,
        covers_from_ts=floor_ts if floor_ts is not None else _covers_from(considered, now),
    )
