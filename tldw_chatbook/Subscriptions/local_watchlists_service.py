"""Local watchlists adapter backed by the existing subscriptions database."""

from __future__ import annotations

import asyncio
import json
import inspect
import hashlib
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence

from loguru import logger

from ..DB.Subscriptions_DB import SubscriptionsDB
from ..Utils.egress import (
    MAX_FETCH_BYTES_PAGE,
    MAX_FETCH_BYTES_SITEMAP,
    guarded_fetch_httpx_async,
    origin_set,
)
from .db_offload import run_db_off_loop
from .item_persist import (
    CONTENT_FORMAT_TEXT,
    CONTENT_KIND_ARTICLE,
    persist_subscription_item,
)
from .watchlist_content_alert_service import WatchlistContentAlertService
from .watchlist_bundle_service import WatchlistBundleService
from .watchlist_filter_service import WatchlistFilterService
from .watchlist_normalizers import (
    WATCHLIST_NAME_SEPARATOR,
    build_watchlist_item_id,
    normalize_local_subscription_row,
    normalize_watchlist_alert_rule,
    normalize_watchlist_item,
    normalize_watchlist_run,
)
from .watchlist_item_page import WatchlistItemCursor, WatchlistItemPage


_ALERT_CONDITION_TYPES = frozenset(
    {
        "no_items",
        "error_rate_above",
        "items_below",
        "items_above",
        "run_failed",
    }
)

#: The source types `_default_run_executor`'s feed arm handles.
_FEED_SOURCE_TYPES = frozenset({"rss", "atom", "json_feed", "podcast"})

#: Every source type a local run can execute, i.e. exactly the arms of
#: `_default_run_executor` below.
#:
#: TASK-1383. Exported because the scheduled-check handler
#: (`Scheduling/scheduler/handlers/watchlist_check_handler.py`) has to decide
#: whether a subscription is executable *before* it launches a run, and it
#: used to answer that question from its own private tuples. Those tuples had
#: drifted: they omitted `sitemap` entirely, so every scheduled sitemap source
#: took an "unknown subscription type" branch and was never checked. One
#: definition, read by both callers, is what stops that recurring.
EXECUTABLE_SOURCE_TYPES = _FEED_SOURCE_TYPES | frozenset(
    {"url", "url_list", "sitemap", "api"}
)

#: Every counter `_disposition_counts` can return, in the order the Runs pane
#: renders them. Named here rather than inline so the zero-fill and the
#: binding below cannot disagree about which counters exist.
_DISPOSITION_COUNTERS: tuple[str, ...] = (
    "changed",
    "unchanged",
    "withheld",
    "baseline",
    "rebaselined",
    # TASK-1394. Appended last rather than inserted in "kind order" so nothing
    # that reads this tuple positionally (none of the current readers do, but
    # a future one might) sees the existing five counters shift place.
    "error",
    # task-16838. Same appended-last rationale. A URL this run never checked
    # at all, because another check of the same (subscription, url) pair was
    # already in flight -- see `_check_url_guarded`.
    "skipped",
)

_RUN_CLAIM_WAIT_TIMEOUT_SECONDS = 300.0
_RUN_CLAIM_POLL_INITIAL_SECONDS = 0.01
_RUN_CLAIM_POLL_MAX_SECONDS = 0.5


def _disposition_count_keys() -> dict[tuple[str, str | None], str]:
    """`check_url`'s `(kind, reason)` -> the run-stats counter it increments.

    Keyed off `monitoring_engine`'s real ``DISPOSITION_*``/``REASON_*``
    constants rather than re-spelled string literals, so the two cannot drift
    apart (TASK-1362 ledgered Minor from Task 3's review: a re-spelled literal
    here would silently `KeyError` inside a run the moment `monitoring_engine`
    renamed a kind, discarding every item that run collected). `withheld` is
    shortened from `DISPOSITION_WITHHELD`'s value for the stats key, which is
    read by the Runs pane as a one-line summary.

    The key is the `(kind, reason)` PAIR, not the kind alone (whole-branch
    review, Critical 1). `DISPOSITION_BASELINE_STORED` has two causes and they
    mean opposite things to the user -- `first_check` discarded nothing,
    `extraction_settings_changed` threw away a real diff window in which a
    change could have been lost -- so they get separate counters. Collapsing
    them back into one leaves the `reason` with no production consumer at all,
    which is what made spec §3's "the Runs pane says why" untrue.

    Five of the seven pairs below are exactly the five `_disposition` call
    sites in `check_url`; an unlisted pair raises `KeyError` in
    `_disposition_counts`, deliberately. The sixth, `DISPOSITION_ERROR`, is
    NOT one of `check_url`'s own outcomes -- it is synthesized by
    `_default_run_executor`'s `url_list`/`sitemap` loops around a
    `check_url` call that raised instead of returning (task-1394), so one
    dead URL is counted rather than failing the whole run. The seventh,
    `DISPOSITION_SKIPPED_IN_FLIGHT`, is likewise caller-synthesized
    (`_check_url_guarded`, task-16838): the call was never made because a
    concurrent check of the same (subscription, url) pair held the claim.

    The `monitoring_engine` import is deliberately local, not module-level:
    this module loads unconditionally from `Subscriptions/__init__.py`
    (its own import is not wrapped in the `try/except` that guards the
    package's `monitoring_engine` re-export), but `monitoring_engine` carries
    a hard `beautifulsoup4`/`defusedxml` import that not every install has
    (see the `websearch` extras group) -- a module-level import here would
    make importing this module fail on any install that lacks them, exactly
    like `_default_run_executor`'s existing local import of `URLMonitor`.
    """
    from .monitoring_engine import (
        DISPOSITION_BASELINE_STORED,
        DISPOSITION_CHANGED,
        DISPOSITION_ERROR,
        DISPOSITION_SKIPPED_IN_FLIGHT,
        DISPOSITION_UNCHANGED,
        DISPOSITION_WITHHELD,
        REASON_BELOW_CHANGE_THRESHOLD,
        REASON_EXTRACTION_SETTINGS_CHANGED,
        REASON_FIRST_CHECK,
    )

    return {
        (DISPOSITION_CHANGED, None): "changed",
        (DISPOSITION_UNCHANGED, None): "unchanged",
        (DISPOSITION_WITHHELD, REASON_BELOW_CHANGE_THRESHOLD): "withheld",
        (DISPOSITION_BASELINE_STORED, REASON_FIRST_CHECK): "baseline",
        (
            DISPOSITION_BASELINE_STORED,
            REASON_EXTRACTION_SETTINGS_CHANGED,
        ): "rebaselined",
        # task-1394: `reason` is always `None` here, unlike the exception
        # detail logged at the catch site -- the counter answers "how many
        # URLs errored", not "which exception", so it needs exactly one
        # stable pair rather than one per exception type.
        (DISPOSITION_ERROR, None): "error",
        # task-16838: a URL this run never checked because a concurrent
        # check of the same (subscription, url) pair was already in flight.
        (DISPOSITION_SKIPPED_IN_FLIGHT, None): "skipped",
    }


def _disposition_counts(dispositions: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate one run's per-URL dispositions into the six counters.

    Spec §4. A run that produced no items used to be indistinguishable from a
    run that produced no items *because it withheld them*; these counters are
    what makes the difference visible. All six keys are always present, so the
    reader never has to distinguish "zero" from "not recorded".

    ``error`` (task-1394) is the odd one out: it does not come from
    `check_url` completing with an outcome, it comes from `check_url` never
    returning at all. A `url_list`/`sitemap` run with one dead URL among many
    still reports `"error": 1` here rather than raising out of the whole run.

    Args:
        dispositions: One disposition dict per URL checked, in check order.

    Returns:
        ``{"changed": n, "unchanged": n, "withheld": n, "baseline": n,
        "rebaselined": n, "error": n}``.

    Raises:
        KeyError: If a disposition carries a ``(kind, reason)`` pair outside
            the vocabulary -- deliberately loud, because a silently dropped
            disposition is exactly the ambiguity this record exists to remove.
    """
    count_keys = _disposition_count_keys()
    counts = {counter: 0 for counter in _DISPOSITION_COUNTERS}
    for disposition in dispositions:
        key = (str(disposition.get("kind")), disposition.get("reason"))
        counts[count_keys[key]] += 1
    return counts


#: The `_DISPOSITION_COUNTERS` entries that mean "this URL's check_url call
#: actually succeeded" -- every counter except `"error"` and `"skipped"`.
#: Named as a tuple comprehension over `_DISPOSITION_COUNTERS` rather than
#: re-spelled here so the all-error detection below cannot silently drift
#: from the counters it is reading (same rationale as
#: `_disposition_count_keys`'s docstring). `"skipped"` (task-16838) is
#: excluded because a skip proves nothing about the source's reachability:
#: the URL was never contacted, so it must not count as the "genuine
#: progress" that resets the auto-pause breaker in
#: `_all_error_check_message`. This exclusion matters for the MIXED case --
#: a run where every URL actually checked errored and the rest were skipped
#: still records the failure. An ENTIRELY-skipped run never reaches
#: `_all_error_check_message` at all: `execute_run` short-circuits its
#: source-health accounting before that point (review B1 -- see
#: `_entirely_skipped_dispositions` there), because with zero `error`
#: dispositions the helper would return `None` and route the run into
#: `record_check_result`'s SUCCESS branch, resetting the breaker and
#: stamping `last_successful_check` for a run that contacted nothing.
_SUCCESS_DISPOSITION_COUNTERS: tuple[str, ...] = tuple(
    counter
    for counter in _DISPOSITION_COUNTERS
    if counter not in ("error", "skipped")
)


#: task-16838: every URL check currently in flight, keyed
#: `(id(db), subscription_id, url)`. This is the serialization mechanism the
#: TASK-15764 review established did not exist: the scheduler (an async
#: worker on the app's event loop) and a UI "Check Now" (a coroutine worker
#: on the same loop) can otherwise interleave checks of the SAME source
#: across `check_url`'s awaits (the network fetch plus the off-loop sqlite
#: and CPU hops), both read the same baseline before either writes, and one
#: page change is reported twice with two snapshots written.
#:
#: MODULE-level, not service-instance state, deliberately: production wiring
#: (`app.py`, task-15463) holds TWO `LocalWatchlistsService` instances over
#: the ONE shared `SubscriptionsDB` -- the UI's `self.local_watchlists_
#: service` and the `WatchlistCheckHandler`'s own default-constructed one --
#: so instance state would never see the exact cross-entrant interleave this
#: exists to stop. `id(db)` scopes the key to that shared database object:
#: `WatchlistPreviewService` runs the same executor against a throwaway
#: in-memory `SubscriptionsDB` whose row ids can collide with live ones, and
#: without the scope a preview could falsely skip (or be skipped by) a real
#: check. The id cannot alias across lives — but note WHERE that guarantee
#: comes from: the set entry is a plain `(int, int, str)` and holds NO
#: reference to `db`. It is `_check_url_guarded`'s own frame — its `db`
#: parameter, alive across the awaited check — that keeps the object from
#: being collected (and its id reused) while its key is registered. A
#: refactor that extracts the claim into a helper receiving only the
#: precomputed key would silently drop that liveness guarantee.
#:
#: A plain `set` with no lock, on the single-loop argument: every entrant
#: that reaches `_default_run_executor` runs on the app's one event loop
#: (scheduler: `run_worker(self.scheduler_loop.run(), ...)`; Check Now /
#: Rerun: coroutine workers; the scheduled handler awaits the service
#: directly), and `_check_url_guarded`'s claim-check-and-add / discard are
#: synchronous between awaits, hence atomic with respect to that loop. If a
#: check entrant ever moves off the app loop, this needs a real lock -- see
#: `_check_url_guarded`'s docstring.
_IN_FLIGHT_URL_CHECKS: set[tuple[int, Any, str]] = set()


def _entirely_skipped_dispositions(
    dispositions_counts: Mapping[str, Any] | None,
) -> bool:
    """Whether a run's dispositions say it checked NOTHING because of the guard.

    task-16838, review B1. True when at least one URL was `skipped` (a
    concurrent check of the same (subscription, url) held the in-flight
    claim) and every other counter is zero -- i.e. the run made no network
    call and observed nothing at all. `execute_run` short-circuits the
    source-health accounting for such a run: without this, its zero `error`
    dispositions routed it through `record_check_result`'s SUCCESS branch,
    which resets the auto-pause breaker, clears `last_error`, and stamps
    `last_successful_check` -- a "successful check" by a run that never
    contacted the source (the 16838 review's PROBE 1: a two-failure streak
    wiped to clean by a skip). A PARTIALLY skipped run is deliberately not
    matched: it did real work on the URLs it checked and keeps the ordinary
    accounting.

    Args:
        dispositions_counts: The run's `_disposition_counts()` output, or
            `None`/absent for source types with no dispositions (feed/API
            runs, whose dispositions are ``None`` on the ordinary path).
            task-19562: feed/API runs CAN now skip -- when they do they carry
            a single `DISPOSITION_SKIPPED_IN_FLIGHT` disposition precisely so
            they match here and `execute_run` skips the health accounting,
            exactly like an entirely-skipped url-family run.

    Returns:
        True only for a non-empty counts mapping that is all zeros except
        `skipped`.
    """
    if not isinstance(dispositions_counts, Mapping):
        return False
    skipped = int(dispositions_counts.get("skipped", 0) or 0)
    if skipped <= 0:
        return False
    others = sum(
        int(dispositions_counts.get(counter, 0) or 0)
        for counter in _DISPOSITION_COUNTERS
        if counter != "skipped"
    )
    return others == 0


def _all_error_check_message(
    dispositions_counts: Mapping[str, Any] | None, item_count: int
) -> str | None:
    """A type-only synthetic error for a `url_list`/`sitemap` run where every URL failed.

    Fix wave for the task-1394 whole-branch review (Finding #1, MAJOR): the
    per-URL isolation in `_check_url_isolated` correctly turns one dead URL
    among many into a single `"error"` disposition rather than failing the
    whole run -- but `execute_run`'s success path always called
    `db.record_check_result(source_id, items=None, stats=stats)` with
    `error=None`, which hits that method's success branch
    (`DB/Subscriptions_DB.py:1504-1517`) and unconditionally RESETS the
    subscription's auto-pause circuit breaker
    (`consecutive_failures`/`error_count` -> 0), even when every single URL in
    the run errored and nothing was found. A permanently-broken `url_list`/
    `sitemap` source could then never reach `auto_pause_threshold`: its
    failure streak was wiped every run instead of accumulating, exactly the
    behaviour `record_check_error` (the pre-fix path, reached via
    `record_run_failure`) used to provide.

    This distinguishes that "every URL errored" case from the ordinary
    partial-failure case the isolation was written for (some URLs succeed,
    one or two do not): a partial run made genuine progress on a reachable
    source and should keep resetting the breaker, exactly as a clean run
    would. Only when there is not one single successful check in the whole
    run does the source's own health tracking need to see a failure.

    Args:
        dispositions_counts: The run's `_disposition_counts()` output (the
            `stats["dispositions"]` dict), or `None` for source types that
            carry no dispositions at all (the feed and API arms) -- those are
            deliberately unaffected and always return `None` here.
        item_count: How many items the run produced overall (pre-filter), so
            a run that somehow reported all-error dispositions yet still
            surfaced an item is never treated as a total failure.

    Returns:
        A message counting only URLs, e.g. ``"all 2 checked URL(s) failed"``
        -- no URL, no exception message, matching `_check_url_isolated`'s own
        type-only logging -- when every disposition in the run was an error
        and zero items were produced. `None` otherwise (nothing to record, or
        this is a feed/api run with no dispositions to judge by).
    """
    if not isinstance(dispositions_counts, Mapping):
        return None
    error_count = int(dispositions_counts.get("error", 0) or 0)
    if error_count == 0 or item_count:
        return None
    successful_count = sum(
        int(dispositions_counts.get(counter, 0) or 0)
        for counter in _SUCCESS_DISPOSITION_COUNTERS
    )
    if successful_count:
        return None
    return f"all {error_count} checked URL(s) failed"


def _max_withheld_percentage(dispositions: list[dict[str, Any]]) -> float | None:
    """The largest change any check in this run held back, display-scaled.

    Spec §1 requires the app to tell the user *what* it is withholding, not
    merely that it withheld something (whole-branch review, Critical 1): a
    bare "2 withheld" gives no way to judge whether the threshold is set too
    high. The maximum is the useful single number -- it is the one the user
    would have to lower the threshold past to see anything at all.

    Args:
        dispositions: One disposition dict per URL checked.

    Returns:
        The largest ``withheld_percentage`` present, or ``None`` when this run
        withheld nothing (so the key can be omitted from stats entirely rather
        than fabricating a 0.0 that reads as "withheld 0%").
    """
    percentages = [
        float(disposition["withheld_percentage"])
        for disposition in dispositions
        if disposition.get("withheld_percentage") is not None
    ]
    return max(percentages) if percentages else None


class LocalWatchlistsService:
    """Thin adapter over `SubscriptionsDB` for the shared watchlists seam."""

    CREATE_FORM_SOURCE_TYPES = ("rss", "atom", "url")

    #: TASK-2305. Every local run read goes through this projection, so a run
    #: arrives already knowing which source produced it and which watchlists
    #: that source belongs to. `local_watchlist_runs` stores only a
    #: `source_id`, and nothing on the Runs pane's path had ever resolved it
    #: -- so a whole run history rendered as "Untitled". `LEFT JOIN`, not
    #: `JOIN`: a run whose source cannot be resolved must still be listed
    #: (unnameable is a better history than absent).
    #:
    #: The watchlist names arrive as one `WATCHLIST_NAME_SEPARATOR`-joined
    #: column rather than a second query per run, ordered by name so the
    #: display is stable between reads.
    _RUN_SELECT = f"""
        SELECT r.*,
               s.name AS source_title,
               (
                   SELECT group_concat(name, '{WATCHLIST_NAME_SEPARATOR}')
                   FROM (
                       SELECT w.name AS name
                       FROM watchlist_sources ws
                       JOIN watchlists w ON w.id = ws.watchlist_id
                       WHERE ws.subscription_id = r.source_id
                       ORDER BY w.name
                   )
               ) AS watchlist_names
        FROM local_watchlist_runs r
        LEFT JOIN subscriptions s ON s.id = r.source_id
    """

    def __init__(
        self,
        *,
        db_factory: Callable[[], SubscriptionsDB],
        notification_dispatcher: Any | None = None,
        notification_app: Any | None = None,
        run_executor: Callable[[Mapping[str, Any]], Any] | None = None,
        filter_service: WatchlistFilterService | None = None,
        content_alert_service: WatchlistContentAlertService | None = None,
    ):
        self._db_factory = db_factory
        self._db_instance: SubscriptionsDB | None = None
        # Guards `_db_instance` only -- see `_db` for why a lock is needed at
        # all. Plain `Lock`, not `RLock`: a factory that re-entered `_db()`
        # would recurse forever anyway, and a deadlock is a louder failure
        # than the silent double-construction this exists to prevent.
        self._db_lock = threading.Lock()
        self.notification_dispatcher = notification_dispatcher
        self.notification_app = notification_app
        self.run_executor = run_executor
        self.filter_service = filter_service or WatchlistFilterService()
        self.content_alert_service = content_alert_service or WatchlistContentAlertService()

    @property
    def db_factory(self) -> Callable[[], SubscriptionsDB]:
        """How this service obtains its database. Resolved once; see `_db`."""
        return self._db_factory

    @db_factory.setter
    def db_factory(self, factory: Callable[[], SubscriptionsDB]) -> None:
        """Repoint the service, dropping whatever the old factory produced.

        A property purely so the cache cannot outlive the factory that filled
        it: assigning `db_factory` is a live test seam (`Tests/UI/
        test_watchlists_inspector.py` repoints a running app's service at a
        spied database mid-test), and a stale cached instance would leave that
        assignment silently inert.
        """
        with self._db_lock:
            self._db_factory = factory
            self._db_instance = None

    def _db(self) -> SubscriptionsDB:
        """The service's database, constructed at most once.

        task-15463. This used to be `return self.db_factory()`, and the
        production factory (`app.py`'s `_wire_watchlists_and_notifications_
        services`) constructed a whole new `SubscriptionsDB` on every call:
        a ~52-statement `executescript` plus migration probes for each of the
        five-plus loads a single Watchlists refresh fires -- measured at
        3.4 ms against 0.04 ms for the same query on a held instance (~85x),
        and 35 ms for the first construction.

        One instance shared across threads is safe: `SubscriptionsDB` keeps
        **thread-local** connections (`DB/Subscriptions_DB.py`'s `conn`
        property), so each `asyncio.to_thread` worker that touches this
        instance opens and reuses its own connection to the same file. That
        is what lets the check path in this module hop threads (see
        `db_offload.run_db_off_loop`) while still holding one instance. It is
        also why an in-memory database must NOT hop -- each connection would
        be a private empty database -- which that helper enforces.

        The cache IS locked, because this is genuinely called from more than
        one thread and the first call is not guaranteed to be the loop's
        (review round 1, Important). `list_home_run_snapshot` below is a
        synchronous method that calls `_db()` itself, and Home runs it inside
        `asyncio.to_thread` (`Home/active_work_adapter.py`'s
        `_compute_active_work_fields`) -- so a Home dashboard build on a
        worker thread can reach an unprimed cache at the same moment the
        event loop does. Unlocked double-checked assignment would then call a
        CONSTRUCTING factory twice, which is precisely the second
        `_initialize_schema` hazard this task exists to remove: a connection
        opened during the second schema rewrite caches a view missing the
        tables it is rewriting (see `app.py`'s `_backfill_subscription_items_
        fts` for the measured incident). The lock is taken once per service
        for real; after that it is an uncontended acquire in front of an
        attribute read.
        """
        if self._db_instance is not None:
            return self._db_instance
        with self._db_lock:
            # Re-checked under the lock: the thread that waited here while
            # another built the instance must return THAT one, not build a
            # second.
            if self._db_instance is None:
                self._db_instance = self._db_factory()
            return self._db_instance

    async def list_sources(
        self, *, limit: int = 100, offset: int = 0, q: str | None = None
    ) -> list[dict[str, Any]]:
        normalized_limit = int(limit)
        normalized_offset = int(offset)
        fetch_limit = (
            normalized_limit
            if not q
            else max(normalized_limit + normalized_offset, 1000)
        )
        db = self._db()
        rows = await run_db_off_loop(
            db,
            db.get_all_subscriptions,
            include_inactive=True,
            limit=fetch_limit,
            offset=0 if q else normalized_offset,
        )
        items = [normalize_local_subscription_row(row) for row in rows]
        if not q:
            return items
        needle = str(q).strip().lower()
        filtered = [
            item
            for item in items
            if needle in str(item.get("title") or "").lower()
            or needle in str(item.get("url") or "").lower()
        ]
        return filtered[normalized_offset : normalized_offset + normalized_limit]

    async def get_source(self, source_id: Any) -> dict[str, Any]:
        db = self._db()
        row = await run_db_off_loop(db, db.get_subscription, int(source_id))
        if row is None:
            raise KeyError(f"Subscription not found: {source_id}")
        return normalize_local_subscription_row(row)

    async def list_items(
        self,
        *,
        source_id: Any = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        run_id: Any = None,
        watchlist_id: Any = None,
        unassigned_only: bool = False,
        statuses: list[str] | None = None,
        is_flagged: bool | None = None,
        search: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """List watchlist items from the local subscriptions database.

        TASK-2301. `status=None` used to be collapsed to `"new"` here, so
        "list every item" was not expressible through this API at all: the
        Items tab asks with `status=None` and got a new-only list back, which
        its own "All statuses" filter then had nothing else to filter. An
        ingested or ignored item was not stale in that result -- it was
        absent, and therefore unreachable anywhere in the tab. `None` now
        means what it says and reaches `get_new_items(status=None)`, which
        drops the status predicate entirely.

        Review wave, Minor 7 -- the empty string changed meaning too, and it
        is called out here rather than glossed. Before TASK-2301 any falsey
        `status` (`None` OR `""`) became `"new"`; now any falsey `status`
        means EVERY status. `""` is deliberately kept on the same side as
        `None`: it is not a status any row holds, so the alternative would be
        a query guaranteed to return nothing. Audited at the time of the
        change -- `WatchlistsCollectionsScreen._load_items` is the only caller
        in the tree (via `WatchlistScopeService.list_items` /
        `WatchlistsBackendController.list_items`) and it passes `None` or a
        real status, never `""` -- so nothing relied on the old default. A
        future caller that wants the unread bucket must now ask for it by
        name.

        Args:
            source_id: Restrict to one source, or `None` for all.
            status: A single item status. Falsey (`None` or `""`) means every
                status -- NOT `"new"`, which is what it used to mean.
            limit: Page size.
            offset: Page offset.
            run_id: Restrict to the items one run produced (TASK-2306), or
                `None` for every run's.
            watchlist_id: Restrict to items of the sources in one watchlist
                (TASK-2513), or `None` for every watchlist's.
            unassigned_only: Restrict to items of sources belonging to no
                watchlist (TASK-2513).
            statuses: Restrict to any of several statuses (TASK-2513), or
                `None` to defer to `status`. Safe to combine with the default
                falsey `status` only -- `get_new_items` rejects passing both
                a truthy `status` and `statuses`.
            is_flagged: Restrict to starred rows (`True`) or unstarred rows
                (`False`), or `None` -- the default -- to not filter by the
                flag at all (TASK-3072).
            search: Full-text terms over title/content/author (TASK-3791 --
                the reader's `/`), or `None`/blank for no search predicate.
                Forwarded verbatim to `get_new_items`, which owns the
                FTS5-or-LIKE mechanics.
            since: Effective-date floor (TASK-3791 -- the Today feed), or
                `None` for no floor.

        Returns:
            Normalized item dicts for the requested window.
        """
        db = self._db()
        subscription_id = int(source_id) if source_id is not None else None
        status_filter = status if status else None
        fetch_limit = int(limit) + int(offset)
        rows = await run_db_off_loop(
            db,
            db.get_new_items,
            subscription_id=subscription_id,
            status=status_filter,
            limit=fetch_limit,
            run_id=int(run_id) if run_id is not None else None,
            watchlist_id=int(watchlist_id) if watchlist_id is not None else None,
            unassigned_only=bool(unassigned_only),
            statuses=list(statuses) if statuses is not None else None,
            is_flagged=is_flagged,
            search=search,
            since=since,
        )
        normalized = [normalize_watchlist_item("local", row) for row in rows]
        return normalized[int(offset) : int(offset) + int(limit)]

    async def list_reader_items_page(
        self,
        *,
        source_id: Any = None,
        status: str | None = None,
        limit: int = 50,
        run_id: Any = None,
        watchlist_id: Any = None,
        unassigned_only: bool = False,
        statuses: list[str] | None = None,
        is_flagged: bool | None = None,
        search: str | None = None,
        since: str | None = None,
        snapshot_max_item_id: int | None = None,
        after: WatchlistItemCursor | None = None,
    ) -> WatchlistItemPage:
        """Return one normalized page from the local Reader snapshot.

        Args:
            source_id: Optional subscription scope.
            status: Optional single status scope.
            limit: Maximum rows in the page.
            run_id: Optional producing-run scope.
            watchlist_id: Optional watchlist scope.
            unassigned_only: Whether to include only unassigned sources.
            statuses: Optional multiple-status scope.
            is_flagged: Optional starred-state scope.
            search: Optional full-text terms.
            since: Optional effective-date floor.
            snapshot_max_item_id: Existing snapshot high-water.
            after: Optional continuation cursor.

        Returns:
            A typed page whose item rows use the local watchlist shape.
        """
        db = self._db()
        page = await run_db_off_loop(
            db,
            db.get_reader_items_page,
            subscription_id=int(source_id) if source_id is not None else None,
            status=status if status else None,
            limit=int(limit),
            run_id=int(run_id) if run_id is not None else None,
            watchlist_id=int(watchlist_id) if watchlist_id is not None else None,
            unassigned_only=bool(unassigned_only),
            statuses=list(statuses) if statuses is not None else None,
            is_flagged=is_flagged,
            search=search,
            since=since,
            snapshot_max_item_id=snapshot_max_item_id,
            after=after,
        )
        return WatchlistItemPage(
            items=tuple(
                normalize_watchlist_item("local", row) for row in page.items
            ),
            has_more=page.has_more,
            snapshot_max_item_id=page.snapshot_max_item_id,
            snapshot_count=page.snapshot_count,
            next_cursor=page.next_cursor,
        )

    async def count_reader_item_arrivals(
        self,
        *,
        snapshot_max_item_id: int,
        source_id: Any = None,
        status: str | None = None,
        run_id: Any = None,
        watchlist_id: Any = None,
        unassigned_only: bool = False,
        statuses: list[str] | None = None,
        is_flagged: bool | None = None,
        search: str | None = None,
        since: str | None = None,
    ) -> int:
        """Count matching local Reader items created after a snapshot.

        Args:
            snapshot_max_item_id: Snapshot high-water that arrivals exceed.
            source_id: Optional subscription scope.
            status: Optional single status scope.
            run_id: Optional producing-run scope.
            watchlist_id: Optional watchlist scope.
            unassigned_only: Whether to include only unassigned sources.
            statuses: Optional multiple-status scope.
            is_flagged: Optional starred-state scope.
            search: Optional full-text terms.
            since: Optional effective-date floor.

        Returns:
            Number of matching rows created after the high-water.
        """
        db = self._db()
        return int(
            await run_db_off_loop(
                db,
                db.count_reader_item_arrivals,
                snapshot_max_item_id=int(snapshot_max_item_id),
                subscription_id=int(source_id) if source_id is not None else None,
                status=status if status else None,
                run_id=int(run_id) if run_id is not None else None,
                watchlist_id=(
                    int(watchlist_id) if watchlist_id is not None else None
                ),
                unassigned_only=bool(unassigned_only),
                statuses=list(statuses) if statuses is not None else None,
                is_flagged=is_flagged,
                search=search,
                since=since,
            )
        )

    def _source_batch_rows(
        self, payloads: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for payload in payloads:
            rows.append(
                {
                    "name": str(payload.get("name") or "Untitled subscription"),
                    "type": self._local_type_for_source_type(payload.get("source_type")),
                    "source": str(
                        payload.get("url")
                        or payload.get("source")
                        or self._first_configured_url(payload)
                        or ""
                    ),
                    "tags": list(payload.get("tags") or []),
                    "description": payload.get("description"),
                    "is_active": bool(payload.get("active", True)),
                    **self._subscription_config_fields(payload),
                }
            )
        return rows

    @staticmethod
    def _create_source_rows_exact_batch(
        db: SubscriptionsDB, rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        outcomes = db.create_sources_exact_batch(
            rows, result_mode="watchlist_source"
        )
        return [
            {
                "input_index": int(outcome["input_index"]),
                "outcome": str(outcome["outcome"]),
                "source": outcome["source"],
            }
            for outcome in outcomes
        ]

    def create_sources_exact_batch_sync(
        self, payloads: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create an ordered source batch on the current Console worker."""
        rows = self._source_batch_rows(payloads)
        db = self._db()
        return self._create_source_rows_exact_batch(db, rows)

    async def create_sources_exact_batch(
        self, payloads: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create an ordered source batch through the database-owner lock."""
        rows = self._source_batch_rows(payloads)
        db = self._db()
        return await run_db_off_loop(
            db,
            self._create_source_rows_exact_batch,
            db,
            rows,
        )

    async def create_source(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Create or resolve one exact configured source."""
        result = await self.create_sources_exact_batch([payload])
        return {
            **result[0]["source"],
            "creation_outcome": result[0]["outcome"],
        }

    async def update_source(
        self, source_id: Any, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        db = self._db()
        changes: dict[str, Any] = {}
        if "name" in payload:
            changes["name"] = payload["name"]
        if "url" in payload:
            changes["source"] = payload["url"]
        elif "source" in payload:
            changes["source"] = payload["source"]
        elif "extraction_rules" in payload:
            configured_url = self._first_configured_url(payload)
            if configured_url:
                changes["source"] = configured_url
        if "tags" in payload:
            changes["tags"] = payload["tags"]
        if "active" in payload:
            changes["is_active"] = bool(payload["active"])
        if "description" in payload:
            changes["description"] = payload["description"]
        if "source_type" in payload:
            changes["type"] = self._local_type_for_source_type(payload["source_type"])
        changes.update(self._subscription_config_fields(payload))
        if changes:
            await run_db_off_loop(
                db, db.update_subscription, int(source_id), **changes
            )
        row = await run_db_off_loop(db, db.get_subscription, int(source_id))
        return normalize_local_subscription_row(row)

    #: Statuses a watchlist item may be moved to from the UI. Mirrors
    #: `ItemsPane._STATUS_OPTIONS` minus its "all" filter entry.
    ITEM_STATUSES = ("new", "reviewed", "ingested", "ignored", "error")

    async def get_item_status(self, item_id: Any) -> str:
        """Read one item's current status, authoritatively.

        Added for the reader's `Mark unread` guard (PR #1091 review, F1). The
        guard previously inferred a status by listing each candidate status
        and looking for the item in the result, which is a paged query: an
        `ingested` item beyond the page depth simply was not in the list, and
        "absent from a truncated page" was read as "does not hold this
        status", so the guard let the destructive write through. This reads
        the one row instead, so page size cannot enter into it.

        Args:
            item_id: The item's local row id (bare, not namespaced).

        Returns:
            The item's current status.

        Raises:
            ValueError: If `item_id` is not an integer id.
            KeyError: If no item has that id.
        """
        try:
            row_id = int(item_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid watchlist item id: {item_id!r}") from exc
        db = self._db()
        return await run_db_off_loop(db, db.get_item_status, row_id)

    async def get_item_content(self, item_id: Any) -> str | None:
        """Read one item's full body text -- the reader's DETAIL fetch.

        TASK-15464 counterpart to `get_item_status` immediately above:
        `list_items`'s underlying `get_new_items` no longer selects
        `content` for list rows (the audit's named cost -- full scraped
        article/diff text on up to 100 rows per refresh), so the reader
        fetches it here, once, only for the item actually opened.

        Args:
            item_id: The item's local row id (bare, not namespaced).

        Returns:
            The stored content, or `None` if no row has this id, or the row
            has one but its content is itself NULL (see
            `SubscriptionsDB.get_item_content`'s docstring for why the two
            are not distinguished).

        Raises:
            ValueError: If `item_id` is not an integer id.
        """
        try:
            row_id = int(item_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid watchlist item id: {item_id!r}") from exc
        db = self._db()
        return await run_db_off_loop(db, db.get_item_content, row_id)

    async def get_url_snapshots(
        self, source_id: Any, url: str, *, limit: int = 2
    ) -> list[dict[str, Any]]:
        """The reader's `[full page]`/`[previous snapshot]` affordances (TASK-1494).

        A thin passthrough to `SubscriptionsDB.get_url_snapshots` -- no
        normalization needed on the way out; the three columns it returns
        (`id`, `extracted_content`, `created_at`) are exactly what the
        screen's `ViewSnapshotRequested` handler and `SnapshotViewModal`
        read. task-19562 B: now hops through `run_db_off_loop` like every
        other read on this service -- the comment that used to justify
        leaving this one inline (every other read already did) no longer
        applies, since none of them are inline anymore either.

        Args:
            source_id: Owning subscription id (bare, not namespaced) --
                `normalize_watchlist_item`'s `source_id` field.
            url: The exact URL the snapshot was captured for --
                `normalize_watchlist_item`'s `url` field.
            limit: How many rows to return, newest first.

        Returns:
            Up to `limit` dicts, newest first; empty when the (source, url)
            pair has no snapshot yet.
        """
        db = self._db()
        return await run_db_off_loop(
            db, db.get_url_snapshots, int(source_id), str(url), limit=limit
        )

    async def update_item(self, *, item_id: Any, status: str) -> dict[str, Any]:
        """Move one watchlist item to a new status.

        TASK-1120 AC#3. `SubscriptionsDB.mark_item_status` has always existed
        and nothing reached it: no service exposed an item-status method, so
        `WatchlistsBackendController.update_item_status` fell through its
        candidate-method loop and raised `NotImplementedError`. `Mark
        reviewed`, `Ingest` and `Ignore` therefore could not have worked even
        once the Inspector started offering them.

        Args:
            item_id: The item's local row id (bare, not namespaced).
            status: One of `ITEM_STATUSES`.

        Returns:
            The normalized item id, backend and new status.

        Raises:
            ValueError: If `status` is not a known item status, or `item_id`
                is not an integer id.
            KeyError: If no item has that id.
        """
        normalized_status = str(status or "").strip().lower()
        if normalized_status not in self.ITEM_STATUSES:
            raise ValueError(
                f"Unknown watchlist item status: {status!r}. "
                f"Expected one of {', '.join(self.ITEM_STATUSES)}."
            )
        try:
            row_id = int(item_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid watchlist item id: {item_id!r}") from exc
        db = self._db()
        updated = await run_db_off_loop(
            db, db.mark_item_status, row_id, normalized_status
        )
        if not updated:
            raise KeyError(f"Watchlist item not found: {item_id}")
        return {
            "success": True,
            "id": build_watchlist_item_id("local", "watchlist_item", row_id),
            "backend": "local",
            "entity_kind": "watchlist_item",
            "item_id": row_id,
            "status": normalized_status,
        }

    async def mark_all_read(
        self,
        *,
        source_id: Any = None,
        watchlist_id: Any = None,
        unassigned_only: bool = False,
    ) -> list[int]:
        """Mark every ``new`` item in scope ``reviewed``; return the affected ids.

        The bulk half of the reader's mark-all-read affordance (TASK-2513).
        Thin delegate to `SubscriptionsDB.mark_all_read`, with the same
        ``int(...) if ... is not None else None`` id normalization
        `list_items` applies to its scope arguments. The returned ids are the
        undo batch the screen hands to `restore_items_new` on `u`.

        Args:
            source_id: Restrict to one source, or `None` for all.
            watchlist_id: Restrict to items of the sources in one watchlist,
                or `None` for every watchlist's.
            unassigned_only: Restrict to items of sources belonging to no
                watchlist.

        Returns:
            The local row ids moved to ``reviewed``.
        """
        db = self._db()
        return await run_db_off_loop(
            db,
            db.mark_all_read,
            subscription_id=int(source_id) if source_id is not None else None,
            watchlist_id=int(watchlist_id) if watchlist_id is not None else None,
            unassigned_only=bool(unassigned_only),
        )

    async def restore_items_new(self, *, item_ids: list[Any]) -> int:
        """Move the given ids back to ``new`` — the undo half of `mark_all_read`.

        Only rows still ``reviewed`` are restored (`SubscriptionsDB`
        enforces that guard), so an item the user has since ingested or
        ignored is not yanked back to unread.

        Args:
            item_ids: Local row ids (bare, not namespaced) — the batch
                `mark_all_read` returned.

        Returns:
            How many rows were actually restored.

        Raises:
            ValueError: If any id is not an integer id.
        """
        row_ids: list[int] = []
        for item_id in item_ids or []:
            try:
                row_ids.append(int(item_id))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid watchlist item id: {item_id!r}") from exc
        db = self._db()
        return await run_db_off_loop(db, db.restore_items_new, row_ids)

    async def set_item_flagged(self, *, item_id: Any, flagged: bool) -> None:
        """Star or unstar one item (TASK-3072 plan task 7).

        The write behind the reader's `s` key and Star button. Thin delegate
        to `SubscriptionsDB.set_item_flagged` -- one row, one global flag,
        with the same ``int(...)`` id normalization `restore_items_new`
        applies to its ids.

        Args:
            item_id: The local row id (bare, already denamespaced by the
                scope service).
            flagged: `True` to star the item, `False` to unstar it.

        Raises:
            ValueError: If the id is not an integer id.
        """
        try:
            row_id = int(item_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid watchlist item id: {item_id!r}") from exc
        db = self._db()
        await run_db_off_loop(db, db.set_item_flagged, row_id, bool(flagged))

    async def find_source_id_by_url(self, url: str) -> int | None:
        """The id of the source carrying exactly this URL, or `None`.

        TASK-3604 (ADR-043 rule 6): OPML import dedupes against the existing
        roster through this lookup -- `create_source` is a plain INSERT, so
        without it a re-import duplicates every feed. Thin delegate to
        `SubscriptionsDB.get_subscription_id_by_source`.

        Args:
            url: The exact source URL to match.

        Returns:
            The subscription's id, or `None` when no row carries it.
        """
        db = self._db()
        return await run_db_off_loop(db, db.get_subscription_id_by_source, str(url))

    async def resolve_or_create_watchlist(self, name: str) -> tuple[dict[str, Any], bool]:
        """The watchlist named `name` (case-insensitive), creating it if missing.

        TASK-3604 (ADR-043 rule 4): OPML folder names map to watchlists by
        case-insensitive match on the stripped name -- `"AI"` reuses `AI`
        rather than creating a duplicate the rail cannot tell apart. The
        membership SQL stays in `WatchlistBundleService`; this is the seam
        the scope service reaches it through.

        Args:
            name: The folder's raw name (stripped here).

        Returns:
            ``(watchlist_dict, created)`` -- `created` is `True` when the
            watchlist was just inserted.

        Raises:
            ValueError: If the stripped name is empty.
        """
        bundle = WatchlistBundleService(self._db())
        wanted = str(name).strip()
        if not wanted:
            raise ValueError("watchlist name cannot be empty or whitespace-only")
        result = bundle.create_with_sources(
            wanted,
            description=None,
            tags=None,
            source_ids=(),
            if_exists="return_existing",
        )
        return result["watchlist"], result["outcome"] == "created"

    async def add_source_to_watchlist(self, *, watchlist_id: Any, source_id: Any) -> None:
        """Add a source to a watchlist (idempotent), via the bundle service.

        Args:
            watchlist_id: The watchlist's row id.
            source_id: The subscription's bare row id.
        """
        WatchlistBundleService(self._db()).add_source(int(watchlist_id), int(source_id))

    async def list_watchlists(self) -> list[dict[str, Any]]:
        """Every watchlist, via the bundle service (TASK-3604 export)."""
        return WatchlistBundleService(self._db()).list_watchlists(limit=10000)

    async def list_watchlist_source_rows(self, *, watchlist_id: Any) -> list[dict[str, Any]]:
        """One watchlist's member feeds, in the serializer's vocabulary.

        TASK-3604: maps the bundle's tree-row keys (`type`) onto the OPML
        payload keys (`source_type`) so the serializer's input contract
        stays in the OPML vocabulary on both sides of the round-trip.

        Args:
            watchlist_id: The watchlist whose members to list.

        Returns:
            One dict per member feed with ``name``, ``url`` and
            ``source_type``.
        """
        rows = WatchlistBundleService(self._db()).list_source_rows(int(watchlist_id))
        return [
            {"name": row["name"], "url": row["url"], "source_type": row["type"]}
            for row in rows
        ]

    async def list_unassigned_source_rows(self) -> list[dict[str, Any]]:
        """Feeds belonging to no watchlist, same vocabulary as above."""
        rows = WatchlistBundleService(self._db()).list_unassigned_source_rows()
        return [
            {"name": row["name"], "url": row["url"], "source_type": row["type"]}
            for row in rows
        ]

    async def delete_source(self, source_id: Any) -> dict[str, Any]:
        db = self._db()
        success = await run_db_off_loop(db, db.delete_subscription, int(source_id))
        return {
            "success": success,
            "id": f"local:subscription:{source_id}",
            "backend": "local",
            "entity_kind": "subscription",
            "source_id": int(source_id),
        }

    async def resume_source(self, source_id: Any) -> dict[str, Any]:
        """Clear an auto-paused source's pause and failure counters (task-2050).

        The UI's one-press recourse for a source auto-paused by repeated
        check failures (task-1410's `_advance_failure_and_maybe_pause`).
        Delegates to `SubscriptionsDB.reset_subscription_errors`, which
        already performs exactly this reset (`error_count`,
        `consecutive_failures`, `last_error`, `is_paused` all cleared) for
        the success branch of `record_check_result` -- this method is that
        reset's first caller reachable from outside the DB layer, giving it
        an explicit trigger instead of only ever firing as a side effect of
        a successful check.

        Safe to call on a source that is not currently paused: the
        underlying write zeroes counters that are already zero and clears an
        already-clear pause flag, so it is a harmless no-op. The UI never
        offers this action for a non-paused source (see
        `InspectorPane._is_paused_subscription`), but this method does not
        need that guard to stay correct on its own.

        Args:
            source_id: The subscription's raw database id.

        Returns:
            The resumed source, freshly normalized.

        Raises:
            KeyError: `source_id` does not name a subscription.
        """
        db = self._db()
        await run_db_off_loop(db, db.reset_subscription_errors, int(source_id))
        row = await run_db_off_loop(db, db.get_subscription, int(source_id))
        if row is None:
            raise KeyError(f"Subscription not found: {source_id}")
        return normalize_local_subscription_row(row)

    async def launch_run(
        self, *, source_id: Any = None, job_id: Any = None
    ) -> dict[str, Any]:
        """Insert a `queued` run row for one source and return it.

        Raises:
            KeyError: `source_id` does not name a subscription -- including
                the case where it stopped naming one part-way through this
                call (see below).
        """
        resolved_source_id = int(source_id if source_id is not None else job_id)
        db = self._db()
        # task-15463: both statements hop to a worker thread. The KeyError is
        # still raised here, on the caller's thread, so its traceback and its
        # position relative to the INSERT are unchanged.
        subscription = await run_db_off_loop(
            db, db.get_subscription, resolved_source_id
        )
        if subscription is None:
            raise KeyError(f"Subscription not found: {resolved_source_id}")
        try:
            receipt = await run_db_off_loop(
                db, self._insert_queued_run, db, resolved_source_id, self._utc_now()
            )
        except sqlite3.IntegrityError as exc:
            # Review round 1, Minor 3. The existence check above and this
            # INSERT are now two awaits apart rather than two straight-line
            # statements, so the window in which the user can delete the
            # source between them is real rather than theoretical -- a
            # scheduled check and a Delete Source press land on different
            # threads. `local_watchlist_runs.source_id` carries
            # `FOREIGN KEY ... REFERENCES subscriptions(id)` and every
            # connection sets `PRAGMA foreign_keys = ON`, so the loser of
            # that race got a raw `IntegrityError` where callers (the
            # scheduled-check handler, the Check Now path) were written
            # against `KeyError`. Mapped back to the documented contract, with
            # the original chained so the cause is not lost.
            raise KeyError(
                f"Subscription not found: {resolved_source_id}"
            ) from exc
        run = await self.get_run(receipt["id"])
        run["_claim_acquired"] = receipt["_claim_acquired"]
        return run

    @staticmethod
    def _insert_queued_run(
        db: SubscriptionsDB, source_id: int, now: str
    ) -> dict[str, Any]:
        """Accept one database-owned source claim."""
        return db.accept_watchlist_run(source_id, created_at=now)

    async def execute_run(self, run_id: Any) -> dict[str, Any]:
        """Execute a queued local watchlist run and persist its observed result.

        task-15463: every synchronous sqlite call below goes through
        `run_db_off_loop`, one awaited hop each, in the order they were
        already in -- a scheduled check is dispatched straight onto the event
        loop (`SchedulerLoop`), so this bookkeeping used to run on it. The
        fetch itself was already async and is untouched.
        """
        db = self._db()
        current = await self.get_run(run_id)
        source_id = int(current.get("source_id") or current.get("job_id"))
        subscription = await run_db_off_loop(db, db.get_subscription, source_id)
        if subscription is None:
            raise KeyError(f"Subscription not found: {source_id}")

        await run_db_off_loop(db, self._mark_run_started, db, int(run_id))
        start_time = time.time()
        try:
            result = await self._execute_subscription(subscription, db)
            raw_items = list(result.get("items") or [])
            stats = dict(result.get("stats") or {})
            stats.setdefault("items_found", len(raw_items))
            stats.setdefault("response_time_ms", int((time.time() - start_time) * 1000))

            # task-16838, review B1: a run the in-flight guard made check
            # NOTHING (every disposition `skipped` -- a concurrent check of
            # the same source held every URL's claim) must not travel the
            # ordinary completion accounting. It contacted the source zero
            # times, so it is not evidence of anything: `record_check_result`
            # below would land in its SUCCESS branch (zero `error`
            # dispositions -> `_all_error_check_message` returns None) and
            # reset the auto-pause breaker, clear `last_error`, stamp
            # `last_successful_check`, and inflate `subscription_stats`;
            # alert evaluation would fire `no_items` over an absence the run
            # never observed. The winner -- the check that actually holds the
            # claim -- records the source's real health moments later. Only
            # the run ROW is written (with its skipped stats), so the Runs
            # pane still shows honestly what this run did: nothing.
            if _entirely_skipped_dispositions(stats.get("dispositions")) and (
                not raw_items
            ):
                stats["items_ingested"] = 0
                stats["new_items_found"] = 0
                return await self.record_run_result(
                    run_id,
                    status=str(result.get("status") or "completed"),
                    stats=stats,
                    error_msg=result.get("error_msg"),
                    log_text=result.get("log_text"),
                    evaluate_alerts=False,
                )

            filters = await run_db_off_loop(
                db, self._load_source_filters, db, source_id
            )
            content_alert_rules = await run_db_off_loop(
                db, self._load_content_alert_rules, db, source_id
            )
            kept_items = self._apply_filters_and_alerts(
                raw_items, filters, content_alert_rules, int(run_id)
            )
            stats["items_ingested"] = len(kept_items)
            stats["new_items_found"] = len(kept_items)
            # No separate `items_filtered` here (review wave, Minor 3). It was
            # written for one round and removed: `items_found` above is the
            # ONLY writer of that key in the whole package, so on every row
            # this pipeline can produce the recorded value is exactly
            # `items_found - items_ingested`, which is what
            # `_run_accounting` derives. The recorded form was justified by an
            # injected `run_executor` reporting a feed's total rather than
            # what it handed over -- but the only production injection point
            # is `WatchlistPreviewService`, which previews and records no run.
            # Rather than keep a key that cannot differ (and a test pinning a
            # shape nothing emits), the derivation is the single answer until
            # an executor actually diverges.

            await run_db_off_loop(
                db,
                self._upsert_subscription_items,
                db,
                source_id,
                int(run_id),
                kept_items,
            )
            # task-1394 fix wave (review Finding #1): a `url_list`/`sitemap`
            # run where every URL errored still needs its own failure to
            # reach the subscription's auto-pause breaker, or a permanently
            # dead source can never auto-pause. A partial run (>=1 success)
            # is unaffected -- see `_all_error_check_message`'s docstring.
            all_error_message = _all_error_check_message(
                stats.get("dispositions"), len(raw_items)
            )
            await run_db_off_loop(
                db,
                db.record_check_result,
                source_id,
                items=None,
                stats=stats,
                error=all_error_message,
            )

            status = str(result.get("status") or "completed")
            if all_error_message and status == "completed":
                # More honest than "completed" with zero items: every URL
                # this run checked failed, so the run itself failed, even
                # though it did not raise (that is exactly the point of the
                # per-URL isolation this run status is not undoing).
                status = "failed"

            return await self.record_run_result(
                run_id,
                status=status,
                stats=stats,
                error_msg=result.get("error_msg") or all_error_message,
                log_text=result.get("log_text"),
            )
        except asyncio.CancelledError:
            # Batch-4 review, C1 (CRITICAL). This worker is cancelled whenever
            # the widget it is registered on is unmounted -- Textual's
            # `Widget._on_unmount` calls `self.workers.cancel_node(self)`,
            # which cancels every worker on that widget regardless of group
            # name (the named "wc_check_now" group only protects against a
            # SECOND `run_worker` call in the same group; it does nothing
            # about the screen itself being torn down). This app's screens are
            # never cached (`app.py`'s `_create_navigation_screen`), so
            # switching tabs while a check is running is an entirely ordinary
            # action that reaches exactly this path.
            #
            # `asyncio.CancelledError` is `BaseException`, not `Exception`
            # (Python >=3.8), so the sibling `except Exception` below never
            # saw it -- the run row `_mark_run_started` set to `running`
            # moments ago was never transitioned to anything else. Recorded
            # here, as an honest terminal state, before the cancellation is
            # allowed to keep propagating: a single `Task.cancel()` delivers
            # exactly one `CancelledError` at the next suspension point and
            # does not re-arm on the next `await`, so awaiting
            # `record_run_failure` here is safe and will not itself be cut
            # off. Re-raised afterward (not swallowed) so the coroutine still
            # finishes looking cancelled to asyncio/Textual, matching what a
            # caller further up the stack (there is none in the check-now
            # path today, but `execute_run` is not a private implementation
            # detail of it) would correctly expect.
            #
            # task-15463 changed what "safe" costs here, and the honest
            # version is this (review round 1, Minor 4). Before, every await
            # inside `record_run_failure` was a coroutine doing SYNCHRONOUS
            # sqlite -- none of them ever yielded to the loop, so once this
            # handler was entered the terminal write was effectively atomic.
            # It now takes about five real suspension points (its own
            # `record_check_error` hop, then `record_run_result`'s `get_run`,
            # UPDATE, alert-rule read and final `get_run`). The
            # single-cancel case this branch exists for -- the user switching
            # tabs -- is unaffected: that cancel has already been delivered,
            # the loop is alive, and the hops complete in about a
            # millisecond. What is genuinely new is that a SECOND
            # cancellation, in practice only interpreter/loop shutdown
            # cancelling every task, can now interrupt the recovery write and
            # leave the row reading `running`.
            #
            # Deliberately NOT shielded. `asyncio.shield` would not make the
            # write land: the outer await still raises at once, the inner
            # task is left detached, and the same shutdown that cancelled us
            # destroys it -- trading a stale row for a stale row plus a
            # "Task was destroyed but it is pending" warning and a write we
            # can no longer log the failure of. A row left at `running` by a
            # process that is exiting is visible in the Runs pane and can be
            # re-run; the failure mode is bounded and honest, which a
            # background write racing the interpreter is not.
            try:
                await self.record_run_failure(
                    run_id,
                    source_id=source_id,
                    error=(
                        "Check cancelled: navigated away before it finished."
                    ),
                    elapsed_ms=int((time.time() - start_time) * 1000),
                )
            except Exception:
                logger.opt(exception=True).warning(
                    f"Watchlists: could not record the cancellation of run "
                    f"{run_id!r}; it may still read 'running'."
                )
            raise
        except Exception as exc:
            return await self.record_run_failure(
                run_id,
                source_id=source_id,
                error=exc,
                elapsed_ms=int((time.time() - start_time) * 1000),
            )

    async def record_run_failure(
        self,
        run_id: Any,
        *,
        source_id: Any = None,
        error: BaseException | str,
        elapsed_ms: int = 0,
    ) -> dict[str, Any]:
        """Mark a run failed and its source errored, durably.

        TASK-1090. Extracted from `execute_run`'s own `except` branch so the
        caller that *launched* the run can use it too. `execute_run` only
        guarded the fetch itself: anything that went wrong around it -- the
        namespaced-id `ValueError` of TASK-1100, a subscription deleted
        between launch and execution -- left the row it had just inserted
        sitting at `queued` forever, with no error on it and nothing written
        to `subscriptions.last_error` either. The user had no way to find out
        that a check had failed, or even that one had been attempted.

        Args:
            run_id: The run to mark failed.
            source_id: Its source, so `last_error` is written too. Resolved
                from the run when omitted.
            error: The exception (or message) that stopped it.
            elapsed_ms: How long it ran before failing.

        Returns:
            The recorded run.
        """
        error_msg = str(error)
        db = self._db()
        if source_id is None:
            try:
                current = await self.get_run(run_id)
                source_id = current.get("source_id") or current.get("job_id")
            except Exception:
                # A run we cannot even read cannot name its source; the run
                # record below is still worth writing. Warned, not debugged --
                # this whole method exists because a swallowed failure here
                # left no trace at all.
                logger.opt(exception=True).warning(
                    f"Watchlists: could not resolve the source of failed run "
                    f"{run_id}; subscriptions.last_error will not be updated."
                )
        if source_id is not None:
            await run_db_off_loop(
                db, db.record_check_error, int(source_id), error_msg
            )
        return await self.record_run_result(
            run_id,
            status="failed",
            stats={
                "items_found": 0,
                "items_ingested": 0,
                "error_msg": error_msg,
                "response_time_ms": elapsed_ms,
            },
            error_msg=error_msg,
            log_text=f"Local watchlist execution failed: {error_msg}",
        )

    async def list_runs(
        self,
        *,
        source_id: Any = None,
        job_id: Any = None,
        limit: int = 100,
        offset: int = 0,
        **_: Any,
    ) -> list[dict[str, Any]]:
        db = self._db()
        filters: list[str] = []
        values: list[Any] = []
        resolved_source_id = source_id if source_id is not None else job_id
        if resolved_source_id is not None:
            filters.append("r.source_id = ?")
            values.append(int(resolved_source_id))
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        values.extend([int(limit), int(offset)])
        # task-19562 B: not one of the originally enumerated 19 methods (same
        # reason as `get_alert_rule` -- it reads through a bare
        # `db.conn.cursor()`, not a named `SubscriptionsDB` method), but it is
        # exactly the same class of loop-blocking read as everything else in
        # this sweep.
        rows = await run_db_off_loop(
            db, self._select_run_rows, db, where_clause, values
        )
        return [self._normalize_run_row(row) for row in rows]

    def _select_run_rows(
        self, db: SubscriptionsDB, where_clause: str, values: list[Any]
    ) -> list[Any]:
        """Read a page of run rows (task-19562 B hop body)."""
        cursor = db.conn.cursor()
        cursor.execute(
            f"""
            {self._RUN_SELECT}
            {where_clause}
            ORDER BY r.id DESC
            LIMIT ? OFFSET ?
            """,
            values,
        )
        return cursor.fetchall()

    def list_home_run_snapshot(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent local watchlist runs from a synchronous Home-safe path."""
        db = self._db()
        cursor = db.conn.cursor()
        cursor.execute(
            f"""
            {self._RUN_SELECT}
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        return [self._normalize_run_row(row) for row in cursor.fetchall()]

    async def get_run(self, run_id: Any) -> dict[str, Any]:
        db = self._db()
        # task-15463: on the scheduled path this runs three times per check
        # (launch, execute, record), so it hops like the writes around it.
        # The row is fully materialized by `fetchone` inside the hop; only
        # pure normalization happens back on the caller's thread.
        row = await run_db_off_loop(db, self._select_run_row, db, int(run_id))
        if row is None:
            raise KeyError(f"Watchlist run not found: {run_id}")
        return self._normalize_run_row(row)

    async def wait_for_terminal_run(self, run_id: Any) -> dict[str, Any]:
        """Observe a durable winner until terminal, deadline, or cancellation.

        Args:
            run_id: Durable run identifier returned by a losing claim receipt.

        Returns:
            The terminal normalized run receipt.

        Raises:
            TimeoutError: The winner stayed active through the bounded wait.
            asyncio.CancelledError: The caller cancelled observation.
        """
        deadline = time.monotonic() + _RUN_CLAIM_WAIT_TIMEOUT_SECONDS
        delay = _RUN_CLAIM_POLL_INITIAL_SECONDS
        while True:
            winner = await self.get_run(run_id)
            if str(winner.get("status") or "").lower() not in {"queued", "running"}:
                return winner
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for watchlist run {run_id}")
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, _RUN_CLAIM_POLL_MAX_SECONDS)

    def _select_run_row(self, db: SubscriptionsDB, run_id: int) -> Any:
        """Read one run row, source title and watchlist names included."""
        cursor = db.conn.cursor()
        cursor.execute(f"{self._RUN_SELECT} WHERE r.id = ?", (run_id,))
        return cursor.fetchone()

    @staticmethod
    def _write_run_result(
        db: SubscriptionsDB,
        run_id: int,
        status: str,
        now: str,
        stats_json: str,
        error_msg: str | None,
        log_text: str | None,
    ) -> int:
        """Write one run's terminal state; returns rows updated (task-15463 hop body).

        The caller raises the missing-run `KeyError` instead of this method,
        which is why the count comes back rather than the exception. That
        moves the raise from inside the transaction (where it rolled back) to
        after its commit -- provably equivalent, because the only way to get
        here with zero rows updated is an UPDATE that matched nothing, so
        there is nothing for either path to undo.
        """
        row = db.transition_watchlist_run(
            run_id,
            status=status,
            finished_at=now,
            stats_json=stats_json,
            error_msg=error_msg,
            log_text=log_text,
        )
        return int(row is not None)

    async def get_run_detail(self, run_id: Any, **_: Any) -> dict[str, Any]:
        return await self.get_run(run_id)

    async def cancel_run(self, run_id: Any) -> dict[str, Any]:
        db = self._db()
        now = self._utc_now()
        # task-19562 B: the whole `db.transaction()` block hops as one unit
        # -- `run_db_off_loop` must not be handed a callable that holds a
        # transaction open across the await boundary, so the UPDATE and its
        # commit/rollback happen entirely inside `_cancel_run_row`, on the
        # worker thread.
        updated_rows = await run_db_off_loop(
            db, self._cancel_run_row, db, int(run_id), now
        )
        if updated_rows == 0:
            raise KeyError(f"Watchlist run not found: {run_id}")
        return await self.get_run(run_id)

    @staticmethod
    def _cancel_run_row(db: SubscriptionsDB, run_id: int, now: str) -> int:
        """Mark one run cancelled; returns rows updated (task-19562 B hop body)."""
        row = db.transition_watchlist_run(
            run_id,
            status="cancelled",
            finished_at=now,
        )
        return int(row is not None)

    async def record_run_result(
        self,
        run_id: Any,
        *,
        status: str,
        stats: Mapping[str, Any] | None = None,
        error_msg: str | None = None,
        log_text: str | None = None,
        dispatch_notifications: bool = True,
        evaluate_alerts: bool = True,
    ) -> dict[str, Any]:
        """Persist a completed local run and emit notifications for matching alert rules.

        task-15463: the UPDATE and the alert-rule read each take one
        `run_db_off_loop` hop, in their existing order; the notification
        dispatch after them is unchanged and still runs on the caller's
        thread.

        ``evaluate_alerts=False`` (task-16838, review B1) skips the alert-rule
        evaluation entirely -- stronger than ``dispatch_notifications=False``,
        which still evaluates and returns the triggered alerts. Used by
        `execute_run` for an entirely-skipped run: a run that checked nothing
        must not fire `no_items` (or any status-agnostic rule) over an absence
        it never actually observed. The run row itself is still written.
        """
        db = self._db()
        current = await self.get_run(run_id)
        now = self._utc_now()
        stats_payload = dict(stats or {})
        if error_msg and "error_msg" not in stats_payload:
            stats_payload["error_msg"] = error_msg
        updated_rows = await run_db_off_loop(
            db,
            self._write_run_result,
            db,
            int(run_id),
            str(status),
            now,
            json.dumps(stats_payload, sort_keys=True),
            error_msg,
            log_text,
        )
        if updated_rows == 0:
            # Raised on the caller's thread, after the hop, so the missing-run
            # KeyError reads exactly as it did when the UPDATE was inline.
            raise KeyError(f"Watchlist run not found: {run_id}")

        # Reads `local_watchlist_alert_rules`; the dispatch below does not
        # touch sqlite and stays on the caller's thread, where the
        # notification dispatcher expects to be called.
        triggered_alerts: list[dict[str, Any]] = []
        if evaluate_alerts:
            triggered_alerts = await run_db_off_loop(
                db,
                self._evaluate_alert_rules_for_run,
                run_id=int(run_id),
                job_id=int(current.get("job_id") or current.get("source_id")),
                stats=stats_payload,
                status=str(status),
            )
        if dispatch_notifications:
            for alert in triggered_alerts:
                notification = self._dispatch_alert_notification(alert)
                if notification is not None:
                    alert["notification_id"] = notification.get("id")

        updated = await self.get_run(run_id)
        updated["triggered_alerts"] = triggered_alerts
        return updated

    async def list_alert_rules(
        self, *, job_id: Any = None, source_id: Any = None
    ) -> list[dict[str, Any]]:
        db = self._db()
        resolved_job_id = job_id if job_id is not None else source_id
        # task-19562 B: same reasoning as `list_runs`/`get_alert_rule` above --
        # a bare `db.conn.cursor()` read, now hopped off the loop.
        rows = await run_db_off_loop(
            db, self._select_alert_rule_rows, db, resolved_job_id
        )
        return [
            normalize_watchlist_alert_rule("local", self._alert_rule_row_to_dict(row))
            for row in rows
        ]

    def _select_alert_rule_rows(self, db: SubscriptionsDB, job_id: Any) -> list[Any]:
        """Read every alert rule, optionally scoped to one job (task-19562 B hop body)."""
        cursor = db.conn.cursor()
        if job_id is None:
            cursor.execute(
                "SELECT * FROM local_watchlist_alert_rules ORDER BY created_at DESC"
            )
        else:
            cursor.execute(
                """
                SELECT * FROM local_watchlist_alert_rules
                WHERE job_id = ? OR job_id IS NULL
                ORDER BY created_at DESC
                """,
                (int(job_id),),
            )
        return cursor.fetchall()

    async def get_alert_rule(self, rule_id: Any) -> dict[str, Any]:
        db = self._db()
        # task-19562 B: not one of the originally enumerated 19 methods --
        # the AST scan that produced that list only matched calls to named
        # `SubscriptionsDB` methods, and this reads through a bare
        # `db.conn.cursor()` -- but `create_alert_rule` and
        # `update_alert_rule` both end by awaiting this method, so leaving
        # it inline would have left those two still blocking the loop on
        # their last statement.
        row = await run_db_off_loop(
            db, self._select_alert_rule_row, db, int(rule_id)
        )
        if row is None:
            raise KeyError(f"Watchlist alert rule not found: {rule_id}")
        return normalize_watchlist_alert_rule(
            "local", self._alert_rule_row_to_dict(row)
        )

    def _select_alert_rule_row(self, db: SubscriptionsDB, rule_id: int) -> Any:
        """Read one alert rule row (task-19562 B hop body)."""
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT * FROM local_watchlist_alert_rules WHERE id = ?", (rule_id,)
        )
        return cursor.fetchone()

    async def create_alert_rule(
        self,
        *,
        name: str,
        condition_type: str,
        condition_value: Mapping[str, Any] | None = None,
        job_id: Any = None,
        source_id: Any = None,
        severity: str = "warning",
    ) -> dict[str, Any]:
        db = self._db()
        normalized_condition_type = self._validate_condition_type(condition_type)
        resolved_job_id = job_id if job_id is not None else source_id
        if resolved_job_id is not None:
            subscription = await run_db_off_loop(
                db, db.get_subscription, int(resolved_job_id)
            )
            if subscription is None:
                raise KeyError(f"Subscription not found: {resolved_job_id}")
        now = self._utc_now()
        # task-19562 B: the whole `db.transaction()` block hops as one unit
        # -- `run_db_off_loop` must not be handed a callable that holds a
        # transaction open across the await boundary, so the INSERT and its
        # commit/rollback happen entirely inside `_insert_alert_rule`, on the
        # worker thread.
        rule_id = await run_db_off_loop(
            db,
            self._insert_alert_rule,
            db,
            int(resolved_job_id) if resolved_job_id is not None else None,
            name,
            normalized_condition_type,
            self._serialize_condition_value(condition_value),
            severity,
            now,
        )
        return await self.get_alert_rule(rule_id)

    @staticmethod
    def _insert_alert_rule(
        db: SubscriptionsDB,
        job_id: int | None,
        name: str,
        condition_type: str,
        condition_value_json: str | None,
        severity: str,
        now: str,
    ) -> int:
        """Insert one alert rule row and return its id (task-19562 B hop body)."""
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO local_watchlist_alert_rules (
                    job_id, name, enabled, condition_type, condition_value_json, severity, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    name,
                    1,
                    condition_type,
                    condition_value_json,
                    severity,
                    now,
                    now,
                ),
            )
            return cursor.lastrowid

    async def update_alert_rule(self, rule_id: Any, **fields: Any) -> dict[str, Any]:
        db = self._db()
        current = await self.get_alert_rule(rule_id)
        updates: dict[str, Any] = {}
        if "name" in fields:
            updates["name"] = fields["name"]
        if "enabled" in fields:
            updates["enabled"] = 1 if bool(fields["enabled"]) else 0
        if "condition_type" in fields:
            updates["condition_type"] = self._validate_condition_type(
                fields["condition_type"]
            )
        if "condition_value" in fields:
            updates["condition_value_json"] = self._serialize_condition_value(
                fields["condition_value"]
            )
        if "severity" in fields:
            updates["severity"] = fields["severity"]
        if "job_id" in fields:
            job_id = fields["job_id"]
            if job_id is not None:
                subscription = await run_db_off_loop(
                    db, db.get_subscription, int(job_id)
                )
                if subscription is None:
                    raise KeyError(f"Subscription not found: {job_id}")
            updates["job_id"] = int(job_id) if job_id is not None else None
        if "source_id" in fields:
            source_id = fields["source_id"]
            if source_id is not None:
                subscription = await run_db_off_loop(
                    db, db.get_subscription, int(source_id)
                )
                if subscription is None:
                    raise KeyError(f"Subscription not found: {source_id}")
            updates["job_id"] = int(source_id) if source_id is not None else None
        if not updates:
            return current

        updates["updated_at"] = self._utc_now()
        assignments = ", ".join(f"{field} = ?" for field in updates)
        values = list(updates.values()) + [int(rule_id)]
        # task-19562 B: whole transaction block hops as one unit, same
        # reasoning as `create_alert_rule` above.
        updated_rows = await run_db_off_loop(
            db, self._update_alert_rule_row, db, assignments, values
        )
        if updated_rows == 0:
            raise KeyError(f"Watchlist alert rule not found: {rule_id}")
        return await self.get_alert_rule(rule_id)

    @staticmethod
    def _update_alert_rule_row(
        db: SubscriptionsDB, assignments: str, values: list[Any]
    ) -> int:
        """Apply one alert rule UPDATE; returns rows updated (task-19562 B hop body)."""
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE local_watchlist_alert_rules SET {assignments} WHERE id = ?",
                values,
            )
            return cursor.rowcount

    async def delete_alert_rule(self, rule_id: Any) -> dict[str, Any]:
        db = self._db()
        # task-19562 B: whole transaction block hops as one unit, same
        # reasoning as `create_alert_rule` above.
        deleted = await run_db_off_loop(
            db, self._delete_alert_rule_row, db, int(rule_id)
        )
        if not deleted:
            raise KeyError(f"Watchlist alert rule not found: {rule_id}")
        return {
            "deleted": True,
            "id": f"local:watchlist_alert_rule:{rule_id}",
            "backend": "local",
            "entity_kind": "watchlist_alert_rule",
            "rule_id": int(rule_id),
        }

    @staticmethod
    def _delete_alert_rule_row(db: SubscriptionsDB, rule_id: int) -> bool:
        """Delete one alert rule row (task-19562 B hop body)."""
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM local_watchlist_alert_rules WHERE id = ?", (rule_id,)
            )
            return cursor.rowcount > 0

    @staticmethod
    def _local_type_for_source_type(source_type: Any) -> str:
        normalized = str(source_type or "rss").strip()
        if normalized == "site":
            return "url"
        if normalized in {
            "rss",
            "atom",
            "json_feed",
            "url",
            "url_list",
            "podcast",
            "sitemap",
            "api",
        }:
            return normalized
        raise ValueError(f"Unsupported local watchlist source type: {normalized}")

    @classmethod
    def _first_configured_url(cls, payload: Mapping[str, Any]) -> str | None:
        extraction_rules = cls._parse_json_value(payload.get("extraction_rules"))
        urls = cls._coerce_url_list(
            extraction_rules.get("urls")
            if isinstance(extraction_rules, Mapping)
            else None
        )
        if urls:
            return urls[0]
        return None

    @staticmethod
    def _subscription_config_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed_fields = (
            "check_frequency",
            "extraction_method",
            "extraction_rules",
            "processing_options",
            "notification_config",
            "change_threshold",
            "ignore_selectors",
            "custom_headers",
            "rate_limit_config",
            "auto_pause_threshold",
        )
        return {
            field: payload[field]
            for field in allowed_fields
            if field in payload and payload[field] is not None
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _mark_run_started(self, db: SubscriptionsDB, run_id: int) -> None:
        now = self._utc_now()
        if db.mark_watchlist_run_started(run_id, started_at=now) is None:
            raise RuntimeError(f"Watchlist run is no longer queued: {run_id}")

    async def _execute_subscription(
        self,
        subscription: Mapping[str, Any],
        db: SubscriptionsDB,
    ) -> dict[str, Any]:
        executor = self.run_executor
        if executor is None:
            result = await self._default_run_executor(subscription, db)
        else:
            result = await self._maybe_await(executor(subscription))
        if result is None:
            return {"items": []}
        if isinstance(result, list):
            return {"items": result}
        if not isinstance(result, Mapping):
            raise ValueError(
                "Local watchlist run executor must return a mapping or list of items."
            )
        return dict(result)

    async def _default_run_executor(
        self,
        subscription: Mapping[str, Any],
        db: SubscriptionsDB,
    ) -> dict[str, Any]:
        from .monitoring_engine import FeedMonitor, URLMonitor

        subscription_config = self._subscription_execution_config(subscription)
        source_type = str(subscription_config.get("type") or "").strip()
        # `None` for the feed and API arms on the ordinary path, which have
        # no dispositions at all (spec §4) -- distinguished from `[]`, which
        # would record four zeros. task-19562: those arms now REASSIGN this
        # to a one-element skip list when the in-flight guard turns them
        # away, so a skipped check is not mistaken for a clean zero-item run.
        dispositions: list[dict[str, Any]] | None = None
        if source_type in _FEED_SOURCE_TYPES:
            items, dispositions = await self._run_guarded_source_check(
                db,
                subscription_config,
                "feed",
                lambda: FeedMonitor().check_feed(subscription_config),
            )
        elif source_type == "url":
            result, disposition = await self._check_url_guarded(
                URLMonitor(db),
                subscription_config,
                str(subscription_config.get("source") or ""),
                db,
                isolated=False,
            )
            items = [result] if result else []
            dispositions = [disposition]
        elif source_type == "url_list":
            monitor = URLMonitor(db)
            items = []
            dispositions = []
            for url in self._urls_for_url_list(subscription_config):
                result, disposition = await self._check_url_guarded(
                    monitor, subscription_config, url, db, isolated=True
                )
                dispositions.append(disposition)
                if result:
                    items.append(result)
        elif source_type == "sitemap":
            monitor = URLMonitor(db)
            items = []
            dispositions = []
            # The sitemap FETCH that produces this URL list happens above, in
            # the `await` the `for` iterates -- it is NOT covered by
            # `_check_url_isolated` and a failure there still fails the whole
            # run (task-1394's isolation is only for the per-URL loop body;
            # if the sitemap itself cannot be fetched there is no per-URL
            # work to isolate).
            for url in await self._urls_for_sitemap(subscription_config):
                result, disposition = await self._check_url_guarded(
                    monitor, subscription_config, url, db, isolated=True
                )
                dispositions.append(disposition)
                if result:
                    items.append(result)
        elif source_type == "api":
            items, dispositions = await self._run_guarded_source_check(
                db,
                subscription_config,
                "api",
                lambda: self._items_for_api_source(subscription_config),
            )
        else:
            raise ValueError(
                f"Unsupported local watchlist source type for execution: {source_type}"
            )
        result_payload: dict[str, Any] = {
            "items": items,
            "log_text": f"Local watchlist execution completed with {len(items)} item(s).",
        }
        if dispositions is not None:
            # `execute_run` already does `stats = dict(result.get("stats") or {})`
            # and persists it to the run's `stats_json`, so this reaches the Runs
            # pane with nothing further to wire.
            run_stats: dict[str, Any] = {
                "dispositions": _disposition_counts(dispositions)
            }
            # A sibling key rather than a sixth entry inside `dispositions`,
            # which is a dict of counters and stays one: a float in among the
            # integers would break every whole-dict comparison of the counts.
            # Omitted entirely when nothing was withheld (spec §1) -- see
            # `_max_withheld_percentage`.
            max_withheld = _max_withheld_percentage(dispositions)
            if max_withheld is not None:
                run_stats["max_withheld_pct"] = max_withheld
            result_payload["stats"] = run_stats
        return result_payload

    async def _run_guarded_source_check(
        self,
        db: Any,
        subscription_config: Mapping[str, Any],
        claim_kind: str,
        check: "Callable[[], Awaitable[list[Any]]]",
    ) -> tuple[list[Any], list[dict[str, Any]] | None]:
        """A whole-source check behind the same in-flight guard the URL arms use.

        task-19562 part A: only the url-family arms went through
        `_check_url_guarded`. The **feed** and **API** arms -- and feeds are
        the commonest source type -- ran unguarded, so a scheduler tick
        overlapping a manual "Check Now" ran the check twice: the alert
        notification fired twice and statistics double-counted.

        These arms have no per-URL granularity, so the claim is scoped to the
        whole source with a ``claim_kind`` sentinel in the URL slot. The
        sentinel is NUL-prefixed, which no real URL can contain, so it can
        never collide with a url-family claim for the same subscription.

        A skip returns a single `DISPOSITION_SKIPPED_IN_FLIGHT` disposition
        rather than the arm's usual ``None``. That is load-bearing, not
        cosmetic: `_entirely_skipped_dispositions` -> `execute_run` uses it to
        short-circuit source-health accounting, so a skipped check cannot take
        `record_check_result`'s SUCCESS branch and reset the auto-pause
        breaker, clear `last_error` and stamp `last_successful_check` for a
        run that never contacted the source. Returning ``[]`` items with
        ``None`` dispositions would have reported "nothing new" for a check
        that never happened.

        ``db`` is taken as a parameter and held in THIS frame across the
        await on purpose: the claim key stores ``id(db)`` and no reference to
        it, so it is the live parameter that stops the object being collected
        and its id reused while the claim is registered (see
        `_IN_FLIGHT_URL_CHECKS`).

        Args:
            db: The database this run writes to; part of the claim key.
            subscription_config: The source's execution config; must carry ``id``.
            claim_kind: Short arm name (``"feed"``/``"api"``) used to build
                the sentinel claim slot.
            check: Zero-arg coroutine factory performing the actual check.

        Returns:
            ``(items, dispositions)`` -- the check's items with ``None``
            dispositions when this entrant won the claim, or ``([], [skip])``
            when a concurrent check of the same source already held it.
        """
        from .monitoring_engine import DISPOSITION_SKIPPED_IN_FLIGHT

        subscription_id = subscription_config.get("id")
        claim_slot = f"\x00{claim_kind}"
        key = (id(db), subscription_id, claim_slot)
        if key in _IN_FLIGHT_URL_CHECKS:
            logger.info(
                f"watchlist check skipped: subscription {subscription_id} "
                f"already has a {claim_kind} check in flight"
            )
            return [], [
                {
                    "kind": DISPOSITION_SKIPPED_IN_FLIGHT,
                    "reason": None,
                    "withheld_percentage": None,
                }
            ]
        _IN_FLIGHT_URL_CHECKS.add(key)
        try:
            return await check(), None
        finally:
            # `finally` so a raise or a cancellation (the user navigating away
            # mid-check) can never strand the source as permanently in flight.
            _IN_FLIGHT_URL_CHECKS.discard(key)

    async def _check_url_guarded(
        self,
        monitor: Any,
        subscription_config: Mapping[str, Any],
        url: str,
        db: Any,
        *,
        isolated: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """One URL check behind the per-(subscription, url) in-flight guard.

        task-16838, from the TASK-15764 review (PR #1679, finding 1): a
        scheduled check and a manual Check Now of the same source both run on
        the app's event loop and can interleave across `check_url`'s awaits
        -- both read the same baseline before either writes, so one page
        change is double-reported with two snapshots written. This is the
        choke point every url-family arm of `_default_run_executor` (`url`,
        `url_list`, `sitemap`) goes through, so it covers every entrant that
        can write: the scheduler handler, the UI's Check Now / Rerun, and any
        direct `launch_run`/`execute_run` caller. (Shadow mode's direct
        `check_url` probe is deliberately outside it: `persist_snapshots=
        False` means it cannot write a snapshot or report an item, and the
        scheduler loop already serializes shadow probes against each other.)

        The second entrant SKIPS -- it does not queue or wait. A concurrent
        Check Now while a scheduled check runs means the user gets the
        scheduled check's result moments later; queuing a duplicate check
        behind it would re-fetch a page whose fresh snapshot the winner just
        wrote and report "unchanged", i.e. do network work to say nothing.
        The skip is honest, not silent: an INFO log here, a dedicated
        `DISPOSITION_SKIPPED_IN_FLIGHT` disposition (-> the run's `skipped`
        stats counter, rendered by the Runs pane), and the Check Now toast
        for an entirely-skipped run says a check was already running.

        Same-run re-entry cannot deadlock or self-skip: the `url_list`/
        `sitemap` loops await each check to completion, so for a duplicate
        URL within one run's list the claim registered here has already been
        released (the `finally` below) before the loop reaches the
        duplicate.

        Single-loop atomicity: the membership test and `add` below run
        synchronously between awaits on the app's one event loop, which every
        entrant runs on (see `_IN_FLIGHT_URL_CHECKS`). If any entrant ever
        moves off that loop, this check-then-add becomes a real race and
        needs a lock keyed the same way.

        Args:
            monitor: The `URLMonitor` (or fake, in tests) to check with.
            subscription_config: The source's execution config; must carry
                the subscription's ``id``.
            url: The one URL this call would check -- the claim key, and
                (when ``isolated``) the per-URL override passed through to
                `_check_url_isolated`.
            db: The database this run writes to. Part of the claim key so a
                preview's throwaway in-memory DB can never collide with the
                live one (see `_IN_FLIGHT_URL_CHECKS`).
            isolated: ``True`` for the `url_list`/`sitemap` loops, where a
                raise must become a `DISPOSITION_ERROR` for this URL rather
                than failing the whole run (task-1394); ``False`` for the
                single-`url` arm, where a raise still fails the run exactly
                as before.

        Returns:
            Whatever the underlying check returned, unchanged, when this
            entrant won the claim. ``(None, {"kind":
            DISPOSITION_SKIPPED_IN_FLIGHT, ...})`` when a concurrent check
            of the same (subscription, url) already held it.
        """
        from .monitoring_engine import DISPOSITION_SKIPPED_IN_FLIGHT

        subscription_id = subscription_config.get("id")
        key = (id(db), subscription_id, url)
        if key in _IN_FLIGHT_URL_CHECKS:
            # Subscription id only, never the URL -- it can carry a query
            # string with sensitive data (`_check_url_isolated`'s rule).
            logger.info(
                f"watchlist check skipped: subscription {subscription_id} "
                f"already has a check of this URL in flight"
            )
            return None, {
                "kind": DISPOSITION_SKIPPED_IN_FLIGHT,
                "reason": None,
                "withheld_percentage": None,
            }
        _IN_FLIGHT_URL_CHECKS.add(key)
        try:
            if isolated:
                return await self._check_url_isolated(
                    monitor, subscription_config, url
                )
            return await monitor.check_url(subscription_config)
        finally:
            # `finally`, so a raise (the non-isolated arm), a cancellation
            # (the user navigating away mid-check), or an isolated error can
            # never strand the pair as permanently "in flight" (AC #3).
            _IN_FLIGHT_URL_CHECKS.discard(key)

    @staticmethod
    async def _check_url_isolated(
        monitor: Any,
        subscription_config: Mapping[str, Any],
        url: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """One URL of a `url_list`/`sitemap` run's `check_url`, failure-isolated.

        task-1394: the `url_list`/`sitemap` loops used to call `check_url`
        with no per-URL `try/except`, so one failing URL (timeout, SSRF
        block, HTTP error) raised out of the whole loop, failed the entire
        run via `record_run_failure`, and discarded the items already
        collected from the URLs that succeeded -- a 50-URL source with one
        dead link yielded nothing at all.

        A raise here is turned into a `DISPOSITION_ERROR` disposition and a
        `None` item instead, so the caller's loop can `continue`: the run
        still completes, the OTHER urls' items and dispositions still
        persist, and `_disposition_counts` reports how many URLs errored
        rather than the run reporting clean zeros or failing outright.

        Args:
            monitor: The `URLMonitor` (or fake, in tests) to check with.
            subscription_config: The source's execution config; only
                ``source``/``type`` are overridden per URL, same as the
                caller did inline before this was extracted.
            url: The one URL this call checks.

        Returns:
            Whatever `monitor.check_url` returned, unchanged, on success.
            ``(None, {"kind": DISPOSITION_ERROR, "reason": None,
            "withheld_percentage": None})`` if it raised.
        """
        from .monitoring_engine import DISPOSITION_ERROR

        try:
            return await monitor.check_url(
                {**subscription_config, "source": url, "type": "url"}
            )
        except Exception as exc:
            # Type-only: never the exception message or the URL itself, both
            # of which can carry fetched page content or a query string with
            # sensitive data.
            logger.debug(
                f"watchlist URL check failed, isolated: {type(exc).__name__}"
            )
            return None, {
                "kind": DISPOSITION_ERROR,
                "reason": None,
                "withheld_percentage": None,
            }

    @classmethod
    def _subscription_execution_config(
        cls, subscription: Mapping[str, Any]
    ) -> dict[str, Any]:
        config = dict(subscription)
        for field in (
            "extraction_rules",
            "processing_options",
            "notification_config",
            "rate_limit_config",
            "custom_headers",
        ):
            if field in config:
                config[field] = cls._parse_json_value(config[field])
        return config

    @classmethod
    def _urls_for_url_list(cls, subscription: Mapping[str, Any]) -> list[str]:
        extraction_rules = subscription.get("extraction_rules")
        urls = []
        if isinstance(extraction_rules, Mapping):
            urls = cls._coerce_url_list(extraction_rules.get("urls"))
        if not urls:
            urls = cls._coerce_url_list(subscription.get("source"))

        return cls._apply_max_urls(urls, subscription)

    @classmethod
    async def _urls_for_sitemap(cls, subscription: Mapping[str, Any]) -> list[str]:
        import httpx

        try:
            import defusedxml.ElementTree as ET
        except ImportError:
            import xml.etree.ElementTree as ET

        source = str(subscription.get("source") or "").strip()
        if not source:
            return []

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await guarded_fetch_httpx_async(
                source,
                client=client,
                max_bytes=MAX_FETCH_BYTES_SITEMAP,
                trusted_origins=origin_set(source),
            )
            response.raise_for_status()

        root = ET.fromstring(response.text)
        urls: list[str] = []
        for url_node in root.iter():
            if cls._xml_local_name(url_node.tag) != "url":
                continue
            for child in list(url_node):
                if cls._xml_local_name(child.tag) == "loc" and child.text:
                    normalized_url = child.text.strip()
                    if normalized_url:
                        urls.append(normalized_url)
                    break
        return cls._apply_max_urls(urls, subscription)

    @staticmethod
    def _xml_local_name(tag: Any) -> str:
        text = str(tag)
        if "}" in text:
            return text.rsplit("}", 1)[-1]
        return text

    @staticmethod
    def _apply_max_urls(urls: list[str], subscription: Mapping[str, Any]) -> list[str]:
        processing_options = subscription.get("processing_options")
        max_urls = None
        if (
            isinstance(processing_options, Mapping)
            and processing_options.get("max_urls") is not None
        ):
            try:
                max_urls = max(int(processing_options["max_urls"]), 0)
            except (TypeError, ValueError):
                max_urls = None
        return urls[:max_urls] if max_urls is not None else urls

    @classmethod
    async def _items_for_api_source(
        cls, subscription: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        import httpx

        source = str(subscription.get("source") or "").strip()
        if not source:
            return []

        headers = {
            "Accept": "application/json",
            "User-Agent": "tldw-chatbook/1.0 (+https://github.com/tldw/chatbook)",
        }
        custom_headers = subscription.get("custom_headers")
        if isinstance(custom_headers, Mapping):
            headers.update(
                {str(key): str(value) for key, value in custom_headers.items()}
            )

        extraction_rules = subscription.get("extraction_rules")
        request_options = (
            extraction_rules if isinstance(extraction_rules, Mapping) else {}
        )
        params = request_options.get("params") or request_options.get("query")
        request_params = dict(params) if isinstance(params, Mapping) and params else None

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await guarded_fetch_httpx_async(
                source,
                client=client,
                max_bytes=MAX_FETCH_BYTES_PAGE,
                trusted_origins=origin_set(source),
                headers=headers,
                params=request_params,
            )
            response.raise_for_status()

        payload = response.json()
        items_payload = cls._api_items_payload(
            payload, request_options.get("items_path")
        )
        if not isinstance(items_payload, list):
            items_payload = [items_payload] if items_payload is not None else []
        items_payload = cls._apply_max_items(items_payload, subscription)

        field_map = request_options.get("field_map")
        normalized_field_map = field_map if isinstance(field_map, Mapping) else {}
        return [
            cls._normalize_api_item(item, normalized_field_map, source)
            for item in items_payload
        ]

    @classmethod
    def _api_items_payload(cls, payload: Any, items_path: Any = None) -> Any:
        if items_path:
            return cls._json_path(payload, str(items_path))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            for key in ("items", "entries", "results", "data"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return candidate
        return payload

    @classmethod
    def _json_path(cls, value: Any, path: str) -> Any:
        current = value
        for raw_part in path.split("."):
            part = raw_part.strip()
            if not part:
                continue
            if isinstance(current, Mapping):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
            if current is None:
                return None
        return current

    @classmethod
    def _normalize_api_item(
        cls,
        item: Any,
        field_map: Mapping[str, Any],
        source_url: str,
    ) -> dict[str, Any]:
        title = cls._api_item_field(
            item, field_map, "title", ("title", "name", "headline")
        )
        url = (
            cls._api_item_field(
                item, field_map, "url", ("url", "link", "html_url", "permalink")
            )
            or source_url
        )
        content = cls._api_item_field(
            item, field_map, "content", ("content", "summary", "description", "body")
        )
        published_date = cls._api_item_field(
            item,
            field_map,
            "published_date",
            ("published_date", "published", "date", "created_at", "updated_at"),
        )
        author = cls._api_item_field(
            item, field_map, "author", ("author", "by", "user")
        )
        content_hash = cls._api_item_field(
            item, field_map, "content_hash", ("content_hash", "hash", "id")
        )
        if not content_hash:
            content_hash = hashlib.sha256(
                f"{title or ''}{content or ''}".encode("utf-8")
            ).hexdigest()

        normalized = {
            "url": str(url),
            "title": str(title or url),
            "content": content,
            "content_hash": str(content_hash),
            "published_date": published_date,
            "author": author,
            "extracted_data": item if isinstance(item, Mapping) else {"value": item},
            # TASK-1343. An API source produces articles, not site changes, so
            # it must dispatch to `render_article` explicitly rather than rely
            # on `render_for`'s fallback -- which is what silently rendered
            # every kind the same way. `content` here is whatever JSON field
            # the source's `field_map` points at, in whatever format the API
            # chose; nothing converts it, and `_VALID_PAIRINGS` permits only
            # "text" or "markdown" for an article, so "text" is the honest
            # answer. Written even when the API supplied no content at all:
            # the kind is still `article` (`render_article` then explains the
            # missing body), and both values are non-None so they survive the
            # filter below.
            "content_kind": CONTENT_KIND_ARTICLE,
            "content_format": CONTENT_FORMAT_TEXT,
        }
        return {key: value for key, value in normalized.items() if value is not None}

    @classmethod
    def _api_item_field(
        cls,
        item: Any,
        field_map: Mapping[str, Any],
        field_name: str,
        fallback_paths: tuple[str, ...],
    ) -> Any:
        mapped_path = field_map.get(field_name)
        if mapped_path:
            value = cls._json_path(item, str(mapped_path))
            if value not in (None, ""):
                return value
        for path in fallback_paths:
            value = cls._json_path(item, path)
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _apply_max_items(
        items: list[Any], subscription: Mapping[str, Any]
    ) -> list[Any]:
        processing_options = subscription.get("processing_options")
        max_items = None
        if isinstance(processing_options, Mapping):
            configured = processing_options.get(
                "max_items", processing_options.get("max_urls")
            )
            if configured is not None:
                try:
                    max_items = max(int(configured), 0)
                except (TypeError, ValueError):
                    max_items = None
        return items[:max_items] if max_items is not None else items

    @staticmethod
    def _parse_json_value(value: Any) -> Any:
        if value in (None, ""):
            return {}
        if isinstance(value, (Mapping, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value

    @staticmethod
    def _coerce_url_list(value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            parts = value.replace(",", "\n").splitlines()
            return [part.strip() for part in parts if part.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _run_row_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        stats: dict[str, Any] = {}
        if payload.get("stats_json"):
            try:
                parsed = json.loads(payload["stats_json"])
                if isinstance(parsed, dict):
                    stats = parsed
            except json.JSONDecodeError:
                stats = {}
        return {
            "id": payload["id"],
            "source_id": payload.get("source_id"),
            "job_id": payload.get("job_id") or payload.get("source_id"),
            "status": payload.get("status"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "stats": stats,
            "error_msg": payload.get("error_msg"),
            "log_text": payload.get("log_text"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            # TASK-2305: joined identity. Carried through the row dict so
            # `normalize_watchlist_run` -- which also serves the server
            # backend, where neither exists -- reads them the same way it
            # reads every other field.
            "source_title": payload.get("source_title"),
            "watchlist_names": payload.get("watchlist_names"),
        }

    def _normalize_run_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize one `_RUN_SELECT` row.

        The single normalizer for every local run read (`list_runs`,
        `get_run`, `list_home_run_snapshot`). Before TASK-2305 only the Home
        snapshot resolved a run's source name, with its own hand-written JOIN,
        and the Runs pane's own list did not -- so the Runs tab showed
        "Untitled" for every run while Home, reading the same table, showed
        the real name. One query and one normalizer is what keeps those two
        from drifting again.
        """
        payload = dict(row)
        normalized = normalize_watchlist_run("local", self._run_row_to_dict(payload))
        source_title = normalized.get("source_title")
        if source_title:
            # Home's active-work rail reads `title` (see
            # `HomeActiveWorkAdapter._local_watchlist_run_items`).
            normalized["title"] = source_title
        return normalized

    @staticmethod
    def _alert_rule_row_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "user_id": "local",
            "job_id": payload.get("job_id"),
            "source_id": payload.get("job_id"),
            "name": payload.get("name"),
            "enabled": bool(payload.get("enabled", True)),
            "condition_type": payload.get("condition_type"),
            "condition_value": payload.get("condition_value_json") or "{}",
            "severity": payload.get("severity"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    @staticmethod
    def _serialize_condition_value(value: Mapping[str, Any] | str | None) -> str:
        if value is None:
            return "{}"
        if isinstance(value, str):
            return value
        return json.dumps(dict(value))

    @staticmethod
    def _validate_condition_type(condition_type: Any) -> str:
        normalized = str(condition_type or "").strip()
        if normalized not in _ALERT_CONDITION_TYPES:
            raise ValueError(
                "Invalid watchlist alert condition_type. "
                f"Expected one of: {', '.join(sorted(_ALERT_CONDITION_TYPES))}"
            )
        return normalized

    def _evaluate_alert_rules_for_run(
        self,
        *,
        run_id: int,
        job_id: int,
        stats: Mapping[str, Any],
        status: str,
    ) -> list[dict[str, Any]]:
        db = self._db()
        rules = db.conn.execute(
            """
            SELECT * FROM local_watchlist_alert_rules
            WHERE enabled = 1 AND (job_id = ? OR job_id IS NULL)
            ORDER BY created_at DESC
            """,
            (job_id,),
        ).fetchall()
        triggered: list[dict[str, Any]] = []
        for row in rules:
            rule = normalize_watchlist_alert_rule(
                "local", self._alert_rule_row_to_dict(row)
            )
            message = self._alert_message_for_rule(rule, stats=stats, status=status)
            if message is None:
                continue
            rule_id = int(rule["rule_id"])
            triggered.append(
                {
                    "rule_id": rule_id,
                    "rule_name": rule["name"],
                    "condition_type": rule["condition_type"],
                    "severity": rule["severity"],
                    "message": message,
                    "notification_payload": {
                        "kind": "watchlist_alert",
                        "source_job_id": str(job_id),
                        "source_domain": "watchlists",
                        "source_job_type": "watchlist_run",
                        "link_type": "watchlist_run",
                        "link_id": str(run_id),
                        "dedupe_key": f"watchlist-alert:{rule_id}:{run_id}",
                    },
                }
            )
        return triggered

    def _dispatch_alert_notification(
        self, alert: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        dispatcher = self.notification_dispatcher
        if dispatcher is None:
            return None
        return dispatcher.dispatch(
            app=self.notification_app,
            category="watchlists",
            title=f"Alert: {alert['rule_name']}",
            message=str(alert["message"]),
            severity=str(alert["severity"]),
            source_backend="local",
            source_entity_kind="watchlist_run",
            source_entity_id=str(alert["notification_payload"]["link_id"]),
            payload=dict(alert["notification_payload"]),
        )

    def _alert_message_for_rule(
        self,
        rule: Mapping[str, Any],
        *,
        stats: Mapping[str, Any],
        status: str,
    ) -> str | None:
        condition_type = rule.get("condition_type")
        items_found = self._coerce_int(stats.get("items_found"), default=0)
        items_ingested = self._coerce_int(stats.get("items_ingested"), default=0)
        error_rate = 1.0 - (items_ingested / items_found) if items_found > 0 else 0.0
        condition_value = dict(rule.get("condition_value") or {})

        if condition_type == "no_items":
            if items_ingested == 0:
                return f"Run produced 0 items (found {items_found})"
            return None
        if condition_type == "error_rate_above":
            threshold = self._coerce_float(
                condition_value.get("threshold"), default=0.5
            )
            if threshold is None:
                return None
            if error_rate > threshold:
                return f"Error rate {error_rate:.0%} exceeds {threshold:.0%} threshold"
            return None
        if condition_type == "items_below":
            threshold = self._coerce_optional_int(
                condition_value.get("threshold"), default=1
            )
            if threshold is None:
                return None
            if items_ingested < threshold:
                return f"Only {items_ingested} items ingested (threshold: {threshold})"
            return None
        if condition_type == "items_above":
            threshold = self._coerce_optional_int(
                condition_value.get("threshold"), default=1000
            )
            if threshold is None:
                return None
            if items_ingested > threshold:
                return f"{items_ingested} items ingested exceeds {threshold} threshold"
            return None
        if condition_type == "run_failed":
            if status == "failed":
                return f"Run failed: {stats.get('error_msg') or 'unknown error'}"
            return None
        return None

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_optional_int(value: Any, *, default: int) -> int | None:
        value = default if value is None else value
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float | None:
        value = default if value is None else value
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_source_filters(self, db: SubscriptionsDB, source_id: int) -> list[dict[str, Any]]:
        """Load active include/exclude/flag filters for a source."""
        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT id, name, is_active, conditions, action, action_params, priority, is_include_required
            FROM subscription_filters
            WHERE (subscription_id = ? OR subscription_id IS NULL)
            AND is_active = 1
            AND action IN ('include', 'exclude', 'flag')
            ORDER BY priority ASC, id ASC
            """,
            (source_id,),
        )
        filters: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            filters.append({
                "id": row["id"],
                "name": row["name"],
                "is_active": bool(row["is_active"]),
                "conditions": self._parse_json_value(row["conditions"]),
                "action": row["action"],
                "action_params": self._parse_json_value(row["action_params"]),
                "priority": int(row["priority"] or 0),
                "is_include_required": bool(row["is_include_required"]),
            })
        return filters

    def _load_content_alert_rules(self, db: SubscriptionsDB, source_id: int) -> list[dict[str, Any]]:
        """Load active content-alert rules for a source."""
        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT id, name, is_active, conditions, action, action_params, priority
            FROM subscription_filters
            WHERE (subscription_id = ? OR subscription_id IS NULL)
            AND is_active = 1
            AND action = 'notify'
            ORDER BY priority ASC, id ASC
            """,
            (source_id,),
        )
        rules: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            rules.append({
                "id": row["id"],
                "name": row["name"],
                "is_active": bool(row["is_active"]),
                "conditions": self._parse_json_value(row["conditions"]),
                "action": row["action"],
                "action_params": self._parse_json_value(row["action_params"]),
                "priority": int(row["priority"] or 0),
                "severity": (self._parse_json_value(row["action_params"]) or {}).get("severity", "warning"),
            })
        return rules

    def _apply_filters_and_alerts(
        self,
        items: list[dict[str, Any]],
        filters: list[dict[str, Any]],
        content_alert_rules: list[dict[str, Any]],
        run_id: int,
    ) -> list[dict[str, Any]]:
        """Apply filters and content-alert rules to raw fetched items."""
        evaluated = self.filter_service.evaluate(items, filters)
        kept: list[dict[str, Any]] = []
        for item, evaluation in zip(items, evaluated):
            decision = evaluation.get("filter_decision")
            if decision == "exclude":
                continue
            enriched = dict(item)
            enriched["filter_decision"] = decision
            enriched["matched_filter_id"] = evaluation.get("matched_filter_id")
            enriched["run_id"] = run_id
            alert_matches = self.content_alert_service.evaluate(enriched, content_alert_rules)
            enriched["alert_matches"] = alert_matches if alert_matches else None
            kept.append(enriched)
        return kept

    @staticmethod
    def _upsert_subscription_items(
        db: SubscriptionsDB,
        source_id: int,
        run_id: int,
        items: list[dict[str, Any]],
    ) -> None:
        """Persist or update subscription items for a run."""
        if not items:
            return
        now = LocalWatchlistsService._utc_now()
        with db.transaction() as conn:
            for item in items:
                url = str(item.get("url") or "")
                content_hash = str(item.get("content_hash") or "")
                if not url or not content_hash:
                    continue
                alert_matches = item.get("alert_matches")
                persist_subscription_item(
                    conn,
                    source_id,
                    {
                        **item,
                        "alert_matches": alert_matches,
                    },
                    run_id=run_id,
                    now=now,
                )
