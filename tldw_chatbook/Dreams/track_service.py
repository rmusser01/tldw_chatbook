"""Dreams track mechanism: watch a page or a question, judge its changes.

Phase 2 (Track) Task 3. ``track_page`` is the "track this page" path behind
the story modal's ``t`` key: attach-or-create a watchlists subscription for
the story URL, join it to the shared ``Dreams Tracked`` watchlist, pin an
``items_above 0`` alert rule so any new item on the page notifies the shared
inbox, and wrap all of it in one ``dream_tracked_items`` row (schema v2).
``untrack`` retires that wrapper -- dream-created subscriptions are DISABLED
(``is_active = 0``), never deleted: deletion would cascade the watchlist
membership and alert rules a user may have adopted, and an attached
(not dream-created) subscription is never touched at all.

Phase 2 (Track) Task 4 adds the question mechanism and its check loop:
``track_question`` wraps a plain search template (no subscription, no
watchlist -- the template IS the watch), and ``run_track_check`` is what the
scheduled ``dream_track_check`` task dispatches: budget-guard, run the query,
digest the result set, and compare against the last anchoring digest. An
identical digest is ``unchanged``; a different one gets exactly one judge
LLM call whose prompt carries ONLY the query, the CURRENT top snippets, and
the JSON-verdict question -- prior state exists only as a digest hash, so
there is no stored prior page text to leak. A ``changed`` verdict writes a
``changed`` run and dispatches one ``dreams_track`` notification;
``rebaseline_track`` re-anchors the comparison so the same change never
re-alerts.

Re-track after untrack (ruling P7, fix round 1) or after a sweep
auto-retire (ruling P10, final review): the URL lookup ignores
``is_active``, so a found source may be one a prior untrack disabled. When
it is (a dream-created tracked row for the subscription pins the
provenance), the re-track RE-ENABLES it and reports ``"re-enabled"``; a
disabled source with no dream provenance is refused with
``TrackSourceDisabled`` -- never adopted, never re-enabled behind the
user's back. A sweep auto-retire now also DISABLES the dream-created
subscription itself (Qodo #1, PR #2890) and the linked event reminder
(Qodo #13), so nothing Dreams scheduled outlives its wrapper; the
provenance lookup below still consults every status, because a wrapper
retired by an OLDER build (or a foreign paused row) must keep pinning
its subscription the same way.

Re-tracking a story that already has an ACTIVE same-mechanism wrapper
(Qodo #2, PR #2890) reuses it -- the cadence and event date refresh on
the existing row, nothing is created, and the outcome is
``"attached"`` -- so one story can never own two live wrappers around
the same watch and an untrack can never strand the older one.

Scheduling needs no registration here: ``WatchlistProjection`` fabricates
``watchlist:<subscription_id>`` tasks straight from subscription rows, so
creating the subscription IS the registration; ``DreamsProjection``
fabricates ``dream_track:<item_id>`` tasks straight from tracked-item rows.

Phase 2 Task 5 adds event reminders: both track paths promote an
event-dated wrapper into one ``one_time`` scheduled task firing a week
ahead (``promote_to_reminder``), through the app's ``ScheduledTasksDB``
``create_reminder_task`` seam -- degrading silently whenever that DB is
missing, unwired, or unwritable, because the tracked item already exists
by then and is the real deliverable.

Phase 2 Task 6 adds the lifecycle sweep (``sweep_track_lifecycle``) and
makes the caps/floors single-sourced: the cap, the check-interval floor,
and the quiet-retire count resolve through ``dreams_setting`` against
``DREAMS_DEFAULTS`` (``tracked_item_cap`` 20,
``track_min_check_interval_hours`` 12, ``track_quiet_retire_count`` 14)
with no module-local fallback literals left.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_chatbook.Chat.Chat_Functions import extract_response_content
from tldw_chatbook.Utils.input_validation import validate_url
from tldw_chatbook.Utils.timestamps import to_utc_iso

from . import discovery
from .discovery import Candidate
from .settings import DREAMS_DEFAULTS, dreams_setting

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .cycle_service import CycleDeps

#: Every Dreams-tracked page joins this one shared watchlist.
_TRACK_WATCHLIST_NAME = "Dreams Tracked"

#: ``dream_tracked_items.mechanism`` for the page path (schema vocabulary).
_PAGE_MECHANISM = "page"

#: ``dream_tracked_items.mechanism`` for the question path (schema
#: vocabulary): the template is re-run as a search, no page is watched.
_QUESTION_MECHANISM = "question"

#: How many top results the check search requests and the judge prompt
#: carries (spec's bounded egress: the CURRENT top snippets only).
_TRACK_CHECK_RESULT_COUNT = 5

#: Fixed sampling for the one judge call: a boolean verdict plus a short
#: note needs far less room than a story, and a near-zero temperature keeps
#: the verdict deterministic.
_JUDGE_MAX_TOKENS = 256
_JUDGE_TEMPERATURE = 0.1

_JUDGE_SYSTEM_PROMPT = (
    "You judge whether a tracked web query's results materially changed "
    "since a prior check. You see only the query and the CURRENT top "
    "results; the prior result set is kept solely as a hash and is not "
    "available to you. Judge semantic freshness signals -- dates, prices, "
    "availability, sold-out markers -- and answer with exactly one JSON "
    'object of the form {"changed": <bool>, "note": "<short explanation>"}. '
    "Your verdict is heuristic; keep the note under 200 characters."
)

_JUDGE_QUESTION = (
    "Compared with the prior result set for this query, is there material "
    'change? Answer exactly one JSON object: {"changed": <bool>, '
    '"note": "<short explanation>"}.'
)

#: Run statuses that never anchor a comparison: they carry no trustworthy
#: digest (error/skipped) or a deliberately suppressed one (withheld). The
#: most recent run OUTSIDE this set is the baseline.
_NON_BASELINE_RUN_STATUSES = frozenset({"error", "skipped", "withheld"})

#: How many recent runs ``_baseline_digest`` scans for an anchoring digest
#: (``DreamsDB.MAX_LIST_LIMIT`` -- the full bounded read).
_BASELINE_SCAN_LIMIT = 200

#: Days before a tracked ``event_date`` that the promoted reminder fires
#: (Phase 2 Task 5): one nudge with a week of runway, not a day-of alarm.
_REMINDER_LEAD_DAYS = 7

#: Grace after a tracked ``event_date`` before the sweep retires the watch
#: (Phase 2 Task 6): the event day itself plus one full day of aftermath
#: coverage; only once that day has fully passed is the watch dead.
_EVENT_PASSED_GRACE_DAYS = 1

#: Consecutive ``error`` runs that auto-pause a tracked item (Task 6),
#: mirroring the subscriptions loop's ``auto_pause_threshold`` default: a
#: watch that keeps failing its checks stops spending budget until a human
#: looks at it.
_FAILURE_PAUSE_THRESHOLD = 3

#: How many characters of each result's snippet feed the check digest
#: (Qodo #3, PR #2890). Whitespace-collapsed and capped: enough of the
#: snippet to move the digest when a price/date/availability line
#: changes, without letting engine re-crawls that append trailing prose
#: flip every check to ``changed``.
_SNIPPET_DIGEST_CHARS = 200

#: Event-loop-only claim set for in-flight checks (Qodo #16, PR #2890):
#: item ids with a ``run_track_check`` currently executing. Same shape
#: and reasoning as ``cycle_service._ACTIVE_CYCLE_DATES`` -- the claim is
#: checked and added with no await between, so a re-dispatch of the same
#: still-due item (the scheduler reloads while a slow search or judge
#: call runs) can never interleave past the guard and double-notify.
_INFLIGHT_TRACK_CHECKS: set[int] = set()


class TrackCapReached(RuntimeError):
    """The active tracked-item budget is exhausted; nothing was created.

    Attributes:
        reason_code: Stable machine identifier (``track_cap_reached``) the
            UI can branch on without string-matching the message.
    """

    reason_code = "track_cap_reached"


class TrackSourceDisabled(RuntimeError):
    """The URL belongs to a source that exists but is disabled (foreign).

    Ruling P7 (fix round 1): a disabled source with no dream provenance is
    never adopted, never re-enabled behind the user's back, and never given
    an active tracked item against a dead watch. Raised BEFORE any write.

    Attributes:
        reason_code: Stable machine identifier
            (``track_source_disabled``).
    """

    reason_code = "track_source_disabled"


class TrackInvalidURL(ValueError):
    """The URL to watch is not a well-formed http(s) address (Qodo #8).

    Raised by :func:`track_page` BEFORE any lookup or write. The story
    modal's ``_ingestable`` gate already refuses non-http rows, but the
    service is also reachable from direct callers, and a malformed
    address (bad scheme, embedded whitespace or credentials) persisted as
    a subscription would be processed by every later watchlist run.

    Attributes:
        reason_code: Stable machine identifier (``track_invalid_url``).
    """

    reason_code = "track_invalid_url"


def _cap() -> int:
    """The ``tracked_item_cap`` setting, coerced to a usable int.

    Single-source (Task 6): the value -- including the fallback for a
    missing or garbage config value -- comes from ``DREAMS_DEFAULTS``
    through :func:`dreams_setting`; no literal lives here.
    """
    try:
        return max(0, int(dreams_setting("tracked_item_cap")))
    except (TypeError, ValueError):
        return int(DREAMS_DEFAULTS["tracked_item_cap"])


def _min_interval_seconds() -> int:
    """``track_min_check_interval_hours`` converted to seconds."""
    try:
        hours = max(0, int(dreams_setting("track_min_check_interval_hours")))
    except (TypeError, ValueError):
        hours = int(DREAMS_DEFAULTS["track_min_check_interval_hours"])
    return hours * 3600


def _quiet_retire_count() -> int:
    """The ``track_quiet_retire_count`` setting, coerced to a usable int.

    Floored at 1: a zero would retire every active item on the first sweep,
    which no configuration can mean.
    """
    try:
        return max(1, int(dreams_setting("track_quiet_retire_count")))
    except (TypeError, ValueError):
        return int(DREAMS_DEFAULTS["track_quiet_retire_count"])


def _dream_owned_subscription(dreams_db: Any, subscription_id: int) -> bool:
    """True when any tracked row pins dream ownership of a subscription.

    Final review (ruling P10): provenance is a fact about the SUBSCRIPTION,
    not about one wrapper's lifecycle state, so every status is consulted --
    a sweep-retired (``event_passed``/``quiet``) or failure-paused row
    proves ownership exactly as well as an active one. Sync; the caller
    thread-offloads.
    """
    for status in ("active", "paused", "retired"):
        rows = dreams_db.list_tracked_items(status)
        if any(
            int(row.get("subscription_id") or 0) == subscription_id
            and int(row.get("created_by_dreams") or 0) == 1
            for row in rows
        ):
            return True
    return False


def _has_dreams_change_rule(rules: list[dict]) -> bool:
    """Whether an enabled Dreams change rule already covers the page.

    Qodo #15 (PR #2890): ``track_page`` pins one ``items_above 0`` rule
    named ``"Change: <title>"`` per subscription; the title suffix
    differs between tracks, so identity is the name PREFIX plus the
    condition -- exactly the tuple this path itself creates.

    Args:
        rules: ``LocalWatchlistsService.list_alert_rules`` rows
            (normalized, ``condition_value`` already coerced).

    Returns:
        True when any enabled rule on the subscription is a Dreams
        change rule (``Change: `` name, ``items_above`` threshold 0).
    """
    for rule in rules:
        if not bool(rule.get("enabled", True)):
            continue
        if not str(rule.get("name") or "").startswith("Change: "):
            continue
        if str(rule.get("condition_type") or "") != "items_above":
            continue
        value = rule.get("condition_value") or {}
        try:
            threshold = int(value.get("threshold"))
        except (AttributeError, TypeError, ValueError):
            continue
        if threshold == 0:
            return True
    return False


async def _refresh_wrapper_plan(
    dreams_db: Any, wrapper: dict, *, cadence: int, event_date: str | None
) -> None:
    """Rewrite an active wrapper's cadence/event date when they moved.

    The reuse branch of Qodo #2 (PR #2890). Only writes when a value
    actually differs -- re-tracking with identical arguments is a pure
    no-op read path.

    Args:
        dreams_db: The app's ``DreamsDB``.
        wrapper: The active tracked-item row being reused.
        cadence: The caller's (already clamped) check cadence.
        event_date: The caller's target-event date, or ``None``.
    """
    if (
        int(wrapper.get("cadence_seconds") or 0) == cadence
        and wrapper.get("event_date") == (event_date or None)
    ):
        return
    await asyncio.to_thread(
        dreams_db.set_tracked_plan, int(wrapper["id"]),
        cadence_seconds=cadence, event_date=event_date,
    )


async def _disable_linked_reminders(
    scheduling_db_getter: Any, tracked_item_id: int
) -> None:
    """Disable the one-time reminder linked to a tracked item; degrade.

    Qodo #13 (PR #2890): ``promote_to_reminder`` links the reminder with
    ``link_type="dream_tracked_item"``; untrack and sweep-retire stop the
    watch, and the linked reminder must stop with it. Every failure mode
    (no getter, raising getter, no scheduling DB, unwritable DB) logs and
    returns -- retirement itself already happened and must not unwind.

    Args:
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` or None.
        tracked_item_id: The retired tracked item whose reminder dies.
    """
    scheduling_db = None
    if scheduling_db_getter is not None:
        try:
            scheduling_db = scheduling_db_getter()
        except Exception as exc:  # noqa: BLE001 - reminder degrades
            logger.warning(
                "Dreams reminder getter failed while retiring: {}",
                type(exc).__name__)
            return
    if scheduling_db is None:
        return
    try:
        await asyncio.to_thread(
            scheduling_db.disable_reminders_by_link,
            "dream_tracked_item", str(tracked_item_id),
        )
    except Exception as exc:  # noqa: BLE001 - reminder degrades
        logger.warning(
            "Dreams reminder disable failed for item {}: {}",
            tracked_item_id, type(exc).__name__)


async def track_page(
    subs_service: Any,
    dreams_db: Any,
    *,
    url: str,
    title: str,
    intent: str,
    event_date: str | None = None,
    origin_story_id: int | None = None,
    cadence_seconds: int | None = None,
    scheduling_db_getter: Any = None,
) -> dict:
    """Attach or create the watch for one page, wrapping it as a tracked item.

    Args:
        subs_service: The app's ``LocalWatchlistsService`` (verified seam).
        dreams_db: The app's ``DreamsDB`` (schema v2).
        url: The exact page URL to watch (http(s) -- the modal gates this).
        title: Story title, used for the subscription / alert-rule names.
        intent: What the user wants out of the watch
            (``event``/``deal``/``topic``).
        event_date: Date the target event is pinned to, if any.
        origin_story_id: Dream story that prompted tracking, if any.
        cadence_seconds: Desired check cadence; clamped up to the
            configured minimum interval.
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` or None (Phase 2 Task 5); when the item
            carries an ``event_date`` a one-time reminder is promoted a
            week ahead. Degrades, never raises: a missing or failing
            getter leaves the tracking result unchanged.

    Returns:
        ``{"tracked_item_id": int, "subscription_id": int,
        "outcome": "created" | "attached" | "re-enabled",
        "watchlist_id": int}``. ``re-enabled`` means the URL resolved to a
        subscription a prior untrack had disabled (a dream-created tracked
        row for it exists), and it was re-activated. An attach to a source
        with dream provenance (ruling P10: any dream-created tracked row
        for the subscription pins it -- active, paused, or retired, e.g.
        one a sweep auto-retired while leaving the subscription live)
        wraps the new item ``created_by_dreams=1``, so a later untrack
        still disables the subscription. When the story already owns an
        ACTIVE page wrapper (Qodo #2, PR #2890) the existing row is
        reused (cadence/event date refreshed, nothing created) and
        reported as ``"attached"``.

    Raises:
        TrackInvalidURL: When ``url`` fails the shared ``validate_url``
            check (scheme, whitespace, credentials, host shape) --
            raised BEFORE any lookup or write.
        TrackCapReached: When ``count_active_tracked()`` already meets the
            cap -- raised BEFORE any subscription, watchlist, alert, or
            tracked-item write.
        TrackSourceDisabled: When the URL resolves to a DISABLED source
            with no dream provenance (the user disabled it themselves) --
            also raised before any write.
    """
    # URL validation FIRST (Qodo #8, PR #2890): the modal gates http(s),
    # but a direct caller can pass anything, and a malformed address
    # persisted as a subscription would ride every later watchlist run.
    if not validate_url(str(url)):
        raise TrackInvalidURL(f"refusing to track a non-http(s) URL: {url!r}")

    cadence = max(int(cadence_seconds or 0), _min_interval_seconds())

    # Reuse before anything is written (Qodo #2, PR #2890): a story that
    # already owns an ACTIVE page wrapper keeps it -- refresh the plan
    # (cadence, event date) on the existing row instead of stacking a
    # second wrapper around the same watch, which a later untrack of the
    # newer row would strand. Reuse consumes nothing, so the cap guard
    # below is deliberately not consulted (and must not refuse it).
    if origin_story_id is not None:
        wrapper = await asyncio.to_thread(
            dreams_db.find_tracked_by_story, origin_story_id)
        if wrapper is not None and wrapper.get("mechanism") == \
                _PAGE_MECHANISM:
            await _refresh_wrapper_plan(
                dreams_db, wrapper, cadence=cadence,
                event_date=event_date)
            # The shared watchlist necessarily exists (this wrapper's
            # subscription joined it); resolving is idempotent and keeps
            # the response contract honest.
            watchlist, _created = await subs_service.resolve_or_create_watchlist(
                _TRACK_WATCHLIST_NAME
            )
            logger.debug(
                "Dreams tracked page reused: subscription={} item={}",
                wrapper.get("subscription_id"), wrapper.get("id"),
            )
            return {
                "tracked_item_id": int(wrapper["id"]),
                "subscription_id": wrapper.get("subscription_id"),
                "outcome": "attached",
                "watchlist_id": int(watchlist["id"]),
            }

    # Cap guard FIRST: the whole point is that a capped runtime creates
    # nothing, so this precedes every write below.
    cap = _cap()
    active = await asyncio.to_thread(dreams_db.count_active_tracked)
    if active >= cap:
        raise TrackCapReached(
            f"track cap reached: {active} active tracked items (cap {cap})"
        )

    # Attach-or-create. An existing subscription for exactly this URL is
    # adopted untouched (never renamed, never re-cadenced, never disabled
    # by a later untrack): the user may already depend on it.
    existing = await subs_service.find_source_id_by_url(str(url))
    if existing is not None:
        subscription_id = int(existing)
        outcome = "attached"
        # P10 (final review): dream provenance is pinned by ANY tracked row
        # for this subscription, whatever its status. A sweep auto-retire
        # now disables the dream-created subscription too (Qodo #1), but
        # rows retired by older builds -- and any future writer that
        # retires a wrapper without touching the subscription -- must
        # still pin provenance, or a re-track would wrap the watch as
        # created_by_dreams=0 and a later untrack would (correctly)
        # never disable it: an orphaned active subscription.
        dream_owned = await asyncio.to_thread(
            _dream_owned_subscription, dreams_db, subscription_id
        )
        created_by_dreams = 1 if dream_owned else 0
        # P7 (fix round 1): the URL lookup ignores is_active, so a found
        # source may be one this flow disabled on a prior untrack, or a
        # foreign source the user disabled themselves. Attaching to either
        # as-is would record an active tracked item against a dead watch.
        source = await subs_service.get_source(subscription_id)
        if not bool(source.get("active")) and not bool(source.get("paused")):
            # "active=False, paused=False" is exactly the DISABLED state
            # untrack writes (is_active=0, is_paused untouched at 0); an
            # auto-paused source (task-1410, paused=True) keeps its own
            # Resume lifecycle and rides the plain attach branch.
            if not dream_owned:
                raise TrackSourceDisabled(
                    "source exists but is disabled (no dream provenance): "
                    f"subscription {subscription_id}"
                )
            # WE disabled it on a prior untrack (dream-created rows pin
            # this subscription): re-enable and say so.
            await subs_service.update_source(
                subscription_id, {"active": True}
            )
            outcome = "re-enabled"
    else:
        # Payload keys are the ones ``_source_batch_rows`` consumes:
        # ``source_type`` (not "type") and ``active`` (not "is_active").
        result = await subs_service.create_source(
            {
                "name": f"Dreams: {str(title)[:40]}",
                "source_type": "url",
                "source": str(url),
                "check_frequency": cadence,
                "active": True,
            }
        )
        subscription_id = int(result["source_id"])
        # ``create_source`` is exact-identity: a concurrent creator can win
        # the insert between the lookup above and this call. Its outcome
        # word is authoritative, and a subscription we did not insert is
        # NOT dream-owned -- untrack must leave it alone.
        if result["creation_outcome"] == "created":
            outcome = "created"
            created_by_dreams = 1
        else:
            outcome = "attached"
            created_by_dreams = 0

    watchlist, _created = await subs_service.resolve_or_create_watchlist(
        _TRACK_WATCHLIST_NAME
    )
    await subs_service.add_source_to_watchlist(
        watchlist_id=watchlist["id"], source_id=subscription_id
    )

    # Any new item emerging from the watched page fires a notification into
    # the shared inbox (the Watchlists Notifications pane displays it).
    # Idempotent (Qodo #15, PR #2890): a re-track on a NEW story with the
    # same URL resolves to the same subscription, and pinning a second
    # identical rule there would notify twice per qualifying run. An
    # existing enabled Dreams rule ("Change: " name + items_above 0) is
    # reused, never duplicated.
    rules = await subs_service.list_alert_rules(job_id=subscription_id)
    if not _has_dreams_change_rule(rules):
        await subs_service.create_alert_rule(
            name=f"Change: {str(title)[:40]}",
            condition_type="items_above",
            condition_value={"threshold": 0},
            job_id=subscription_id,
            severity="information",
        )

    tracked_item_id = await asyncio.to_thread(
        dreams_db.create_tracked_item,
        origin_story_id=origin_story_id,
        mechanism=_PAGE_MECHANISM,
        intent=intent,
        subscription_id=subscription_id,
        query_template=None,
        event_date=event_date,
        cadence_seconds=cadence,
        created_by_dreams=created_by_dreams,
    )
    # Phase 2 Task 5: an event-dated watch gets one scheduled reminder a
    # week ahead. The wrapper row above already exists, so a reminder
    # failure (or a runtime without the scheduled-tasks DB) only logs --
    # the tracking result is returned unchanged either way.
    await promote_to_reminder(
        scheduling_db_getter,
        {"id": tracked_item_id, "query_template": None,
         "event_date": event_date},
    )
    logger.debug(
        "Dreams tracked page: outcome={} subscription={} item={}",
        outcome, subscription_id, tracked_item_id,
    )
    return {
        "tracked_item_id": tracked_item_id,
        "subscription_id": subscription_id,
        "outcome": outcome,
        "watchlist_id": int(watchlist["id"]),
    }


async def untrack(
    subs_service: Any,
    dreams_db: Any,
    tracked_item_id: int,
    *,
    scheduling_db_getter: Any = None,
) -> dict:
    """Retire one tracked item, disabling a dream-created subscription.

    The wrapper always moves to ``retired`` (``retired_reason="manual"``).
    A subscription Dreams created is DISABLED via the existing service seam
    -- ``update_source(subscription_id, {"active": False})`` routing into
    ``SubscriptionsDB.update_subscription``'s allowlisted ``is_active``
    field -- never deleted, so its watchlist membership and alert rules
    survive for a user who adopted them. A subscription Dreams merely
    attached to (``created_by_dreams = 0``) is never touched. The linked
    event reminder is disabled too (Qodo #13, PR #2890), so a stopped
    watch cannot still fire its one-time nudge.

    Args:
        subs_service: The app's ``LocalWatchlistsService``.
        dreams_db: The app's ``DreamsDB``.
        tracked_item_id: The ``dream_tracked_items`` row to retire.
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` or None; when present, the reminder
            linked to this item (``link_type="dream_tracked_item"``) is
            disabled. Degrades, never raises: retirement already happened
            by the time this runs.

    Returns:
        ``{"tracked_item_id": int, "status": "retired",
        "subscription_disabled": bool}``.

    Raises:
        KeyError: If no tracked item has that id.
    """
    item = await asyncio.to_thread(dreams_db.get_tracked_item, tracked_item_id)
    if item is None:
        raise KeyError(f"tracked item not found: {tracked_item_id}")

    await asyncio.to_thread(
        dreams_db.set_tracked_status,
        tracked_item_id,
        "retired",
        retired_reason="manual",
    )
    await _disable_linked_reminders(scheduling_db_getter, tracked_item_id)

    subscription_id = item.get("subscription_id")
    subscription_disabled = False
    if item.get("created_by_dreams") and subscription_id:
        try:
            await subs_service.update_source(
                int(subscription_id), {"active": False}
            )
            subscription_disabled = True
        except Exception as exc:  # noqa: BLE001 - wrapper is retired either way
            logger.warning(
                "Dreams untrack: disabling subscription {} failed: {}",
                subscription_id, type(exc).__name__,
            )
    return {
        "tracked_item_id": tracked_item_id,
        "status": "retired",
        "subscription_disabled": subscription_disabled,
    }


# --- Lifecycle sweep (Phase 2 Task 6) -------------------------------------------


def _sweep_sync(
    dreams_db: Any, *, now_date: date, quiet_count: int
) -> tuple[list[str], list[dict]]:
    """The sweep's reads and writes; sync, the caller thread-offloads.

    One ``asyncio.to_thread`` hop for the whole sweep (the controller's
    stage discipline): every hop can land on a fresh executor thread, and
    each new thread opens another held SQLite connection, so per-operation
    hops would churn connections for no concurrency gain -- the rules are
    strictly sequential.

    Returns:
        ``(notes, retired)`` -- the human-readable degradation notes, and
        the full row dicts this pass moved to ``retired`` (NOT the paused
        ones): the async caller uses ``retired`` to disable each row's
        dream-created subscription (Qodo #1) and linked reminder
        (Qodo #13) through the service seams, which cannot run inside
        this sync hop.
    """
    notes: list[str] = []
    retired: list[dict] = []
    for item in dreams_db.list_tracked_items("active"):
        item_id = int(item["id"])

        # Rule 1: the event's day-after has fully passed. An unparsable
        # event_date leaves the item alone (degrade, never guess).
        event_date = item.get("event_date")
        if event_date:
            try:
                passed = (
                    date.fromisoformat(str(event_date))
                    + timedelta(days=_EVENT_PASSED_GRACE_DAYS)
                ) < now_date
            except ValueError:
                logger.warning(
                    "Dreams sweep: unparsable event_date {!r}", event_date)
                passed = False
            if passed:
                dreams_db.set_tracked_status(
                    item_id, "retired", retired_reason="event_passed")
                retired.append(item)
                notes.append(
                    f"track sweep: retired item {item_id} (event passed)")
                continue

        # Rule 2: the quiet streak. consecutive_track_dispositions stops
        # at the newest differing run, so a recent changed/error verdict
        # resets the count by construction.
        unchanged_run = dreams_db.consecutive_track_dispositions(
            item_id, "unchanged")
        if unchanged_run >= quiet_count:
            dreams_db.set_tracked_status(
                item_id, "retired", retired_reason="quiet")
            retired.append(item)
            notes.append(
                f"track sweep: retired item {item_id} "
                f"(quiet: {unchanged_run} unchanged)")
            continue

        # Rule 3: repeated failures pause the watch, never retire it.
        error_run = dreams_db.consecutive_track_dispositions(item_id, "error")
        if error_run >= _FAILURE_PAUSE_THRESHOLD:
            dreams_db.set_tracked_status(item_id, "paused")
            notes.append(
                f"track sweep: paused item {item_id} "
                f"(repeated failures: {error_run})")
    if notes:
        logger.info("Dreams track sweep: {}", "; ".join(notes))
    return notes, retired


async def sweep_track_lifecycle(
    dreams_db: Any,
    *,
    now: Any,
    subs_service_getter: Any = None,
    scheduling_db_getter: Any = None,
) -> list[str]:
    """Retire or pause active tracked items whose watch has gone dead.

    Three rules, applied to each ACTIVE item (paused and retired items are
    never revisited, and a retired item's ``retired_reason`` is never
    rewritten):

    * ``event_date + 1 day`` fully past (judged on the UTC date of
      ``now``) → ``retired("event_passed")`` -- the grace keeps the watch
      alive through the day after the event;
    * ``track_quiet_retire_count`` (default 14) CONSECUTIVE ``unchanged``
      runs → ``retired("quiet")`` -- a streak broken by any other
      disposition resets, so a watch the judge recently called ``changed``
      is never retired;
    * ``_FAILURE_PAUSE_THRESHOLD`` (3) consecutive ``error`` runs →
      ``paused`` (mirroring the subscriptions loop's
      ``auto_pause_threshold``): a repeatedly failing check stops spending
      budget without erasing the watch.

    One rule fires per item per sweep (first match wins, ``continue``),
    and every hit appends a human-readable note to the returned list --
    the callers' degradation-notes channel. The whole sweep runs under a
    single ``asyncio.to_thread`` hop (see :func:`_sweep_sync`).

    A wrapper the sweep retires also loses the things Dreams scheduled
    around it (Qodo #1/#13, PR #2890): a dream-created page subscription
    is DISABLED through the ``update_source({"active": False})`` seam
    (attached foreign subscriptions are never touched), and the linked
    one-time event reminder is disabled. Both degrade per item -- a
    missing service, a raising getter, or a failed write logs and moves
    on; the retirement itself already stands.

    Args:
        dreams_db: The app's ``DreamsDB`` (schema v2).
        now: The sweep clock (UTC-aware ``datetime``, ``CycleDeps.now``
            shaped); event-past judging uses its UTC date.
        subs_service_getter: Zero-arg callable returning the app's
            ``LocalWatchlistsService`` or None; None (the default) means
            retired page items keep their subscription -- the legacy
            sweep behavior -- never an error.
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` or None; None means linked reminders are
            left enabled.

    Returns:
        Degradation notes naming every retirement/pause the sweep made.
    """
    notes, retired = await asyncio.to_thread(
        _sweep_sync, dreams_db,
        now_date=now.date(), quiet_count=_quiet_retire_count())

    subs_service = None
    if subs_service_getter is not None:
        try:
            subs_service = subs_service_getter()
        except Exception as exc:  # noqa: BLE001 - sweep degrades, never raises
            logger.warning(
                "Dreams sweep: watchlists service getter failed: {}",
                type(exc).__name__)
    for item in retired:
        # Qodo #13 first: the linked reminder dies for EVERY retired item
        # (page and question alike -- track_question promotes reminders
        # too).
        await _disable_linked_reminders(scheduling_db_getter, int(item["id"]))
        subscription_id = item.get("subscription_id")
        # Qodo #1: a dream-created PAGE subscription is disabled so the
        # independent watchlist projection stops scheduling it; a foreign
        # attached subscription (created_by_dreams=0) is never ours to
        # stop, and a question item has no subscription at all.
        if (
            subs_service is not None
            and item.get("mechanism") == _PAGE_MECHANISM
            and int(item.get("created_by_dreams") or 0) == 1
            and subscription_id
        ):
            try:
                await subs_service.update_source(
                    int(subscription_id), {"active": False}
                )
            except Exception as exc:  # noqa: BLE001 - retire stands either way
                logger.warning(
                    "Dreams sweep: disabling subscription {} failed: {}",
                    subscription_id, type(exc).__name__,
                )
    return notes


# --- Event reminders (Phase 2 Task 5) -------------------------------------------


async def promote_to_reminder(
    scheduling_db_getter: Any,
    tracked_item: Any,
    *,
    lead_days: int = _REMINDER_LEAD_DAYS,
    owner_id: str = "local",
) -> str | None:
    """Promote one event-dated tracked item into a scheduled reminder.

    Only a tracked item with an ``event_date`` promotes (a dateless watch
    has nothing on the calendar). The reminder is one ``one_time`` task
    firing ``lead_days`` before the event, linked back to the tracked item
    (``link_type="dream_tracked_item"``, ``link_id=str(id)``) so the
    scheduling UI can trace it.

    Degrade, never raise -- the wrapper row already exists when this is
    called, so every failure mode (no getter, getter returning None --
    a runtime whose scheduled-tasks DB is not wired yet -- a raising
    getter, an unwritable scheduling DB) logs and returns ``None``;
    tracking itself is unaffected.

    Args:
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` (``create_reminder_task`` seam) or None.
        tracked_item: Mapping with at least ``id``, ``query_template`` and
            ``event_date`` -- the tracked row (or its just-created view).
        lead_days: Days before ``event_date`` the reminder fires.
        owner_id: Reminder owner (the local scheduling owner id).

    Returns:
        The new reminder task id, or ``None`` when nothing was promoted.
    """
    event_date = tracked_item.get("event_date")
    if not event_date:
        return None

    scheduling_db = None
    if scheduling_db_getter is not None:
        try:
            scheduling_db = scheduling_db_getter()
        except Exception as exc:  # noqa: BLE001 - reminder degrades
            logger.warning(
                "Dreams reminder getter failed: {}", type(exc).__name__)
            return None
    if scheduling_db is None:
        # Wiring order: a runtime without the scheduled-tasks DB still
        # tracks; the reminder is a bonus, not a precondition.
        return None

    try:
        run_at = (
            date.fromisoformat(str(event_date)) - timedelta(days=int(lead_days))
        ).isoformat()
    except ValueError:
        logger.warning(
            "Dreams reminder: unparsable event_date {!r}", event_date)
        return None

    try:
        return await asyncio.to_thread(
            scheduling_db.create_reminder_task,
            owner_id,
            f"Dreams: {tracked_item.get('query_template') or 'tracked event'}",
            body=f"Tracked Dreams event on {event_date}",
            schedule_kind="one_time",
            run_at=run_at,
            next_run_at=run_at,
            link_type="dream_tracked_item",
            link_id=str(tracked_item.get("id")),
        )
    except Exception as exc:  # noqa: BLE001 - reminder degrades
        logger.warning(
            "Dreams reminder creation failed: {}", type(exc).__name__)
        return None


# --- Question mechanism + judged change checks (Phase 2 Task 4) ----------------


def _local_date(now: Any) -> str:
    """The user's local calendar date bucketing a check's budget spend.

    Same bucketing rule ``cycle_service._local_date`` applies to cycles;
    kept as its own copy so importing this module does not pull the whole
    cycle-service chain (discovery, story generation, web search).
    """
    return now.astimezone().strftime("%Y-%m-%d")


async def track_question(
    dreams_db: Any,
    *,
    query_template: str,
    intent: str,
    event_date: str | None = None,
    origin_story_id: int | None = None,
    cadence_seconds: int | None = None,
    scheduling_db_getter: Any = None,
) -> int:
    """Wrap one search question as a watched tracked item.

    No LLM synthesizes anything here and no subscription is created: the
    template IS the watch, re-run verbatim (region-substituted) by each
    :func:`run_track_check`. The same cap guard and cadence floor as
    :func:`track_page` apply.

    Args:
        dreams_db: The app's ``DreamsDB`` (schema v2).
        query_template: Search query template; ``{region}`` is substituted
            from ``[dreams] region`` at check time, any other placeholder
            is the caller's bug (rendering degrades to the literal).
        intent: What the user wants out of the watch
            (``event``/``deal``/``topic``).
        event_date: Date the target event is pinned to, if any.
        origin_story_id: Dream story that prompted tracking, if any.
        cadence_seconds: Desired check cadence; clamped up to the
            configured minimum interval.
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` or None (Phase 2 Task 5); when the item
            carries an ``event_date`` a one-time reminder is promoted a
            week ahead. Degrades, never raises.

    Returns:
        The tracked item id -- a NEW row, or the EXISTING active row when
        the story already owns a question wrapper for this template
        (Qodo #2, PR #2890: the plan refreshes on the existing row and
        nothing is created).

    Raises:
        ValueError: If ``query_template`` is empty/blank -- an empty watch
            is a caller bug, not a degradation.
        TrackCapReached: When ``count_active_tracked()`` already meets the
            cap -- raised BEFORE the tracked-item row is written.
    """
    template = str(query_template)
    if not template.strip():
        raise ValueError("track_question requires a non-empty query_template")

    cadence = max(int(cadence_seconds or 0), _min_interval_seconds())

    # Reuse before anything is written (Qodo #2, PR #2890), same shape as
    # ``track_page``'s branch: the story's existing ACTIVE question
    # wrapper keeps watching; only its cadence/event date refresh.
    if origin_story_id is not None:
        wrapper = await asyncio.to_thread(
            dreams_db.find_tracked_by_story, origin_story_id)
        if wrapper is not None and wrapper.get("mechanism") == \
                _QUESTION_MECHANISM:
            await _refresh_wrapper_plan(
                dreams_db, wrapper, cadence=cadence,
                event_date=event_date)
            # Qodo #11: identifiers and lengths only -- the template is
            # user text and must not reach the logs.
            logger.debug(
                "Dreams tracked question reused: item={} template_len={}",
                wrapper["id"], len(str(wrapper.get("query_template") or "")),
            )
            return int(wrapper["id"])

    # Cap guard FIRST, identical discipline to ``track_page``.
    cap = _cap()
    active = await asyncio.to_thread(dreams_db.count_active_tracked)
    if active >= cap:
        raise TrackCapReached(
            f"track cap reached: {active} active tracked items (cap {cap})"
        )

    tracked_item_id = await asyncio.to_thread(
        dreams_db.create_tracked_item,
        origin_story_id=origin_story_id,
        mechanism=_QUESTION_MECHANISM,
        intent=intent,
        subscription_id=None,
        query_template=template,
        event_date=event_date,
        cadence_seconds=cadence,
        # The Dreams loop owns the whole watch (no user subscription to
        # adopt); with ``subscription_id`` NULL the flag's only effect --
        # untrack disabling a subscription -- is inert by construction.
        created_by_dreams=1,
    )
    # Phase 2 Task 5: same event reminder as the page path, same degrade.
    await promote_to_reminder(
        scheduling_db_getter,
        {"id": tracked_item_id, "query_template": template,
         "event_date": event_date},
    )
    # Qodo #11 (PR #2890): identifiers and lengths only -- the template is
    # caller-supplied text (it can carry anything the user typed next to
    # a secret), so it must never be logged, not even a prefix.
    logger.debug("Dreams tracked question: item={} template_len={}",
                 tracked_item_id, len(template))
    return tracked_item_id


def _render_query(template: str) -> str:
    """Substitute ``{region}``; a malformed template degrades to literal.

    ``str.format`` raises ``KeyError``/``IndexError``/``ValueError`` on
    stray braces or unknown fields -- a template is user input, so the
    check treats it as the literal query rather than crashing the loop.
    """
    try:
        return template.format(region=dreams_setting("region", ""))
    except (KeyError, IndexError, ValueError):
        return template


def _snippet_component(snippet: Any) -> str:
    """Normalized snippet fragment for the digest (Qodo #3, PR #2890).

    Whitespace-collapsed (``"a  b\\n c"`` → ``"a b c"``) and capped at
    ``_SNIPPET_DIGEST_CHARS``: re-crawl noise -- trailing prose drift,
    re-wrapped lines -- must not flip every check to ``changed``, while a
    price/date/availability change near the top of the snippet must.
    """
    return " ".join(str(snippet or "").split())[:_SNIPPET_DIGEST_CHARS]


def _results_digest(results: list[Candidate]) -> str:
    """Content digest of one check's result set.

    Hashes each result's normalized URL, title, and (normalized,
    bounded) snippet. The snippet component is what makes
    price/date/availability changes inside an unchanged URL+title
    reachable by the judge (Qodo #3, PR #2890) -- without it the digest
    short-circuit declared such results ``unchanged`` forever.
    """
    joined = "\n".join(
        discovery.normalize_url(result.url)
        + result.title
        + _snippet_component(result.snippet)
        for result in results
    )
    return hashlib.sha256(joined.encode()).hexdigest()


def _baseline_digest(dreams_db: Any, tracked_item_id: int) -> str | None:
    """Digest of the most recent run that may anchor a comparison.

    ``error``/``skipped``/``withheld`` runs never anchor (no trustworthy
    digest); everything that follows -- ``baseline``, ``unchanged``,
    ``changed``, ``rebaselined`` -- carries the digest the next check
    compares against. Sync; the caller thread-offloads.
    """
    for run in dreams_db.list_recent_track_runs(
        tracked_item_id, limit=_BASELINE_SCAN_LIMIT
    ):
        if str(run["status"]) in _NON_BASELINE_RUN_STATUSES:
            continue
        digest = run["digest_hash"]
        return str(digest) if digest else None
    return None


def _record_run(
    dreams_db: Any,
    tracked_item_id: int,
    *,
    status: str,
    digest: str | None,
    note: str,
    notified: int,
    now_iso: str,
) -> int:
    """Insert one run row and stamp the item's due clock (one thread hop).

    Plain synchronous SQLite -- ``run_track_check`` wraps this in a single
    ``asyncio.to_thread`` call per stage so a check never blocks the event
    loop and never pays two hops for two dependent writes.
    """
    run_id = dreams_db.insert_track_run(
        tracked_item_id, status=status, digest_hash=digest,
        verdict_note=note, notified=notified,
    )
    dreams_db.touch_tracked_checked(tracked_item_id, now_iso)
    return run_id


def _first_json_object(text: str) -> dict | None:
    """The first balanced JSON object in ``text``, or None.

    Tolerant parse for model prose: verdicts may arrive wrapped in
    "Sure! Here is my verdict: {...}" -- scanning for the first brace
    pair that parses as an object extracts them without a regex grammar.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:index + 1])
                    except ValueError:
                        break  # unparseable pair; try the next brace
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def _parse_verdict(text: str) -> tuple[bool, str] | None:
    """``{"changed": bool, "note": str}`` out of a judge reply, or None.

    ``None`` means "no usable verdict" (missing/non-boolean ``changed``,
    unparseable text) -- the caller records an ``error`` run, never a
    guess.
    """
    obj = _first_json_object(str(text or ""))
    if obj is None:
        return None
    changed = obj.get("changed")
    if not isinstance(changed, bool):
        return None
    return changed, str(obj.get("note") or "").strip()


def _judge_call_kwargs(query: str, results: list[Candidate]) -> dict[str, Any]:
    """Payload for the one judge call (PRIVACY, spec-binding).

    Carries ONLY the query, the CURRENT top snippets (title/url/snippet
    per result, capped at the check's result count), and the JSON-verdict
    question. No stored prior page text exists to send -- prior state is
    a digest hash -- and the digest itself never leaves the machine.
    """
    user = json.dumps(
        {
            "query": query,
            "current_results": [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                }
                for result in results[:_TRACK_CHECK_RESULT_COUNT]
            ],
            "question": _JUDGE_QUESTION,
        },
        ensure_ascii=False,
    )
    return {
        "messages_payload": [{"role": "user", "content": user}],
        "system_message": _JUDGE_SYSTEM_PROMPT,
        "streaming": False,
        "max_tokens": _JUDGE_MAX_TOKENS,
        "temp": _JUDGE_TEMPERATURE,
    }


def _dispatch_changed(
    deps: "CycleDeps",
    *,
    query_template: str,
    note: str,
    tracked_item_id: int,
) -> bool:
    """Dispatch the one ``dreams_track`` notification; degrade, never raise.

    Qodo #4 (PR #2890): the caller learns whether the alert actually
    reached a dispatcher, so the run row can record ``notified=0`` and a
    degrade marker instead of claiming delivery that never happened.

    Args:
        deps: Injected collaborators (only ``dispatch_getter`` is read).
        query_template: The watch's template (notification title prefix).
        note: The judge's verdict note (notification body).
        tracked_item_id: Item the notification is about.

    Returns:
        True when a dispatcher resolved and ``dispatch`` returned
        without raising; False for every degraded outcome (no getter, a
        raising getter, no dispatcher wired, or a dispatch failure --
        each logged, never raised).
    """
    getter = getattr(deps, "dispatch_getter", None)
    dispatcher = None
    if getter is not None:
        try:
            dispatcher = getter()
        except Exception as exc:  # noqa: BLE001 - notification degrades
            logger.warning(
                "Dreams track dispatch getter failed: {}", type(exc).__name__)
    if dispatcher is None:
        return False
    try:
        dispatcher.dispatch(
            category="dreams_track",
            title=f"Tracked update: {query_template[:50]}",
            message=note,
            severity="information",
            source_entity_kind="dream_tracked_item",
            source_entity_id=str(tracked_item_id),
        )
    except Exception as exc:  # noqa: BLE001 - notification degrades
        logger.warning(
            "Dreams track notification dispatch failed: {}",
            type(exc).__name__)
        return False
    return True


async def run_track_check(deps: "CycleDeps", tracked_item_id: int) -> dict:
    """Run one scheduled check of a tracked item; every outcome is a row.

    Per-item in-flight claim (Qodo #16, PR #2890): the projection keeps
    an item due until ``touch_tracked_checked`` lands at the END of a
    check, and the queue reloads (~every 30 minutes) while a slow search
    or judge call is still running -- a re-dispatch of the same item
    could otherwise run a second concurrent check and notify twice. The
    claim mirrors ``cycle_service._ACTIVE_CYCLE_DATES``: a module-level
    event-loop-only set, checked and added with NO await between, and
    discarded in a ``finally`` so cancellations cannot leak it.

    Args:
        deps: Injected collaborators (see ``CycleDeps``; only
            ``dreams_db``/``perform_search``/``chat_getter``/``now``/
            ``dispatch_getter`` are read).
        tracked_item_id: The ``dream_tracked_items`` row to check.

    Returns:
        ``{"status": <disposition>, "notified": bool}`` -- one of
        ``baseline``/``unchanged``/``changed``/``withheld``/``error``/
        ``skipped``, or ``inflight`` when this item already has a check
        executing (nothing spent, nothing recorded).
    """
    # No await between the check and the add: a second dispatch of the
    # same item running on the same event loop cannot interleave here.
    if tracked_item_id in _INFLIGHT_TRACK_CHECKS:
        return {"status": "inflight", "notified": False}
    _INFLIGHT_TRACK_CHECKS.add(tracked_item_id)
    try:
        return await _run_track_check_claimed(deps, tracked_item_id)
    finally:
        _INFLIGHT_TRACK_CHECKS.discard(tracked_item_id)


async def _run_track_check_claimed(
    deps: "CycleDeps", tracked_item_id: int
) -> dict:
    """The check pipeline proper; ``run_track_check`` holds the claim.

    Pipeline (spec §track loop): sweep the lifecycle first (Task 6,
    below), then load the item (a missing one -- retired between emission
    and dispatch -- is a clean ``skipped`` with no row; a present but
    non-ACTIVE one -- paused, or retired by the sweep just above --
    records one ``skipped`` "not active" run and spends nothing, Qodo #5,
    PR #2890; a PAGE-mechanism one is skipped the same way, Qodo #7 --
    its subscription's own loop watches the page, and a Dreams search on
    an empty template would only burn the shared budget), budget-guard
    BEFORE spending anything, re-run the template as one search, digest
    the result set, and compare against the most recent anchoring digest:

    * no baseline yet → ``baseline`` run, ``notified=0``;
    * identical digest → ``unchanged`` run, no LLM call at all;
    * different digest → exactly one judge ``chat`` call whose prompt
      carries only the query, the current top snippets, and the
      JSON-verdict question (prior digests are hashes; prior snippets are
      not stored). Verdict ``{"changed": true, ...}`` → ``changed`` run
      plus one ``dreams_track`` notification, with ``notified=1`` ONLY
      when the dispatch actually delivered (Qodo #4: a missing or failed
      dispatcher records ``notified=0`` and a "(delivery degraded)"
      verdict-note suffix -- the run no longer claims an alert that never
      arrived); a judge failure or unparseable verdict → ``error`` run.
      Notification delivery degrades (logged, noted on the run), never
      raises.

    Budgets: the check is skipped before any spend when either daily
    counter is exhausted; the search bumps ``searches=1`` after it runs,
    and any ATTEMPTED judge call bumps ``llm_calls=1`` (the attempted-call
    accounting ``discovery.run_queries`` already applies to searches).

    Args:
        deps: Injected collaborators (see ``CycleDeps``).
        tracked_item_id: The ``dream_tracked_items`` row to check.

    Returns:
        ``{"status": <disposition>, "notified": bool}``.
    """
    # Lifecycle sweep FIRST (Task 6): the item this task dispatches may
    # already be past its event or quiet -- and other active items may be
    # too. Degrade-never-abort: a sweep failure logs and the check itself
    # proceeds unchanged.
    try:
        await sweep_track_lifecycle(
            deps.dreams_db, now=deps.now(),
            subs_service_getter=getattr(deps, "subs_service_getter", None),
            scheduling_db_getter=getattr(
                deps, "scheduling_db_getter", None),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the sweep must not kill a check
        logger.warning("Dreams track sweep before check failed: {}",
                       type(exc).__name__)

    dreams_db = deps.dreams_db
    now = deps.now()
    local_date = _local_date(now)

    item = await asyncio.to_thread(dreams_db.get_tracked_item, tracked_item_id)
    if item is None:
        return {"status": "skipped", "notified": False}
    # Qodo #5 (PR #2890): existence is not liveness. A task dispatched
    # before an untrack/pause/sweep finds a row whose watch is already
    # stopped; it must record one cheap skip, not search and spend.
    if str(item.get("status") or "") != "active":
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="skipped", digest=None, note="not active", notified=0,
            now_iso=to_utc_iso(now))
        return {"status": "skipped", "notified": False}
    # Qodo #7 (PR #2890), defense in depth under the projection's
    # question-only filter: a page item has no template to search -- its
    # subscription's own loop watches the page -- and a blank template
    # from any other writer has nothing to search either.
    if str(item.get("mechanism") or "") != _QUESTION_MECHANISM:
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="skipped", digest=None,
            note="page items are watched by subscription", notified=0,
            now_iso=to_utc_iso(now))
        return {"status": "skipped", "notified": False}
    query_template = str(item.get("query_template") or "")
    if not query_template.strip():
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="skipped", digest=None, note="empty query template",
            notified=0, now_iso=to_utc_iso(now))
        return {"status": "skipped", "notified": False}

    # Budget guard FIRST: an exhausted day skips before the search, the
    # judge, and every bump -- spending nothing is the point.
    usage = await asyncio.to_thread(dreams_db.usage_get, local_date)
    if (int(usage["searches"]) >= int(dreams_setting("max_searches_per_day"))
            or int(usage["llm_calls"]) >=
            int(dreams_setting("max_llm_calls_per_day"))):
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="skipped", digest=None, note="budget", notified=0,
            now_iso=to_utc_iso(now))
        return {"status": "skipped", "notified": False}

    # Question synthesis uses NO LLM: the template IS the query.
    # ``date_range=None`` deliberately overrides ``run_queries``'s cycle
    # default of ``"m"`` (ruling P8): a track check compares a watch's
    # CURRENT result set against its anchor, and a long-lived watch whose
    # engine results fall outside a one-month recency window would return
    # empty forever -- perpetual ``withheld``, never anchored, never
    # notified. Change detection must see the unfiltered result set.
    query = _render_query(query_template)
    results, _searches_used = await discovery.run_queries(
        deps.perform_search, engine=str(dreams_setting("search_engine")),
        queries=[query], result_count=_TRACK_CHECK_RESULT_COUNT,
        date_range=None)
    await asyncio.to_thread(
        dreams_db.usage_bump, local_date, searches=1)

    if not results:
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="withheld", digest=None, note="no results", notified=0,
            now_iso=to_utc_iso(now))
        return {"status": "withheld", "notified": False}

    digest = _results_digest(results)
    baseline = await asyncio.to_thread(
        _baseline_digest, dreams_db, tracked_item_id)
    if baseline is None:
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="baseline", digest=digest, note="", notified=0,
            now_iso=to_utc_iso(now))
        return {"status": "baseline", "notified": False}
    if digest == baseline:
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="unchanged", digest=digest, note="", notified=0,
            now_iso=to_utc_iso(now))
        return {"status": "unchanged", "notified": False}

    # Different digest: the one judge call. Resolving the chat seam can
    # fail without any spend (nothing was invoked); once the call is
    # ATTEMPTED the budget is spent either way.
    verdict: tuple[bool, str] | None = None
    chat: Any = None
    llm_spent = False
    try:
        chat = await asyncio.to_thread(deps.chat_getter)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider resolution degrades
        logger.warning("Dreams track check: chat resolution failed: {}",
                       type(exc).__name__)
    if chat is not None:
        try:
            response = await asyncio.to_thread(
                chat, **_judge_call_kwargs(query, results))
            verdict = _parse_verdict(extract_response_content(response))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - judge call degrades
            logger.warning("Dreams track judge call failed: {}",
                           type(exc).__name__)
        finally:
            llm_spent = True
    if llm_spent:
        await asyncio.to_thread(
            dreams_db.usage_bump, local_date, llm_calls=1)
    if verdict is None:
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="error", digest=digest, note="judge unavailable",
            notified=0, now_iso=to_utc_iso(now))
        return {"status": "error", "notified": False}

    changed, note = verdict
    if not changed:
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="unchanged", digest=digest, note=note, notified=0,
            now_iso=to_utc_iso(now))
        return {"status": "unchanged", "notified": False}

    delivered = _dispatch_changed(
        deps, query_template=query_template, note=note,
        tracked_item_id=tracked_item_id)
    # Qodo #4 (PR #2890): ``notified`` records DELIVERY, not intent -- a
    # degraded dispatch writes 0 and marks the note so the user can see
    # the alert never arrived.
    stored_note = note if delivered else (
        f"{note}; notification delivery degraded")
    await asyncio.to_thread(
        _record_run, dreams_db, tracked_item_id,
        status="changed", digest=digest, note=stored_note,
        notified=1 if delivered else 0,
        now_iso=to_utc_iso(now))
    return {"status": "changed", "notified": delivered}


def rebaseline_track(dreams_db: Any, tracked_item_id: int) -> int | None:
    """Re-anchor a tracked item's comparison at its latest known digest.

    Inserts a ``rebaselined`` run carrying the latest non-``error``/
    ``skipped``/``withheld`` digest, so the next identical-results check is
    ``unchanged`` -- re-alerting on a change the user has already seen
    stops here. With no anchoring run yet there is nothing to re-anchor:
    a no-op returning ``None``.

    Args:
        dreams_db: The app's ``DreamsDB``.
        tracked_item_id: The tracked item to re-anchor.

    Returns:
        The new run id, or ``None`` when no baseline exists.
    """
    digest = _baseline_digest(dreams_db, tracked_item_id)
    if digest is None:
        return None
    return dreams_db.insert_track_run(
        tracked_item_id, status="rebaselined", digest_hash=digest,
        verdict_note="", notified=0)
