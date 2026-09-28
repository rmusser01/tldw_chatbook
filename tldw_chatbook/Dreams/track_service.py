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

Re-track after untrack (ruling P7, fix round 1): the URL lookup ignores
``is_active``, so a found source may be one a prior untrack disabled. When
it is (a retired, dream-created tracked row pins the provenance), the
re-track RE-ENABLES it and reports ``"re-enabled"``; a disabled source with
no dream provenance is refused with ``TrackSourceDisabled`` -- never
adopted, never re-enabled behind the user's back.

Scheduling needs no registration here: ``WatchlistProjection`` fabricates
``watchlist:<subscription_id>`` tasks straight from subscription rows, so
creating the subscription IS the registration; ``DreamsProjection``
fabricates ``dream_track:<item_id>`` tasks straight from tracked-item rows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_chatbook.Chat.Chat_Functions import extract_response_content
from tldw_chatbook.Utils.timestamps import to_utc_iso

from . import discovery
from .discovery import Candidate
from .settings import dreams_setting

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .cycle_service import CycleDeps

#: Fallbacks until Task 6 adds the ``[dreams]`` keys; ``dreams_setting``
#: keeps reading them live so the future keys win without a change here.
_DEFAULT_TRACKED_ITEM_CAP = 20
_DEFAULT_MIN_CHECK_INTERVAL_HOURS = 12

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


def _cap() -> int:
    """The ``tracked_item_cap`` setting, coerced to a usable int."""
    try:
        return max(0, int(dreams_setting("tracked_item_cap",
                                         _DEFAULT_TRACKED_ITEM_CAP)))
    except (TypeError, ValueError):
        return _DEFAULT_TRACKED_ITEM_CAP


def _min_interval_seconds() -> int:
    """``track_min_check_interval_hours`` converted to seconds."""
    try:
        hours = max(0, int(dreams_setting("track_min_check_interval_hours",
                                          _DEFAULT_MIN_CHECK_INTERVAL_HOURS)))
    except (TypeError, ValueError):
        hours = _DEFAULT_MIN_CHECK_INTERVAL_HOURS
    return hours * 3600


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

    Returns:
        ``{"tracked_item_id": int, "subscription_id": int,
        "outcome": "created" | "attached" | "re-enabled",
        "watchlist_id": int}``. ``re-enabled`` means the URL resolved to a
        subscription a prior untrack had disabled (retired dream-created
        row with this subscription), and it was re-activated.

    Raises:
        TrackCapReached: When ``count_active_tracked()`` already meets the
            cap -- raised BEFORE any subscription, watchlist, alert, or
            tracked-item write.
        TrackSourceDisabled: When the URL resolves to a DISABLED source
            with no dream provenance (the user disabled it themselves) --
            also raised before any write.
    """
    # Cap guard FIRST: the whole point is that a capped runtime creates
    # nothing, so this precedes every write below.
    cap = _cap()
    active = await asyncio.to_thread(dreams_db.count_active_tracked)
    if active >= cap:
        raise TrackCapReached(
            f"track cap reached: {active} active tracked items (cap {cap})"
        )

    cadence = max(int(cadence_seconds or 0), _min_interval_seconds())

    # Attach-or-create. An existing subscription for exactly this URL is
    # adopted untouched (never renamed, never re-cadenced, never disabled
    # by a later untrack): the user may already depend on it.
    existing = await subs_service.find_source_id_by_url(str(url))
    if existing is not None:
        subscription_id = int(existing)
        outcome = "attached"
        created_by_dreams = 0
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
            retired_rows = await asyncio.to_thread(
                dreams_db.list_tracked_items, "retired"
            )
            dream_owned = any(
                int(row.get("subscription_id") or 0) == subscription_id
                and int(row.get("created_by_dreams") or 0) == 1
                for row in retired_rows
            )
            if not dream_owned:
                raise TrackSourceDisabled(
                    "source exists but is disabled (no dream provenance): "
                    f"subscription {subscription_id}"
                )
            # WE disabled it on a prior untrack (retired + dream-created +
            # this subscription): re-enable and say so.
            await subs_service.update_source(
                subscription_id, {"active": True}
            )
            outcome = "re-enabled"
            created_by_dreams = 1
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


async def untrack(subs_service: Any, dreams_db: Any, tracked_item_id: int) -> dict:
    """Retire one tracked item, disabling a dream-created subscription.

    The wrapper always moves to ``retired`` (``retired_reason="manual"``).
    A subscription Dreams created is DISABLED via the existing service seam
    -- ``update_source(subscription_id, {"active": False})`` routing into
    ``SubscriptionsDB.update_subscription``'s allowlisted ``is_active``
    field -- never deleted, so its watchlist membership and alert rules
    survive for a user who adopted them. A subscription Dreams merely
    attached to (``created_by_dreams = 0``) is never touched.

    Args:
        subs_service: The app's ``LocalWatchlistsService``.
        dreams_db: The app's ``DreamsDB``.
        tracked_item_id: The ``dream_tracked_items`` row to retire.

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

    Returns:
        The new tracked item id.

    Raises:
        ValueError: If ``query_template`` is empty/blank -- an empty watch
            is a caller bug, not a degradation.
        TrackCapReached: When ``count_active_tracked()`` already meets the
            cap -- raised BEFORE the tracked-item row is written.
    """
    template = str(query_template)
    if not template.strip():
        raise ValueError("track_question requires a non-empty query_template")

    # Cap guard FIRST, identical discipline to ``track_page``.
    cap = _cap()
    active = await asyncio.to_thread(dreams_db.count_active_tracked)
    if active >= cap:
        raise TrackCapReached(
            f"track cap reached: {active} active tracked items (cap {cap})"
        )

    cadence = max(int(cadence_seconds or 0), _min_interval_seconds())
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
    logger.debug("Dreams tracked question: item={} template={}",
                 tracked_item_id, template[:50])
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


def _results_digest(results: list[Candidate]) -> str:
    """Content digest of one check's result set (URL identity + titles)."""
    joined = "\n".join(
        discovery.normalize_url(result.url) + result.title for result in results
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
) -> str:
    """Dispatch the one ``dreams_track`` notification; degrade, never raise.

    Returns the verdict note, with a degrade marker appended when the
    notification could not be delivered (no dispatcher wired, getter
    failure, or dispatch failure) -- the run row is written either way.
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
        return f"{note}; notification unavailable"
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
        return f"{note}; notification delivery degraded"
    return note


async def run_track_check(deps: "CycleDeps", tracked_item_id: int) -> dict:
    """Run one scheduled check of a tracked item; every outcome is a row.

    Pipeline (spec §track loop): load the item (a missing one -- retired
    between emission and dispatch -- is a clean ``skipped`` with no row),
    budget-guard BEFORE spending anything, re-run the template as one
    search, digest the result set, and compare against the most recent
    anchoring digest:

    * no baseline yet → ``baseline`` run, ``notified=0``;
    * identical digest → ``unchanged`` run, no LLM call at all;
    * different digest → exactly one judge ``chat`` call whose prompt
      carries only the query, the current top snippets, and the
      JSON-verdict question (prior digests are hashes; prior snippets are
      not stored). Verdict ``{"changed": true, ...}`` → ``changed`` run
      (``notified=1``) plus one ``dreams_track`` notification; a judge
      failure or unparseable verdict → ``error`` run. Notification
      delivery degrades (logged, noted on the run), never raises.

    Budgets: the check is skipped before any spend when either daily
    counter is exhausted; the search bumps ``searches=1`` after it runs,
    and any ATTEMPTED judge call bumps ``llm_calls=1`` (the attempted-call
    accounting ``discovery.run_queries`` already applies to searches).

    Args:
        deps: Injected collaborators (see ``CycleDeps``; only
            ``dreams_db``/``perform_search``/``chat_getter``/``now``/
            ``dispatch_getter`` are read).
        tracked_item_id: The ``dream_tracked_items`` row to check.

    Returns:
        ``{"status": <disposition>, "notified": bool}`` -- one of
        ``baseline``/``unchanged``/``changed``/``withheld``/``error``/
        ``skipped``.
    """
    dreams_db = deps.dreams_db
    now = deps.now()
    local_date = _local_date(now)

    item = await asyncio.to_thread(dreams_db.get_tracked_item, tracked_item_id)
    if item is None:
        return {"status": "skipped", "notified": False}
    query_template = str(item.get("query_template") or "")

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

    stored_note = _dispatch_changed(
        deps, query_template=query_template, note=note,
        tracked_item_id=tracked_item_id)
    await asyncio.to_thread(
        _record_run, dreams_db, tracked_item_id,
        status="changed", digest=digest, note=stored_note, notified=1,
        now_iso=to_utc_iso(now))
    return {"status": "changed", "notified": True}


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
