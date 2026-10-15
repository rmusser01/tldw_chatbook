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
the same watch and an untrack can never strand the older one. The
wrapper's reminder follows the refreshed plan (Qodo #13, PR #2921):
created when missing, moved with the event date, disabled when the date
is cleared -- and otherwise left exactly as it stands, so a re-track
never re-arms, replaces or re-creates a reminder the user changed or one
that has fired (review of #2921, items 4 and 5).

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

The master switch (Qodo #5, PR #2921): ``[dreams] enabled`` gates only
what the Dreams projection emits. A page watch's subscription and an
event watch's reminder live in the watchlist and reminder queues, which
never read that flag, so ``set_tracking_suspended`` switches them -- off
stops them for every live wrapper and records, on the wrapper, exactly
what it stopped; on restores what was recorded and nothing the user had
stopped themselves (review of #2921, item 4). The record
(:func:`_suspension_record`) merges on every switch-off, its stops are
re-applied until they have all landed, and it carries a ``settled``
marker the startup check reads, so a switch that failed or was cut short
part-way is finished by the next pass (round 3, T2/T3). ``reconcile_tracking`` is
how it is called: it picks the direction from the live flag and runs the
pass as a task this module owns, so the Settings toggle and the startup
reconcile (``Dreams/track_reconcile.py``) can both ask for it, a
cancelled caller cannot cut it short, and overlapping requests coalesce
(item 3c).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from functools import partial
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker
from tldw_chatbook.Chat.Chat_Functions import extract_response_content
from tldw_chatbook.Utils.egress import (
    URL_PROVENANCE_OPTION,
    UrlProvenance,
    evaluate_url_policy_async,
)
from tldw_chatbook.Utils.input_validation import validate_url
from tldw_chatbook.Utils.timestamps import to_utc_iso

from . import discovery
from .discovery import Candidate
from .settings import (
    DREAMS_DEFAULTS,
    dreams_execution_allowed,
    dreams_region,
    dreams_setting,
)
from .track_reconcile import (
    SETTLED_OFF,
    SETTLED_ON,
    changed_after,
    is_dreams_source,
    settled_state,
)

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

#: ``link_type`` every Dreams event reminder is created with: the one
#: spelling shared by promote, disable, and the suspend/resume switch.
_REMINDER_LINK_TYPE = "dream_tracked_item"

#: Days before a tracked ``event_date`` that the promoted reminder fires
#: (Phase 2 Task 5): one nudge with a week of runway, not a day-of alarm.
_REMINDER_LEAD_DAYS = 7

#: Local wall-clock time the reminder fires on its lead date (Qodo #14,
#: PR #2921). An ``event_date`` is a calendar day, so the nudge needs an
#: hour on the USER's calendar; nine in the morning is the hour the
#: reminder forms' own examples use (``schedule_input_parsing``).
_REMINDER_LOCAL_TIME = time(9, 0)

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

    Watchlists has no control that switches an existing source back on
    (it can create one inactive, resume an auto-PAUSED one, and delete
    one), so the notice for this must not send the user looking for one
    (round 4, A2): what they can do is delete that source and track again.

    Attributes:
        reason_code: Stable machine identifier
            (``track_source_disabled``).
    """

    reason_code = "track_source_disabled"


class TrackDreamsOff(RuntimeError):
    """Dreams is switched off, so nothing is tracked (TASK-34928).

    The Library hides Track and Watch while Dreams is off, but a call that
    reached the service anyway re-enabled a page's source and armed an
    event reminder -- work the master switch had stopped. Raised before
    any read of another store and before any write, and asked again just
    before a re-track would switch a source back on, after the page's
    name lookup (review of #3108, X2): nothing is switched on then.

    Attributes:
        reason_code: Stable machine identifier (``track_dreams_off``).
    """

    reason_code = "track_dreams_off"


def _require_dreams_on() -> None:
    """Refuse to track while ``[dreams] enabled`` is off."""
    if not bool(dreams_setting("enabled")):
        raise TrackDreamsOff("Dreams is switched off; nothing was tracked")


#: The lock a switch-off pass and a re-track's switch-on share, one per event
#: loop (an ``asyncio.Lock`` belongs to the loop it first waits on), kept for
#: as long as that loop lives, so callers on two loops never swap each
#: other's lock out from under a holder.
_SWITCH_LOCKS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)


def _switch_lock() -> asyncio.Lock:
    """Serialize a re-track's switch check and switch-on with switch-off.

    Review of #3108, round 2, Z1. A re-track asks the switch and then
    switches a source on; a switch-off pass landing in between stopped
    what it found and was then undone by the switch-on, leaving the source
    running with Dreams off. With both under this lock the pass either ran
    before the check -- which then refuses -- or runs after the switch-on
    and stops that source with the rest. Held only across the check, the
    un-settling and the switch-on (no name lookup, no wait on a pass), and
    taken by the switch-off direction only: a switch-on pass enabling
    sources is not in conflict with a re-track doing the same.
    """
    loop = asyncio.get_running_loop()
    lock = _SWITCH_LOCKS.get(loop)
    if lock is None:
        lock = _SWITCH_LOCKS[loop] = asyncio.Lock()
    return lock


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


class TrackPrivateURL(TrackInvalidURL):
    """The URL to watch is not a public destination (Qodo #1, PR #2921).

    A story URL comes out of a web search or a feed item, never out of the
    user's keyboard, and the watchlist loop trusts a subscription's own
    configured source when it fetches. Storing a loopback, private-network
    or metadata address as a Dreams subscription would therefore let an
    untrusted result point the watcher at the user's network. Raised by
    :func:`track_page` BEFORE the subscription is created or re-enabled,
    with a message that never echoes the URL. A subclass of
    :class:`TrackInvalidURL`, so a caller that only handles the invalid
    refusal still refuses.

    Attributes:
        reason_code: Stable machine identifier (``track_private_url``).
    """

    reason_code = "track_private_url"


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


#: What a subscription Dreams creates records about its URL: nobody typed
#: it (review of #2921, item 2). The watchlist fetch reads this back and
#: grants the URL no trusted egress origin -- see
#: ``Subscriptions.monitoring_engine._source_provenance``.
_STORY_URL_OPTIONS = {URL_PROVENANCE_OPTION: UrlProvenance.UNKNOWN.value}


async def _public_url_refusal(url: str) -> str | None:
    """Why the egress policy would not fetch ``url`` on its own, or None.

    The question is asked with NO trusted origins: a story URL never
    vouches for itself (``UrlProvenance.UNKNOWN`` in ``Utils/egress.py``'s
    terms). The policy resolves the host and refuses loopback, private,
    link-local, reserved, multicast and metadata addresses (IPv4-mapped
    and NAT64 wrappers included), and an unresolvable host fails closed.
    ``[web_security] allowed_hosts`` and ``enabled`` apply exactly as they
    do to every other guarded fetch.

    Returns:
        ``None`` when the destination is public, else the policy's reason
        slug (``private``/``metadata``/``dns_failure``/``scheme``).
    """
    decision = await evaluate_url_policy_async(url)
    return None if decision.allowed else str(decision.reason)


async def _require_public_url(url: str) -> None:
    """Refuse a story URL the egress policy would not fetch on its own.

    Qodo #1 (PR #2921, security). ``validate_url`` checks syntax only, and
    a story's URL was supplied by a search result or a feed item, not
    typed by the user. Before Dreams stores (or re-enables) one as a
    subscription it must be a public destination. This is the check at
    the door; the same rule is applied again on every fetch, because the
    subscription is created marked as story-derived
    (``_STORY_URL_OPTIONS``) and the watchlist loop gives such a source
    no trusted origin -- a host that resolves somewhere private LATER is
    refused then (review of #2921, item 2). Neither check pins the
    address it resolved (``Utils/egress.py``, a documented non-goal): a
    name that changes its answer between the check and the connection is
    outside what this covers.

    Args:
        url: The syntactically valid http(s) story URL.

    Raises:
        TrackPrivateURL: When the policy refuses the destination. The
            message carries the policy's reason slug, never the URL.
    """
    reason = await _public_url_refusal(url)
    if reason is not None:
        raise TrackPrivateURL(
            f"refusing to track a non-public destination ({reason})"
        )


def _source_options(source: Any) -> dict:
    """A copy of the processing options a ``get_source`` row carries."""
    return dict(
        ((source or {}).get("settings") or {}).get("processing_options") or {})


def _needs_story_mark(source: Any) -> bool:
    """Whether a source has no usable explicit provenance record.

    Preserve an explicit user entry; normalize absent or malformed lineage to
    UNKNOWN without treating an old row as evidence of user intent.
    """
    return _source_options(source).get(URL_PROVENANCE_OPTION) not in (
        UrlProvenance.UNKNOWN.value, UrlProvenance.USER_ENTERED.value
    )


def _is_users_address(source: Any) -> bool:
    """Whether the source records that a user entered its URL."""
    return _source_options(source).get(
        URL_PROVENANCE_OPTION) == UrlProvenance.USER_ENTERED.value


def _story_options(source: Any) -> dict:
    """A source's processing options with the story-derived mark added.

    Options the source already carries are kept -- a provenance record
    it already has included.
    """
    options = _source_options(source)
    if options.get(URL_PROVENANCE_OPTION) != UrlProvenance.USER_ENTERED.value:
        options[URL_PROVENANCE_OPTION] = UrlProvenance.UNKNOWN.value
    return options


async def _enable_dream_subscription(
    subs_service: Any, subscription_id: int, source: Any
) -> None:
    """Turn a subscription Dreams created back on, marked story-derived.

    Every path that re-enables a dream-created source goes through here,
    so the source is marked whenever Dreams turns it on. (The mark does
    not depend on a re-enable: ownership is enough, see
    :func:`backfill_egress_marks`.)

    Args:
        subs_service: The app's ``LocalWatchlistsService``.
        subscription_id: The dream-created subscription.
        source: Its ``get_source`` row (for the options it already has).
    """
    await subs_service.update_source(
        subscription_id,
        {"active": True, "processing_options": _story_options(source)})


async def backfill_egress_marks(
    subs_service: Any, dreams_db: Any, *, verify: bool = False,
) -> int:
    """Stamp every subscription Dreams created that carries no mark yet.

    Review of #2921, round 3, T1. When this was written, a source with no
    mark was fetched as one the user typed: its URL was a trusted egress
    origin. Since the URL-provenance rework (ADR-207) that is no longer
    so: only an explicit user-entered record earns trust, and a source
    with no record, or a malformed one, is fetched under the public-only
    rule like a marked one (``Utils.egress.provenance_of_stored_source``).
    The mark is still written -- every Dreams source says in so many words
    that nobody typed its URL -- and it is owed by OWNERSHIP: every
    subscription a ``created_by_dreams`` wrapper names, whatever the
    wrapper's status and whatever the master switch says.

    Runs from the startup reconcile, once: ``DreamsDB`` keeps the list of
    what is still to do (``subscriptions_awaiting_egress_mark``), each
    subscription leaves it when stamped -- or found deleted -- and an
    ordinary start then finds the list empty and never comes here. A
    source whose stamp fails, or a start without the watchlists service,
    stays on the list for the next start. Marking never changes whether
    a source is active.

    That list is a HINT (round 4, B4). It lives in the Dreams store and
    the mark in the subscriptions store, and a backup restores them as
    parts of two different groups ("conversations" and "research",
    ``Backup_Recovery.data_groups``): restore the subscriptions from a
    generation taken before the mark and Dreams' sources come back
    unmarked while the list says "done". ``verify=True`` therefore
    ignores the list and reads the real options of every source Dreams
    owns. It is what runs whenever Dreams is reading the subscriptions
    anyway -- a start whose reconcile has work, and every cycle run --
    and the suspend pass, an attach and a re-enable stamp from the real
    options too. A start with no other work does not verify: an
    unmarked source is not trusted any more (above), so there is nothing
    for the cost to buy -- the subscriptions store opened and one read
    per source Dreams ever created, on every boot. Verifying writes only
    where something changed (TASK-34928).

    Args:
        subs_service: The app's ``LocalWatchlistsService``, or None.
        dreams_db: A ``DreamsDB``.
        verify: Walk every subscription Dreams owns and decide from its
            actual options, instead of trusting the "still to do" list.

    Returns:
        How many subscriptions were stamped.
    """
    if subs_service is None:
        return 0
    marked = 0
    # The still-to-do list, read once: verifying writes the marker only for
    # a source it stamped or one the list names, so a run that finds every
    # source marked writes nothing (TASK-34928).
    awaiting = set(await asyncio.to_thread(
        dreams_db.subscriptions_awaiting_egress_mark))
    # ponytail: one get_source per Dreams-owned source when verifying;
    # batch the read in SubscriptionsDB if a profile ever owns thousands.
    for subscription_id in (await asyncio.to_thread(
            dreams_db.dream_owned_subscription_ids) if verify
            else sorted(awaiting)):
        try:
            try:
                source = await subs_service.get_source(subscription_id)
            except KeyError:
                source = None  # deleted by the user: nothing to stamp
            if source is not None and not await _still_dreams_source(
                    dreams_db, subscription_id, source):
                source = None  # no longer Dreams' source (review of #3123, M1)
            stamped = source is not None and _needs_story_mark(source)
            if stamped:
                await subs_service.update_source(
                    subscription_id,
                    {"processing_options": _story_options(source)})
                marked += 1
            if stamped or subscription_id in awaiting:
                await asyncio.to_thread(
                    dreams_db.set_egress_marked, subscription_id)
        except Exception as exc:  # noqa: BLE001 - stays pending for a retry
            logger.warning(
                "Dreams egress mark: stamping subscription {} failed: {}",
                subscription_id, type(exc).__name__)
    return marked


def _is_change_condition(rule: dict) -> bool:
    """Whether a rule has the condition Dreams' change rule is made with."""
    if str(rule.get("condition_type") or "") != "items_above":
        return False
    try:
        return int((rule.get("condition_value") or {}).get("threshold")) == 0
    except (AttributeError, TypeError, ValueError):
        return False


async def _dreams_rules_on(
    subs_service: Any, dreams_db: Any, subscription_id: int,
) -> dict[int, dict]:
    """The alert rule(s) Dreams itself created on a subscription, by id.

    Round 3, T11. Dreams used to recognise "its" rule by name prefix
    (``Change: ``) plus condition, so the user's own ``Change: my price
    alert`` on their own source was adopted at track time and switched
    off together with Dreams -- and because ``list_alert_rules(job_id=)``
    also returns GLOBAL rules, a matching global rule would have been
    silenced for every source. The rule is now known by the id recorded
    on the wrappers (``dream_tracked_items.alert_rule_id``) and by
    nothing else.

    Wrappers from before the id was recorded (NULL) are looked at ONCE:
    a rule scoped to this very subscription, with Dreams' condition, and
    named exactly ``Change: <title>`` for a story one of those wrappers
    tracks, is taken to be the one ``track_page`` created for it and its
    id is recorded; anything else is not Dreams', and ``0`` ("none") is
    recorded so the question is not asked again.

    Args:
        subs_service: The app's ``LocalWatchlistsService``.
        dreams_db: The app's ``DreamsDB``.
        subscription_id: The subscription to look at.

    Returns:
        ``{rule_id: rule}`` for every recorded rule that still exists on
        the subscription (enabled or not), as ``list_alert_rules`` rows.
    """
    wrappers = await asyncio.to_thread(run_finite_local_worker,
        dreams_db.list_tracked_for_subscription, subscription_id)
    scoped = {
        int(rule["rule_id"]): rule
        for rule in await subs_service.list_alert_rules(job_id=subscription_id)
        if rule.get("rule_id") is not None
        # A global rule (no job) is every source's, never Dreams'.
        and str(rule.get("job_id")) == str(subscription_id)
    }
    recorded = {int(w["alert_rule_id"]) for w in wrappers
                if w.get("alert_rule_id")}
    unrecorded = [w for w in wrappers if w.get("alert_rule_id") is None]
    if unrecorded:
        adopted = 0
        if not recorded:
            def expected_names() -> set[str]:
                names = set()
                for wrapper in unrecorded:
                    story = (dreams_db.get_story(int(wrapper["origin_story_id"]))
                             if wrapper.get("origin_story_id") else None)
                    if story is not None:
                        names.add(_change_rule_name(story.get("title")))
                        names.add(_change_rule_name(
                            story.get("title"), as_written_before=True))
                return names

            names = await asyncio.to_thread(run_finite_local_worker, expected_names)
            adopted = next(
                (rule_id for rule_id, rule in sorted(scoped.items())
                 if str(rule.get("name")) in names
                 and _is_change_condition(rule)), 0)
        else:
            # A later wrapper of a page whose rule IS recorded shares it.
            adopted = min(recorded)
        await asyncio.to_thread(run_finite_local_worker,
            dreams_db.set_tracked_alert_rule, subscription_id, adopted,
            only_unrecorded=True)
        if adopted:
            recorded.add(adopted)
    return {rule_id: scoped[rule_id] for rule_id in sorted(recorded)
            if rule_id in scoped}


_SQUARE_BRACKETS = str.maketrans("[]", "()")


def _label(text: Any) -> str:
    """Text as Dreams writes it into a title in ANOTHER feature's store.

    Review of #2921, round 4, C1. A story title and a watched query are
    ordinary text and may contain square brackets ("[PDF] Rust
    conference", "see [/r/boardgames] meetup"). Dreams copies them into a
    reminder's title, an alert rule's name, a source's name and a
    notification's title; Schedules and Watchlists show those rows on
    screens that read ``[...]`` as Rich markup (pinned rows in
    ``scripts/markup_interpolation_census.tsv``, not Dreams' to edit),
    where an unmatched closing tag is a ``MarkupError`` at render time
    and a tag-shaped word is swallowed. So what Dreams writes there has
    its square brackets written as parentheses. The text itself -- the
    story's title, the wrapper's ``query_template`` -- is stored as it
    is; only these copies change.
    """
    return str(text).translate(_SQUARE_BRACKETS)


def _change_rule_name(title: Any, *, as_written_before: bool = False) -> str:
    """The name ``track_page`` gives the change rule it creates.

    Args:
        title: The story's title.
        as_written_before: The name as builds before round 4 wrote it,
            square brackets and all -- for recognising a rule one of
            them made (:func:`_dreams_rules_on`).
    """
    name = f"Change: {str(title or 'Untitled dream story')[:40]}"
    return name if as_written_before else _label(name)


async def _reusable_wrapper(
    dreams_db: Any, origin_story_id: int, mechanism: str,
) -> dict | None:
    """The story's live wrapper of one mechanism, put back to work.

    The reuse lookup of both track paths. It also finds a PAUSED wrapper
    (round 3, T5): a watch the sweep paused after repeated failures is
    still the story's watch, and tracking the story again used to stack a
    second wrapper -- and a second armed reminder for the same event --
    beside it. A paused wrapper is reactivated, which is what pressing
    "track" on it asks for; that makes one more active watch, so the cap
    is consulted for it exactly as for a new one.

    Raises:
        TrackCapReached: When reactivating a paused wrapper would exceed
            the cap; the wrapper stays paused.
    """
    wrapper = await asyncio.to_thread(
        dreams_db.find_tracked_by_story, origin_story_id,
        mechanism=mechanism, include_paused=True)
    if wrapper is None or wrapper.get("status") == "active":
        return wrapper
    cap = _cap()
    active = await asyncio.to_thread(dreams_db.count_active_tracked)
    if active >= cap:
        raise TrackCapReached(
            f"track cap reached: {active} active tracked items (cap {cap})"
        )
    def reactivate() -> None:
        dreams_db.set_tracked_status(int(wrapper["id"]), "active")
        # A fresh start in the same step (round 4, B1). The failed checks
        # that paused the watch are still its newest runs, and the sweep
        # pauses on that streak: without this row the next sweep paused
        # it again before a single check could run. Any other disposition
        # ends a streak; the due clock is deliberately not touched.
        dreams_db.insert_track_run(
            int(wrapper["id"]), status="skipped", digest_hash=None,
            verdict_note="reactivated", notified=0)

    await asyncio.to_thread(reactivate)
    wrapper["status"] = "active"
    return wrapper


async def _refresh_wrapper_plan(
    dreams_db: Any,
    wrapper: dict,
    *,
    cadence: int,
    event_date: str | None,
    scheduling_db_getter: Any = None,
) -> None:
    """Bring an active wrapper, and its reminder, in line with a new plan.

    The reuse branch of Qodo #2 (PR #2890). The plan row is only written
    when a value actually differs. The reminder is looked at EVERY time
    (Qodo #13, PR #2921): it is a separate row in the scheduling DB, so a
    plan refresh alone left it missing (the first track ran before the
    scheduling DB was wired, or the watch had no date then) or armed for
    the old event (the date moved or was cleared).
    :func:`promote_to_reminder` is told the date the wrapper carried
    before, so it moves the reminder only when the event moved and
    otherwise leaves an existing one exactly as the user left it.

    Args:
        dreams_db: The app's ``DreamsDB``.
        wrapper: The active tracked-item row being reused.
        cadence: The caller's (already clamped) check cadence.
        event_date: The caller's target-event date, or ``None``.
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` or None; None leaves reminders alone.
    """
    if (
        int(wrapper.get("cadence_seconds") or 0) != cadence
        or wrapper.get("event_date") != (event_date or None)
    ):
        await asyncio.to_thread(
            dreams_db.set_tracked_plan, int(wrapper["id"]),
            cadence_seconds=cadence, event_date=event_date,
        )
    if wrapper.get("event_date") != (event_date or None):
        await _forget_superseded_reminders(dreams_db, wrapper)
    await promote_to_reminder(
        scheduling_db_getter,
        {"id": wrapper["id"], "query_template": wrapper.get("query_template"),
         "event_date": event_date,
         "origin_story_id": wrapper.get("origin_story_id")},
        previous_event_date=wrapper.get("event_date"),
        dreams_db=dreams_db,
    )


async def _forget_superseded_reminders(dreams_db: Any, wrapper: dict) -> None:
    """Drop the reminders a suspension record owes once the event moved.

    Round 3, T12(b). A record can still owe a reminder -- a resume ran
    without the scheduling DB. That reminder is for the wrapper's OLD
    event; when the event moves, :func:`promote_to_reminder` makes the
    one for the new date, and a later resume re-arming the recorded one
    left two armed reminders for two dates. The debt is dropped with the
    date it belonged to; everything else the record says stands.
    """
    record = _suspension_record(wrapper)
    if not record:
        return
    subscription, reminders, rules = _owed(record)
    if not reminders:
        return
    settled = settled_state(wrapper.get("suspension"))
    text: str | None = None
    if subscription or rules or settled == SETTLED_OFF:
        text = _record_text(
            subscription=subscription, reminders=[], rules=rules,
            held=record.get("held"), settled=settled, at=record.get("at"))
    await asyncio.to_thread(
        dreams_db.set_tracked_suspension, int(wrapper["id"]), text)


def _reminder_heir(dreams_db: Any, item: dict) -> dict | None:
    """The live watch that inherits a retiring watch's event reminder.

    One story has ONE reminder per event date, linked to whichever of its
    watches was made first (round 3, T5). When that watch ends, the
    reminder belongs to the story's other watch of the same event -- an
    active or paused one -- if there is one (round 4, A3).
    """
    story_id = item.get("origin_story_id")
    event_date = item.get("event_date")
    if dreams_db is None or story_id is None or not event_date:
        return None
    for sibling in dreams_db.list_tracked_for_story(int(story_id)):
        if (int(sibling["id"]) != int(item["id"])
                and sibling.get("status") in ("active", "paused")
                and sibling.get("event_date") == event_date):
            return sibling
    return None


def _retire_reminders_sync(
    scheduling_db: Any, dreams_db: Any, item: dict, hand_over: bool,
    dreams_on: bool = True,
) -> tuple[list[str], list[str]]:
    """Sync body of :func:`_retire_linked_reminders`.

    TASK-34923. Nothing moves before it is written down: the heir's record
    takes the debt of a reminder Dreams disabled BEFORE that reminder is
    re-linked to it, and if that write fails the reminder stays where it
    was, owed on the retiring watch's record, for the next switch-on to
    hand on (:func:`_orphaned_reminders`). While Dreams is off nothing is
    handed on here at all: a switch-off pass may be writing the heir's
    record at this very moment, and the reminder must end up recorded on
    exactly one watch. An armed one is disarmed and an owed one stays
    owed -- both on the retiring watch's record -- and switching Dreams on
    hands them on.
    """
    item_id = str(item["id"])
    heir = _reminder_heir(dreams_db, item) if hand_over and dreams_on else None
    body = _reminder_body(item.get("event_date"))
    rows = scheduling_db.list_reminder_tasks_by_link(
        _REMINDER_LINK_TYPE,
        [item_id] if heir is None else [item_id, str(heir["id"])])
    if heir is not None and any(
            str(row.get("link_id")) == str(heir["id"])
            and row.get("body") == body for row in rows):
        heir = None  # it already has a reminder of its own for this event
    owed = set(_owed(_suspension_record(item))[1])
    to_disarm: list[str] = []
    handed: list[str] = []
    debt: list[str] = []
    titles = {str(row["id"]): row.get("title") for row in rows}
    for row in rows:
        if str(row.get("link_id")) != item_id:
            continue
        reminder_id = str(row["id"])
        inherit = (heir is not None and not row.get("last_run_at")
                   and row.get("body") == body)
        if row.get("enabled") and inherit:
            handed.append(reminder_id)
        elif row.get("enabled"):
            to_disarm.append(reminder_id)
        elif inherit and reminder_id in owed:
            # Disabled by the master switch and owed a re-arm: the debt
            # moves with the reminder, or switching Dreams on restores
            # none for the watch that is left.
            handed.append(reminder_id)
            debt.append(reminder_id)
    if debt:
        record = _suspension_record(heir) or {}
        subscription, reminders, rules = _owed(record)
        # "off" stays true (the reminder is disabled); anything else is
        # unfinished again, so the next resume pass re-arms it.
        still_off = settled_state(heir.get("suspension")) == SETTLED_OFF
        try:
            dreams_db.set_tracked_suspension(int(heir["id"]), _record_text(
                subscription=subscription, reminders=reminders + debt,
                rules=rules, held=record.get("held"),
                settled=SETTLED_OFF if still_off else None,
                at=record.get("at") if still_off else None))
        except Exception as exc:  # noqa: BLE001 - stays owed where it is
            logger.warning(
                "Dreams reminder hand-over to item {} failed: {}",
                heir.get("id"), type(exc).__name__)
            handed = [value for value in handed if value not in debt]
    for reminder_id in to_disarm:
        scheduling_db.update_reminder_task(reminder_id, enabled=False)
    for reminder_id in handed:
        scheduling_db.update_reminder_task(
            reminder_id, link_id=str(heir["id"]),
            **_handed_title(titles.get(reminder_id), item, heir))
    if handed:
        dreams_db.set_tracked_reminder_date(
            int(heir["id"]), str(heir["event_date"]))
    return to_disarm, handed


def _orphaned_reminders(
    scheduling_db: Any, dreams_db: Any, item: dict,
) -> list[str]:
    """Reminders a live watch takes over when Dreams is switched on (sync).

    TASK-34923: the second half of a hand-over that did not happen when
    its watch retired -- Dreams was off, or the heir's record could not be
    written. The story's reminder for this watch's event that a retired
    watch of the same story and event still owes (one Dreams disabled) is
    this watch's now, when no live watch of the story has a reminder for
    the event: not for an event that is over (the sweep's
    ``event_passed``), not one that fired, not one the user switched off
    (no record owes those). Read-only; :func:`_take_over_reminders` moves
    them once the watch's record lists them.
    """
    story_id = item.get("origin_story_id")
    event_date = item.get("event_date")
    if story_id is None or not event_date:
        return []
    siblings = dreams_db.list_tracked_for_story(int(story_id))
    owed: set[str] = set()
    for sibling in siblings:
        if (sibling.get("status") == "retired"
                and sibling.get("event_date") == event_date
                and sibling.get("retired_reason") != "event_passed"):
            owed.update(_owed(_suspension_record(sibling))[1])
    if not owed:
        return []
    live = {str(sibling["id"]) for sibling in siblings
            if sibling.get("status") in ("active", "paused")}
    body = _reminder_body(event_date)
    for_date = [
        row for row in scheduling_db.list_reminder_tasks_by_link(
            _REMINDER_LINK_TYPE, [str(sibling["id"]) for sibling in siblings])
        if row.get("body") == body]
    if any(str(row.get("link_id")) in live for row in for_date):
        return []  # a live watch of the story has its reminder
    return [str(row["id"]) for row in for_date
            if str(row["id"]) in owed and not row.get("enabled")
            and not row.get("last_run_at")][:1]


def _take_over_reminders(
    scheduling_db: Any, dreams_db: Any, item: dict, reminder_ids: list[str],
) -> None:
    """Link reminders to a live watch, under its title (sync).

    The title changes only when the reminder still carries its retired
    holder's own (:func:`_handed_title`).
    """
    for reminder_id in reminder_ids:
        row = scheduling_db.get_reminder_task(reminder_id) or {}
        holder = (dreams_db.get_tracked_item(int(row["link_id"]))
                  if str(row.get("link_id") or "").isdigit() else None)
        scheduling_db.update_reminder_task(
            reminder_id, link_id=str(item["id"]),
            **(_handed_title(row.get("title"), holder, item)
               if holder is not None else {}))
    if reminder_ids:
        dreams_db.set_tracked_reminder_date(
            int(item["id"]), str(item["event_date"]))


async def _retire_linked_reminders(
    scheduling_db_getter: Any,
    dreams_db: Any,
    item: dict,
    *,
    hand_over: bool = True,
) -> tuple[list[str], list[str]]:
    """Stop a retiring watch's reminders -- or hand them on; degrade.

    Qodo #13 (PR #2890): ``promote_to_reminder`` links the reminder with
    ``link_type="dream_tracked_item"``; untrack and sweep-retire stop the
    watch, and the linked reminder must stop with it.

    Unless the story has another live watch of the same event (round 4,
    A3). The story's one reminder is linked to whichever watch was made
    first; disarming it with that watch left the other one -- still
    running, still showing its event date -- with none: untrack the page
    of a story that is also watched as a question, or let the question
    watch retire as quiet, and the reminder was gone with no way back.
    An unfired reminder for the event is therefore RE-LINKED to that
    watch instead (:func:`_reminder_heir`): the same row, with whatever
    time the user moved it to, armed if it was armed. One the user had
    switched off stays where it is, off. One the master switch disabled
    moves too, and the debt to re-arm it moves onto the heir's record. It
    takes the heir's title. While Dreams is off nothing is handed on here,
    and a hand-over whose record write failed does not happen: either way
    the reminder stays owed on this watch's record and the next switch-on
    hands it on (TASK-34923, see :func:`_retire_reminders_sync`).

    Every failure mode (no getter, raising getter, no scheduling DB,
    unwritable DB) logs and returns -- retirement itself already happened
    and must not unwind.

    Args:
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` or None.
        dreams_db: The app's ``DreamsDB``.
        item: The retired tracked item's row (as read before it retired).
        hand_over: False when the event itself is over (the sweep's
            ``event_passed``): nobody needs that reminder any more.

    Returns:
        ``(disarmed, handed)``: the ids of the reminders that were armed
        and are now disabled -- what Dreams itself switched off (see
        :func:`_record_retirement`) -- and the ids now linked to the
        story's other watch, which this wrapper no longer answers for.
    """
    scheduling_db = None
    if scheduling_db_getter is not None:
        try:
            scheduling_db = scheduling_db_getter()
        except Exception as exc:  # noqa: BLE001 - reminder degrades
            logger.warning(
                "Dreams reminder getter failed while retiring: {}",
                type(exc).__name__)
            return [], []
    if scheduling_db is None:
        return [], []
    try:
        return await asyncio.to_thread(run_finite_local_worker,
            _retire_reminders_sync, scheduling_db, dreams_db, item, hand_over,
            bool(dreams_setting("enabled")))
    except Exception as exc:  # noqa: BLE001 - reminder degrades
        logger.warning(
            "Dreams reminder disable failed for item {}: {}",
            item.get("id"), type(exc).__name__)
        return [], []


async def _record_retirement(
    dreams_db: Any, item: dict, *, reminders: list[str], subscription: bool,
    handed: list[str] | tuple[str, ...] = (),
) -> None:
    """Leave on a retired wrapper the record of what Dreams stopped then.

    Round 3, T5/T10. Retiring a watch (untrack, the lifecycle sweep)
    disables its armed reminders and, for the last rider of a source
    Dreams created, the source. Until now nothing said so afterwards, and
    a later re-track could not tell "Dreams switched this off when the
    watch was retired" from "the user switched it off". The record uses
    the suspension record's shape (:func:`_suspension_record`), merged
    with whatever the switch already owed the wrapper, and
    ``settled: "retired"``. A reminder handed to the story's other watch
    (``handed``, round 4, A3) is that watch's now and is not listed.

    The same goes for the source (TASK-34920). When the record owes it a
    restart -- the master switch stopped it -- and another live watch
    still rides it, that watch is the one switching Dreams on must restart
    it for, so the debt moves onto its record first. A debt left on a
    retired watch is one no switch pass reads. Should that write fail,
    the debt stays here, where tracking the story again still finds it
    (:func:`_retry_held_source`).
    """
    owed_subscription, owed_reminders, owed_rules = _owed(
        _suspension_record(item))
    owed_reminders = [value for value in owed_reminders if value not in handed]
    if owed_subscription and not subscription:
        owed_subscription = not await _hand_on_source_debt(dreams_db, item)
    try:
        await asyncio.to_thread(run_finite_local_worker,
            dreams_db.set_tracked_suspension, int(item["id"]), _record_text(
                subscription=subscription or owed_subscription,
                # The rules the switch silenced stay owed (TASK-34985):
                # tracking the page again re-enables them
                # (:func:`_restore_rules_left_off`).
                reminders=owed_reminders + reminders, rules=owed_rules,
                settled=_SETTLED_RETIRED, at=_now_iso()))
    except Exception as exc:  # noqa: BLE001 - the retirement stands
        logger.warning(
            "Dreams retire: recording item {} failed: {}",
            item.get("id"), type(exc).__name__)


async def _hand_on_source_debt(dreams_db: Any, item: dict) -> bool:
    """Move a retiring watch's restart debt onto a live watch of its source.

    See :func:`_record_retirement` (TASK-34920). The heir keeps whatever
    its own record owed, and its settled state when that was "off" (the
    source IS stopped); anything else leaves it unsettled, so the next
    switch-on takes it up.

    Returns:
        True when a live watch now owes the restart (it already did, or
        its record was written); False when there is none, or the write
        failed.
    """
    subscription_id = int(item.get("subscription_id") or 0)
    if not subscription_id:
        return False
    try:
        riders = await asyncio.to_thread(run_finite_local_worker,
            dreams_db.list_tracked_for_subscription, subscription_id)
        heir = next((rider for rider in riders
                     if int(rider["id"]) != int(item["id"])
                     and rider.get("status") in ("active", "paused")), None)
        if heir is None:
            return False
        record = _suspension_record(heir) or {}
        subscription, reminders, rules = _owed(record)
        if not subscription:
            still_off = settled_state(heir.get("suspension")) == SETTLED_OFF
            await asyncio.to_thread(run_finite_local_worker,
                partial(dreams_db.set_tracked_suspension, only_live=True),
                int(heir["id"]), _record_text(
                    subscription=True, reminders=reminders, rules=rules,
                    held=record.get("held"),
                    settled=SETTLED_OFF if still_off else None,
                    at=record.get("at") if still_off else None))
        return True
    except Exception as exc:  # noqa: BLE001 - the debt stays on the retiree
        logger.warning(
            "Dreams retire: handing subscription {} on failed: {}",
            subscription_id, type(exc).__name__)
        return False


def _dreams_can_account_for_disable(riders: list[dict]) -> bool:
    """Whether Dreams itself switched a now-disabled subscription off.

    Round 3, T10. ``track_page`` may switch a disabled source back on
    only when Dreams can account for its being off:

    * a rider's record owes the subscription -- the master switch stopped
      it, a resume is holding it, or the wrapper's retirement (untrack,
      sweep) disabled it and said so (:func:`_record_retirement`); or
    * no active wrapper rides it and a retired rider carries no record at
      all -- retired by a build from before such records, when retiring
      the last watch always disabled the source.

    Anything else was switched off by the user in Watchlists.

    Args:
        riders: Every wrapper of the subscription, any status
            (``DreamsDB.list_tracked_for_subscription``).
    """
    records = [(rider, _suspension_record(rider)) for rider in riders]
    if any(record is not None and _owed(record)[0] for _, record in records):
        return True
    if any(rider.get("status") == "active" for rider in riders):
        return False
    return any(rider.get("status") == "retired" and record is None
               for rider, record in records)


async def _settle_source_restored(dreams_db: Any, riders: list[dict]) -> None:
    """After a re-track switched a source back on: nobody owes it any more.

    Every rider's record gives up the subscription (and the ``held``
    reason with it). A live rider's record that then owes nothing is
    cleared; a retired rider keeps its retirement record -- and one that
    had none gets an empty one, so that an old, record-less retirement
    cannot account for a LATER disable the user makes.
    """
    for rider in riders:
        record = _suspension_record(rider)
        retired = rider.get("status") == "retired"
        if record is None and not retired:
            continue
        _subscription, reminders, rules = _owed(record)
        text: str | None = None
        if retired:
            text = _record_text(
                subscription=False, reminders=reminders, rules=rules,
                settled=_SETTLED_RETIRED, at=(record or {}).get("at"))
        elif reminders or rules:
            text = _record_text(
                subscription=False, reminders=reminders, rules=rules)
        await asyncio.to_thread(
            dreams_db.set_tracked_suspension, int(rider["id"]), text)


async def _unsettle_held(
    dreams_db: Any, riders: list[dict], *,
    still_owed: Mapping[int, tuple[bool, list[str], list[int]]] | None = None,
) -> None:
    """Un-settle held records just before their source runs again.

    A record settled "on" that owes only its held subscription says the
    page is paused, and the boot check takes it as done. Once the page has
    passed the check, and before the source is switched on, such a record
    is written back unsettled (no ``held``), so a closing write that then
    fails leaves a record the next pass finishes -- not a running source
    reported as held (review of #3101, round 2: S1 for the resume pass, and
    the re-track path). A write that fails here raises, and the source
    stays off; the riders already un-settled get their hold back first, so
    every row still reads paused and a start does not look the name up
    (round 3, T3).

    Args:
        dreams_db: The app's ``DreamsDB``.
        riders: The subscription's wrappers, as the caller read them.
        still_owed: The resume pass's view of what each rider it worked
            on still owes. Such a rider that carried a hold kept it in the
            pass's writes (TASK-34921); it is rewritten without it too
            (review of #3108, X1), or a failed closing write left a running
            page reported as held.
    """
    async def write(rider: dict, text: str | None) -> None:
        await asyncio.to_thread(run_finite_local_worker,
            partial(dreams_db.set_tracked_suspension, only_live=True),
            int(rider["id"]), text)

    # (rider, the record without its hold, the record to put back)
    plan: list[tuple[dict, str, str | None]] = []
    for rider in riders:
        raw = rider.get("suspension")
        if _settled_hold_only(raw):
            plan.append((rider, _record_text(
                subscription=True, reminders=[], rules=[]), raw))
            continue
        held = (_suspension_record(rider) or {}).get("held")
        if held and still_owed and int(rider["id"]) in still_owed:
            subscription, reminders, rules = still_owed[int(rider["id"])]
            plan.append((
                rider,
                _record_text(subscription=subscription, reminders=reminders,
                             rules=rules),
                _record_text(subscription=subscription, reminders=reminders,
                             rules=rules, held=held)))
    done: list[tuple[dict, str, str | None]] = []
    try:
        for entry in plan:
            await write(entry[0], entry[1])
            done.append(entry)
    except Exception:
        for rider, _text, back in done:
            # Best effort: the failure being raised is the one reported;
            # a rider whose hold cannot be put back is only asked about at
            # the next start.
            with contextlib.suppress(Exception):
                await write(rider, back)
        raise


async def _retry_held_source(
    subs_service: Any, dreams_db: Any, wrapper: dict,
) -> None:
    """Ask again about a reused wrapper's page that a resume left off.

    A watch whose page was not a public address when Dreams was switched
    back on is held, and is asked about again by the next switch-on or
    by tracking it again (round 3, T19) -- this is the "tracking it
    again" half, for the story that already owns the wrapper.

    Raises:
        TrackPrivateURL: The page is still not a public destination; the
            source stays off and the hold stands.
    """
    subscription_id = int(wrapper.get("subscription_id") or 0)
    record = _suspension_record(wrapper)
    if (not subscription_id
            or settled_state(wrapper.get("suspension")) == SETTLED_OFF):
        return
    if record is None or not _owed(record)[0]:
        # Not this watch's own debt -- but its source may still be off for
        # Dreams' reasons, with the debt on another watch of it, a retired
        # one included (TASK-34920: a hand-over that failed, or an earlier
        # build). Tracking the story again is the user asking for the
        # watch to work, so that source is restarted too. One the user
        # switched off stays off.
        if not int(wrapper.get("created_by_dreams") or 0):
            return
        try:
            source = await subs_service.get_source(subscription_id)
        except KeyError:
            return
        if bool(source.get("active")) or bool(source.get("paused")):
            return
        if not _dreams_can_account_for_disable(await asyncio.to_thread(
                dreams_db.list_tracked_for_subscription, subscription_id)):
            return

    async def unsettle() -> None:
        # The switch is asked again here (review of #3108, X2): Dreams may
        # have been switched off while the page's name was looked up.
        _require_dreams_on()
        await _unsettle_held(dreams_db, await asyncio.to_thread(
            dreams_db.list_tracked_for_subscription, subscription_id))

    resolved, held = await _restore_source(
        subs_service, subscription_id, before_enable=unsettle,
        enable_guard=_switch_lock, dreams_db=dreams_db)
    if not resolved:
        raise TrackPrivateURL(
            f"refusing to track a non-public destination ({held})")
    # The source runs again and the story keeps its wrapper: the re-track
    # is done. A failure to settle the records now is not the user's to
    # see (review of #3101, round 3, T2) -- they were un-settled before
    # the switch-on, so the next start finishes them.
    try:
        await _settle_source_restored(
            dreams_db, await asyncio.to_thread(
                dreams_db.list_tracked_for_subscription, subscription_id))
    except Exception as exc:  # noqa: BLE001 - the next start finishes it
        logger.warning(
            "Dreams re-track: settling subscription {} failed: {}",
            subscription_id, type(exc).__name__)


async def _restore_rules_left_off(
    subs_service: Any, dreams_db: Any, subscription_id: int,
    dreams_rules: dict[int, dict],
) -> None:
    """Re-enable the Dreams rules a retired watch's record still owes.

    TASK-34985. The switch-off silences Dreams' change rule and the watch's
    record says so; a watch retired while Dreams was off keeps that debt on
    its retirement record, and no switch pass reads a retired watch. A page
    tracked again (Dreams is on: :func:`_require_dreams_on`) gets its rule
    back here, and the debt leaves the record. Only a rule a record owes is
    touched: one the user switched off themselves, or a user's own rule,
    never is.

    Review of #3123, N4: an owed rule the user changed after the record
    was written (its ``updated_at`` later than the record's ``at``) is
    theirs and is left as it is; the debt still leaves the record. A
    record from before ``at`` was written, or a rule row without
    ``updated_at``, cannot tell the two apart, and the owed rule is
    re-enabled as before.
    """
    riders = await asyncio.to_thread(
        dreams_db.list_tracked_for_subscription, subscription_id)
    for rider in riders:
        if rider.get("status") != "retired":
            continue
        record = _suspension_record(rider) or {}
        subscription, reminders, rules = _owed(record)
        owed = [rule_id for rule_id in rules if rule_id in dreams_rules]
        if not owed:
            continue
        for rule_id in owed:
            rule = dreams_rules[rule_id]
            if changed_after(rule.get("updated_at"), record.get("at")):
                continue  # the user's change since; it stands
            if not bool(rule.get("enabled", True)):
                try:
                    await subs_service.update_alert_rule(rule_id, enabled=True)
                except KeyError:
                    pass  # the user deleted the rule
        await asyncio.to_thread(
            dreams_db.set_tracked_suspension, int(rider["id"]), _record_text(
                subscription=subscription, reminders=reminders,
                rules=[rule_id for rule_id in rules if rule_id not in owed],
                held=record.get("held"), settled=_SETTLED_RETIRED,
                at=record.get("at")))


async def _still_dreams_source(
    dreams_db: Any, subscription_id: int, source: Any,
) -> bool:
    """Whether ``source`` is still the one Dreams made for its watches.

    Review of #3123, M1 (see ``track_reconcile.is_dreams_source``): after a
    restore of the subscriptions alone, the id a Dreams watch names can
    belong to the user's own new source. Every Dreams path that changes a
    source's state by id asks this first, and a "no" is handled like a
    deleted source.
    """
    urls = await asyncio.to_thread(run_finite_local_worker,
        dreams_db.story_urls_for_subscription, int(subscription_id))
    return is_dreams_source(_source_options(source), source.get("url"), urls)


async def _source_is_gone(
    subs_service: Any, wrapper: dict, dreams_db: Any = None,
) -> bool:
    """Whether a page watch's source was deleted from Watchlists.

    A Dreams-created watch's id that now names another source -- the
    user's, after a restore (review of #3123, M1) -- is gone too.
    """
    subscription_id = int(wrapper.get("subscription_id") or 0)
    if not subscription_id:
        return False
    try:
        source = await subs_service.get_source(subscription_id)
    except KeyError:
        return True
    return (dreams_db is not None
            and bool(int(wrapper.get("created_by_dreams") or 0))
            and not await _still_dreams_source(
                dreams_db, subscription_id, source))


async def _retire_replaced(
    subs_service: Any, dreams_db: Any, dead: dict, scheduling_db_getter: Any,
) -> None:
    """Retire a page watch whose source the user deleted; log a failure.

    Called once its replacement exists (review of #3108, X4), so the story
    keeps a live watch to inherit its reminder, and the user's track has
    already succeeded: a failure here is logged, not raised (round 2, Z2),
    and the story's next track retires the dead watch then
    (:func:`_retire_dead_siblings`).
    """
    try:
        await untrack(subs_service, dreams_db, int(dead["id"]),
                      scheduling_db_getter=scheduling_db_getter)
    except Exception as exc:  # noqa: BLE001 - the next track retries it
        logger.warning(
            "Dreams track: retiring replaced item {} failed: {}",
            dead.get("id"), type(exc).__name__)


async def _retire_dead_siblings(
    subs_service: Any, dreams_db: Any, wrapper: dict, scheduling_db_getter: Any,
) -> None:
    """Retire the story's other live page watches whose source is gone.

    Review of #3108, round 2, Z2: a replacement whose retirement step
    failed left two live watches, the dead one still holding the story's
    reminder and counting toward the cap. Tracking the story again reuses
    the live one, so the dead one is retired here, with it as heir.
    """
    story_id = wrapper.get("origin_story_id")
    if story_id is None:
        return
    siblings = await asyncio.to_thread(
        dreams_db.list_tracked_for_story, int(story_id))
    for sibling in siblings:
        if (int(sibling["id"]) != int(wrapper["id"])
                and sibling.get("mechanism") == _PAGE_MECHANISM
                and sibling.get("status") in ("active", "paused")
                and await _source_is_gone(subs_service, sibling, dreams_db)):
            await _retire_replaced(
                subs_service, dreams_db, sibling, scheduling_db_getter)


async def _disable_dream_source(
    subs_service: Any, subscription_id: int, dreams_db: Any = None,
) -> bool:
    """Switch off a source Dreams created; say whether DREAMS did it.

    Used when the last watch on it is retired. A source that is already
    off -- the user switched it off in Watchlists -- or gone is left
    alone and is NOT claimed, so the retirement record never accounts for
    a disable that was not Dreams' (round 3, T10). So is one that is no
    longer the source Dreams made (review of #3123, M1).

    Returns:
        True when the source was running and Dreams switched it off.
    """
    try:
        source = await subs_service.get_source(subscription_id)
    except KeyError:
        return False
    if dreams_db is not None and not await _still_dreams_source(
            dreams_db, subscription_id, source):
        return False
    if not (bool(source.get("active")) or bool(source.get("paused"))):
        return False
    await subs_service.update_source(subscription_id, {"active": False})
    return True


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
        TrackDreamsOff: When Dreams is switched off -- raised first, before
            anything is read or written (TASK-34928).
        TrackInvalidURL: When ``url`` fails the shared ``validate_url``
            check (scheme, whitespace, credentials, host shape) --
            raised BEFORE any lookup or write.
        TrackPrivateURL: When Dreams would have to create or re-enable
            the subscription and the URL is not a public destination
            under the egress policy (loopback, private network, metadata
            endpoint, or an unresolvable host) -- raised before any
            write. Attaching to a subscription the user already owns is
            never refused on this ground.
        TrackCapReached: When ``count_active_tracked()`` already meets the
            cap -- raised BEFORE any subscription, watchlist, alert, or
            tracked-item write.
        TrackSourceDisabled: When the URL resolves to a DISABLED source
            with no dream provenance (the user disabled it themselves) --
            also raised before any write.
    """
    _require_dreams_on()
    # URL validation FIRST (Qodo #8, PR #2890): the modal gates http(s),
    # but a direct caller can pass anything, and a malformed address
    # persisted as a subscription would ride every later watchlist run.
    if not validate_url(str(url)):
        raise TrackInvalidURL(f"refusing to track a non-http(s) URL: {url!r}")

    cadence = max(int(cadence_seconds or 0), _min_interval_seconds())
    stale: dict | None = None  # a watch whose source the user deleted

    # Reuse before anything is written (Qodo #2, PR #2890): a story that
    # already owns an ACTIVE page wrapper keeps it -- refresh the plan
    # (cadence, event date) on the existing row instead of stacking a
    # second wrapper around the same watch, which a later untrack of the
    # newer row would strand. Reuse consumes nothing, so the cap guard
    # below is deliberately not consulted (and must not refuse it). The
    # lookup asks for the PAGE wrapper (Qodo #3, PR #2921): the story may
    # also own a newer question watch, and finding that one instead
    # skipped reuse and stacked a duplicate page wrapper.
    if origin_story_id is not None:
        wrapper = await _reusable_wrapper(
            dreams_db, origin_story_id, _PAGE_MECHANISM)
        if wrapper is not None and await _source_is_gone(
                subs_service, wrapper, dreams_db):
            # TASK-34927. The user deleted the watch's source in Watchlists
            # -- what the notices and the guide tell them to do with one
            # that is stuck -- and is tracking again: the old watch has
            # nothing behind it. A new watch is made below, on a new
            # source, and only THEN is the old one retired, with the new
            # one there to inherit the story's reminder as the user left
            # it (review of #3108, X4).
            stale, wrapper = wrapper, None
        if wrapper is not None:
            await _refresh_wrapper_plan(
                dreams_db, wrapper, cadence=cadence,
                event_date=event_date,
                scheduling_db_getter=scheduling_db_getter)
            await _retire_dead_siblings(
                subs_service, dreams_db, wrapper, scheduling_db_getter)
            await _retry_held_source(subs_service, dreams_db, wrapper)
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
    # nothing, so this precedes every write below. A dead watch about to be
    # replaced does not count.
    cap = _cap()
    active = await asyncio.to_thread(dreams_db.count_active_tracked)
    if stale is not None and stale.get("status") == "active":
        active -= 1
    if active >= cap:
        raise TrackCapReached(
            f"track cap reached: {active} active tracked items (cap {cap})"
        )

    # Attach-or-create. An existing subscription for exactly this URL is
    # adopted untouched (never renamed, never re-cadenced, never disabled
    # by a later untrack): the user may already depend on it.
    # Review of #3108, round 2, Z1: from the switch check after the name
    # lookup to the new watch's row and reminder, this is one step to a
    # switch-off pass (:func:`_switch_lock`). A pass that waited runs after
    # it and finds the watch, so it stops what was switched on, created or
    # armed; one that ran first leaves the switch off, and the check
    # refuses. No name lookup happens while the lock is held.
    locked = False
    async with contextlib.AsyncExitStack() as switch_held:
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
            # One targeted read (Qodo #6, PR #2921), never a scan of the
            # capped status lists: the owning row of an old subscription may
            # be older than the newest 500 retired wrappers.
            dream_owned = await asyncio.to_thread(
                dreams_db.subscription_dream_owned, subscription_id
            )
            created_by_dreams = 1 if dream_owned else 0
            # P7 (fix round 1): the URL lookup ignores is_active, so a found
            # source may be one this flow disabled on a prior untrack, or a
            # foreign source the user disabled themselves. Attaching to either
            # as-is would record an active tracked item against a dead watch.
            source = await subs_service.get_source(subscription_id)
            if dream_owned and not await _still_dreams_source(
                    dreams_db, subscription_id, source):
                # The Dreams store names this id, but it is not the source
                # Dreams made -- the user's, after a restore (review of
                # #3123, M1): attached as theirs, never switched by Dreams.
                dream_owned, created_by_dreams = False, 0
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
                # A source Dreams created is off. Dreams switches it back on
                # only when it can account for the disable (round 3, T10) --
                # a wrapper's record says Dreams stopped it -- and never
                # undoes the user's own "off" in Watchlists.
                riders = await asyncio.to_thread(
                    dreams_db.list_tracked_for_subscription, subscription_id)
                if not _dreams_can_account_for_disable(riders):
                    raise TrackSourceDisabled(
                        "source exists but was switched off in Watchlists, "
                        f"not by Dreams: subscription {subscription_id}"
                    )
                # Re-enabling restarts fetches of a URL Dreams stored, so the
                # destination is checked again now (Qodo #1, PR #2921) --
                # still before any write.
                await _require_public_url(str(url))
                # ...and the switch again, after that lookup (review of #3108,
                # X2): a switch-off that landed meanwhile leaves it off. The
                # check and the switch-on are one step to a switch-off pass
                # (round 2, Z1, :func:`_switch_lock`).
                await switch_held.enter_async_context(_switch_lock())
                locked = True
                _require_dreams_on()
                await _unsettle_held(dreams_db, riders)
                await _enable_dream_subscription(
                    subs_service, subscription_id, source)
                await _settle_source_restored(dreams_db, riders)
                outcome = "re-enabled"
            elif dream_owned and _needs_story_mark(source):
                # A live source Dreams created before the mark existed (round
                # 3, T1): attaching used to leave it unmarked, so it stayed a
                # trusted egress origin. Ownership is enough to stamp it.
                await subs_service.update_source(
                    subscription_id,
                    {"processing_options": _story_options(source)})
        else:
            # Dreams is about to store this URL as a subscription the
            # watchlist loop will fetch as a trusted source, and nobody typed
            # it: it must be a public destination (Qodo #1, PR #2921). An
            # ATTACH above is exempt on purpose -- that subscription is the
            # user's own, already being fetched, and tracking adds no egress.
            await _require_public_url(str(url))
            await switch_held.enter_async_context(_switch_lock())
            locked = True
            _require_dreams_on()
            # Payload keys are the ones ``_source_batch_rows`` consumes:
            # ``source_type`` (not "type") and ``active`` (not "is_active").
            result = await subs_service.create_source(
                {
                    "name": _label(f"Dreams: {str(title)[:40]}"),
                    "source_type": "url",
                    "source": str(url),
                    "check_frequency": cadence,
                    "active": True,
                    # Nobody typed this URL: the watchlist fetch must never
                    # treat it as its own trusted origin (review item 2).
                    "processing_options": dict(_STORY_URL_OPTIONS),
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

        if not locked:
            await switch_held.enter_async_context(_switch_lock())
            _require_dreams_on()
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
        # The rule is found by the id Dreams recorded when it made it, never
        # by its name (round 3, T11). One that exists is shared as it stands
        # -- enabled or not: a rule the user switched off stays off.
        dreams_rules = await _dreams_rules_on(
            subs_service, dreams_db, subscription_id)
        if dreams_rules:
            alert_rule_id = min(dreams_rules)
            await _restore_rules_left_off(
                subs_service, dreams_db, subscription_id, dreams_rules)
        else:
            created_rule = await subs_service.create_alert_rule(
                name=_change_rule_name(title),
                condition_type="items_above",
                condition_value={"threshold": 0},
                job_id=subscription_id,
                severity="information",
            )
            alert_rule_id = int(created_rule["rule_id"])

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
            # Every branch above that leaves a Dreams-owned source leaves it
            # marked; this is the record that it is (see
            # ``backfill_egress_marks``).
            egress_marked=created_by_dreams,
            alert_rule_id=alert_rule_id,
        )
        if created_by_dreams:
            # ...for the older wrappers of the same source too.
            await asyncio.to_thread(dreams_db.set_egress_marked, subscription_id)
        # Phase 2 Task 5: an event-dated watch gets one scheduled reminder a
        # week ahead. The wrapper row above already exists, so a reminder
        # failure (or a runtime without the scheduled-tasks DB) only logs --
        # the tracking result is returned unchanged either way.
        await promote_to_reminder(
            scheduling_db_getter,
            {"id": tracked_item_id, "query_template": None,
             "event_date": event_date, "origin_story_id": origin_story_id},
            dreams_db=dreams_db,
        )
    if stale is not None:
        await _retire_replaced(
            subs_service, dreams_db, stale, scheduling_db_getter)
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
    survive for a user who adopted them; and only once no other ACTIVE
    wrapper rides the same subscription (Qodo #4, PR #2921: two stories
    with one URL share it). A subscription Dreams merely
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
        "subscription_disabled": bool}`` -- the flag is False for an
        attached subscription and for one another wrapper still watches.

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
    disarmed, handed = await _retire_linked_reminders(
        scheduling_db_getter, dreams_db, item)

    subscription_id = item.get("subscription_id")
    subscription_disabled = False
    # Qodo #4 (PR #2921): another story with the same URL may own an
    # active wrapper on this very subscription. The item above is already
    # retired, so the count is of the OTHER wrappers; the source is only
    # disabled once none is left watching it.
    if (
        item.get("created_by_dreams")
        and subscription_id
        and not await asyncio.to_thread(
            dreams_db.count_active_tracked_for_subscription,
            int(subscription_id))
    ):
        try:
            subscription_disabled = await _disable_dream_source(
                subs_service, int(subscription_id), dreams_db)
        except Exception as exc:  # noqa: BLE001 - wrapper is retired either way
            logger.warning(
                "Dreams untrack: disabling subscription {} failed: {}",
                subscription_id, type(exc).__name__,
            )
    await _record_retirement(
        dreams_db, item, reminders=disarmed,
        subscription=subscription_disabled, handed=handed)
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
                retired.append({**item, "retired_reason": "event_passed"})
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
    notes, retired = await asyncio.to_thread(run_finite_local_worker,
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
        # ...unless the story's other watch of the same event is still
        # live: then it inherits the reminder (round 4, A3). Not when the
        # event itself is over.
        disarmed, handed = await _retire_linked_reminders(
            scheduling_db_getter, dreams_db, item,
            hand_over=item.get("retired_reason") != "event_passed")
        subscription_disabled = False
        subscription_id = item.get("subscription_id")
        # Qodo #1: a dream-created PAGE subscription is disabled so the
        # independent watchlist projection stops scheduling it; a foreign
        # attached subscription (created_by_dreams=0) is never ours to
        # stop, and a question item has no subscription at all.
        # ...and, like untrack, only once no OTHER active wrapper still
        # rides it (Qodo #4, PR #2921). Every row this pass retired is
        # already retired by now, so two wrappers of one subscription
        # retiring together still take it with them.
        if (
            subs_service is not None
            and item.get("mechanism") == _PAGE_MECHANISM
            and int(item.get("created_by_dreams") or 0) == 1
            and subscription_id
            and not await asyncio.to_thread(run_finite_local_worker,
                dreams_db.count_active_tracked_for_subscription,
                int(subscription_id))
        ):
            try:
                subscription_disabled = await _disable_dream_source(
                    subs_service, int(subscription_id), dreams_db)
            except Exception as exc:  # noqa: BLE001 - retire stands either way
                logger.warning(
                    "Dreams sweep: disabling subscription {} failed: {}",
                    subscription_id, type(exc).__name__,
                )
        await _record_retirement(
            dreams_db, item, reminders=disarmed,
            subscription=subscription_disabled, handed=handed)
    return notes


# --- Event reminders (Phase 2 Task 5) -------------------------------------------


def _reminder_run_at(event_date: Any, lead_days: int) -> datetime:
    """The instant an event's reminder fires: ``lead_days`` before it.

    The lead date is a day on the user's own calendar (Qodo #14,
    PR #2921). Storing it date-only made the scheduler read it as
    midnight UTC -- the evening BEFORE that day anywhere west of UTC, so
    the nudge arrived a local day early. The instant is therefore built
    as ``_REMINDER_LOCAL_TIME`` on the lead date in the machine's local
    timezone (the same "a naive time is local time" rule the reminder
    forms apply), resolved for that date so a DST change in between
    cannot shift it, and handed over timezone-aware.

    Args:
        event_date: The tracked item's ISO ``event_date``.
        lead_days: Days of runway before the event.

    Returns:
        A timezone-aware UTC instant: nine in the morning, local time, on
        the lead date.

    Raises:
        ValueError: If ``event_date`` is not an ISO date.
    """
    lead_date = date.fromisoformat(str(event_date)) - timedelta(
        days=int(lead_days))
    # A naive datetime's astimezone() reads it as local wall-clock time
    # under the platform's rule for the date it carries.
    return datetime.combine(
        lead_date, _REMINDER_LOCAL_TIME).astimezone(UTC)


#: ``previous_event_date`` default for :func:`promote_to_reminder`: the
#: wrapper's event did not move (a new wrapper, or a re-track with the
#: same date). A sentinel, because ``None`` is a real previous value
#: ("the watch had no date").
_EVENT_NOT_MOVED: Any = object()


def _reminder_body(event_date: Any) -> str:
    """The body Dreams writes on an event reminder; it names the event."""
    return f"Tracked Dreams event on {event_date}"


def _reminder_title(item: Any) -> str:
    """The title of the reminder a watch holds: its question, or a generic one."""
    return _label(f"Dreams: {item.get('query_template') or 'tracked event'}")


def _handed_title(title: Any, holder: Any, heir: Any) -> dict:
    """The title change a hand-over makes, as update fields (maybe none).

    Schedules lets the user rename a reminder (review of #3108, X3): only
    a reminder still carrying its holder's own title -- as Dreams writes it
    now, or with its square brackets, as builds before round 4 did (the
    same two forms :func:`_change_rule_name` recognises) -- takes the new
    holder's title. A title the user chose stays.
    """
    own = {_reminder_title(holder),
           f"Dreams: {holder.get('query_template') or 'tracked event'}"}
    return {"title": _reminder_title(heir)} if title in own else {}


#: ``settled`` value of the record a wrapper is left with when it is
#: retired (untrack, lifecycle sweep): what Dreams itself stopped then.
_SETTLED_RETIRED = "retired"


def _dreams_disarmed(row: dict, wrappers: dict[str, dict]) -> bool:
    """Whether Dreams itself can account for a reminder being disabled.

    True when the reminder is linked to a wrapper Dreams retired and that
    wrapper's record lists it among the reminders Dreams disabled then --
    or carries no record at all (retired before such records existed, when
    retiring a watch always disabled its reminder). Anything else was
    switched off by the user, and stays their decision.
    """
    wrapper = wrappers.get(str(row.get("link_id")))
    if wrapper is None or wrapper.get("status") != "retired":
        return False
    record = _suspension_record(wrapper)
    return record is None or str(row["id"]) in _owed(record)[1]


def _ensure_reminder(
    scheduling_db: Any,
    tracked_item: Any,
    run_at: datetime | None,
    owner_id: str,
    moved: bool,
    siblings: list[dict],
) -> tuple[str | None, bool]:
    """Sync body of :func:`promote_to_reminder`; the caller thread-offloads.

    What changes a watch's reminder is its EVENT DATE changing -- never a
    comparison of the stored instant with the one that would be computed
    today (review of #2921, items 4 and 5). And whether a reminder is
    owed at all is decided per STORY and event date, not per wrapper
    (round 3, T5): the reminders looked at are those linked to ANY
    wrapper the story ever had, so untracking and tracking again, or a
    second way of watching the same story, cannot bring back a nudge
    that was already delivered.

    For a wrapper whose event is ``D``:

    1. When the event moved, this wrapper's armed reminders for another
       date are disabled.
    2. A reminder for ``D`` that has FIRED, on any of the story's
       wrappers, settles it: never again.
    3. No reminder row for ``D`` is left but one of the story's wrappers
       recorded that it made one (``reminder_event_date``): the user
       deleted it. Not re-created.
    4. This wrapper already has its reminder for ``D``: it stands as the
       user left it -- armed, disabled or rescheduled. Only when the
       event has just moved BACK to ``D`` is it re-armed (Dreams
       disarmed it when the event moved away).
    5. Another of the story's wrappers has it armed: nothing to add (when
       that wrapper is retired the reminder is handed on, see
       :func:`_retire_linked_reminders`).
    6. It exists only disabled, on other wrappers: a new one is made when
       Dreams itself disarmed every one of them by retiring their watch
       (:func:`_dreams_disarmed`); otherwise the user switched it off,
       and it stays off.
    7. Otherwise one reminder is created.

    Args:
        siblings: Every wrapper of the same story (any status, this one
            included); empty for a watch that has no story.

    Returns:
        ``(created_id, exists)``: the id of the reminder created (or
        None), and whether a reminder for this event exists or existed --
        the caller records the date on the wrapper when it does.
    """
    link_id = str(tracked_item.get("id"))
    body = _reminder_body(tracked_item.get("event_date"))
    wrappers = {str(row["id"]): row for row in siblings}
    rows = scheduling_db.list_reminder_tasks_by_link(
        _REMINDER_LINK_TYPE, {link_id, *wrappers})
    own = [row for row in rows if str(row.get("link_id")) == link_id]
    if moved:
        for row in own:
            if row.get("enabled") and row.get("body") != body:
                scheduling_db.update_reminder_task(row["id"], enabled=False)
    if run_at is None:
        return None, False
    for_date = [row for row in rows if row.get("body") == body]
    if any(row.get("last_run_at") for row in for_date):
        return None, True
    if not for_date:
        recorded = str(tracked_item.get("event_date"))
        if any(str(row.get("reminder_event_date") or "") == recorded
               for row in siblings):
            return None, True
    own_for_date = [row for row in for_date
                    if str(row.get("link_id")) == link_id]
    if own_for_date:
        if moved:
            for row in own_for_date:
                if not row.get("enabled"):
                    scheduling_db.update_reminder_task(row["id"], enabled=True)
        return None, True
    if for_date and (
        any(row.get("enabled") for row in for_date)
        or not all(_dreams_disarmed(row, wrappers) for row in for_date)
    ):
        return None, True
    created = scheduling_db.create_reminder_task(
        owner_id,
        _reminder_title(tracked_item),
        body=body,
        schedule_kind="one_time",
        run_at=run_at,
        next_run_at=run_at,
        link_type=_REMINDER_LINK_TYPE,
        link_id=link_id,
    )
    return created, True


async def promote_to_reminder(
    scheduling_db_getter: Any,
    tracked_item: Any,
    *,
    previous_event_date: Any = _EVENT_NOT_MOVED,
    lead_days: int = _REMINDER_LEAD_DAYS,
    owner_id: str = "local",
    dreams_db: Any = None,
) -> str | None:
    """Give one tracked item the reminder its event date calls for.

    An event-dated item owns one ``one_time`` reminder firing
    ``lead_days`` before the event, at nine in the morning local time
    (see :func:`_reminder_run_at`), linked back to the tracked item
    (``link_type="dream_tracked_item"``, ``link_id=str(id)``) so the
    scheduling UI can trace it; a dateless item owns none.

    The call is safe to repeat: it creates a reminder only when the
    story has none for that event, and it touches existing reminders only
    when the event date itself moved (``previous_event_date`` differs
    from the item's date) -- see :func:`_ensure_reminder`. A reminder the
    user disabled or rescheduled, and one that has fired, are never
    re-armed, replaced or re-created by a re-track of the same event --
    whichever wrapper of the story is asking.

    Degrade, never raise -- the wrapper row already exists when this is
    called, so every failure mode (no getter, getter returning None --
    a runtime whose scheduled-tasks DB is not wired yet -- a raising
    getter, an unwritable scheduling DB, an unparsable date) logs and
    returns ``None``; tracking itself is unaffected. An unparsable NEW
    date means the event is no longer known: no reminder is made, and
    the one pending for the previous date is disarmed.

    Args:
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` or None.
        tracked_item: Mapping with at least ``id``, ``query_template`` and
            ``event_date`` -- the tracked row (or its just-created view) --
            and ``origin_story_id`` when the watch has a story.
        previous_event_date: The event date the wrapper carried before
            this call's ``event_date`` replaced it, when the caller
            refreshed an existing wrapper's plan; omit it for a new
            wrapper.
        lead_days: Days before ``event_date`` the reminder fires.
        owner_id: Reminder owner (the local scheduling owner id).
        dreams_db: The app's ``DreamsDB``. With it the decision spans the
            story's wrappers and the date is recorded on this one;
            without it only this wrapper's own reminders are considered.

    Returns:
        The id of the reminder created, or ``None`` when none was (no
        event date, a reminder already stands or was delivered, or
        nothing could be promoted).
    """
    event_date = tracked_item.get("event_date")
    moved = previous_event_date is not _EVENT_NOT_MOVED and (
        (previous_event_date or None) != (event_date or None))
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

    run_at: datetime | None = None
    if event_date:
        try:
            run_at = _reminder_run_at(event_date, lead_days)
        except ValueError:
            # The event is no longer known (round 3, T16, owner ruling).
            # No reminder can be made for it -- but the one still pending
            # for the OLD date must not stay armed, so this goes on to
            # ``_ensure_reminder`` with nothing to create: when the date
            # moved it disarms the old event's reminder, and a fired one
            # is left alone.
            logger.warning(
                "Dreams reminder: unparsable event_date {!r}", event_date)

    def promote() -> str | None:
        siblings: list[dict] = []
        story_id = tracked_item.get("origin_story_id")
        if dreams_db is not None and story_id is not None:
            siblings = dreams_db.list_tracked_for_story(int(story_id))
        elif dreams_db is not None:
            # A watch with no story is its own only sibling.
            row = dreams_db.get_tracked_item(int(tracked_item["id"]))
            siblings = [row] if row is not None else []
        created, exists = _ensure_reminder(
            scheduling_db, tracked_item, run_at, owner_id, moved, siblings)
        if dreams_db is not None and exists:
            dreams_db.set_tracked_reminder_date(
                int(tracked_item["id"]), str(event_date))
        return created

    try:
        return await asyncio.to_thread(promote)
    except Exception as exc:  # noqa: BLE001 - reminder degrades
        logger.warning(
            "Dreams reminder creation failed: {}", type(exc).__name__)
        return None


# --- The master switch (Qodo #5, PR #2921) --------------------------------------


def _scheduling_db(scheduling_db_getter: Any) -> Any:
    """Resolve the app's ``ScheduledTasksDB`` for the switch, or None.

    A missing getter, a getter returning None, and a raising getter all
    mean "no reminders can be touched right now"; the switch degrades.
    """
    if scheduling_db_getter is None:
        return None
    try:
        return scheduling_db_getter()
    except Exception as exc:  # noqa: BLE001 - the switch degrades
        logger.warning(
            "Dreams switch: reminder getter failed: {}", type(exc).__name__)
        return None


def _suspension_record(item: Any) -> dict | None:
    """A wrapper's suspension record, or None when it carries none.

    The record is JSON on ``dream_tracked_items.suspension``::

        {"subscription": bool,        # Dreams stopped this watch's
                                      #   subscription and owes it a
                                      #   re-enable
         "reminders": ["<id>", ...],  # reminders Dreams disabled
         "rules": [<id>, ...],        # alert rules Dreams silenced
         "held": "<reason>",          # optional: why the subscription
                                      #   could not be re-enabled
         "settled": "off" | "on"}     # optional, see below

    ``settled`` says whether a switch pass still has work on the wrapper
    (review of #2921, round 3, T2/T3/T19). ``"off"``: every stop the
    record names has been applied -- the watch is fully suspended.
    ``"on"``: a resume restored everything it may; what the record still
    owes is ``held`` (the page is not a public address) until the next
    switch-on or re-track -- and, when the reason may pass (any DNS
    answer: a failed lookup, a private or metadata address; see
    ``_LASTING_HOLDS``), until the daily cycle or any resume pass asks
    again (round 4, B2; TASK-34922). Absent: unfinished -- the write-ahead record
    of a stop that has not (all) landed, or a resume that could not reach
    a service -- and the next pass in the switch's direction takes it up.

    Unreadable text still means "a record" -- with nothing known to
    restore -- so a damaged record is cleared by resume, never mistaken
    for "no record".
    """
    raw = item.get("suspension")
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return {}
    return record if isinstance(record, dict) else {}


#: Refusal reasons that are facts about the address itself: only a scheme
#: that is not a web one. Every other hold Dreams meets is a DNS answer --
#: a failed lookup, and also a private or cloud-metadata address, which a
#: hotel, airport or VPN resolver gives for every name -- and may be gone
#: tomorrow, so it is asked about again without the user doing anything
#: (round 4, B2; TASK-34922, owner decision 2026-10-13: retry private and
#: metadata holds on later cycles). A private IP literal or a metadata
#: hostname would be a fact about the address too, but Dreams does not
#: hold a page for either: ``_require_public_url`` refuses both when the
#: page is tracked (review of #3101, N2). A retry is the same
#: public-address check, so it can never switch on a page that still
#: resolves privately.
_LASTING_HOLDS = frozenset({"scheme"})


def _hold_may_pass(raw: Any) -> bool:
    """Whether a stored record holds its source for a passing reason."""
    held = (_suspension_record({"suspension": raw}) or {}).get("held")
    return bool(held) and held not in _LASTING_HOLDS


def _resume_has_work(raw: Any) -> bool:
    """Whether a resume pass still has something to do for this record.

    A record that is not settled "on" -- and one that is, when what it
    holds may pass.
    """
    return bool(raw) and (
        settled_state(raw) != SETTLED_ON or _hold_may_pass(raw))


def _settled_hold_only(raw: Any) -> bool:
    """Whether a record is settled "on" and owes only its held subscription.

    Every writer of "on" today writes only that (``only_held``), but the
    resume pass skips its write-ahead and un-settles on this answer, not on
    the settled state alone (review of #3101, round 2, S2).
    """
    if settled_state(raw) != SETTLED_ON:
        return False
    subscription, reminders, rules = _owed(_suspension_record({"suspension": raw}))
    return subscription and not reminders and not rules


def _owed(record: dict | None) -> tuple[bool, list[str], list[int]]:
    """What a record owes: ``(subscription, reminder ids, rule ids)``."""
    record = record or {}
    try:
        rules = [int(value) for value in record.get("rules") or []]
    except (TypeError, ValueError):
        rules = []
    return (
        bool(record.get("subscription")),
        [str(value) for value in record.get("reminders") or []],
        rules,
    )


def _record_text(
    *,
    subscription: bool,
    reminders: list[str],
    rules: list[int],
    held: str | None = None,
    settled: str | None = None,
    at: str | None = None,
) -> str:
    """Serialize a suspension record (see :func:`_suspension_record`).

    ``at`` is when the switch-off settled the record, or when the watch
    retired (review of #3123, M2): a reminder or rule changed after it is
    one the user changed, which the start check leaves alone. Rewrites
    that keep a record's settled state keep its ``at``.
    """
    record: dict[str, Any] = {
        "subscription": bool(subscription),
        "reminders": sorted(set(reminders)),
        "rules": sorted(set(rules)),
    }
    if held:
        record["held"] = held
    if settled:
        record["settled"] = settled
    if at:
        record["at"] = at
    return json.dumps(record)


def _now_iso() -> str:
    """The ``at`` a record is written with: now, UTC, ISO 8601."""
    return datetime.now(UTC).isoformat()


def _enabled_linked_reminder_ids(scheduling_db: Any, item_id: int) -> list[str]:
    """Ids of the item's linked reminders that are armed right now (sync)."""
    return [
        str(row["id"])
        for row in scheduling_db.list_reminder_tasks_by_link(
            _REMINDER_LINK_TYPE, [item_id])
        if row.get("enabled")
    ]


def _disable_reminders(scheduling_db: Any, reminder_ids: list[str]) -> None:
    """Disable exactly these reminders (sync; the caller thread-offloads)."""
    for reminder_id in reminder_ids:
        scheduling_db.update_reminder_task(reminder_id, enabled=False)


def _restore_reminders(scheduling_db: Any, reminder_ids: list[str]) -> None:
    """Re-arm the reminders the switch disabled (sync).

    Only a reminder that still exists, is still disabled, and has not
    fired is re-enabled; its schedule is left exactly as it stands.
    """
    for reminder_id in reminder_ids:
        row = scheduling_db.get_reminder_task(reminder_id)
        if row and not row.get("enabled") and not row.get("last_run_at"):
            scheduling_db.update_reminder_task(reminder_id, enabled=True)


def _riders_by_subscription(items: list[dict]) -> dict[int, list[dict]]:
    """Every live wrapper, grouped by the subscription it rides."""
    riders: dict[int, list[dict]] = {}
    for item in items:
        subscription_id = int(item.get("subscription_id") or 0)
        if subscription_id:
            riders.setdefault(subscription_id, []).append(item)
    return riders


@dataclass
class _SourceStop:
    """What one suspend pass decides for one subscription, ONCE.

    Several wrappers can ride one subscription, and they need not agree
    about who created it (round 3, T13). The decision is therefore made
    per subscription, from all its riders, and every rider records it.

    Attributes:
        known: The watchlists service answered; ``owed`` and ``rules``
            are facts. False leaves every rider unsettled for a later
            pass (no service, or a failed read).
        owed: Dreams stops this subscription -- it is Dreams' own and
            running now, or an earlier record of a rider already owes it.
        rules: Dreams' change rules that are enabled on it right now.
        applied: Every stop for it landed in this pass.
    """

    known: bool = False
    owed: bool = False
    rules: list[int] = field(default_factory=list)
    applied: bool = False
    #: Options to stamp on a Dreams-owned source that is not marked yet
    #: (round 3, T1); None when there is nothing to stamp.
    mark: dict | None = None


async def _plan_source_stop(
    subs_service: Any, dreams_db: Any, subscription_id: int,
    riders: list[dict],
) -> _SourceStop:
    """Read what is running on one subscription and decide its stop."""
    plan = _SourceStop(owed=any(
        _owed(_suspension_record(rider))[0] for rider in riders))
    if subs_service is None:
        return plan
    try:
        try:
            source = await subs_service.get_source(subscription_id)
        except KeyError:
            # The user deleted the source (round 3, T4) -- the one by-hand
            # action Watchlists offers on a "Dreams: ..." source. There is
            # no subscription to stop and none to restore; the wrapper's
            # reminders are still stopped, and it settles.
            return _SourceStop(known=True, applied=True)
        owned = any(
            int(rider.get("created_by_dreams") or 0) == 1 for rider in riders
        ) or await asyncio.to_thread(run_finite_local_worker,
            dreams_db.subscription_dream_owned, subscription_id)
        if owned and not await _still_dreams_source(
                dreams_db, subscription_id, source):
            # The id names another source now -- the user's, after a
            # restore (review of #3123, M1). The watch's own source is
            # gone: handled as deleted, the other one never touched.
            return _SourceStop(known=True, applied=True)
        # Running means "would be fetched, now or after one press of
        # Resume" (round 3, T9): the normalized ``active`` is ``is_active
        # and not paused``, so an auto-paused source reads inactive while
        # its ``is_active`` is still 1.
        running = bool(source.get("active")) or bool(source.get("paused"))
        plan.owed = plan.owed or (owned and running)
        if owned and _needs_story_mark(source):
            plan.mark = _story_options(source)
        plan.rules = [
            rule_id for rule_id, rule in (await _dreams_rules_on(
                subs_service, dreams_db, subscription_id)).items()
            if bool(rule.get("enabled", True))]
        plan.known = True
    except Exception as exc:  # noqa: BLE001 - the riders stay unsettled
        logger.warning(
            "Dreams suspend: reading subscription {} failed: {}",
            subscription_id, type(exc).__name__)
    return plan


async def _suspend_pass(
    subs_service: Any, dreams_db: Any, scheduling_db: Any, items: list[dict],
) -> int:
    """Stop the outside work of every live wrapper not settled "off".

    Three steps, in this order:

    1. **Record** (write-ahead). Each wrapper's record becomes what it
       already owed MERGED with what is running now -- its armed
       reminders, the enabled Dreams alert rule on its page, its
       subscription when Dreams stops it. Merging, not skipping, is what
       makes a left-over record harmless (round 3, T2): the record a held
       resume leaves behind used to make the next switch-off skip the
       wrapper while its reminder and rule were running again.
    2. **Stop** everything the records name: the reminders per wrapper;
       the rules and the subscription once per subscription. Every stop
       is idempotent, so a record whose stops did not (all) land -- the
       write failed, the app quit in between, a service was missing -- is
       simply applied again (T3).
    3. **Settle.** A wrapper is marked ``settled: "off"`` only when every
       one of its stops landed in this pass. Anything else leaves it
       unsettled, which is exactly what the startup check looks for.

    Returns:
        How many wrappers the pass worked on (settled or not).
    """
    pending = [item for item in items
               if settled_state(item.get("suspension")) != SETTLED_OFF]
    if not pending:
        return 0
    riders = _riders_by_subscription(items)
    plans: dict[int, _SourceStop] = {}
    for item in pending:
        subscription_id = int(item.get("subscription_id") or 0)
        if subscription_id and subscription_id not in plans:
            plans[subscription_id] = await _plan_source_stop(
                subs_service, dreams_db, subscription_id,
                riders[subscription_id])

    staged: list[tuple[dict, bool, list[str], list[int]]] = []
    for item in pending:
        item_id = int(item["id"])
        plan = plans.get(int(item.get("subscription_id") or 0))
        subscription, reminders, rules = _owed(_suspension_record(item))
        try:
            if scheduling_db is not None:
                reminders = reminders + await asyncio.to_thread(run_finite_local_worker,
                    _enabled_linked_reminder_ids, scheduling_db, item_id)
            if plan is not None and plan.applied:
                subscription, rules = False, []  # its source is gone
            elif plan is not None:
                subscription = subscription or plan.owed
                rules = rules + plan.rules
            await asyncio.to_thread(run_finite_local_worker,
                partial(dreams_db.set_tracked_suspension, only_live=True),
                item_id, _record_text(
                    subscription=subscription, reminders=reminders,
                    rules=rules))
            if reminders and scheduling_db is not None:
                await asyncio.to_thread(run_finite_local_worker,
                    _disable_reminders, scheduling_db, reminders)
        except Exception as exc:  # noqa: BLE001 - one failed wrapper stops nothing
            logger.warning(
                "Dreams suspend: switching tracked item {} failed: {}",
                item_id, type(exc).__name__)
            continue
        staged.append((item, subscription, reminders, rules))

    for subscription_id, plan in plans.items():
        if not plan.known or plan.applied:
            continue
        recorded = [
            entry for entry in staged
            if int(entry[0].get("subscription_id") or 0) == subscription_id]
        if plan.owed and not any(entry[1] for entry in recorded):
            # No rider's write-ahead record says "Dreams stopped this"
            # (round 4, A2): the staging step failed for every one of
            # them. A stop nobody recorded is a stop nobody can undo --
            # the next pass would find the source not running, record
            # "subscription: false" and settle, and switching Dreams on
            # would restore nothing. Left running and unsettled; the next
            # pass records first and stops then.
            continue
        rule_ids = {rule_id for entry in recorded for rule_id in entry[3]}
        try:
            for rule_id in sorted(rule_ids):
                try:
                    await subs_service.update_alert_rule(
                        rule_id, enabled=False)
                except KeyError:
                    pass  # the rule was deleted: nothing left to silence
            payload: dict[str, Any] = {}
            if plan.owed:
                payload["active"] = False
            if plan.mark is not None:
                payload["processing_options"] = plan.mark
            if payload:
                await subs_service.update_source(subscription_id, payload)
            if plan.mark is not None:
                await asyncio.to_thread(run_finite_local_worker,
                    dreams_db.set_egress_marked, subscription_id)
            plan.applied = True
        except Exception as exc:  # noqa: BLE001 - its riders stay unsettled
            logger.warning(
                "Dreams suspend: stopping subscription {} failed: {}",
                subscription_id, type(exc).__name__)

    for item, subscription, reminders, rules in staged:
        plan = plans.get(int(item.get("subscription_id") or 0))
        if scheduling_db is None or (plan is not None and not plan.applied):
            continue  # unfinished: a later pass (the next start) retries
        try:
            await asyncio.to_thread(run_finite_local_worker,
                partial(dreams_db.set_tracked_suspension, only_live=True),
                int(item["id"]),
                _record_text(subscription=subscription, reminders=reminders,
                             rules=rules, settled=SETTLED_OFF, at=_now_iso()))
        except Exception as exc:  # noqa: BLE001 - stays unsettled
            logger.warning(
                "Dreams suspend: switching tracked item {} failed: {}",
                item.get("id"), type(exc).__name__)
    return len(staged)


async def _restore_source(
    subs_service: Any, subscription_id: int, *, before_enable: Any = None,
    enable_guard: Any = None, dreams_db: Any = None,
) -> tuple[bool, str | None]:
    """Re-enable one subscription Dreams stopped, if its page is public.

    Re-enabling restarts fetches of a URL Dreams stored, so the same
    question ``track_page`` asks before a re-enable is asked here (review
    of #2921, item 2).

    Args:
        subs_service: The app's ``LocalWatchlistsService``.
        subscription_id: The subscription to re-enable.
        before_enable: Optional coroutine function awaited once the page
            has passed the check and just before the source is switched
            on -- the resume pass un-settles held records there (review of
            #3101, round 2, S1). If it raises, the source stays off.
        enable_guard: Optional zero-arg callable returning an async context
            manager held across ``before_enable`` and the switch-on (a
            re-track passes :func:`_switch_lock`).

    Returns:
        ``(resolved, held)``. ``resolved`` is True when nothing is owed
        any more -- the source runs again, or the user deleted it.
        Otherwise ``held`` carries the policy's reason and the source
        stays disabled.
    """
    try:
        source = await subs_service.get_source(subscription_id)
    except KeyError:
        return True, None  # the user deleted the source
    if dreams_db is not None and not await _still_dreams_source(
            dreams_db, subscription_id, source):
        # Not the source Dreams made any more (review of #3123, M1): as
        # good as deleted, and never switched on.
        return True, None
    # The question is about a STORY address. One the user has since typed
    # over it is theirs (round 4): Watchlists fetches it, private or not,
    # and holding it here as "not public" would hold it for good.
    refusal = None if _is_users_address(source) else (
        await _public_url_refusal(str(source.get("url") or "")))
    if refusal is not None:
        # A held source is not an active one (round 4). It can be running:
        # an earlier resume enabled it and was cut before it could say
        # so, and by now the address is no longer public.
        if bool(source.get("active")) or bool(source.get("paused")):
            await subs_service.update_source(
                subscription_id, {"active": False})
        logger.warning(
            "Dreams resume: subscription {} left disabled, its page is not "
            "a public destination ({})", subscription_id, refusal)
        return False, refusal
    async with (enable_guard() if enable_guard is not None
                else contextlib.nullcontext()):
        if before_enable is not None:
            await before_enable()
        await _enable_dream_subscription(subs_service, subscription_id, source)
    return True, None


async def _resume_pass(
    subs_service: Any, dreams_db: Any, scheduling_db: Any, items: list[dict],
) -> int:
    """Restore what the records say, for every wrapper not settled "on".

    Per wrapper: the recorded reminders are re-armed (those that still
    exist, are still disabled and have not fired) and the recorded alert
    rules re-enabled. Per subscription, once: when a record owes it, it
    is re-enabled after the public-destination check. What could be
    restored leaves the record; an empty record is cleared.

    A subscription whose page is not a public address stays disabled; its
    riders keep the debt with the reason as ``held`` and are marked
    ``settled: "on"`` (round 3, T19) -- the Tracked row shows it, and the
    next switch-on or re-track asks again, not every start (a start
    asks only when it has other reconcile work, and then this pass looks
    every passing hold up too). A hold whose
    reason may pass -- every DNS answer: the name could not be looked up,
    or it resolved to a private or metadata address -- is also taken up
    again by every resume pass that runs, and the daily cycle runs one
    for it (round 4, B2; TASK-34922, :func:`retry_unfinished_resume`);
    only a non-web scheme is not asked about again by itself. A debt that is left only
    because a service could not be reached leaves the wrapper unsettled
    instead, so the next pass does retry it -- unless the watch was held
    already: it stays held for the reason it recorded, through every write
    of a pass that does not finish (TASK-34921).

    Returns:
        How many wrappers the pass worked on.
    """
    pending = [
        item for item in items if _resume_has_work(item.get("suspension"))]
    if not pending:
        return 0
    worked = 0
    # (subscription still owed, reminders still owed, rules still owed)
    left: dict[int, tuple[bool, list[str], list[int]]] = {}
    # Wrappers whose restore failed part-way: the record written last is
    # the truth about them, and the stale row read at pass start is not.
    failed: set[int] = set()
    # Subscriptions whose held riders could not be un-settled, so their
    # source stayed off (review of #3101, round 3, T3).
    unsettle_failed: set[int] = set()
    # Subscriptions whose page passed and whose held riders were
    # un-settled: their old hold reason no longer applies (TASK-34921).
    unsettled: set[int] = set()
    # Dreams rules this pass re-enabled, by subscription (TASK-34919).
    restored_rules: dict[int, set[int]] = {}

    async def write_unsettled(
        item_id: int, subscription: bool, reminders: list[str],
        rules: list[int], held: str | None,
    ) -> None:
        # ``held`` rides along (TASK-34921): until this pass decides the
        # source, the watch is still held for the reason it recorded, and
        # its Tracked row says so even if the pass stops here.
        await asyncio.to_thread(run_finite_local_worker,
            partial(dreams_db.set_tracked_suspension, only_live=True),
            item_id, _record_text(
                subscription=subscription, reminders=reminders,
                rules=rules, held=held))

    async def unsettle_held_riders(subscription_id: int) -> None:
        # Review of #3101, round 2, S1. A held record skipped the
        # write-ahead (N1); its page has now passed the check, so before
        # the source runs again it stops saying "held, settled on". A rider
        # whose own step failed is included (round 3, T1): its skipped
        # write-ahead would have written exactly this, and the closing
        # loop below leaves its record as last written.
        try:
            await _unsettle_held(dreams_db, [
                rider for rider in items
                if int(rider.get("subscription_id") or 0) == subscription_id],
                still_owed=left)
        except Exception:
            unsettle_failed.add(subscription_id)
            raise
        unsettled.add(subscription_id)

    for item in pending:
        item_id = int(item["id"])
        subscription, reminders, rules = _owed(_suspension_record(item))
        held_before = (_suspension_record(item) or {}).get("held")
        try:
            # The list was read when the pass began. A lifecycle sweep (a
            # cycle, a track check) may have retired this watch since, and
            # torn its outside work down; restoring from the stale row
            # would re-arm its reminder and switch its source back on
            # with no live watch left (round 3, T14b). Its record stays
            # with the retired wrapper.
            fresh = await asyncio.to_thread(run_finite_local_worker, dreams_db.get_tracked_item, item_id)
            if fresh is None or fresh.get("status") not in ("active", "paused"):
                continue
            # A hand-over that did not happen when the story's other watch
            # retired (TASK-34923) is finished here: the reminder goes into
            # this watch's record first, then moves, then is re-armed below.
            taken: list[str] = []
            if scheduling_db is not None:
                taken = await asyncio.to_thread(run_finite_local_worker,
                    _orphaned_reminders, scheduling_db, dreams_db, fresh)
                reminders = reminders + taken
            # The record stops saying "settled off" BEFORE anything is
            # restored (round 4, A1), and is rewritten as each restore
            # lands (round 3, T12). A wrapper whose reminder runs again is
            # not suspended: the record used to be rewritten once, after
            # the reminders AND the rules, so a failure in between left
            # work running under a record the next switch-off -- and the
            # boot check -- skip as done. And what has been restored is no
            # longer owed, whatever happens to the rest: a reminder left
            # listed was re-armed by the next resume over a disable the
            # user had made in between. A record settled "on" that owes
            # only its held subscription says nothing of the kind, and
            # nothing is restored before that decision, so it is not
            # rewritten here: a held page asked about again on every cycle
            # trigger would otherwise be written twice each time (review of
            # #3101, N1). It is un-settled only once its page passes the
            # check (``unsettle_held_riders``, round 2, S1).
            if taken or not _settled_hold_only(item.get("suspension")):
                await write_unsettled(
                    item_id, subscription, reminders, rules, held_before)
            if taken:
                await asyncio.to_thread(run_finite_local_worker,
                    _take_over_reminders, scheduling_db, dreams_db, fresh,
                    taken)
            if reminders and scheduling_db is not None:
                await asyncio.to_thread(run_finite_local_worker,
                    _restore_reminders, scheduling_db, reminders)
                reminders = []
                await write_unsettled(
                    item_id, subscription, reminders, rules, held_before)
            if rules and subs_service is not None:
                for rule_id in rules:
                    try:
                        await subs_service.update_alert_rule(
                            rule_id, enabled=True)
                    except KeyError:
                        pass  # the user deleted the rule
                    restored_rules.setdefault(
                        int(item.get("subscription_id") or 0), set()).add(rule_id)
                rules = []
                await write_unsettled(
                    item_id, subscription, reminders, rules, held_before)
        except Exception as exc:  # noqa: BLE001 - one failed wrapper stops nothing
            logger.warning(
                "Dreams resume: switching tracked item {} failed: {}",
                item_id, type(exc).__name__)
            failed.add(item_id)
            continue
        left[item_id] = (subscription, reminders, rules)
        worked += 1

    # One decision per subscription, shared by every rider that owes it.
    outcomes: dict[int, tuple[bool, str | None]] = {}
    for item in pending:
        subscription_id = int(item.get("subscription_id") or 0)
        item_id = int(item["id"])
        # A failed un-settling is that subscription's decision too: its
        # source stays off for this pass (round 3, T3).
        if (item_id not in left or not left[item_id][0]
                or not subscription_id or subscription_id in outcomes
                or subscription_id in unsettle_failed
                or subs_service is None):
            continue
        try:
            outcomes[subscription_id] = await _restore_source(
                subs_service, subscription_id,
                before_enable=partial(unsettle_held_riders, subscription_id),
                dreams_db=dreams_db)
        except Exception as exc:  # noqa: BLE001 - its riders stay unsettled
            logger.warning(
                "Dreams resume: restoring subscription {} failed: {}",
                subscription_id, type(exc).__name__)

    # TASK-34919. A watch can be retired while this pass works on it -- an
    # untrack or a sweep landing during a name lookup -- after the pass
    # read it as live. Its source was still off then, so the retirement
    # left it alone, and later passes read live watches only: the source
    # ran on, its rule enabled, with no watch behind it. Every
    # subscription this pass switched on or re-armed a rule on is looked
    # at once more; with no live watch left, both go back off.
    # ponytail: the riders are read and the rules and source switched off
    # with awaits in between, so a story attaching to the running source in
    # that gap ends on a source this switches off (review of #3108, X5). A
    # second read inside the disable step would only narrow the gap.
    # Tracking that story again restarts it -- the retired watch's record
    # owes the restart (TASK-34920). Closing it needs track_page's attach
    # and this step under one lock.
    touched = set(restored_rules) | {
        subscription_id for subscription_id, (resolved, _held)
        in outcomes.items() if resolved}
    for subscription_id in sorted(touched - {0}):
        try:
            riders = await asyncio.to_thread(run_finite_local_worker,
                dreams_db.list_tracked_for_subscription, subscription_id)
            if any(rider.get("status") in ("active", "paused")
                   for rider in riders):
                continue
            for rule_id in sorted(restored_rules.get(subscription_id, ())):
                try:
                    await subs_service.update_alert_rule(
                        rule_id, enabled=False)
                except KeyError:
                    pass  # the user deleted the rule
            if subscription_id in outcomes:
                await _disable_dream_source(
                    subs_service, subscription_id, dreams_db)
        except Exception as exc:  # noqa: BLE001 - the next pass is no help here
            logger.warning(
                "Dreams resume: re-checking subscription {} failed: {}",
                subscription_id, type(exc).__name__)

    for item in items:
        item_id = int(item["id"])
        record = _suspension_record(item)
        if record is None:
            continue
        subscription_id = int(item.get("subscription_id") or 0)
        outcome = outcomes.get(subscription_id)
        if item_id in left:
            subscription, reminders, rules = left[item_id]
        elif item_id in failed:
            continue  # its record stands as last written, for the retry
        elif outcome is not None:
            # A held rider this pass did not otherwise touch: its
            # subscription was decided through another rider.
            subscription, reminders, rules = _owed(record)
        else:
            continue
        # Undecided -- the service failed or was missing, or un-settling
        # failed (round 3, T3) -- the hold stands for the reason recorded
        # (TASK-34921), not as a bare debt the next start would look up.
        # Only a page that passed the check, its riders un-settled, drops
        # it.
        held: str | None = (
            None if subscription_id in unsettled else record.get("held"))
        if subscription and outcome is not None:
            resolved, held = outcome
            subscription = not resolved
        text: str | None = None
        if subscription or reminders or rules:
            only_held = bool(held) and not reminders and not rules
            text = _record_text(
                subscription=subscription, reminders=reminders, rules=rules,
                held=held, settled=SETTLED_ON if only_held else None)
        # Still held for the same reason: nothing to write (review of
        # #3101, N1). Comparing with the row read at pass start is enough;
        # no "did this pass write it" guard is needed (round 2, S3). A
        # final text equal to that row is either a held record that is
        # still held -- skipped its write-ahead and was never un-settled,
        # or un-settled and put back when un-settling failed (best effort,
        # see ``_unsettle_held``) -- or an unsettled one that owes no
        # reminder or rule (restoring one changes the text, a failed
        # restore skips this loop), whose write-ahead wrote exactly that
        # text: a held reason it carried would have made the final text
        # "settled: on" (TASK-34921). A rider whose page passed never
        # matches: its final text has no hold, and its start row had one.
        if text is not None and json.loads(text) == record:
            continue
        try:
            # ``only_live``: a watch retired since the pass began carries
            # its retirement record, which this pass must not rewrite
            # from the row it read at its start (round 4).
            await asyncio.to_thread(run_finite_local_worker,
                partial(dreams_db.set_tracked_suspension, only_live=True),
                item_id, text)
        except Exception as exc:  # noqa: BLE001 - the record stands for a retry
            logger.warning(
                "Dreams resume: switching tracked item {} failed: {}",
                item_id, type(exc).__name__)
    return worked


async def set_tracking_suspended(
    subs_service: Any,
    dreams_db: Any,
    *,
    suspended: bool,
    scheduling_db_getter: Any = None,
    now: datetime | None = None,
) -> int:
    """Suspend or resume the work Dreams scheduled outside its own loop.

    ``[dreams] enabled`` gates the cycle and the question checks (their
    projection emits nothing while it is off), but two things a watch
    creates live in OTHER queues that never read that flag: a page watch
    is a watchlists subscription the watchlist projection schedules by
    itself, and an event-dated watch owns a reminder task. Switching
    Dreams off therefore has to stop those too, or a disabled Dreams
    keeps fetching pages and notifying.

    Suspending (:func:`_suspend_pass`) walks every tracked item that is
    not retired (active, and paused -- a paused watch keeps its reminder
    armed) and is not settled "off" yet. For each it RECORDS what it is
    about to stop -- the reminders linked to the wrapper that are armed
    right now, the enabled ``Change:`` alert rule Dreams pinned on its
    page (item 3b), and its subscription when that is Dreams' own and
    running -- merged with whatever its record already owed, and then
    stops exactly that through the same seams untrack uses. A
    subscription Dreams merely attached to is the user's and keeps
    running; only Dreams' own alert rule on it goes quiet. No wrapper
    changes lifecycle state.

    Resuming (:func:`_resume_pass`) restores what the records say and
    nothing else (review of #2921, item 4): the recorded reminders are
    re-armed with whatever schedule they now carry, the recorded rules
    re-enabled, and the recorded subscription re-enabled after the same
    public-destination check a re-track makes (item 2). A subscription
    the user had disabled by hand, a reminder they had disabled, and a
    reminder they had rescheduled were never in a record, or are restored
    as they stand -- the switch cannot bring back what it did not stop.
    The lifecycle sweep then runs (a watch whose event passed while
    Dreams was off is retired, with the usual teardown, instead of left
    to nag about a date already gone).

    Both passes are idempotent and resumable: a wrapper whose stops or
    restores did not all land keeps an unsettled record, and the next
    pass in the same direction -- the next start runs one, see
    ``track_reconcile`` -- takes it up again (round 3, T2/T3).

    Degrade, never raise: a missing service or scheduling DB, or one
    failed write, logs and moves on.

    Args:
        subs_service: The app's ``LocalWatchlistsService``, or None (page
            subscriptions and alert rules are then left for a later
            pass).
        dreams_db: The app's ``DreamsDB``.
        suspended: True when Dreams is off, False when it is on.
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` or None (reminders are then left for a
            later pass).
        now: Clock for the resume sweep (UTC-aware); defaults to the
            current time.

    Returns:
        How many wrappers the pass worked on.
    """
    scheduling_db = _scheduling_db(scheduling_db_getter)

    async def live_items() -> list[dict]:
        # Every watch that is not retired: a PAUSED one (the sweep paused
        # its failing checks) still owns an armed event reminder, and a
        # disabled Dreams must not send that either. One thread hop for
        # both reads.
        return await asyncio.to_thread(run_finite_local_worker,
            lambda: [
                *dreams_db.list_tracked_items("active"),
                *dreams_db.list_tracked_items("paused"),
            ]
        )

    if suspended:
        # Never between a re-track's switch check and its switch-on, and
        # the watches are read under the lock too, so one a re-track has
        # just made is among them (review of #3108, round 2, Z1).
        async with _switch_lock():
            return await _suspend_pass(
                subs_service, dreams_db, scheduling_db, await live_items())
    items = await live_items()
    switched = await _resume_pass(
        subs_service, dreams_db, scheduling_db, items)
    # AFTER the restore, so every record is still on a live wrapper when
    # it is read; a watch that died while Dreams was off is then retired
    # with the sweep's own teardown.
    try:
        await sweep_track_lifecycle(
            dreams_db, now=now or datetime.now(UTC),
            subs_service_getter=lambda: subs_service,
            scheduling_db_getter=scheduling_db_getter)
    except Exception as exc:  # noqa: BLE001 - resume degrades, never raises
        logger.warning(
            "Dreams resume: lifecycle sweep failed: {}", type(exc).__name__)
    return switched


#: The reconcile pass in flight, owned HERE (review of #2921, item 3c).
#: A strong reference -- a bare task is only weakly held by the event
#: loop -- and the reason a cancelled caller cannot stop the switch: the
#: caller awaits this task through ``asyncio.shield``. Event-loop-only,
#: like ``_INFLIGHT_TRACK_CHECKS``.
_RECONCILE_TASK: asyncio.Task | None = None

#: The latest request that arrived while a pass was running; the running
#: task takes it up when its pass ends, so overlapping requests coalesce
#: into one more pass instead of running side by side.
_RECONCILE_PENDING: tuple | None = None


async def _reconcile_until_settled(request: tuple) -> int:
    """Run switch passes until no newer request is waiting (the task body).

    Each pass reads ``[dreams] enabled`` as it is at that moment, so the
    watches end where the LATEST switch says whatever order the requests
    arrived in. Nothing escapes: this is the whole body of a task nobody
    may be awaiting any more.
    """
    global _RECONCILE_PENDING
    switched = 0
    settle: Any = None
    while True:
        subs_service, dreams_db, scheduling_db_getter, now, on_settled = request
        settle = on_settled or settle
        try:
            switched += await set_tracking_suspended(
                subs_service, dreams_db,
                suspended=not bool(dreams_setting("enabled")),
                scheduling_db_getter=scheduling_db_getter, now=now)
        except Exception as exc:  # noqa: BLE001 - must never escape the task
            logger.warning(
                "Dreams reconcile: switching tracked watches failed: {}",
                type(exc).__name__)
        request, _RECONCILE_PENDING = _RECONCILE_PENDING, None
        if request is not None:
            continue
        # Only now, with no further pass waiting (round 3, T14c): the
        # callback reloads the scheduler queue, and reloading it between
        # an "off" pass and the "on" pass already queued behind it loaded
        # the first pass's state.
        if settle is not None:
            try:
                settle()
            except Exception as exc:  # noqa: BLE001 - must never escape the task
                logger.warning(
                    "Dreams reconcile: settle callback failed: {}",
                    type(exc).__name__)
        return switched


async def reconcile_tracking(
    subs_service: Any,
    dreams_db: Any,
    *,
    scheduling_db_getter: Any = None,
    now: datetime | None = None,
    on_settled: Any = None,
) -> int:
    """Bring Dream-owned outside work in line with ``[dreams] enabled``.

    The robust way to apply the master switch (review of #2921, item 3c).
    :func:`set_tracking_suspended` is one pass in one direction; this
    decides the direction from the live flag and makes the pass safe to
    request from anywhere:

    * it cannot be cut short. The pass runs as a task this module owns,
      and callers await it shielded, so cancelling a caller -- a Settings
      worker cancelled by another settings action, a screen dismissed
      mid-switch -- leaves the switch running to the end;
    * it never runs twice at once. A request that arrives while a pass is
      running is remembered (the latest one wins) and taken up when the
      pass ends, and every caller is answered when the whole thing has
      settled;
    * it is idempotent. Wrappers already on the right side of the switch
      are skipped (their suspension record says which side they are on),
      so it is equally the startup reconcile for a hand-edited
      ``config.toml`` or a switch that never finished.

    Args:
        subs_service: The app's ``LocalWatchlistsService``, or None.
        dreams_db: The app's ``DreamsDB``.
        scheduling_db_getter: Zero-arg callable returning the app's
            ``ScheduledTasksDB`` or None.
        now: Clock for the resume sweep; defaults to the current time.
        on_settled: Zero-arg callable run once everything has settled --
            after the last pass, not between coalesced ones (the app asks
            its scheduler queue to reload, so work already loaded for a
            suspended watch drops out). Called from the owned task, so it
            still runs when the caller was cancelled.

    Returns:
        How many wrappers were switched by the time everything settled.
    """
    global _RECONCILE_TASK, _RECONCILE_PENDING
    request = (subs_service, dreams_db, scheduling_db_getter, now, on_settled)
    loop = asyncio.get_running_loop()
    task = _RECONCILE_TASK
    # No await between the check and the hand-off: two requests on one
    # loop cannot both start a task. A task left by another (closed) loop
    # is not this loop's to await.
    if task is not None and not task.done() and task.get_loop() is loop:
        _RECONCILE_PENDING = request
    else:
        _RECONCILE_PENDING = None
        task = _RECONCILE_TASK = loop.create_task(
            _reconcile_until_settled(request), name="dreams_track_reconcile")
    return await asyncio.shield(task)


async def stop_restored_leftovers(
    subs_service: Any, scheduling_db: Any,
    source_ids: list[int], reminder_ids: list[str], dreams_db: Any = None,
) -> None:
    """Stop again what a restore brought back for retired watches.

    TASK-34924. With Dreams off the start check (``track_reconcile.
    _restore_drift``) finds a source a retirement record says Dreams
    switched off, running again with no live watch behind it, and a
    reminder such a record lists as disarmed, armed again. No switch pass
    reads a retired watch, so they are switched off here, through the
    same seams the retirement used. The records stay as they are: they
    already say Dreams stopped these.

    Raises:
        Exception: Whatever a service raises; the caller keeps holding
            Dreams' dispatch then.
    """
    for source_id in source_ids:
        if subs_service is not None:
            await _disable_dream_source(
                subs_service, int(source_id), dreams_db)
    if reminder_ids and scheduling_db is not None:
        await asyncio.to_thread(run_finite_local_worker,
            _disable_reminders, scheduling_db, list(reminder_ids))


async def retry_unfinished_resume(
    subs_service: Any,
    dreams_db: Any,
    *,
    scheduling_db_getter: Any = None,
    now: datetime | None = None,
) -> int:
    """The daily cycle's turn at what a resume could not finish.

    Round 4, B2. A tracked page that was left off because its name could
    not be looked up when Dreams was switched on stayed off until the
    user switched Dreams off and on, or tracked it again: every hold was
    settled for good, whatever the reason. Since TASK-34922 the same goes
    for a page held because its name resolved to a private or metadata
    address (a portal or VPN resolver answers every name that way); the
    retry is the same public-address check, so it can never switch on a
    page that still resolves privately. The startup check must not
    look addresses up on every start (round 3, T19), so the cycle does it
    instead, on every cycle run while Dreams is on -- the daily schedule, a
    startup catch-up, and each run the user starts -- because it runs
    before the cycle checks whether the day's collection is already taken:
    when any live watch's record still has work for a resume pass
    (:func:`_resume_has_work`: unfinished, or held for a reason that may
    pass) it asks for one, through :func:`reconcile_tracking` so that it
    cannot run beside a switch the user makes at that moment.

    Cost: one read when there is nothing to do; otherwise one lookup per
    held source on every retry, one after another and with no backoff. A
    rider whose record would come out the same (still held, for the same
    reason) is not rewritten (review of #3101, N1). No scheduler reload is
    asked for: a source this switches back on is picked up by the queue's
    own periodic reload.

    Returns:
        How many wrappers the pass worked on (0 when none was needed).
    """
    records = await asyncio.to_thread(dreams_db.live_tracked_suspensions)
    if not any(_resume_has_work(raw) for raw in records):
        return 0
    return await reconcile_tracking(
        subs_service, dreams_db,
        scheduling_db_getter=scheduling_db_getter, now=now)


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
        TrackDreamsOff: When Dreams is switched off -- raised first.
    """
    _require_dreams_on()
    template = str(query_template)
    if not template.strip():
        raise ValueError("track_question requires a non-empty query_template")

    cadence = max(int(cadence_seconds or 0), _min_interval_seconds())

    # Reuse before anything is written (Qodo #2, PR #2890), same shape as
    # ``track_page``'s branch: the story's existing ACTIVE question
    # wrapper keeps watching; only its cadence/event date refresh. Asked
    # for by mechanism (Qodo #3, PR #2921) so a newer page watch on the
    # same story cannot hide it.
    if origin_story_id is not None:
        wrapper = await _reusable_wrapper(
            dreams_db, origin_story_id, _QUESTION_MECHANISM)
        if wrapper is not None:
            await _refresh_wrapper_plan(
                dreams_db, wrapper, cadence=cadence,
                event_date=event_date,
                scheduling_db_getter=scheduling_db_getter)
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
         "event_date": event_date, "origin_story_id": origin_story_id},
        dreams_db=dreams_db,
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
        return template.format(region=dreams_region())
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


def _reserve_search(
    dreams_db: Any, local_date: str, *, max_searches: int, max_llm_calls: int
) -> bool:
    """Take the check's one search slot, or refuse when a budget is spent.

    One thread hop for the budget guard (Qodo #11, PR #2921). The LLM
    budget is only READ here -- a check with no judge budget left would
    search for nothing -- while the search slot is RESERVED atomically:
    checks of different items run concurrently, and a read followed by a
    bump after the search let two of them spend the same last slot.

    Returns:
        True when the search may run (its slot is already counted).
    """
    if int(dreams_db.usage_get(local_date)["llm_calls"]) >= max_llm_calls:
        return False
    return dreams_db.usage_reserve(
        local_date, counter="searches", limit=max_searches) == 1


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


async def _dispatch_changed(
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

    Off the event loop (Qodo #7, PR #2921): ``dispatch`` reads the
    notification settings and writes the inbox row through SQLite, the
    same blocking shape as every other stage of a check, so it runs under
    ``asyncio.to_thread`` -- but only when the dispatcher's store is
    positively file-backed (``store.is_memory_db is False``). The
    notifications store holds one connection per thread on disk and a
    single loop-thread connection for ``:memory:`` (the app's fallback
    when the file cannot open), which sqlite refuses from a worker
    thread; an unknown store shape stays inline for the same reason
    (``Home/active_work_adapter``'s confirmed-file-backed rule).

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
    if (dispatcher is None or not dreams_setting("enabled")
            or not dreams_execution_allowed(deps.dreams_db)):
        return False
    payload = {
        "category": "dreams_track",
        "title": _label(f"Tracked update: {query_template[:50]}"),
        "message": note,
        "severity": "information",
        "source_entity_kind": "dream_tracked_item",
        "source_entity_id": str(tracked_item_id),
    }
    file_backed = getattr(
        getattr(dispatcher, "store", None), "is_memory_db", None) is False
    try:
        if file_backed:
            await asyncio.to_thread(
                discovery.guard_invocation(dispatcher.dispatch, lambda: bool(dreams_setting("enabled")) and dreams_execution_allowed(deps.dreams_db)),
                **payload)
        else:
            dispatcher.dispatch(**payload)
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
        executing (nothing spent, nothing recorded), or ``disabled`` when
        ``[dreams] enabled`` is off -- at the start, or by the time the
        check reached its search, its judge call, or its notification.
        A check stopped that way records no run: the watch stays due and
        its comparison anchor is untouched, so nothing it had found is
        lost once Dreams is back on.
    """
    # The master switch binds the check itself (review of #2921, item
    # 3a): the projection emits no check while Dreams is off, but a check
    # already dispatched would otherwise still search, judge and notify.
    if not dreams_setting("enabled") or not dreams_execution_allowed(deps.dreams_db):
        return {"status": "disabled", "notified": False}
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
    counter is exhausted. Each spend is RESERVED before the work it pays
    for (``DreamsDB.usage_reserve``: a read and a write under one
    ``BEGIN IMMEDIATE``): the search
    slot by the guard itself, the judge slot immediately before the call
    -- so an ATTEMPTED search or judge call is counted whatever its
    outcome (the attempted-call accounting ``discovery.run_queries``
    applies), and concurrent checks of different items can never both
    spend the day's last slot (Qodo #11, PR #2921). A judge slot that
    went to another spender mid-check is a ``skipped`` "budget" run.

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

    disabled = {"status": "disabled", "notified": False}
    halted = False

    def switched_off() -> bool:
        nonlocal halted
        halted = halted or not dreams_setting("enabled") or not dreams_execution_allowed(deps.dreams_db)
        return halted

    if switched_off():
        # Switched off since this check was dispatched (item 3a): the
        # flag is re-read before each spend and before the notification.
        return disabled

    # Budget guard FIRST: an exhausted day skips before the search and
    # the judge -- spending nothing is the point. The guard RESERVES the
    # search slot (Qodo #11, PR #2921) rather than reading the counter
    # and bumping it after the work: checks of different items run side
    # by side, and two of them could both read "one slot left".
    max_searches = int(dreams_setting("max_searches_per_day"))
    max_llm_calls = int(dreams_setting("max_llm_calls_per_day"))
    if not await asyncio.to_thread(
            _reserve_search, dreams_db, local_date,
            max_searches=max_searches, max_llm_calls=max_llm_calls):
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="skipped", digest=None, note="budget", notified=0,
            now_iso=to_utc_iso(now))
        return {"status": "skipped", "notified": False}

    if switched_off():
        return disabled

    # Question synthesis uses NO LLM: the template IS the query.
    # ``date_range=None`` deliberately overrides ``run_queries``'s cycle
    # default of ``"m"`` (ruling P8): a track check compares a watch's
    # CURRENT result set against its anchor, and a long-lived watch whose
    # engine results fall outside a one-month recency window would return
    # empty forever -- perpetual ``withheld``, never anchored, never
    # notified. Change detection must see the unfiltered result set.
    query = _render_query(query_template)
    results, _searches_used = await discovery.run_queries(
        discovery.guard_invocation(deps.perform_search, lambda: not switched_off()),
        engine=str(dreams_setting("search_engine")),
        queries=[query], result_count=_TRACK_CHECK_RESULT_COUNT,
        date_range=None)

    if switched_off():
        return disabled
    if not results:
        await asyncio.to_thread(
            _record_run, dreams_db, tracked_item_id,
            status="withheld", digest=None, note="no results", notified=0,
            now_iso=to_utc_iso(now))
        return {"status": "withheld", "notified": False}

    digest = _results_digest(results)
    # The anchor is read directly (Qodo #12, PR #2921), however many
    # non-anchoring runs came after it.
    baseline = await asyncio.to_thread(
        dreams_db.latest_anchor_digest, tracked_item_id)
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
    # fail without any spend (nothing was invoked); the call's slot is
    # RESERVED immediately before it is attempted, so the budget is spent
    # either way from there and two concurrent checks cannot both take
    # the day's last call (Qodo #11, PR #2921).
    if switched_off():
        return disabled
    verdict: tuple[bool, str] | None = None
    chat: Any = None
    try:
        chat = await asyncio.to_thread(deps.chat_getter)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider resolution degrades
        logger.warning("Dreams track check: chat resolution failed: {}",
                       type(exc).__name__)
    if switched_off():
        return disabled
    if chat is not None:
        if not await asyncio.to_thread(
                dreams_db.usage_reserve, local_date,
                counter="llm_calls", limit=max_llm_calls):
            # Another spender took the last call while this check was
            # searching. A non-anchoring skip: the comparison anchor is
            # untouched, so the next check judges this same change.
            await asyncio.to_thread(
                _record_run, dreams_db, tracked_item_id,
                status="skipped", digest=None, note="budget", notified=0,
                now_iso=to_utc_iso(now))
            return {"status": "skipped", "notified": False}
        if switched_off():
            return disabled
        try:
            response = await asyncio.to_thread(
                discovery.guard_invocation(chat, lambda: not switched_off()),
                **_judge_call_kwargs(query, results))
            verdict = _parse_verdict(extract_response_content(response))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - judge call degrades
            logger.warning("Dreams track judge call failed: {}",
                           type(exc).__name__)
    if switched_off():
        return disabled
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

    if switched_off():
        # No "Tracked update" from a disabled Dreams, and no ``changed``
        # run either: recording it would move the anchor past a change
        # the user was never told about.
        return disabled
    delivered = await _dispatch_changed(
        deps, query_template=query_template, note=note,
        tracked_item_id=tracked_item_id)
    if not delivered and switched_off():
        return disabled
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
    digest = dreams_db.latest_anchor_digest(tracked_item_id)
    if digest is None:
        return None
    return dreams_db.insert_track_run(
        tracked_item_id, status="rebaselined", digest_hash=digest,
        verdict_note="", notified=0)
