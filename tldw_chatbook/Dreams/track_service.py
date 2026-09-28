"""Dreams track mechanism: turn one story page into a watched subscription.

Phase 2 (Track) Task 3. ``track_page`` is the "track this page" path behind
the story modal's ``t`` key: attach-or-create a watchlists subscription for
the story URL, join it to the shared ``Dreams Tracked`` watchlist, pin an
``items_above 0`` alert rule so any new item on the page notifies the shared
inbox, and wrap all of it in one ``dream_tracked_items`` row (schema v2).
``untrack`` retires that wrapper -- dream-created subscriptions are DISABLED
(``is_active = 0``), never deleted: deletion would cascade the watchlist
membership and alert rules a user may have adopted, and an attached
(not dream-created) subscription is never touched at all.

Re-track after untrack (ruling P7, fix round 1): the URL lookup ignores
``is_active``, so a found source may be one a prior untrack disabled. When
it is (a retired, dream-created tracked row pins the provenance), the
re-track RE-ENABLES it and reports ``"re-enabled"``; a disabled source with
no dream provenance is refused with ``TrackSourceDisabled`` -- never
adopted, never re-enabled behind the user's back.

Scheduling needs no registration here: ``WatchlistProjection`` fabricates
``watchlist:<subscription_id>`` tasks straight from subscription rows, so
creating the subscription IS the registration.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from .settings import dreams_setting

#: Fallbacks until Task 6 adds the ``[dreams]`` keys; ``dreams_setting``
#: keeps reading them live so the future keys win without a change here.
_DEFAULT_TRACKED_ITEM_CAP = 20
_DEFAULT_MIN_CHECK_INTERVAL_HOURS = 12

#: Every Dreams-tracked page joins this one shared watchlist.
_TRACK_WATCHLIST_NAME = "Dreams Tracked"

#: ``dream_tracked_items.mechanism`` for the page path (schema vocabulary).
_PAGE_MECHANISM = "page"


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
