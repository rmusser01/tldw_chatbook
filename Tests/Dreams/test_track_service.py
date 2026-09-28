# Tests/Dreams/test_track_service.py
"""``Dreams.track_service`` -- the page track mechanism (Phase 2 Task 3).

Fixture construction mirrors ``Tests/Subscriptions/test_local_watchlists_
service.py``'s real-DB pattern: a real ``SubscriptionsDB(tmp_path)`` behind a
real ``LocalWatchlistsService`` (with ``ClientNotificationsDB`` +
``NotificationDispatchService`` wired, as production does), plus a real
file-backed ``DreamsDB`` (schema v2 tracked items).

Deactivation seam (recorded per the task brief): untrack DISABLES a
dream-created subscription through
``LocalWatchlistsService.update_source(subscription_id, {"active": False})``,
the existing service method that routes into ``SubscriptionsDB.
update_subscription``'s allowlisted ``is_active`` field -- no raw SQL, no new
service method. Attached (not dream-created) subscriptions are never touched.
"""
import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    NotificationDispatchService,
)
from tldw_chatbook.Subscriptions import LocalWatchlistsService
from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
    WatchlistBundleService,
)

from tldw_chatbook.Dreams.track_service import (
    TrackCapReached,
    TrackSourceDisabled,
    track_page,
    untrack,
)

_URL = "https://example.com/flights"
_MIN_INTERVAL_SECONDS = 12 * 3600


@pytest.fixture(autouse=True)
def _subs_config_defaults(monkeypatch):
    """Keep the guarded config loader out of the sandboxed test.

    ``SubscriptionsDB.add_subscription`` (reached via ``create_source``)
    freezes the ``subscriptions.auto_pause_after_failures`` default through
    the guarded config loader, which fails closed under the per-test
    TLDW_CONFIG_PATH redirect (same admission signature as the pre-existing
    reds in ``Tests/Subscriptions/test_local_watchlists_service.py``; see
    Tests/conftest.py TASK-32873 notes). Serving the production default
    directly keeps these tests hermetic without the bootstrap profile.
    """
    monkeypatch.setattr(
        "tldw_chatbook.DB.Subscriptions_DB.get_cli_setting",
        lambda section, key, default=None: default,
    )


@pytest.fixture()
def dreams_db(tmp_path):
    database = DreamsDB(tmp_path / "dreams.sqlite", "track-test")
    yield database
    database.close()


@pytest.fixture()
def settings(monkeypatch):
    """Deterministic [dreams] settings; Task 6 keys flow via fallbacks."""
    overrides: dict = {}

    def get(section, key, default=None):
        if section == "dreams" and key in overrides:
            return overrides[key]
        return default

    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting", get)
    return overrides


@pytest.fixture()
def subs_stack(tmp_path):
    """Real SubscriptionsDB + LocalWatchlistsService (notifications wired).

    The teardown closes what these tests open, keeping the session's
    open-fd accounting clean (Tests/conftest.py's fd-growth warning).
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    dispatcher = NotificationDispatchService(store=store)
    service = LocalWatchlistsService(
        db_factory=lambda: db, notification_dispatcher=dispatcher
    )
    try:
        yield db, service
    finally:
        db.close()
        store.close()


def _count_rows(db, table: str, where: str = "1=1", params: tuple = ()) -> int:
    return int(
        db.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {where}", params
        ).fetchone()[0]
    )


# --- created path -------------------------------------------------------------


@pytest.mark.asyncio
async def test_track_page_creates_subscription_watchlist_alert_and_tracked_item(
    tmp_path, dreams_db, settings, subs_stack
):
    subs_db, service = subs_stack

    result = await track_page(
        service,
        dreams_db,
        url=_URL,
        title="Cheap flights to Japan",
        intent="deal",
        event_date="2026-10-01",
        origin_story_id=7,
    )

    assert result["outcome"] == "created"
    subscription_id = result["subscription_id"]

    # Subscription row: type 'url', active, cadence >= the min interval.
    subscription = subs_db.get_subscription(subscription_id)
    assert subscription is not None
    assert subscription["type"] == "url"
    assert subscription["source"] == _URL
    assert int(subscription["is_active"]) == 1
    assert int(subscription["check_frequency"]) >= _MIN_INTERVAL_SECONDS

    # Watchlist membership: "Dreams Tracked" carries the source.
    (watchlist,) = WatchlistBundleService(subs_db).list_watchlists()
    assert watchlist["name"] == "Dreams Tracked"
    assert result["watchlist_id"] == watchlist["id"]
    member_ids = [
        row["id"]
        for row in WatchlistBundleService(subs_db).list_source_rows(watchlist["id"])
    ]
    assert subscription_id in member_ids

    # Alert rule pinned to the subscription: any new item notifies.
    rules = await service.list_alert_rules(job_id=subscription_id)
    assert [rule["condition_type"] for rule in rules] == ["items_above"]
    assert rules[0]["condition_value"] == {"threshold": 0}
    assert rules[0]["severity"] == "information"
    assert rules[0]["name"].startswith("Change: ")

    # Tracked item row: page mechanism, dream-created, active.
    item = dreams_db.get_tracked_item(result["tracked_item_id"])
    assert item is not None
    assert item["mechanism"] == "page"
    assert item["intent"] == "deal"
    assert item["subscription_id"] == subscription_id
    assert item["origin_story_id"] == 7
    assert item["event_date"] == "2026-10-01"
    assert item["created_by_dreams"] == 1
    assert item["status"] == "active"
    assert item["cadence_seconds"] >= _MIN_INTERVAL_SECONDS


# --- attach path --------------------------------------------------------------


@pytest.mark.asyncio
async def test_track_page_attaches_to_existing_subscription_without_duplicate(
    tmp_path, dreams_db, settings, subs_stack
):
    subs_db, service = subs_stack
    existing = await service.create_source(
        {"name": "User's own feed", "url": _URL, "source_type": "url"}
    )
    count_before = _count_rows(subs_db, "subscriptions")

    result = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="topic"
    )

    assert result["outcome"] == "attached"
    assert result["subscription_id"] == existing["source_id"]
    assert _count_rows(subs_db, "subscriptions") == count_before, (
        "attaching must never duplicate the subscription"
    )
    item = dreams_db.get_tracked_item(result["tracked_item_id"])
    assert item["created_by_dreams"] == 0, "an adopted subscription is not dream-owned"
    # The adopted source still joins the watchlist and gets its change alert.
    member_ids = [
        row["id"]
        for row in WatchlistBundleService(subs_db).list_source_rows(
            result["watchlist_id"]
        )
    ]
    assert existing["source_id"] in member_ids
    rules = await service.list_alert_rules(job_id=existing["source_id"])
    assert [rule["condition_type"] for rule in rules] == ["items_above"]


# --- cap guard (FIRST -- before any creation) ---------------------------------


@pytest.mark.asyncio
async def test_track_page_cap_guard_blocks_before_any_creation(
    tmp_path, dreams_db, settings, subs_stack
):
    settings["tracked_item_cap"] = 2
    subs_db, service = subs_stack
    dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600
    )
    dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600
    )

    with pytest.raises(TrackCapReached) as excinfo:
        await track_page(
            service, dreams_db, url=_URL, title="Cheap flights", intent="deal"
        )

    assert excinfo.value.reason_code == "track_cap_reached"
    # Guard-first: nothing was created anywhere.
    assert _count_rows(subs_db, "subscriptions") == 0, (
        "the cap guard must fire before the subscription is created"
    )
    assert _count_rows(subs_db, "watchlists") == 0
    assert _count_rows(subs_db, "local_watchlist_alert_rules") == 0
    assert dreams_db.count_active_tracked() == 2


@pytest.mark.asyncio
async def test_track_page_cap_counts_only_active_items(
    tmp_path, dreams_db, settings, subs_stack
):
    settings["tracked_item_cap"] = 1
    subs_db, service = subs_stack
    seeded = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600
    )
    dreams_db.set_tracked_status(seeded, "retired", retired_reason="manual")

    result = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal"
    )

    assert result["outcome"] == "created", "retired items do not consume cap"


# --- cadence clamping ---------------------------------------------------------


@pytest.mark.asyncio
async def test_track_page_clamps_cadence_to_min_interval(
    tmp_path, dreams_db, settings, subs_stack
):
    subs_db, service = subs_stack

    clamped = await track_page(
        service,
        dreams_db,
        url="https://example.com/a",
        title="A",
        intent="topic",
        cadence_seconds=600,  # 10 minutes: below the 12-hour floor
    )
    assert clamped["outcome"] == "created"
    subscription = subs_db.get_subscription(clamped["subscription_id"])
    assert int(subscription["check_frequency"]) == _MIN_INTERVAL_SECONDS
    assert (
        dreams_db.get_tracked_item(clamped["tracked_item_id"])["cadence_seconds"]
        == _MIN_INTERVAL_SECONDS
    )

    respected = await track_page(
        service,
        dreams_db,
        url="https://example.com/b",
        title="B",
        intent="topic",
        cadence_seconds=86400,  # 24 hours: above the floor, passes through
    )
    subscription_b = subs_db.get_subscription(respected["subscription_id"])
    assert int(subscription_b["check_frequency"]) == 86400
    assert (
        dreams_db.get_tracked_item(respected["tracked_item_id"])["cadence_seconds"]
        == 86400
    )


# --- untrack ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_untrack_dream_created_retires_wrapper_and_disables_subscription(
    tmp_path, dreams_db, settings, subs_stack
):
    subs_db, service = subs_stack
    tracked = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal"
    )

    outcome = await untrack(service, dreams_db, tracked["tracked_item_id"])

    item = dreams_db.get_tracked_item(tracked["tracked_item_id"])
    assert item["status"] == "retired"
    assert item["retired_reason"] == "manual"
    assert outcome["subscription_disabled"] is True
    # DISABLED, never deleted (controller ruling P2): the subscription row,
    # its watchlist membership, and its alert rule all survive deactivation.
    subscription = subs_db.get_subscription(tracked["subscription_id"])
    assert subscription is not None, "untrack must never delete the subscription"
    assert int(subscription["is_active"]) == 0
    member_ids = [
        row["id"]
        for row in WatchlistBundleService(subs_db).list_source_rows(
            tracked["watchlist_id"]
        )
    ]
    assert tracked["subscription_id"] in member_ids, "membership survives untrack"
    rules = await service.list_alert_rules(job_id=tracked["subscription_id"])
    assert rules, "alert rules survive untrack"


@pytest.mark.asyncio
async def test_untrack_attached_leaves_subscription_active(
    tmp_path, dreams_db, settings, subs_stack
):
    subs_db, service = subs_stack
    existing = await service.create_source(
        {"name": "User's own feed", "url": _URL, "source_type": "url"}
    )
    tracked = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="topic"
    )
    assert tracked["outcome"] == "attached"

    outcome = await untrack(service, dreams_db, tracked["tracked_item_id"])

    item = dreams_db.get_tracked_item(tracked["tracked_item_id"])
    assert item["status"] == "retired"
    assert outcome["subscription_disabled"] is False
    subscription = subs_db.get_subscription(existing["source_id"])
    assert int(subscription["is_active"]) == 1, (
        "an attached (not dream-created) subscription is NEVER touched"
    )


# --- re-track after untrack (fix round 1, ruling P7) ---------------------------


@pytest.mark.asyncio
async def test_retrack_after_untrack_reenables_dream_disabled_source(
    tmp_path, dreams_db, settings, subs_stack
):
    """untrack disables the dream-created source; re-tracking must re-enable.

    ``find_source_id_by_url`` matches on URL alone (no is_active filter), so
    without the P7 re-enable the re-track would attach to a still-disabled
    subscription: a dead watch wearing an active tracked item and an
    untruthful "attached" notice.
    """
    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal"
    )
    assert first["outcome"] == "created"
    await untrack(service, dreams_db, first["tracked_item_id"])
    assert int(
        subs_db.get_subscription(first["subscription_id"])["is_active"]
    ) == 0, "fixture: the prior untrack disabled the dream-created source"

    second = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal"
    )

    assert second["outcome"] == "re-enabled"
    assert second["subscription_id"] == first["subscription_id"], (
        "re-track adopts the same subscription, never a duplicate"
    )
    assert second["watchlist_id"] == first["watchlist_id"]
    subscription = subs_db.get_subscription(second["subscription_id"])
    assert int(subscription["is_active"]) == 1, (
        "the source we disabled on untrack is active again"
    )
    active = dreams_db.list_tracked_items()
    assert len(active) == 1, "exactly one active tracked item after re-track"
    assert active[0]["id"] == second["tracked_item_id"]
    assert active[0]["created_by_dreams"] == 1, (
        "a re-enabled dream source stays dream-owned (untrack disables again)"
    )
    assert _count_rows(subs_db, "subscriptions") == 1
    member_ids = [
        row["id"]
        for row in WatchlistBundleService(subs_db).list_source_rows(
            second["watchlist_id"]
        )
    ]
    assert second["subscription_id"] in member_ids, "membership present"
    rules = await service.list_alert_rules(job_id=second["subscription_id"])
    assert rules, "alert rule present"


@pytest.mark.asyncio
async def test_retrack_refuses_foreign_disabled_source(
    tmp_path, dreams_db, settings, subs_stack
):
    """A disabled source with NO dream provenance is refused, not adopted.

    The user disabled this source themselves; re-track must neither re-enable
    it behind their back nor record an active tracked item against a dead
    watch.
    """
    subs_db, service = subs_stack
    existing = await service.create_source(
        {"name": "User's own feed", "url": _URL, "source_type": "url",
         "active": False}
    )
    assert int(
        subs_db.get_subscription(existing["source_id"])["is_active"]
    ) == 0, "fixture: the foreign source starts disabled"

    with pytest.raises(TrackSourceDisabled) as excinfo:
        await track_page(
            service, dreams_db, url=_URL, title="Cheap flights", intent="topic"
        )

    assert excinfo.value.reason_code == "track_source_disabled"
    # Refused BEFORE any write: no tracked row, no watchlist, no alert rule,
    # and the foreign source stays exactly as disabled as the user left it.
    assert dreams_db.list_tracked_items() == []
    assert int(
        subs_db.get_subscription(existing["source_id"])["is_active"]
    ) == 0
    assert _count_rows(subs_db, "watchlists") == 0
    assert _count_rows(subs_db, "local_watchlist_alert_rules") == 0
