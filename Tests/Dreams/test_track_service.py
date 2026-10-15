# Tests/Dreams/test_track_service.py
"""``Dreams.track_service`` -- the page track mechanism (Phase 2 Task 3)
plus the question mechanism and judged change checks (Phase 2 Task 4).

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
import asyncio
import hashlib
import json
import threading
from datetime import UTC, datetime

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Dreams.cycle_service import CycleDeps
from tldw_chatbook.Dreams.discovery import normalize_url
from tldw_chatbook.Dreams.track_service import (
    TrackCapReached,
    TrackInvalidURL,
    TrackPrivateURL,
    TrackSourceDisabled,
    promote_to_reminder,
    rebaseline_track,
    retry_unfinished_resume,
    run_track_check,
    set_tracking_suspended,
    sweep_track_lifecycle,
    track_page,
    track_question,
    untrack,
)
from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    NotificationDispatchService,
)
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Subscriptions import LocalWatchlistsService
from tldw_chatbook.Utils.egress import UrlProvenance
from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
    WatchlistBundleService,
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


@pytest.fixture(autouse=True)
def _egress_policy_seams(monkeypatch):
    """Serve the egress policy's two outside reads without leaving the box.

    ``track_page`` asks the app-wide egress policy whether a story URL is a
    public destination before it creates (or re-enables) a subscription
    (Qodo #1, PR #2921). That reads ``[web_security]`` through the guarded
    config loader (same fail-closed admission as the fixture above) and
    resolves the host. Defaults are served directly and every hostname
    resolves to one public address through the module's documented
    ``_resolve_async`` seam; tests about private destinations override it.
    """
    from tldw_chatbook.Utils import egress

    monkeypatch.setattr(
        egress, "get_cli_setting", lambda section, key, default=None: default)

    async def _public(_host):
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", _public)


@pytest.fixture()
def dreams_db(tmp_path):
    database = DreamsDB(tmp_path / "dreams.sqlite", "track-test")
    yield database
    database.close_all()  # every to_thread hop opened its own connection


@pytest.fixture()
def settings(monkeypatch):
    """Deterministic [dreams] settings; Task 6 keys flow via fallbacks.

    ``enabled`` starts True: a track check refuses to run -- and stops a
    run in flight -- while the master switch is off (review of #2921,
    item 3a), so the shipped default (off) would make every check test a
    no-op. Tests about the switch set it explicitly.
    """
    overrides: dict = {"enabled": True}

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


@pytest.fixture()
def scheduling_db(tmp_path):
    """A real ScheduledTasksDB: the reminder seam Task 5 promotes into."""
    db = ScheduledTasksDB(tmp_path / "scheduled.db", "track-test")
    yield db
    db.close()


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


# --- question mechanism + judged change checks (Phase 2 Task 4) ---------------
#
# ``run_track_check`` runs against a real file-backed ``DreamsDB`` with every
# external collaborator faked: the ``perform_websearch`` seam (result sets are
# swapped between checks to move the digest), the ``chat_api_call`` seam
# (OpenAI-shaped reply), and the notification dispatcher (a recorder). ``now``
# is fixed so the daily-budget date bucket is deterministic on any machine.


NOW = datetime(2026, 9, 22, 8, 0, tzinfo=UTC)

_TEMPLATE = "flights to {region}"

_SET_A = (
    ("Flights round-up", "https://example.com/flights/roundup",
     "usual round-up, nothing new"),
    ("Deal thread", "https://example.com/deals/thread", "prices unchanged"),
)
_SET_B = (
    ("Flights round-up -- new dates", "https://example.com/flights/roundup",
     "dates moved to Nov"),
    ("New deal thread", "https://example.com/deals/new-thread",
     "sale announced"),
)


def _today() -> str:
    return NOW.astimezone().strftime("%Y-%m-%d")


def _expected_digest(results) -> str:
    """The digest of a result set under the production rule (Qodo #3):
    normalized URL + title + whitespace-collapsed, 200-char-capped
    snippet per result, newline-joined, SHA-256."""
    from tldw_chatbook.Dreams.track_service import _snippet_component

    return hashlib.sha256("\n".join(
        normalize_url(url) + title + _snippet_component(snippet)
        for title, url, snippet in results
    ).encode()).hexdigest()


class _FakeSearch:
    """``perform_websearch``-shaped fake whose result set can be swapped."""

    def __init__(self, results=()):
        self.results = list(results)
        self.calls: list[tuple] = []

    def __call__(self, engine, query, **kwargs):
        self.calls.append((engine, query, kwargs))
        return {"results": [
            {"title": title, "url": url, "content": snippet}
            for title, url, snippet in self.results
        ]}


class _FakeChat:
    """``chat_api_call``-shaped fake recording every call's kwargs."""

    def __init__(self, content='{"changed": true, "note": "new dates announced"}',
                 raises=False):
        self.content = content
        self.raises = raises
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("judge exploded")
        return {"choices": [{"message": {"content": self.content}}]}


class _DispatchRecorder:
    """``NotificationDispatchService.dispatch``-shaped recorder."""

    def __init__(self, raises=False):
        self.raises = raises
        self.dispatched: list[dict] = []

    def dispatch(self, **kwargs):
        self.dispatched.append(kwargs)
        if self.raises:
            raise RuntimeError("dispatch failed")

    def __call__(self):  # usable directly as a dispatch_getter
        return self


def _track_deps(dreams_db, *, perform_search, chat=None, dispatch_getter=None):
    return CycleDeps(
        dreams_db=dreams_db,
        chachanotes_db_getter=lambda: None,
        media_db_getter=lambda: None,
        subs_db_getter=lambda: None,
        pc_service_getter=lambda: None,
        chat_getter=lambda: chat if chat is not None else _FakeChat(),
        perform_search=perform_search,
        now=lambda: NOW,
        dispatch_getter=dispatch_getter,
    )


# --- track_question -----------------------------------------------------------


@pytest.mark.asyncio
async def test_track_question_creates_active_question_wrapper_row(
        dreams_db, settings):
    item_id = await track_question(
        dreams_db,
        query_template=_TEMPLATE,
        intent="deal",
        event_date="2026-10-01",
        origin_story_id=3,
        cadence_seconds=600,  # below the 12-hour floor
    )

    item = dreams_db.get_tracked_item(item_id)
    assert item is not None
    assert item["mechanism"] == "question"
    assert item["intent"] == "deal"
    assert item["query_template"] == _TEMPLATE
    assert item["subscription_id"] is None, (
        "a question watch has no page subscription"
    )
    assert item["origin_story_id"] == 3
    assert item["event_date"] == "2026-10-01"
    assert item["status"] == "active"
    assert item["cadence_seconds"] == _MIN_INTERVAL_SECONDS


@pytest.mark.asyncio
async def test_track_question_cap_guard_blocks_before_any_row(
        dreams_db, settings):
    settings["tracked_item_cap"] = 1
    dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)

    with pytest.raises(TrackCapReached):
        await track_question(
            dreams_db, query_template=_TEMPLATE, intent="topic")

    assert dreams_db.count_active_tracked() == 1


@pytest.mark.asyncio
async def test_track_question_requires_non_empty_template(dreams_db, settings):
    with pytest.raises(ValueError):
        await track_question(
            dreams_db, query_template="   ", intent="topic")
    assert dreams_db.list_tracked_items() == []


# --- run_track_check: baseline / unchanged / changed --------------------------


@pytest.mark.asyncio
async def test_track_check_first_run_is_baseline_notified_zero(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat()
    recorder = _DispatchRecorder()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=recorder)

    result = await run_track_check(deps, item_id)

    assert result == {"status": "baseline", "notified": False}
    # The template IS the query: rendered with the configured region, no LLM.
    assert search.calls[0][1] == "flights to kyoto"
    assert search.calls[0][0] == "duckduckgo"
    assert search.calls[0][2]["result_count"] == 5
    assert chat.calls == [], "the first check never invokes the judge"
    assert recorder.dispatched == []
    runs = dreams_db.list_recent_track_runs(item_id)
    assert [run["status"] for run in runs] == ["baseline"]
    assert runs[0]["digest_hash"] == _expected_digest(_SET_A)
    assert runs[0]["notified"] == 0
    assert dreams_db.get_tracked_item(item_id)["last_checked"] is not None
    assert dreams_db.usage_get(_today()) == {"searches": 1, "llm_calls": 0}


@pytest.mark.asyncio
async def test_track_check_search_is_not_recency_filtered(
        dreams_db, settings):
    """Fix round 1, ruling P8: the check search passes ``date_range=None``.

    ``run_queries`` defaults to ``date_range="m"`` (the cycle's recency
    promise); a track check on a long-lived watch must NOT inherit it --
    an engine whose results fall outside a one-month window would return
    empty forever, the check would read ``withheld`` forever, and change
    detection would silently die without ever anchoring or notifying.
    """
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    deps = _track_deps(dreams_db, perform_search=search)

    await run_track_check(deps, item_id)

    assert search.calls[0][2]["date_range"] is None, (
        "the check's search must be unfiltered (date_range=None), not the "
        "cycle's one-month recency default"
    )


@pytest.mark.asyncio
async def test_track_check_same_digest_is_unchanged_without_judge(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat()
    recorder = _DispatchRecorder()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=recorder)

    await run_track_check(deps, item_id)
    result = await run_track_check(deps, item_id)

    assert result == {"status": "unchanged", "notified": False}
    assert chat.calls == [], "an identical digest never invokes the judge"
    assert recorder.dispatched == []
    assert [run["status"] for run in
            dreams_db.list_recent_track_runs(item_id)] == ["unchanged",
                                                           "baseline"]


@pytest.mark.asyncio
async def test_track_check_changed_digest_notifies_with_captured_dispatch(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat('{"changed": true, "note": "new dates announced"}')
    recorder = _DispatchRecorder()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=recorder)

    await run_track_check(deps, item_id)
    search.results = _SET_B
    result = await run_track_check(deps, item_id)

    assert result == {"status": "changed", "notified": True}
    assert len(chat.calls) == 1, "exactly one judge call per changed digest"
    assert len(recorder.dispatched) == 1
    call = recorder.dispatched[0]
    assert call["category"] == "dreams_track"
    assert call["title"] == f"Tracked update: {_TEMPLATE[:50]}"
    assert call["message"] == "new dates announced"
    assert call["severity"] == "information"
    assert call["source_entity_kind"] == "dream_tracked_item"
    assert call["source_entity_id"] == str(item_id)

    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "changed"
    assert runs[0]["notified"] == 1
    assert runs[0]["verdict_note"] == "new dates announced"
    assert runs[0]["digest_hash"] == _expected_digest(_SET_B)
    assert dreams_db.usage_get(_today()) == {"searches": 2, "llm_calls": 1}

    # PRIVACY (binding): the judge payload carries ONLY the query, the
    # current top snippets, and the JSON-verdict question -- never stored
    # prior page text (prior state exists only as a digest hash) and never
    # the digest itself.
    prompt = chat.calls[0]["messages_payload"][0]["content"]
    payload = json.loads(prompt)
    assert set(payload) == {"query", "current_results", "question"}
    assert payload["query"] == "flights to kyoto"
    assert [r["title"] for r in payload["current_results"]] == \
        [title for title, _u, _s in _SET_B]
    assert "prices unchanged" not in prompt, "prior snippet must not leak"
    assert _expected_digest(_SET_A) not in prompt
    assert "JSON" in chat.calls[0]["system_message"]


@pytest.mark.asyncio
async def test_track_check_judge_says_unchanged_is_unchanged_run(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat('{"changed": false, "note": "just reordering"}')
    recorder = _DispatchRecorder()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=recorder)

    await run_track_check(deps, item_id)
    search.results = _SET_B
    result = await run_track_check(deps, item_id)

    assert result == {"status": "unchanged", "notified": False}
    assert recorder.dispatched == []
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "unchanged"
    assert runs[0]["verdict_note"] == "just reordering"
    assert runs[0]["notified"] == 0


@pytest.mark.asyncio
async def test_track_check_template_stray_brace_falls_back_to_literal(
        dreams_db, settings):
    settings["region"] = "kyoto"
    template = "deals {oops"
    item_id = await track_question(
        dreams_db, query_template=template, intent="deal")
    search = _FakeSearch(_SET_A)
    deps = _track_deps(dreams_db, perform_search=search)

    await run_track_check(deps, item_id)

    assert search.calls[0][1] == template, (
        "a stray brace must degrade to the literal template, not crash"
    )


# --- run_track_check: budget / withheld / missing ------------------------------


@pytest.mark.asyncio
async def test_track_check_budget_exhausted_skips_spending_nothing(
        dreams_db, settings):
    settings["region"] = "kyoto"
    settings["max_searches_per_day"] = 1
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat()
    recorder = _DispatchRecorder()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=recorder)

    await run_track_check(deps, item_id)  # spends the day's one search
    assert dreams_db.usage_get(_today()) == {"searches": 1, "llm_calls": 0}
    search.results = _SET_B
    result = await run_track_check(deps, item_id)

    assert result == {"status": "skipped", "notified": False}
    assert dreams_db.usage_get(_today()) == {"searches": 1, "llm_calls": 0}, (
        "a budget skip must spend nothing"
    )
    assert len(search.calls) == 1, "no second search was attempted"
    assert chat.calls == []
    assert recorder.dispatched == []
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "skipped"
    assert runs[0]["verdict_note"] == "budget"


@pytest.mark.asyncio
async def test_track_check_llm_budget_exhausted_skips_too(
        dreams_db, settings):
    settings["region"] = "kyoto"
    settings["max_llm_calls_per_day"] = 0
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    deps = _track_deps(dreams_db, perform_search=search)

    result = await run_track_check(deps, item_id)

    assert result == {"status": "skipped", "notified": False}
    assert search.calls == []
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}


@pytest.mark.asyncio
async def test_track_check_search_empty_results_withheld(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch()  # engine returned nothing usable
    chat = _FakeChat()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat)

    result = await run_track_check(deps, item_id)

    assert result == {"status": "withheld", "notified": False}
    assert chat.calls == []
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "withheld"
    assert runs[0]["digest_hash"] is None
    assert dreams_db.usage_get(_today()) == {"searches": 1, "llm_calls": 0}


@pytest.mark.asyncio
async def test_track_check_missing_item_skips_cleanly(dreams_db, settings):
    search = _FakeSearch(_SET_A)

    result = await run_track_check(
        _track_deps(dreams_db, perform_search=search), 424242)

    assert result["status"] == "skipped"
    assert result["notified"] is False
    assert search.calls == []
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}


# --- run_track_check: judge failure dispositions --------------------------------


@pytest.mark.asyncio
async def test_track_check_judge_raise_is_error_run_spending_llm(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat(raises=True)
    recorder = _DispatchRecorder()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=recorder)

    await run_track_check(deps, item_id)
    search.results = _SET_B
    result = await run_track_check(deps, item_id)

    assert result == {"status": "error", "notified": False}
    assert recorder.dispatched == []
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "error"
    assert runs[0]["notified"] == 0
    assert dreams_db.usage_get(_today()) == {"searches": 2, "llm_calls": 1}, (
        "an attempted judge call spends the budget either way"
    )


@pytest.mark.asyncio
async def test_track_check_chat_unresolvable_is_error_run_spending_nothing(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    deps = _track_deps(dreams_db, perform_search=search)
    deps.chat_getter = _raising_chat_getter
    # Seed a baseline so the second check reaches the judge stage.
    await run_track_check(_track_deps(dreams_db, perform_search=search),
                          item_id)
    search.results = _SET_B

    result = await run_track_check(deps, item_id)

    assert result == {"status": "error", "notified": False}
    assert dreams_db.usage_get(_today()) == {"searches": 2, "llm_calls": 0}, (
        "only the two searches; an unresolvable judge spends no llm budget"
    )
    assert dreams_db.list_recent_track_runs(item_id)[0]["status"] == "error"


def _raising_chat_getter():
    raise RuntimeError("Dreams provider/model unavailable")


@pytest.mark.asyncio
async def test_track_check_judge_unparseable_is_error_run(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat("I cannot answer that question.")
    deps = _track_deps(dreams_db, perform_search=search, chat=chat)

    await run_track_check(deps, item_id)
    search.results = _SET_B
    result = await run_track_check(deps, item_id)

    assert result == {"status": "error", "notified": False}
    assert dreams_db.list_recent_track_runs(item_id)[0]["status"] == "error"


@pytest.mark.asyncio
async def test_track_check_tolerates_verdict_wrapped_in_prose(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat(
        'Sure! Here is my verdict: {"changed": true, '
        '"note": "sale announced"} -- hope that helps.')
    recorder = _DispatchRecorder()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=recorder)

    await run_track_check(deps, item_id)
    search.results = _SET_B
    result = await run_track_check(deps, item_id)

    assert result == {"status": "changed", "notified": True}
    assert recorder.dispatched[0]["message"] == "sale announced"


# --- notification degrade paths --------------------------------------------------


@pytest.mark.asyncio
async def test_track_check_dispatch_getter_none_still_writes_changed_run(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=None)

    await run_track_check(deps, item_id)
    search.results = _SET_B
    result = await run_track_check(deps, item_id)

    # Qodo #4 (PR #2890): a run whose alert never reached a dispatcher is
    # ``changed`` but NOT ``notified`` -- the row no longer claims
    # delivery that did not happen.
    assert result == {"status": "changed", "notified": False}
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "changed"
    assert runs[0]["notified"] == 0
    assert "notification delivery degraded" in runs[0]["verdict_note"], (
        "the missing dispatcher is noted, never raised"
    )


@pytest.mark.asyncio
async def test_track_check_dispatch_failure_still_writes_changed_run(
        dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat()
    recorder = _DispatchRecorder(raises=True)
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=recorder)

    await run_track_check(deps, item_id)
    search.results = _SET_B
    result = await run_track_check(deps, item_id)

    assert result == {"status": "changed", "notified": False}, (
        "a failed dispatch must not report the run as notified"
    )
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "changed"
    assert runs[0]["notified"] == 0
    assert "new dates announced" in runs[0]["verdict_note"]
    assert "notification delivery degraded" in runs[0]["verdict_note"]


# --- rebaseline_track ------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebaseline_track_anchors_next_comparison(dreams_db, settings):
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    chat = _FakeChat()
    recorder = _DispatchRecorder()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=recorder)

    await run_track_check(deps, item_id)  # baseline over SET_A
    search.results = _SET_B
    changed = await run_track_check(deps, item_id)  # alerts once
    assert changed["status"] == "changed"

    run_id = rebaseline_track(dreams_db, item_id)
    assert run_id is not None
    assert dreams_db.list_recent_track_runs(item_id)[0]["status"] == \
        "rebaselined"

    again = await run_track_check(deps, item_id)  # SET_B is now the anchor
    assert again == {"status": "unchanged", "notified": False}, (
        "a rebaselined item must not re-alert on the same change"
    )
    assert len(recorder.dispatched) == 1
    assert len(chat.calls) == 1, (
        "the rebaselined identical digest short-circuits before the judge"
    )


def test_rebaseline_track_without_baseline_is_noop(dreams_db):
    item_id = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)
    dreams_db.insert_track_run(item_id, status="skipped", digest_hash=None,
                               verdict_note="budget")

    assert rebaseline_track(dreams_db, item_id) is None
    assert [run["status"] for run in
            dreams_db.list_recent_track_runs(item_id)] == ["skipped"]


# --- event reminders (Phase 2 Task 5) -------------------------------------------
#
# ``promote_to_reminder`` is called by both track paths after the wrapper row
# exists; every test here drives it through those paths (the way production
# reaches it) against a real ``ScheduledTasksDB`` on tmp_path.


@pytest.mark.asyncio
async def test_track_page_with_event_date_promotes_reminder(
        dreams_db, settings, subs_stack, scheduling_db):
    service = subs_stack[1]

    result = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal",
        event_date="2026-10-01",
        scheduling_db_getter=lambda: scheduling_db)

    (task,) = scheduling_db.list_reminder_tasks(owner_id="local")
    assert task["schedule_kind"] == "one_time"
    assert _enabled_run_dates(
        scheduling_db, result["tracked_item_id"]) == ["2026-09-24"], (
        "run_at is the event date minus the 7-day lead, on the local calendar"
    )
    assert task["next_run_at"] == task["run_at"]
    assert task["link_type"] == "dream_tracked_item"
    assert task["link_id"] == str(result["tracked_item_id"]), (
        "the reminder links back to the tracked item"
    )
    assert task["title"] == "Dreams: tracked event", (
        "a page item carries no query template"
    )
    assert task["body"] == "Tracked Dreams event on 2026-10-01"


@pytest.mark.asyncio
async def test_track_question_with_event_date_promotes_reminder(
        dreams_db, settings, scheduling_db):
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal",
        event_date="2026-10-01",
        scheduling_db_getter=lambda: scheduling_db)

    (task,) = scheduling_db.list_reminder_tasks(owner_id="local")
    assert task["link_id"] == str(item_id)
    assert task["title"] == f"Dreams: {_TEMPLATE}"
    assert _enabled_run_dates(scheduling_db, item_id) == ["2026-09-24"]


@pytest.mark.asyncio
async def test_track_without_event_date_promotes_no_reminder(
        dreams_db, settings, scheduling_db):
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="topic",
        scheduling_db_getter=lambda: scheduling_db)

    assert scheduling_db.list_reminder_tasks() == []
    # Directly: no event_date means nothing to promote; a None scheduling DB
    # (wiring order) degrades the same silent way.
    assert await promote_to_reminder(
        lambda: scheduling_db,
        {"id": item_id, "query_template": _TEMPLATE, "event_date": None},
    ) is None
    assert await promote_to_reminder(
        lambda: None,
        {"id": item_id, "query_template": _TEMPLATE, "event_date": "2026-10-01"},
    ) is None


@pytest.mark.asyncio
async def test_track_reminder_failure_never_fails_tracking(
        dreams_db, settings, subs_stack):
    service = subs_stack[1]

    def _broken_getter():
        raise RuntimeError("scheduled-tasks DB unavailable")

    result = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal",
        event_date="2026-10-01",
        scheduling_db_getter=_broken_getter)

    assert result["outcome"] == "created", (
        "a reminder failure must leave the tracking result unchanged"
    )
    assert dreams_db.get_tracked_item(result["tracked_item_id"]) is not None


# --- lifecycle sweep (Phase 2 Task 6) -------------------------------------------
#
# ``sweep_track_lifecycle`` walks every ACTIVE tracked item and applies the
# three retirement/pause rules: an event whose day-after has fully passed
# retires (``event_passed``), ``track_quiet_retire_count`` consecutive
# ``unchanged`` runs retire (``quiet``), and 3 consecutive ``error`` runs
# pause (``failures``). ``NOW`` (2026-09-22 08:00 UTC) is the sweep clock.


def _seed_runs(db, item_id: int, statuses: list[str]) -> None:
    for status in statuses:
        db.insert_track_run(
            item_id, status=status,
            digest_hash=None if status in ("error", "skipped", "withheld")
            else "d" + status,
            verdict_note="", notified=0)


@pytest.mark.asyncio
async def test_sweep_retires_item_whose_event_passed(dreams_db, settings):
    item_id = dreams_db.create_tracked_item(
        mechanism="question", intent="event", event_date="2026-09-20",
        cadence_seconds=3600)

    notes = await sweep_track_lifecycle(dreams_db, now=NOW)

    item = dreams_db.get_tracked_item(item_id)
    assert item["status"] == "retired"
    assert item["retired_reason"] == "event_passed"
    assert any("event passed" in note for note in notes), (
        "the sweep returns degradation notes naming what it retired"
    )


@pytest.mark.asyncio
async def test_sweep_spares_item_on_the_day_after_its_event(dreams_db, settings):
    """Event 2026-09-21 + 1 day == 2026-09-22 (``NOW``): not strictly past.

    The +1d grace keeps the watch alive through the day after the event;
    only once that day has fully passed does the sweep retire.
    """
    item_id = dreams_db.create_tracked_item(
        mechanism="question", intent="event", event_date="2026-09-21",
        cadence_seconds=3600)

    notes = await sweep_track_lifecycle(dreams_db, now=NOW)

    assert dreams_db.get_tracked_item(item_id)["status"] == "active"
    assert notes == []


@pytest.mark.asyncio
async def test_sweep_quiet_retires_after_fourteen_consecutive_unchanged(
        dreams_db, settings):
    quiet = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)
    _seed_runs(dreams_db, quiet, ["baseline"] + ["unchanged"] * 14)
    one_short = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)
    _seed_runs(dreams_db, one_short, ["baseline"] + ["unchanged"] * 13)

    notes = await sweep_track_lifecycle(dreams_db, now=NOW)

    assert dreams_db.get_tracked_item(quiet)["status"] == "retired"
    assert dreams_db.get_tracked_item(quiet)["retired_reason"] == "quiet"
    assert dreams_db.get_tracked_item(one_short)["status"] == "active"
    assert any("quiet" in note for note in notes)


@pytest.mark.asyncio
async def test_sweep_broken_unchanged_streak_is_not_quiet(dreams_db, settings):
    """13 unchanged, one ``changed``, then more unchanged: streak reset.

    The judge said the watch changed recently -- retiring it as quiet would
    kill a live watch, so only the CONSECUTIVE trailing run counts.
    """
    item_id = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)
    _seed_runs(dreams_db, item_id,
               ["baseline"] + ["unchanged"] * 13 + ["changed"]
               + ["unchanged"] * 5)

    await sweep_track_lifecycle(dreams_db, now=NOW)

    assert dreams_db.get_tracked_item(item_id)["status"] == "active"


@pytest.mark.asyncio
async def test_sweep_quiet_count_resolves_through_settings(
        dreams_db, settings):
    """Single-source: the threshold is the [dreams] key, not a literal."""
    settings["track_quiet_retire_count"] = 2
    item_id = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)
    _seed_runs(dreams_db, item_id, ["baseline", "unchanged", "unchanged"])

    await sweep_track_lifecycle(dreams_db, now=NOW)

    assert dreams_db.get_tracked_item(item_id)["retired_reason"] == "quiet"


@pytest.mark.asyncio
async def test_sweep_pauses_after_three_consecutive_errors(dreams_db, settings):
    failing = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)
    _seed_runs(dreams_db, failing, ["baseline", "error", "error", "error"])
    two = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)
    _seed_runs(dreams_db, two, ["baseline", "error", "error"])

    notes = await sweep_track_lifecycle(dreams_db, now=NOW)

    paused = dreams_db.get_tracked_item(failing)
    assert paused["status"] == "paused", "3 consecutive errors auto-pause"
    assert paused["retired_reason"] is None, "a pause is not a retirement"
    assert dreams_db.get_tracked_item(two)["status"] == "active"
    assert any("failures" in note for note in notes)


@pytest.mark.asyncio
async def test_sweep_leaves_mixed_healthy_items_untouched(dreams_db, settings):
    fresh = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)
    baselined = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)
    _seed_runs(dreams_db, baselined, ["baseline", "unchanged"])
    already_paused = dreams_db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=3600)
    dreams_db.set_tracked_status(already_paused, "paused")
    already_retired = dreams_db.create_tracked_item(
        mechanism="question", intent="event", event_date="2026-09-20",
        cadence_seconds=3600)
    dreams_db.set_tracked_status(already_retired, "retired",
                                 retired_reason="manual")

    notes = await sweep_track_lifecycle(dreams_db, now=NOW)

    assert dreams_db.get_tracked_item(fresh)["status"] == "active"
    assert dreams_db.get_tracked_item(baselined)["status"] == "active"
    assert dreams_db.get_tracked_item(already_paused)["status"] == "paused"
    retired = dreams_db.get_tracked_item(already_retired)
    assert retired["status"] == "retired"
    assert retired["retired_reason"] == "manual", (
        "the sweep never rewrites an already-retired item's reason"
    )
    assert notes == [], "a sweep with nothing to do records no notes"


@pytest.mark.asyncio
async def test_run_track_check_sweeps_before_checking(dreams_db, settings):
    """The check loop sweeps first -- and a sweep-retired item stops there.

    Qodo #5 (PR #2890): the sweep retires the event-passed item BEFORE
    the check's own row read, so the dispatched task now records one
    cheap ``skipped`` "not active" run instead of searching and spending
    budget on a watch that just died. Existence alone is no longer
    liveness.
    """
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-09-20")
    search = _FakeSearch(_SET_A)
    deps = _track_deps(dreams_db, perform_search=search)

    result = await run_track_check(deps, item_id)

    assert result == {"status": "skipped", "notified": False}
    item = dreams_db.get_tracked_item(item_id)
    assert item["status"] == "retired"
    assert item["retired_reason"] == "event_passed"
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "skipped"
    assert runs[0]["verdict_note"] == "not active"
    assert search.calls == [], "a stopped watch must not search"
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}


# --- re-track after a sweep auto-retire (final review, ruling P10) --------------


@pytest.mark.asyncio
async def test_retrack_after_sweep_retire_keeps_dream_provenance(
        tmp_path, dreams_db, settings, subs_stack):
    """A sweep-retired watch must re-track with its provenance intact.

    Qodo #1 (PR #2890) changed the sweep's teardown: retiring a
    dream-created page wrapper now DISABLES its subscription too, so the
    re-track lands on the P7 re-enable branch (a disabled source with
    dream provenance pinned by the retired row) instead of the plain
    attach -- the outcome word differs, the invariant does not: the
    subscription is dream-owned, re-enabled, and a later untrack disables
    it again. No orphaned active subscription may outlive the wrapper.
    """
    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-09-20", origin_story_id=7,
    )
    assert first["outcome"] == "created"

    # The sweep auto-retires the wrapper (event passed + grace) and, with
    # the service wired, disables the dream-created subscription with it.
    await sweep_track_lifecycle(
        dreams_db, now=NOW, subs_service_getter=lambda: service)
    retired = dreams_db.get_tracked_item(first["tracked_item_id"])
    assert retired["status"] == "retired"
    assert retired["retired_reason"] == "event_passed"
    assert int(
        subs_db.get_subscription(first["subscription_id"])["is_active"]
    ) == 0, "the sweep's teardown disabled the dream-owned subscription"

    second = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        origin_story_id=7,
    )

    assert second["outcome"] == "re-enabled", (
        "the subscription is disabled but dream-owned: re-enabled, "
        "never refused and never duplicated"
    )
    assert second["subscription_id"] == first["subscription_id"], (
        "re-track adopts the same subscription, never a duplicate"
    )
    wrapper = dreams_db.get_tracked_item(second["tracked_item_id"])
    assert wrapper["created_by_dreams"] == 1, (
        "provenance carries from the sweep-retired row onto the new wrapper"
    )

    # The point of the provenance: untrack must disable the subscription.
    outcome = await untrack(service, dreams_db, second["tracked_item_id"])
    assert outcome["subscription_disabled"] is True
    assert int(
        subs_db.get_subscription(first["subscription_id"])["is_active"]
    ) == 0, "no orphaned active subscription may outlive the wrapper"


@pytest.mark.asyncio
async def test_sweep_retire_without_service_keeps_legacy_behavior(
        tmp_path, dreams_db, settings, subs_stack):
    """The teardown getters are optional: a getter-less sweep (the legacy
    call shape) retires the wrapper and leaves the subscription live --
    degrade, never raise."""
    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-09-20", origin_story_id=7,
    )

    notes = await sweep_track_lifecycle(dreams_db, now=NOW)

    assert any("event passed" in note for note in notes)
    assert int(
        subs_db.get_subscription(first["subscription_id"])["is_active"]
    ) == 1, "without the service seam the sweep cannot disable anything"


# --- Qodo review on PR #2890 -----------------------------------------------------


class _RecordingSubsService:
    """``update_source``-only stand-in that records every call."""

    def __init__(self):
        self.updates: list[tuple[int, dict]] = []

    async def update_source(self, source_id, payload):
        self.updates.append((source_id, dict(payload)))
        return {"id": source_id, **dict(payload)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_url",
    [
        "ftp://example.com/file",          # unsupported scheme
        "https://exa mple.com/a",          # raw whitespace
        "https://user:pass@example.com/a",  # embedded credentials
        "example.com/no-scheme",
        "",
    ],
)
async def test_track_page_refuses_invalid_url_before_any_write(
        tmp_path, dreams_db, settings, subs_stack, bad_url):
    """Qodo #8: the shared ``validate_url`` gate runs service-side, before
    any lookup or write -- a malformed address never becomes a
    subscription the watchlist loop would keep processing."""
    subs_db, service = subs_stack

    with pytest.raises(TrackInvalidURL) as excinfo:
        await track_page(
            service, dreams_db, url=bad_url, title="Bad", intent="topic")

    assert excinfo.value.reason_code == "track_invalid_url"
    assert dreams_db.list_tracked_items() == []
    assert _count_rows(subs_db, "subscriptions") == 0
    assert _count_rows(subs_db, "watchlists") == 0
    assert _count_rows(subs_db, "local_watchlist_alert_rules") == 0


@pytest.mark.asyncio
async def test_track_page_twice_same_story_reuses_one_wrapper(
        tmp_path, dreams_db, settings, subs_stack):
    """Qodo #2: re-tracking a story that already owns an ACTIVE page
    wrapper reuses it (plan refreshed) instead of stacking a second
    wrapper whose retirement would strand the first."""
    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    assert first["outcome"] == "created"

    second = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7,
        cadence_seconds=86400, event_date="2026-10-01")

    assert second["outcome"] == "attached"
    assert second["tracked_item_id"] == first["tracked_item_id"]
    assert second["subscription_id"] == first["subscription_id"]
    active = dreams_db.list_tracked_items()
    assert len(active) == 1, "one story, one wrapper"
    # The caller's plan moved onto the reused row.
    assert active[0]["cadence_seconds"] == 86400
    assert active[0]["event_date"] == "2026-10-01"
    assert _count_rows(subs_db, "subscriptions") == 1


@pytest.mark.asyncio
async def test_track_question_twice_same_story_reuses_wrapper(
        dreams_db, settings):
    """Qodo #2, question path: the story's existing ACTIVE question wrapper
    is returned (plan refreshed), never duplicated."""
    first = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal",
        origin_story_id=3, cadence_seconds=86400)
    second = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal",
        origin_story_id=3, cadence_seconds=172800)

    assert second == first
    active = dreams_db.list_tracked_items()
    assert len(active) == 1
    assert active[0]["cadence_seconds"] == 172800


@pytest.mark.asyncio
async def test_track_check_snippet_only_change_reaches_judge(
        dreams_db, settings):
    """Qodo #3: a result keeping its URL and title while its snippet
    changes (price, date, availability) must move the digest and reach
    the judge -- not be declared ``unchanged`` forever."""
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    set_a = (("Flights round-up", "https://example.com/flights/roundup",
              "prices unchanged"),)
    set_b = (("Flights round-up", "https://example.com/flights/roundup",
              "SOLD OUT everywhere"),)
    search = _FakeSearch(set_a)
    chat = _FakeChat()
    recorder = _DispatchRecorder()
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=recorder)

    await run_track_check(deps, item_id)
    search.results = set_b
    result = await run_track_check(deps, item_id)

    assert result == {"status": "changed", "notified": True}
    assert len(chat.calls) == 1, "the snippet change reaches the judge"
    assert _expected_digest(set_a) != _expected_digest(set_b)


@pytest.mark.asyncio
async def test_track_check_on_paused_item_skips_without_spending(
        dreams_db, settings):
    """Qodo #5: a task dispatched after a pause records one cheap skip --
    no search, no budget, no judge."""
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="topic")
    dreams_db.set_tracked_status(item_id, "paused")
    search = _FakeSearch(_SET_A)
    deps = _track_deps(dreams_db, perform_search=search)

    result = await run_track_check(deps, item_id)

    assert result == {"status": "skipped", "notified": False}
    assert search.calls == []
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "skipped"
    assert runs[0]["verdict_note"] == "not active"


@pytest.mark.asyncio
async def test_track_check_on_page_item_skips_no_spend(dreams_db, settings):
    """Qodo #7 (defense in depth under the projection's question-only
    filter): a page-mechanism item records a skip -- its subscription's
    own loop watches the page; a Dreams search on an empty template would
    only burn the shared budget."""
    item_id = dreams_db.create_tracked_item(
        mechanism="page", intent="topic", cadence_seconds=3600,
        subscription_id=99, created_by_dreams=1)
    search = _FakeSearch(_SET_A)
    deps = _track_deps(dreams_db, perform_search=search)

    result = await run_track_check(deps, item_id)

    assert result == {"status": "skipped", "notified": False}
    assert search.calls == []
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "skipped"
    assert runs[0]["verdict_note"] == "page items are watched by subscription"


class _BlockingSearch:
    """Sync search fake that parks inside the worker thread until released."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, engine, query, **kwargs):
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=10)
        return {"results": [
            {"title": "Flights round-up", "url":
             "https://example.com/flights/roundup", "content": "nothing new"},
        ]}


@pytest.mark.asyncio
async def test_track_check_second_concurrent_run_returns_inflight(
        dreams_db, settings):
    """Qodo #16: while one check of an item is still inside its search, a
    re-dispatch returns ``inflight`` immediately -- no second search, no
    second judge, no second notification."""
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _BlockingSearch()
    deps = _track_deps(dreams_db, perform_search=search)

    first = asyncio.create_task(run_track_check(deps, item_id))
    # Yield the loop (a sync Event.wait would block it) until the spawned
    # check is parked inside its worker-thread search.
    for _ in range(200):
        if search.started.is_set():
            break
        await asyncio.sleep(0.05)
    assert search.started.is_set(), "first check reached its search"

    second = await run_track_check(deps, item_id)

    assert second == {"status": "inflight", "notified": False}
    search.release.set()
    result = await first
    assert result == {"status": "baseline", "notified": False}
    assert search.calls == 1, "exactly one search ran"
    assert dreams_db.usage_get(_today()) == {"searches": 1, "llm_calls": 0}
    # The claim is released: a later check of the same item runs normally
    # (identical results -> unchanged, judge skipped).
    same = (("Flights round-up", "https://example.com/flights/roundup",
             "nothing new"),)
    again = await run_track_check(
        _track_deps(dreams_db, perform_search=_FakeSearch(same)), item_id)
    assert again["status"] == "unchanged"


@pytest.mark.asyncio
async def test_sweep_retire_disables_dream_subscription_and_reminder(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Qodo #1 + #13: a sweep-retired dream-created page item takes its
    subscription (disabled, never deleted) and its linked reminder with
    it."""
    subs_db, service = subs_stack
    result = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-09-20", origin_story_id=7,
        scheduling_db_getter=lambda: scheduling_db)
    (task,) = scheduling_db.list_reminder_tasks()
    assert task["enabled"] is True, "fixture: the reminder starts armed"

    notes = await sweep_track_lifecycle(
        dreams_db, now=NOW,
        subs_service_getter=lambda: service,
        scheduling_db_getter=lambda: scheduling_db)

    assert any("event passed" in note for note in notes)
    subscription = subs_db.get_subscription(result["subscription_id"])
    assert subscription is not None, "disabled, never deleted"
    assert int(subscription["is_active"]) == 0
    (task,) = scheduling_db.list_reminder_tasks()
    assert task["enabled"] is False, "the linked reminder stops firing"


@pytest.mark.asyncio
async def test_sweep_retire_of_question_item_touches_no_subscription(
        dreams_db, settings):
    """Qodo #1: a question item has no subscription -- its retirement must
    not attempt any external subscription write (only its reminder dies)."""
    item_id = dreams_db.create_tracked_item(
        mechanism="question", intent="event", event_date="2026-09-20",
        cadence_seconds=3600)
    recorder = _RecordingSubsService()

    await sweep_track_lifecycle(
        dreams_db, now=NOW, subs_service_getter=lambda: recorder)

    assert dreams_db.get_tracked_item(item_id)["status"] == "retired"
    assert recorder.updates == [], "nothing external was touched"


@pytest.mark.asyncio
async def test_sweep_retire_of_attached_page_leaves_subscription_active(
        tmp_path, dreams_db, settings, subs_stack):
    """Qodo #1: a subscription Dreams merely attached to (the user's own)
    is never disabled by the sweep -- only dream-created ones."""
    subs_db, service = subs_stack
    existing = await service.create_source(
        {"name": "User's own feed", "url": _URL, "source_type": "url"})
    tracked = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-09-20", origin_story_id=7)
    assert tracked["outcome"] == "attached"

    await sweep_track_lifecycle(
        dreams_db, now=NOW, subs_service_getter=lambda: service)

    assert dreams_db.get_tracked_item(tracked["tracked_item_id"])[
        "status"] == "retired"
    assert int(
        subs_db.get_subscription(existing["source_id"])["is_active"]
    ) == 1, "an attached subscription is NEVER ours to stop"


@pytest.mark.asyncio
async def test_untrack_disables_linked_reminder(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Qodo #13: manual untrack stops the watch AND its one-time reminder."""
    service = subs_stack[1]
    result = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01",
        scheduling_db_getter=lambda: scheduling_db)
    (task,) = scheduling_db.list_reminder_tasks()
    assert task["enabled"] is True

    await untrack(
        service, dreams_db, result["tracked_item_id"],
        scheduling_db_getter=lambda: scheduling_db)

    (task,) = scheduling_db.list_reminder_tasks()
    assert task["enabled"] is False


@pytest.mark.asyncio
async def test_untrack_without_scheduling_getter_leaves_reminder(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Qodo #13 degrade path: no scheduling getter (wiring order) and the
    retirement still succeeds -- the reminder simply stays armed."""
    service = subs_stack[1]
    result = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01",
        scheduling_db_getter=lambda: scheduling_db)

    outcome = await untrack(service, dreams_db, result["tracked_item_id"])

    assert outcome["status"] == "retired"
    (task,) = scheduling_db.list_reminder_tasks()
    assert task["enabled"] is True, "degrade, never raise"


@pytest.mark.asyncio
async def test_retrack_new_story_same_url_keeps_one_alert_rule(
        tmp_path, dreams_db, settings, subs_stack):
    """Qodo #15: a second wrapper on the same subscription (a new story,
    same URL) must not pin a second identical change rule -- one
    notification per qualifying run, not one per wrapper."""
    _subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    second = await track_page(
        service, dreams_db, url=_URL, title="Flights again", intent="deal",
        origin_story_id=8)
    assert {first["outcome"], second["outcome"]} == {"created", "attached"}

    rules = await service.list_alert_rules(job_id=first["subscription_id"])
    dreams_rules = [rule for rule in rules
                    if rule["name"].startswith("Change: ")]
    assert len(dreams_rules) == 1, "one change rule per subscription"
    assert dreams_rules[0]["condition_type"] == "items_above"
    assert dreams_rules[0]["condition_value"] == {"threshold": 0}


# --- Qodo review on PR #2921 -----------------------------------------------------


def _seed_newer_retired_rows(dreams_db, count: int) -> None:
    """Retired wrappers newer than anything the test tracked (direct SQL)."""
    with dreams_db.transaction() as conn:
        conn.executemany(
            "INSERT INTO dream_tracked_items"
            " (mechanism, intent, cadence_seconds, status, query_template,"
            "  created_at, updated_at)"
            " VALUES ('question', 'topic', 3600, 'retired', ?, ?, ?)",
            [(f"q {i}", "2099-01-01T00:00:00+00:00",
              "2099-01-01T00:00:00+00:00") for i in range(count)],
        )


@pytest.mark.asyncio
async def test_retrack_keeps_dream_provenance_past_the_status_list_cap(
        tmp_path, dreams_db, settings, subs_stack):
    """Qodo #6: ownership is a targeted read, not a scan of capped lists.

    ``list_tracked_items`` returns the newest 500 rows of a status, so once
    500 newer retired wrappers exist the row that pins a subscription as
    dream-created falls outside the read -- and the re-track used to refuse
    the subscription Dreams itself disabled as a foreign one.
    """
    from tldw_chatbook.DB.Dreams_DB import TRACKED_ITEMS_LIST_LIMIT

    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal")
    await untrack(service, dreams_db, first["tracked_item_id"])
    _seed_newer_retired_rows(dreams_db, TRACKED_ITEMS_LIST_LIMIT)

    second = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal")

    assert second["outcome"] == "re-enabled"
    assert second["subscription_id"] == first["subscription_id"]
    wrapper = dreams_db.get_tracked_item(second["tracked_item_id"])
    assert wrapper["created_by_dreams"] == 1, (
        "the new wrapper stays dream-owned, so untrack disables it again"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_memory_db, off_loop",
    [
        (False, True),   # the app's file-backed inbox: one connection per thread
        (True, False),   # the :memory: fallback: one loop-thread connection
        (None, False),   # unknown store shape: unconfirmed, stays inline
    ],
)
async def test_track_check_dispatches_the_changed_alert_off_the_event_loop(
        dreams_db, settings, is_memory_db, off_loop):
    """Qodo #7: ``dispatch`` reads notification settings and writes the
    inbox row through SQLite, so the check hands it to a worker thread
    like every other blocking stage -- when the store is confirmed
    file-backed. The in-memory fallback store owns a single loop-thread
    connection sqlite refuses from a worker, so it (and any unknown
    shape) keeps the inline call rather than losing the alert."""
    from types import SimpleNamespace

    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    dispatch_threads: list[int] = []

    class _ThreadRecorder(_DispatchRecorder):
        store = SimpleNamespace(is_memory_db=is_memory_db)

        def dispatch(self, **kwargs):
            dispatch_threads.append(threading.get_ident())
            super().dispatch(**kwargs)

    recorder = _ThreadRecorder()
    deps = _track_deps(dreams_db, perform_search=search, chat=_FakeChat(),
                       dispatch_getter=recorder)

    await run_track_check(deps, item_id)
    search.results = _SET_B
    result = await run_track_check(deps, item_id)

    assert result == {"status": "changed", "notified": True}
    assert len(recorder.dispatched) == 1
    assert (dispatch_threads[0] != threading.get_ident()) is off_loop


@pytest.mark.asyncio
async def test_track_check_delivers_through_the_real_file_backed_inbox(
        tmp_path, dreams_db, settings):
    """Qodo #7, unmocked: the real dispatcher over a file-backed inbox takes
    the worker-thread path and the alert still lands as an inbox row."""
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    dispatcher = NotificationDispatchService(store=store)
    deps = _track_deps(dreams_db, perform_search=search, chat=_FakeChat(),
                       dispatch_getter=lambda: dispatcher)
    try:
        await run_track_check(deps, item_id)
        search.results = _SET_B
        result = await run_track_check(deps, item_id)

        assert result == {"status": "changed", "notified": True}
        (row,) = store.list_notifications()
        assert row["category"] == "dreams_track"
        assert row["source_entity_id"] == str(item_id)
    finally:
        store.close()


# --- story URLs must be public destinations (Qodo #1, security) -----------------
#
# A story's URL came out of a web search or a feed item, not out of the
# user's keyboard. The watchlist loop trusts a subscription's own configured
# source (``trusted_origins=origin_set(source)``), which is right for a URL
# the user typed and wrong for one an untrusted result supplied: storing it
# as a subscription would authorize it to be fetched on the user's network.


def _assert_nothing_was_written(dreams_db, subs_db) -> None:
    assert dreams_db.list_tracked_items() == []
    assert _count_rows(subs_db, "subscriptions") == 0
    assert _count_rows(subs_db, "watchlists") == 0
    assert _count_rows(subs_db, "local_watchlist_alert_rules") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "private_url",
    [
        "http://127.0.0.1:8080/admin",              # loopback
        "http://192.168.1.10/status",               # RFC 1918
        "http://[::1]/metrics",                     # IPv6 loopback
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://[64:ff9b::7f00:1]/",                # NAT64-wrapped loopback
    ],
)
async def test_track_page_refuses_private_destination_before_any_write(
        tmp_path, dreams_db, settings, subs_stack, private_url):
    subs_db, service = subs_stack

    with pytest.raises(TrackPrivateURL) as excinfo:
        await track_page(
            service, dreams_db, url=private_url, title="Router", intent="topic")

    assert excinfo.value.reason_code == "track_private_url"
    assert isinstance(excinfo.value, TrackInvalidURL), (
        "callers that only know the invalid-URL refusal still catch it"
    )
    assert private_url not in str(excinfo.value), "the message names no URL"
    _assert_nothing_was_written(dreams_db, subs_db)


@pytest.mark.asyncio
async def test_track_page_refuses_a_public_looking_host_that_resolves_private(
        tmp_path, dreams_db, settings, subs_stack, monkeypatch):
    """The check runs AFTER resolution: a hostname is only as public as the
    addresses it resolves to."""
    from tldw_chatbook.Utils import egress

    async def _private(_host):
        return ["10.0.0.5"]

    monkeypatch.setattr(egress, "_resolve_async", _private)
    subs_db, service = subs_stack

    with pytest.raises(TrackPrivateURL):
        await track_page(
            service, dreams_db, url="https://intranet.example.com/wiki",
            title="Wiki", intent="topic")

    _assert_nothing_was_written(dreams_db, subs_db)


@pytest.mark.asyncio
async def test_track_page_refuses_a_host_that_does_not_resolve(
        tmp_path, dreams_db, settings, subs_stack, monkeypatch):
    """Fail closed: an address nobody could verify is not stored as a
    trusted watch."""
    from tldw_chatbook.Utils import egress

    async def _unresolvable(_host):
        raise OSError("name or service not known")

    monkeypatch.setattr(egress, "_resolve_async", _unresolvable)
    subs_db, service = subs_stack

    with pytest.raises(TrackPrivateURL):
        await track_page(
            service, dreams_db, url="https://gone.example.com/a",
            title="Gone", intent="topic")

    _assert_nothing_was_written(dreams_db, subs_db)


@pytest.mark.asyncio
async def test_track_page_still_attaches_to_the_users_own_private_source(
        tmp_path, dreams_db, settings, subs_stack):
    """A subscription the USER configured is the user's intent: tracking a
    story with that URL attaches to it and starts no new egress, so the
    public-destination rule does not apply."""
    subs_db, service = subs_stack
    private_url = "http://192.168.1.10/status"
    existing = await service.create_source(
        {"name": "My NAS", "url": private_url, "source_type": "url",
         "url_provenance": UrlProvenance.USER_ENTERED})

    result = await track_page(
        service, dreams_db, url=private_url, title="NAS", intent="topic")

    assert result["outcome"] == "attached"
    assert result["subscription_id"] == existing["source_id"]
    assert dreams_db.get_tracked_item(
        result["tracked_item_id"])["created_by_dreams"] == 0


@pytest.mark.asyncio
async def test_retrack_never_reenables_a_dream_source_that_is_now_private(
        tmp_path, dreams_db, settings, subs_stack, monkeypatch):
    """Re-enabling restarts fetches of a URL Dreams stored: the destination
    is checked again at that moment, so a host that has since moved onto a
    private address stays disabled."""
    from tldw_chatbook.Utils import egress

    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal")
    await untrack(service, dreams_db, first["tracked_item_id"])

    async def _now_private(_host):
        return ["127.0.0.1"]

    monkeypatch.setattr(egress, "_resolve_async", _now_private)

    with pytest.raises(TrackPrivateURL):
        await track_page(
            service, dreams_db, url=_URL, title="Cheap flights", intent="deal")

    assert int(
        subs_db.get_subscription(first["subscription_id"])["is_active"]
    ) == 0, "the disabled source stays disabled"
    assert dreams_db.list_tracked_items() == []


# --- the master switch reaches the other queues (Qodo #5) -----------------------
#
# ``[dreams] enabled`` gates the Dreams projection only. A page watch's
# subscription is scheduled by the watchlist projection and an event watch's
# reminder by the reminder queue; neither reads that flag, so the switch has
# to stop (and later restore) them itself.


async def _two_page_watches_and_a_question(dreams_db, service, scheduling_db):
    """A dream-created page watch, a user-owned one, and a question watch,
    the first and last carrying an event date (so each owns a reminder)."""
    getter = lambda: scheduling_db  # noqa: E731 - a one-line seam
    owned = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01", origin_story_id=1,
        scheduling_db_getter=getter)
    users_url = "https://example.com/users-own"
    await service.create_source(
        {"name": "User's own", "url": users_url, "source_type": "url",
         "url_provenance": UrlProvenance.USER_ENTERED})
    attached = await track_page(
        service, dreams_db, url=users_url, title="Theirs", intent="topic",
        origin_story_id=2, scheduling_db_getter=getter)
    question = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01", origin_story_id=3,
        scheduling_db_getter=getter)
    return owned, attached, question


def _reminder_states(scheduling_db) -> dict[str, bool]:
    return {task["link_id"]: task["enabled"]
            for task in scheduling_db.list_reminder_tasks(owner_id="local")}


@pytest.mark.asyncio
async def test_suspending_stops_dream_subscriptions_and_reminders_only(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    subs_db, service = subs_stack
    owned, attached, question = await _two_page_watches_and_a_question(
        dreams_db, service, scheduling_db)

    switched = await set_tracking_suspended(
        service, dreams_db, suspended=True,
        scheduling_db_getter=lambda: scheduling_db)

    assert switched == 3
    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 0, (
        "the subscription Dreams created stops with Dreams"
    )
    assert int(subs_db.get_subscription(
        attached["subscription_id"])["is_active"]) == 1, (
        "a subscription Dreams only attached to is the user's"
    )
    assert _reminder_states(scheduling_db) == {
        str(owned["tracked_item_id"]): False, str(question): False}
    assert {item["status"] for item in dreams_db.list_tracked_items()} == {
        "active"}, "suspending retires nothing"
    assert len(dreams_db.list_tracked_items()) == 3


@pytest.mark.asyncio
async def test_resuming_restores_what_suspending_stopped_without_duplicates(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    subs_db, service = subs_stack
    owned, attached, question = await _two_page_watches_and_a_question(
        dreams_db, service, scheduling_db)
    getter = lambda: scheduling_db  # noqa: E731
    # The question watch's reminder already fired before Dreams went off:
    # a consumed one-time nudge must not be re-armed or re-created.
    (fired,) = [task for task in scheduling_db.list_reminder_tasks()
                if task["link_id"] == str(question)]
    scheduling_db.mark_reminder_dispatched(fired["id"], NOW)
    await set_tracking_suspended(
        service, dreams_db, suspended=True, scheduling_db_getter=getter)

    switched = await set_tracking_suspended(
        service, dreams_db, suspended=False, scheduling_db_getter=getter,
        now=NOW)

    assert switched == 3
    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 1
    assert int(subs_db.get_subscription(
        attached["subscription_id"])["is_active"]) == 1
    assert _reminder_states(scheduling_db) == {
        str(owned["tracked_item_id"]): True, str(question): False}
    assert len(scheduling_db.list_reminder_tasks()) == 2, (
        "resume re-enables the suspended reminder, it never duplicates one"
    )


@pytest.mark.asyncio
async def test_resuming_retires_a_watch_whose_event_passed_while_off(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Dreams was off across the event: resuming retires that watch through
    the lifecycle sweep instead of waking its subscription and its reminder
    to nag about a date already gone."""
    subs_db, service = subs_stack
    getter = lambda: scheduling_db  # noqa: E731
    owned = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01", origin_story_id=1,
        scheduling_db_getter=getter)
    await set_tracking_suspended(
        service, dreams_db, suspended=True, scheduling_db_getter=getter)

    after_the_event = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
    switched = await set_tracking_suspended(
        service, dreams_db, suspended=False, scheduling_db_getter=getter,
        now=after_the_event)

    # The record is restored first (it is read off a live wrapper), then
    # the sweep retires the dead watch and tears its outside work down.
    assert switched == 1
    item = dreams_db.get_tracked_item(owned["tracked_item_id"])
    assert (item["status"], item["retired_reason"]) == (
        "retired", "event_passed")
    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 0
    assert _reminder_states(scheduling_db) == {
        str(owned["tracked_item_id"]): False}


@pytest.mark.asyncio
async def test_the_switch_degrades_without_a_service_or_scheduling_db(
        tmp_path, dreams_db, settings, subs_stack):
    """No watchlists service and no scheduling DB (wiring order, or a
    runtime without them): the switch does what it can and never raises."""
    subs_db, service = subs_stack
    owned = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01")

    assert await set_tracking_suspended(None, dreams_db, suspended=True) == 1
    assert await set_tracking_suspended(
        None, dreams_db, suspended=False, now=NOW) == 1
    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 1


# --- concurrent checks share one daily budget (Qodo #11) ------------------------
#
# The per-item in-flight claim stops two checks of the SAME item; checks of
# different items still run side by side. Reading the day's usage and
# bumping it after the work were two separate steps, so two checks could
# both read "one slot left" and both spend it.


class _BlockingChat:
    """Judge fake that parks inside the worker thread until released."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=10)
        return {"choices": [{"message": {
            "content": '{"changed": false, "note": "same"}'}}]}


async def _until(event: threading.Event) -> None:
    """Yield the loop until a worker thread sets ``event``."""
    for _ in range(200):
        if event.is_set():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the first check never reached its blocking call")


@pytest.mark.asyncio
async def test_concurrent_checks_cannot_both_spend_the_last_search(
        dreams_db, settings):
    settings["region"] = "kyoto"
    settings["max_searches_per_day"] = 1
    first_item = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    second_item = await track_question(
        dreams_db, query_template="trains to {region}", intent="deal")
    parked = _BlockingSearch()
    other = _FakeSearch(_SET_A)

    first = asyncio.create_task(run_track_check(
        _track_deps(dreams_db, perform_search=parked), first_item))
    await _until(parked.started)
    second = await run_track_check(
        _track_deps(dreams_db, perform_search=other), second_item)
    parked.release.set()
    await first

    assert second == {"status": "skipped", "notified": False}
    assert other.calls == [], "the day's one search was already taken"
    assert dreams_db.usage_get(_today()) == {"searches": 1, "llm_calls": 0}
    (run,) = dreams_db.list_recent_track_runs(second_item)
    assert (run["status"], run["verdict_note"]) == ("skipped", "budget")


@pytest.mark.asyncio
async def test_concurrent_checks_cannot_both_spend_the_last_judge_call(
        dreams_db, settings):
    settings["region"] = "kyoto"
    settings["max_llm_calls_per_day"] = 1
    first_item = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    second_item = await track_question(
        dreams_db, query_template="trains to {region}", intent="deal")
    for item_id in (first_item, second_item):
        dreams_db.insert_track_run(
            item_id, status="baseline", digest_hash="an-older-result-set")
    parked = _BlockingChat()
    other = _FakeChat()
    other_search = _FakeSearch(_SET_B)

    first = asyncio.create_task(run_track_check(
        _track_deps(dreams_db, perform_search=_FakeSearch(_SET_A),
                    chat=parked), first_item))
    await _until(parked.started)
    second = await run_track_check(
        _track_deps(dreams_db, perform_search=other_search, chat=other),
        second_item)
    parked.release.set()
    await first

    assert second == {"status": "skipped", "notified": False}
    assert other.calls == [], "the day's one judge call was already taken"
    assert other_search.calls == [], "and nothing was searched for it"
    assert parked.calls == 1
    assert dreams_db.usage_get(_today())["llm_calls"] == 1


@pytest.mark.asyncio
async def test_track_check_skips_the_judge_when_its_slot_went_mid_check(
        dreams_db, settings):
    """The check passed the budget guard, then another spender took the
    last LLM call while it searched: the judge is not called, and the
    non-anchoring ``skipped`` run leaves the change for the next check."""
    settings["region"] = "kyoto"
    settings["max_llm_calls_per_day"] = 1
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    dreams_db.insert_track_run(
        item_id, status="baseline", digest_hash="an-older-result-set")
    chat = _FakeChat()

    class _SearchWhileAnotherSpends(_FakeSearch):
        def __call__(self, engine, query, **kwargs):
            dreams_db.usage_bump(_today(), llm_calls=1)
            return super().__call__(engine, query, **kwargs)

    result = await run_track_check(
        _track_deps(dreams_db,
                    perform_search=_SearchWhileAnotherSpends(_SET_A),
                    chat=chat), item_id)

    assert result == {"status": "skipped", "notified": False}
    assert chat.calls == []
    assert dreams_db.usage_get(_today()) == {"searches": 1, "llm_calls": 1}
    newest = dreams_db.list_recent_track_runs(item_id)[0]
    assert (newest["status"], newest["verdict_note"], newest["digest_hash"]) \
        == ("skipped", "budget", None)


# --- the comparison anchor outlives any run of skips (Qodo #12) -----------------


@pytest.mark.asyncio
async def test_track_check_keeps_its_anchor_behind_200_non_anchoring_runs(
        dreams_db, settings):
    """A long-lived watch can log hundreds of non-anchoring runs in a row
    (budget skips, empty result sets). The anchor used to be looked for in
    the newest 200 runs only, so an older valid digest was lost and the
    next real change was filed as a first ``baseline`` -- never judged,
    never notified."""
    settings["region"] = "kyoto"
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    dreams_db.insert_track_run(
        item_id, status="baseline", digest_hash=_expected_digest(_SET_A))
    with dreams_db.transaction() as conn:
        conn.executemany(
            "INSERT INTO dream_track_runs"
            " (tracked_item_id, status, digest_hash, verdict_note, notified,"
            "  created_at) VALUES (?, ?, NULL, ?, 0, ?)",
            [(item_id, "withheld" if i % 2 else "skipped", "quiet",
              f"2099-01-01T00:00:{i % 60:02d}.{i:03d}Z") for i in range(200)],
        )
    chat = _FakeChat()
    recorder = _DispatchRecorder()

    result = await run_track_check(
        _track_deps(dreams_db, perform_search=_FakeSearch(_SET_B), chat=chat,
                    dispatch_getter=recorder), item_id)

    assert result == {"status": "changed", "notified": True}, (
        "the change is judged against the anchor, not filed as a baseline"
    )
    assert len(chat.calls) == 1
    assert len(recorder.dispatched) == 1


# --- re-tracking keeps the reminder in step with the plan (Qodo #13) ------------
#
# The reuse path refreshes an existing wrapper's cadence and event date. The
# reminder is a separate row in the scheduling DB, so it has to be brought
# along explicitly: created when it is missing, rescheduled when the date
# moved, disabled when the date was cleared.


def _linked_reminders(scheduling_db, item_id) -> list[dict]:
    return [task for task in scheduling_db.list_reminder_tasks(owner_id="local")
            if task["link_id"] == str(item_id)]


def _enabled_run_dates(scheduling_db, item_id) -> list[str]:
    """Local calendar dates of the item's ENABLED reminders."""
    return [
        datetime.fromisoformat(task["run_at"]).astimezone().date().isoformat()
        for task in _linked_reminders(scheduling_db, item_id)
        if task["enabled"]
    ]


@pytest.mark.asyncio
async def test_retrack_page_creates_the_reminder_an_earlier_track_could_not(
        dreams_db, settings, subs_stack, scheduling_db):
    """The first track ran before the scheduling DB was wired (the degrade
    path): no reminder. Tracking the story again must not leave it so."""
    service = subs_stack[1]
    first = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01", origin_story_id=7,
        scheduling_db_getter=lambda: None)
    assert scheduling_db.list_reminder_tasks() == []

    again = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01", origin_story_id=7,
        scheduling_db_getter=lambda: scheduling_db)

    assert again["tracked_item_id"] == first["tracked_item_id"]
    assert len(_linked_reminders(scheduling_db, first["tracked_item_id"])) == 1
    assert len(_enabled_run_dates(
        scheduling_db, first["tracked_item_id"])) == 1


@pytest.mark.asyncio
async def test_retrack_page_with_a_new_event_date_moves_the_reminder(
        dreams_db, settings, subs_stack, scheduling_db):
    service = subs_stack[1]
    getter = lambda: scheduling_db  # noqa: E731
    first = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01", origin_story_id=7,
        scheduling_db_getter=getter)
    item_id = first["tracked_item_id"]
    (old_date,) = _enabled_run_dates(scheduling_db, item_id)

    await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-11-01", origin_story_id=7,
        scheduling_db_getter=getter)

    assert dreams_db.get_tracked_item(item_id)["event_date"] == "2026-11-01"
    (new_date,) = _enabled_run_dates(scheduling_db, item_id)
    assert new_date != old_date, "the old schedule is no longer armed"
    assert (datetime.fromisoformat(new_date)
            - datetime.fromisoformat(old_date)).days == 31
    assert len(_linked_reminders(scheduling_db, item_id)) == 2, (
        "the superseded reminder is disabled, not deleted"
    )


@pytest.mark.asyncio
async def test_retrack_question_clearing_the_event_date_disarms_the_reminder(
        dreams_db, settings, scheduling_db):
    getter = lambda: scheduling_db  # noqa: E731
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01", origin_story_id=7,
        scheduling_db_getter=getter)
    assert len(_enabled_run_dates(scheduling_db, item_id)) == 1

    again = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date=None, origin_story_id=7, scheduling_db_getter=getter)

    assert again == item_id
    assert dreams_db.get_tracked_item(item_id)["event_date"] is None
    assert _enabled_run_dates(scheduling_db, item_id) == []


@pytest.mark.asyncio
async def test_retrack_question_adding_an_event_date_creates_the_reminder(
        dreams_db, settings, scheduling_db):
    getter = lambda: scheduling_db  # noqa: E731
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        origin_story_id=7, scheduling_db_getter=getter)
    assert scheduling_db.list_reminder_tasks() == []

    await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01", origin_story_id=7,
        scheduling_db_getter=getter)

    (task,) = _linked_reminders(scheduling_db, item_id)
    assert task["enabled"] is True
    assert task["title"] == f"Dreams: {_TEMPLATE}"


@pytest.mark.asyncio
async def test_retrack_with_an_unchanged_plan_leaves_one_reminder(
        dreams_db, settings, subs_stack, scheduling_db):
    service = subs_stack[1]
    getter = lambda: scheduling_db  # noqa: E731
    for _ in range(3):
        result = await track_page(
            service, dreams_db, url=_URL, title="Festival", intent="event",
            event_date="2026-10-01", origin_story_id=7,
            scheduling_db_getter=getter)

    (task,) = _linked_reminders(scheduling_db, result["tracked_item_id"])
    assert task["enabled"] is True


# --- the reminder lands on the user's own calendar day (Qodo #14) ---------------
#
# "A week before the event" is a LOCAL date. A date-only ``run_at`` is read
# by the scheduler as midnight UTC, which west of UTC is still the evening
# of the day before.


@pytest.fixture()
def local_zone():
    """Run the test under a chosen local timezone (POSIX ``tzset``)."""
    import os
    import time as time_module

    if not hasattr(time_module, "tzset"):
        pytest.skip("needs time.tzset to move the local timezone")
    before = os.environ.get("TZ")

    def set_zone(name: str) -> None:
        os.environ["TZ"] = name
        time_module.tzset()

    try:
        yield set_zone
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time_module.tzset()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "zone",
    ["America/Los_Angeles", "Pacific/Pago_Pago", "UTC", "Asia/Tokyo",
     "Pacific/Kiritimati"],
)
async def test_event_reminder_fires_on_the_local_lead_date_at_nine(
        dreams_db, settings, scheduling_db, local_zone, zone):
    local_zone(zone)

    await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01",
        scheduling_db_getter=lambda: scheduling_db)

    (task,) = scheduling_db.list_reminder_tasks(owner_id="local")
    fires = datetime.fromisoformat(task["run_at"]).astimezone()
    assert fires.date().isoformat() == "2026-09-24", (
        f"{zone}: a week before the event on the user's own calendar"
    )
    assert (fires.hour, fires.minute) == (9, 0)
    assert task["next_run_at"] == task["run_at"]
    assert datetime.fromisoformat(task["run_at"]).utcoffset() is not None, (
        "the stored instant is unambiguous, never a bare date"
    )


# --- reuse is per mechanism (Qodo #3) -------------------------------------------
#
# One story may be tracked by page AND watched by question. The reuse lookup
# asked for the story's newest wrapper and then checked its mechanism, so
# re-tracking whichever mechanism was OLDER found the other one, skipped
# reuse, and stacked a duplicate active wrapper.


def _active_by_mechanism(dreams_db) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for item in dreams_db.list_tracked_items():
        grouped.setdefault(item["mechanism"], []).append(item["id"])
    return grouped


@pytest.mark.asyncio
async def test_retrack_page_reuses_its_wrapper_when_a_question_watch_is_newer(
        tmp_path, dreams_db, settings, subs_stack):
    _subs_db, service = subs_stack
    page = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    question = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal", origin_story_id=7)

    again = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7, cadence_seconds=86400)

    assert again["tracked_item_id"] == page["tracked_item_id"]
    assert again["outcome"] == "attached"
    assert _active_by_mechanism(dreams_db) == {
        "page": [page["tracked_item_id"]], "question": [question]}
    assert dreams_db.get_tracked_item(
        page["tracked_item_id"])["cadence_seconds"] == 86400, (
        "the reused page wrapper took the refreshed plan"
    )


@pytest.mark.asyncio
async def test_retrack_question_reuses_its_wrapper_when_a_page_watch_is_newer(
        tmp_path, dreams_db, settings, subs_stack):
    _subs_db, service = subs_stack
    question = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal", origin_story_id=7)
    page = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)

    again = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal", origin_story_id=7,
        cadence_seconds=86400)

    assert again == question
    assert _active_by_mechanism(dreams_db) == {
        "page": [page["tracked_item_id"]], "question": [question]}
    assert dreams_db.get_tracked_item(question)["cadence_seconds"] == 86400


# --- a shared subscription outlives one of its wrappers (Qodo #4) ---------------
#
# Two stories with the same URL resolve to ONE subscription and each gets
# its own wrapper. Retiring one of them used to disable the subscription
# outright, leaving the other wrapper active on a source nobody checks.


async def _two_wrappers_on_one_subscription(dreams_db, service, **first_kwargs):
    first = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7, **first_kwargs)
    second = await track_page(
        service, dreams_db, url=_URL, title="Flights again", intent="deal",
        origin_story_id=8)
    assert second["subscription_id"] == first["subscription_id"]
    assert dreams_db.get_tracked_item(
        second["tracked_item_id"])["created_by_dreams"] == 1, (
        "fixture: both wrappers carry dream ownership of the shared source"
    )
    return first, second


@pytest.mark.asyncio
async def test_untrack_keeps_a_subscription_another_wrapper_still_watches(
        tmp_path, dreams_db, settings, subs_stack):
    subs_db, service = subs_stack
    first, second = await _two_wrappers_on_one_subscription(dreams_db, service)

    outcome = await untrack(service, dreams_db, first["tracked_item_id"])

    assert outcome["status"] == "retired"
    assert outcome["subscription_disabled"] is False
    assert int(subs_db.get_subscription(
        first["subscription_id"])["is_active"]) == 1, (
        "the other story's watch still needs its source checked"
    )
    assert dreams_db.get_tracked_item(
        second["tracked_item_id"])["status"] == "active"

    last = await untrack(service, dreams_db, second["tracked_item_id"])

    assert last["subscription_disabled"] is True, (
        "the final wrapper takes the dream-created subscription with it"
    )
    assert int(subs_db.get_subscription(
        first["subscription_id"])["is_active"]) == 0


@pytest.mark.asyncio
async def test_sweep_keeps_a_subscription_another_wrapper_still_watches(
        tmp_path, dreams_db, settings, subs_stack):
    subs_db, service = subs_stack
    first, second = await _two_wrappers_on_one_subscription(
        dreams_db, service, event_date="2026-09-20")

    await sweep_track_lifecycle(
        dreams_db, now=NOW, subs_service_getter=lambda: service)

    assert dreams_db.get_tracked_item(
        first["tracked_item_id"])["retired_reason"] == "event_passed"
    assert dreams_db.get_tracked_item(
        second["tracked_item_id"])["status"] == "active"
    assert int(subs_db.get_subscription(
        first["subscription_id"])["is_active"]) == 1


@pytest.mark.asyncio
async def test_sweep_disables_a_subscription_when_all_its_wrappers_retire(
        tmp_path, dreams_db, settings, subs_stack):
    subs_db, service = subs_stack
    first, second = await _two_wrappers_on_one_subscription(
        dreams_db, service, event_date="2026-09-20")
    dreams_db.set_tracked_plan(
        second["tracked_item_id"], cadence_seconds=_MIN_INTERVAL_SECONDS,
        event_date="2026-09-19")

    await sweep_track_lifecycle(
        dreams_db, now=NOW, subs_service_getter=lambda: service)

    assert dreams_db.list_tracked_items() == []
    assert int(subs_db.get_subscription(
        first["subscription_id"])["is_active"]) == 0


@pytest.mark.asyncio
async def test_the_switch_also_covers_a_paused_watchs_reminder(
        tmp_path, dreams_db, settings, scheduling_db):
    """A watch the sweep PAUSED (repeated check failures) keeps its event
    reminder armed -- pausing only stops the checks. Switching Dreams off
    must stop that reminder too, and switching it back on restores it."""
    getter = lambda: scheduling_db  # noqa: E731
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01", scheduling_db_getter=getter)
    dreams_db.set_tracked_status(item_id, "paused")
    assert _reminder_states(scheduling_db) == {str(item_id): True}

    await set_tracking_suspended(
        None, dreams_db, suspended=True, scheduling_db_getter=getter)
    assert _reminder_states(scheduling_db) == {str(item_id): False}, (
        "a disabled Dreams sends no reminder, paused watch or not"
    )

    await set_tracking_suspended(
        None, dreams_db, suspended=False, scheduling_db_getter=getter,
        now=NOW)
    assert _reminder_states(scheduling_db) == {str(item_id): True}
    assert dreams_db.get_tracked_item(item_id)["status"] == "paused", (
        "the switch never changes a watch's own lifecycle state"
    )


# --- the switch restores only what it stopped (review of #2921, item 4) ---------
#
# Suspension used to record nothing, so resume could not tell "the switch
# stopped this" from "the user stopped this" and brought everything back:
# a subscription the user had disabled by hand, a reminder they had
# disabled, and -- by replacing it -- a reminder they had rescheduled.
# Each wrapper now carries a record of exactly what the switch stopped,
# and resume restores that and nothing else.


async def _off_then_on(service, dreams_db, scheduling_db) -> None:
    def getter():
        return scheduling_db

    await set_tracking_suspended(
        service, dreams_db, suspended=True, scheduling_db_getter=getter)
    await set_tracking_suspended(
        service, dreams_db, suspended=False, scheduling_db_getter=getter,
        now=NOW)


@pytest.mark.asyncio
async def test_resume_leaves_a_subscription_the_user_disabled_by_hand(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    subs_db, service = subs_stack
    owned = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    await service.update_source(owned["subscription_id"], {"active": False})

    await _off_then_on(service, dreams_db, scheduling_db)

    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 0, (
        "the user turned this source off; the switch never stopped it, so "
        "it has nothing to restore"
    )
    assert dreams_db.get_tracked_item(
        owned["tracked_item_id"])["suspension"] is None


@pytest.mark.asyncio
async def test_resume_leaves_a_reminder_the_user_disabled_by_hand(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    _subs_db, service = subs_stack
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01", scheduling_db_getter=lambda: scheduling_db)
    (task,) = _linked_reminders(scheduling_db, item_id)
    scheduling_db.update_reminder_task(task["id"], enabled=False)

    await _off_then_on(service, dreams_db, scheduling_db)

    (after,) = _linked_reminders(scheduling_db, item_id)
    assert after["enabled"] is False, "the user's own disable stands"


@pytest.mark.asyncio
async def test_resume_keeps_a_reminder_the_user_rescheduled(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    _subs_db, service = subs_stack
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01", scheduling_db_getter=lambda: scheduling_db)
    (task,) = _linked_reminders(scheduling_db, item_id)
    moved = datetime(2026, 9, 20, 18, 30, tzinfo=UTC)
    scheduling_db.update_reminder_task(
        task["id"], run_at=moved, next_run_at=moved)

    await _off_then_on(service, dreams_db, scheduling_db)

    reminders = _linked_reminders(scheduling_db, item_id)
    assert len(reminders) == 1, (
        "the rescheduled reminder is restored as it was, never replaced"
    )
    (after,) = reminders
    assert after["id"] == task["id"]
    assert after["enabled"] is True
    assert datetime.fromisoformat(after["run_at"]) == moved
    assert datetime.fromisoformat(after["next_run_at"]) == moved


@pytest.mark.asyncio
async def test_suspending_twice_keeps_what_the_first_pass_recorded(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """A second pass over an already suspended watch finds everything
    stopped. It must not overwrite the record with "nothing to restore"."""
    subs_db, service = subs_stack

    def getter():
        return scheduling_db

    owned = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01", origin_story_id=1,
        scheduling_db_getter=getter)
    for _ in range(2):
        await set_tracking_suspended(
            service, dreams_db, suspended=True, scheduling_db_getter=getter)

    await set_tracking_suspended(
        service, dreams_db, suspended=False, scheduling_db_getter=getter,
        now=NOW)

    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 1
    assert _reminder_states(scheduling_db) == {
        str(owned["tracked_item_id"]): True}


@pytest.mark.asyncio
async def test_suspension_is_recorded_before_anything_is_stopped(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Write-ahead: if the switch is interrupted between stopping a source
    and recording it, resume would take the stop for the user's and never
    undo it. The record therefore lands first -- restoring something that
    was never stopped is a harmless no-op."""
    import json

    subs_db, service = subs_stack
    owned = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    seen_records: list = []
    real_update = service.update_source

    async def recording_update(source_id, payload):
        seen_records.append(dreams_db.get_tracked_item(
            owned["tracked_item_id"])["suspension"])
        return await real_update(source_id, payload)

    service.update_source = recording_update

    await set_tracking_suspended(service, dreams_db, suspended=True)

    assert json.loads(seen_records[0])["subscription"] is True
    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 0


# --- a reminder is never re-made behind the user's back (review items 4, 5) -----
#
# Re-tracking used to "converge" the linked reminder on the instant it
# would compute today, and treated any reminder not stored at exactly that
# instant as stale. So a reminder that had already FIRED was re-created
# whenever its stored instant had another shape (a legacy date-only value,
# or one computed under another timezone), a reminder the user had
# rescheduled was disabled and replaced, and one they had disabled was
# re-armed. What moves a reminder now is the EVENT DATE moving, nothing
# else.


async def _question_with_reminder(dreams_db, scheduling_db):
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01", origin_story_id=7,
        scheduling_db_getter=lambda: scheduling_db)
    (task,) = _linked_reminders(scheduling_db, item_id)
    return item_id, task


async def _retrack_question(dreams_db, scheduling_db, event_date="2026-10-01"):
    return await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date=event_date, origin_story_id=7,
        scheduling_db_getter=lambda: scheduling_db)


@pytest.mark.asyncio
async def test_fired_legacy_date_only_reminder_is_never_made_again(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Item 5. The shape already on dev: ``run_at`` stored date-only, read
    back as midnight UTC. Once that reminder has fired, neither a re-track
    nor an off-then-on may create a second one for the same event."""
    _subs_db, service = subs_stack
    item_id, task = await _question_with_reminder(dreams_db, scheduling_db)
    scheduling_db.update_reminder_task(
        task["id"], run_at="2026-09-24", next_run_at="2026-09-24")
    scheduling_db.mark_reminder_dispatched(task["id"], NOW)

    assert await _retrack_question(dreams_db, scheduling_db) == item_id
    await _off_then_on(service, dreams_db, scheduling_db)
    assert await _retrack_question(dreams_db, scheduling_db) == item_id

    (only,) = _linked_reminders(scheduling_db, item_id)
    assert only["id"] == task["id"]
    assert only["enabled"] is False, "a fired one-time reminder stays consumed"


@pytest.mark.asyncio
async def test_fired_reminder_is_never_made_again_after_a_timezone_change(
        tmp_path, dreams_db, settings, scheduling_db, local_zone):
    local_zone("America/Los_Angeles")
    item_id, task = await _question_with_reminder(dreams_db, scheduling_db)
    scheduling_db.mark_reminder_dispatched(task["id"], NOW)

    local_zone("Asia/Tokyo")
    await _retrack_question(dreams_db, scheduling_db)

    (only,) = _linked_reminders(scheduling_db, item_id)
    assert only["id"] == task["id"]


@pytest.mark.asyncio
async def test_retrack_leaves_a_reminder_the_user_disabled_by_hand(
        tmp_path, dreams_db, settings, scheduling_db):
    """Item 4, the re-track half."""
    item_id, task = await _question_with_reminder(dreams_db, scheduling_db)
    scheduling_db.update_reminder_task(task["id"], enabled=False)

    await _retrack_question(dreams_db, scheduling_db)

    (only,) = _linked_reminders(scheduling_db, item_id)
    assert only["enabled"] is False, "the user's own disable stands"


@pytest.mark.asyncio
async def test_retrack_leaves_a_reminder_the_user_rescheduled(
        tmp_path, dreams_db, settings, scheduling_db):
    """Item 4, the re-track half."""
    item_id, task = await _question_with_reminder(dreams_db, scheduling_db)
    moved = datetime(2026, 9, 20, 18, 30, tzinfo=UTC)
    scheduling_db.update_reminder_task(
        task["id"], run_at=moved, next_run_at=moved)

    await _retrack_question(dreams_db, scheduling_db)

    (only,) = _linked_reminders(scheduling_db, item_id)
    assert only["id"] == task["id"], "not disabled and replaced"
    assert only["enabled"] is True
    assert datetime.fromisoformat(only["run_at"]) == moved


@pytest.mark.asyncio
async def test_event_moving_away_and_back_does_not_remake_a_fired_reminder(
        tmp_path, dreams_db, settings, scheduling_db):
    """The event date is what moves a reminder -- but moving back to a date
    whose reminder has already fired must not deliver that nudge again."""
    item_id, task = await _question_with_reminder(dreams_db, scheduling_db)
    scheduling_db.mark_reminder_dispatched(task["id"], NOW)

    await _retrack_question(dreams_db, scheduling_db, event_date="2026-11-01")
    await _retrack_question(dreams_db, scheduling_db, event_date="2026-10-01")

    reminders = _linked_reminders(scheduling_db, item_id)
    assert len(reminders) == 2, "the fired one and the superseded one"
    assert [r["enabled"] for r in reminders] == [False, False]


# --- a tracked URL is never a trusted origin (review of #2921, item 2) -----------
#
# The track-time check alone was not enough: the watchlist loop trusts a
# subscription's own source on every fetch, so a public URL that later
# resolved to a private address was fetched all the same. A subscription
# Dreams creates is marked "nobody typed this URL", the fetch honours the
# mark, and the master switch re-checks before it re-enables one.


@pytest.fixture()
def wire(monkeypatch):
    """Fake the HTTP transport under the real fetch; record what was sent."""
    import httpx

    sent: list[str] = []

    async def send(self, request, **kwargs):
        sent.append(str(request.url))
        return httpx.Response(
            200, request=request, headers={"content-type": "text/html"},
            content=b"<html><body><p>fares from 89</p></body></html>")

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return sent


def _resolve_everything_to(monkeypatch, address: str) -> None:
    from tldw_chatbook.Utils import egress

    async def resolve(_host):
        return [address]

    monkeypatch.setattr(egress, "_resolve_async", resolve)


@pytest.mark.asyncio
async def test_tracked_page_is_refused_at_fetch_time_once_it_resolves_private(
        tmp_path, dreams_db, settings, subs_stack, monkeypatch, wire):
    """The reviewer's attack, end to end through the real check path: the
    URL is public when tracked, and its host later resolves to a private
    address."""
    subs_db, service = subs_stack
    tracked = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal")
    first = await service.launch_run(source_id=tracked["subscription_id"])
    await service.execute_run(first["run_id"])
    assert wire == [_URL], "fixture: the public page is fetched"

    _resolve_everything_to(monkeypatch, "192.168.1.1")
    second = await service.launch_run(source_id=tracked["subscription_id"])
    await service.execute_run(second["run_id"])

    assert wire == [_URL], "no request left for the private address"
    row = subs_db.get_subscription(tracked["subscription_id"])
    assert "network safety policy" in str(row["last_error"])


@pytest.mark.asyncio
async def test_users_own_private_source_is_still_fetched_after_tracking_it(
        tmp_path, dreams_db, settings, subs_stack, monkeypatch, wire):
    """Attaching a Dreams watch to a subscription the user configured must
    not take its trust away: it is still the user's source."""
    subs_db, service = subs_stack
    private_url = "http://nas.internal/status"
    _resolve_everything_to(monkeypatch, "192.168.1.10")
    existing = await service.create_source(
        {"name": "My NAS", "url": private_url, "source_type": "url",
         "url_provenance": UrlProvenance.USER_ENTERED})
    tracked = await track_page(
        service, dreams_db, url=private_url, title="NAS", intent="topic")
    assert tracked["outcome"] == "attached"

    run = await service.launch_run(source_id=existing["source_id"])
    await service.execute_run(run["run_id"])

    assert wire == [private_url]


@pytest.mark.asyncio
async def test_resume_does_not_reenable_a_source_that_is_now_private(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """The master switch re-enables a subscription Dreams stored, so it
    asks the same question track_page asks before a re-enable. A source
    that is no longer public stays disabled, the wrapper says why, and the
    next switch-on asks again (round 3, T19: not every pass)."""
    import json

    from tldw_chatbook.Dreams.dreams_view import (
        format_tracked_row,
        list_tracked_updates,
    )

    subs_db, service = subs_stack

    def getter():
        return scheduling_db

    owned = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01", origin_story_id=1,
        scheduling_db_getter=getter)
    await set_tracking_suspended(
        service, dreams_db, suspended=True, scheduling_db_getter=getter)

    _resolve_everything_to(monkeypatch, "10.0.0.5")
    await set_tracking_suspended(
        service, dreams_db, suspended=False, scheduling_db_getter=getter,
        now=NOW)

    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 0
    record = json.loads(dreams_db.get_tracked_item(
        owned["tracked_item_id"])["suspension"])
    assert record["subscription"] is True, "still owed a restore"
    assert record["held"] == "private"
    assert _reminder_states(scheduling_db) == {
        str(owned["tracked_item_id"]): True}, (
        "the event reminder does not depend on the page; it is restored"
    )
    (row,) = list_tracked_updates(dreams_db)
    assert "paused" in format_tracked_row(row)
    assert "not a public" in format_tracked_row(row)

    # The host is public again: switching off and on completes the restore.
    _resolve_everything_to(monkeypatch, "93.184.216.34")
    await _off_then_on(service, dreams_db, scheduling_db)

    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 1
    assert dreams_db.get_tracked_item(
        owned["tracked_item_id"])["suspension"] is None
    (row,) = list_tracked_updates(dreams_db)
    assert "paused" not in format_tracked_row(row)


def _stored_options(subs_db, subscription_id) -> dict:
    import json

    return json.loads(
        subs_db.get_subscription(subscription_id)["processing_options"] or "{}")


@pytest.mark.asyncio
async def test_subscription_dreams_creates_records_that_nobody_typed_its_url(
        tmp_path, dreams_db, settings, subs_stack):
    subs_db, service = subs_stack

    tracked = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal")

    assert _stored_options(subs_db, tracked["subscription_id"]) == {
        "url_provenance": "unknown"}


@pytest.mark.asyncio
async def test_reenabling_marks_a_source_made_before_the_mark_existed(
        tmp_path, dreams_db, settings, subs_stack):
    """A dream-created source from an earlier build carries no mark. The
    first time Dreams turns it back on it gains one, and options it
    already had are kept."""
    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal")
    subs_db.update_subscription(
        first["subscription_id"], processing_options={"max_items": 5})
    await untrack(service, dreams_db, first["tracked_item_id"])

    again = await track_page(
        service, dreams_db, url=_URL, title="Cheap flights", intent="deal")

    assert again["outcome"] == "re-enabled"
    assert _stored_options(subs_db, first["subscription_id"]) == {
        "max_items": 5, "url_provenance": "unknown"}


# --- a check in flight honours the master switch (review of #2921, item 3a) -----
#
# The projection emits no check while Dreams is off, but a check already
# dispatched never looked at the flag again: it went on to search, call the
# judge, and send "Tracked update" after the user had switched Dreams off.


def _run_rows(dreams_db, item_id) -> list[str]:
    return [run["status"] for run in dreams_db.list_recent_track_runs(item_id)]


@pytest.mark.asyncio
async def test_track_check_does_not_start_while_dreams_is_disabled(
        dreams_db, settings):
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    settings["enabled"] = False
    search = _FakeSearch(_SET_A)

    result = await run_track_check(
        _track_deps(dreams_db, perform_search=search), item_id)

    assert result == {"status": "disabled", "notified": False}
    assert search.calls == []
    assert _run_rows(dreams_db, item_id) == [], (
        "nothing ran, so no run row and the watch stays due"
    )
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}


@pytest.mark.asyncio
async def test_disabling_dreams_during_the_search_stops_before_the_judge(
        dreams_db, settings):
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    dreams_db.insert_track_run(
        item_id, status="baseline", digest_hash="an-older-result-set")
    chat = _FakeChat()
    recorder = _DispatchRecorder()

    class _SearchWhileSwitchedOff(_FakeSearch):
        def __call__(self, engine, query, **kwargs):
            settings["enabled"] = False  # the user flips the switch mid-search
            return super().__call__(engine, query, **kwargs)

    result = await run_track_check(
        _track_deps(dreams_db, perform_search=_SearchWhileSwitchedOff(_SET_A),
                    chat=chat, dispatch_getter=recorder), item_id)

    assert result == {"status": "disabled", "notified": False}
    assert chat.calls == [], "no judge call after the switch went off"
    assert recorder.dispatched == []
    assert _run_rows(dreams_db, item_id) == ["baseline"], (
        "the interrupted check records nothing; its anchor is untouched"
    )


@pytest.mark.asyncio
async def test_disabling_dreams_during_the_judge_sends_no_tracked_update(
        dreams_db, settings):
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    dreams_db.insert_track_run(
        item_id, status="baseline", digest_hash="an-older-result-set")
    recorder = _DispatchRecorder()

    class _JudgeWhileSwitchedOff(_FakeChat):
        def __call__(self, **kwargs):
            settings["enabled"] = False  # switched off while the judge runs
            return super().__call__(**kwargs)

    result = await run_track_check(
        _track_deps(dreams_db, perform_search=_FakeSearch(_SET_A),
                    chat=_JudgeWhileSwitchedOff(), dispatch_getter=recorder),
        item_id)

    assert result == {"status": "disabled", "notified": False}
    assert recorder.dispatched == [], (
        "a disabled Dreams sends no tracked update"
    )
    assert _run_rows(dreams_db, item_id) == ["baseline"], (
        "the change is not recorded as seen: the next check, once Dreams "
        "is back on, judges and reports it"
    )


# --- the switch silences Dreams' own alert rule (review of #2921, item 3b) ------
#
# Tracking a page pins a "Change: <title>" alert rule on its subscription.
# On a source the USER owns the switch must not stop the subscription --
# so the rule kept firing "Change:" alerts for a Dreams that was off.


async def _attach_to_users_source(dreams_db, service, *, story_id=2, provenance=None):
    users_url = "https://example.com/users-own"
    existing = await service.create_source(
        {"name": "User's own", "url": users_url, "source_type": "url",
         "url_provenance": provenance})
    attached = await track_page(
        service, dreams_db, url=users_url, title="Theirs", intent="topic",
        origin_story_id=story_id)
    assert attached["outcome"] == "attached"
    return existing["source_id"], attached


async def _change_rules(service, source_id) -> list[dict]:
    return [rule for rule in await service.list_alert_rules(job_id=source_id)
            if str(rule["name"]).startswith("Change: ")]


@pytest.mark.asyncio
async def test_suspending_silences_dreams_rule_on_a_users_own_source(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    subs_db, service = subs_stack
    source_id, _attached = await _attach_to_users_source(dreams_db, service)
    (rule,) = await _change_rules(service, source_id)
    assert rule["enabled"] is True, "fixture: tracking pinned the rule"

    await set_tracking_suspended(service, dreams_db, suspended=True)

    assert int(subs_db.get_subscription(source_id)["is_active"]) == 1, (
        "the user's subscription keeps running"
    )
    (rule,) = await _change_rules(service, source_id)
    assert rule["enabled"] is False, (
        "but it sends no Dreams alert while Dreams is off"
    )

    await set_tracking_suspended(
        service, dreams_db, suspended=False, now=NOW)

    (rule,) = await _change_rules(service, source_id)
    assert rule["enabled"] is True
    assert int(subs_db.get_subscription(source_id)["is_active"]) == 1


@pytest.mark.asyncio
async def test_resume_leaves_an_alert_rule_the_user_disabled_by_hand(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    _subs_db, service = subs_stack
    source_id, _attached = await _attach_to_users_source(dreams_db, service)
    (rule,) = await _change_rules(service, source_id)
    await service.update_alert_rule(rule["rule_id"], enabled=False)

    await _off_then_on(service, dreams_db, scheduling_db)

    (rule,) = await _change_rules(service, source_id)
    assert rule["enabled"] is False, "the user's own choice stands"


@pytest.mark.asyncio
async def test_two_watches_on_one_source_both_record_its_silenced_rule(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Two stories on one page share one rule. Whichever wrapper the pass
    reaches second must not find the rule already silenced and record
    "nothing to restore": retiring the first would then strand it off."""
    import json

    _subs_db, service = subs_stack
    source_id, first = await _attach_to_users_source(dreams_db, service)
    second = await track_page(
        service, dreams_db, url="https://example.com/users-own",
        title="Theirs again", intent="topic", origin_story_id=3)
    (rule,) = await _change_rules(service, source_id)

    await set_tracking_suspended(service, dreams_db, suspended=True)

    for tracked in (first, second):
        record = json.loads(dreams_db.get_tracked_item(
            tracked["tracked_item_id"])["suspension"])
        assert record["rules"] == [rule["rule_id"]]

    dreams_db.set_tracked_status(
        first["tracked_item_id"], "retired", retired_reason="manual")
    await set_tracking_suspended(
        service, dreams_db, suspended=False, now=NOW)

    (rule,) = await _change_rules(service, source_id)
    assert rule["enabled"] is True


# --- the switch cannot be cut short (review of #2921, item 3c) ------------------
#
# The suspension ran once, inline in a Settings worker that shares an
# exclusive group with the provider/model/region handlers: a second
# settings action cancelled it midway. ``reconcile_tracking`` runs the
# switch as a task the module owns -- a cancelled caller cannot stop it,
# overlapping requests coalesce, and each pass reads the live flag.


#: The bound on each wait in the parked-source tests (TASK-34929; review of
#: #3106, W2). The old 5 s / 10 s bounds were too tight: these tests take
#: about 1.2 s locally under load and ran five times slower than that on a
#: loaded host. 60 s is 10x that slow case, and well inside the lanes' 180 s
#: pytest-timeout, so a source that never parks or a pass that never ends
#: fails here, naming the wait, instead of in the event loop's selector.
_PARKED_WAIT_S = 60


async def _within_bound(awaitable, what: str):
    """Await ``awaitable``; fail naming ``what`` if it takes over 60 s."""
    try:
        return await asyncio.wait_for(awaitable, _PARKED_WAIT_S)
    except TimeoutError:
        pytest.fail(f"{what} did not happen within {_PARKED_WAIT_S} s")


class _ParkedSources:
    """Wraps a watchlists service; ``get_source`` waits until released."""

    def __init__(self, service):
        self._service = service
        self.parked = asyncio.Event()
        self.release = asyncio.Event()

    def __getattr__(self, name):
        return getattr(self._service, name)

    async def get_source(self, source_id):
        self.parked.set()
        await self.release.wait()
        return await self._service.get_source(source_id)


@pytest.mark.asyncio
async def test_reconcile_follows_the_live_switch(
        tmp_path, dreams_db, settings, subs_stack):
    from tldw_chatbook.Dreams.track_service import reconcile_tracking

    subs_db, service = subs_stack
    owned = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    settled: list[int] = []

    settings["enabled"] = False
    await reconcile_tracking(
        service, dreams_db, on_settled=lambda: settled.append(1))
    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 0

    settings["enabled"] = True
    await reconcile_tracking(
        service, dreams_db, now=NOW, on_settled=lambda: settled.append(1))
    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 1
    assert settled == [1, 1], "each settled pass reports once"


@pytest.mark.asyncio
async def test_cancelling_a_reconcile_caller_does_not_cut_the_switch_short(
        tmp_path, dreams_db, settings, subs_stack):
    from tldw_chatbook.Dreams import track_service

    subs_db, service = subs_stack
    owned = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    parked = _ParkedSources(service)
    settings["enabled"] = False

    caller = asyncio.create_task(
        track_service.reconcile_tracking(parked, dreams_db))
    await _within_bound(parked.parked.wait(), "the first pass parking on get_source")
    caller.cancel()  # e.g. another settings action cancelled the worker
    with pytest.raises(asyncio.CancelledError):
        await caller
    parked.release.set()
    await _within_bound(track_service._RECONCILE_TASK, "the owned reconcile task finishing")

    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 0, (
        "the switch finished although its caller was cancelled"
    )
    assert dreams_db.get_tracked_item(
        owned["tracked_item_id"])["suspension"] is not None


@pytest.mark.asyncio
async def test_overlapping_reconciles_settle_on_the_latest_switch(
        tmp_path, dreams_db, settings, subs_stack):
    """Off, then on again while the "off" pass is still running: the
    second request must not run beside the first (they would undo each
    other), and the watches must end where the LATEST switch says."""
    from tldw_chatbook.Dreams import track_service

    subs_db, service = subs_stack
    owned = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    parked = _ParkedSources(service)

    settings["enabled"] = False
    first = asyncio.create_task(
        track_service.reconcile_tracking(parked, dreams_db))
    await _within_bound(parked.parked.wait(), "the first pass parking on get_source")
    settings["enabled"] = True
    second = asyncio.create_task(
        track_service.reconcile_tracking(parked, dreams_db, now=NOW))
    await asyncio.sleep(0)
    parked.release.set()
    await _within_bound(asyncio.gather(first, second), "both reconcile requests settling")

    assert int(subs_db.get_subscription(
        owned["subscription_id"])["is_active"]) == 1
    assert dreams_db.get_tracked_item(
        owned["tracked_item_id"])["suspension"] is None


# --- the region in a tracked question (review of #2921, item 8) -----------------


def test_tracked_question_renders_no_region_that_fails_the_text_rule(settings):
    """``{region}`` is filled from ``[dreams] region`` at check time and the
    result goes out as a web search: an unusable region is left out."""
    from tldw_chatbook.Dreams.track_service import _render_query

    settings["region"] = "kyoto"
    assert _render_query("flights to {region}") == "flights to kyoto"
    settings["region"] = "kyoto\u202e"
    assert _render_query("flights to {region}") == "flights to "
    settings["region"] = "kyoto\nsite:internal.example"
    assert _render_query("flights to {region}") == "flights to "


# === Round 3, part 2: the suspension record redesign ============================
#
# The record now MERGES (what it already owes plus what is running now), every
# recorded stop is re-applied until it has landed, and a wrapper carries an
# explicit ``settled`` marker: "off" once every stop is in place, "on" once a
# resume has restored all it may (what is left is held). The boot check asks
# for live wrappers that are not settled in the direction the switch says.


def _sched_getter(scheduling_db):
    return lambda: scheduling_db


def _record(dreams_db, item_id) -> dict | None:
    raw = dreams_db.get_tracked_item(item_id)["suspension"]
    return json.loads(raw) if raw else None


def _is_active(subs_db, subscription_id) -> int:
    return int(subs_db.get_subscription(subscription_id)["is_active"])


def _boot_needs_switch(dreams_db, *, enabled: bool) -> bool:
    from tldw_chatbook.Dreams import track_reconcile

    return track_reconcile.needs_switch(dreams_db, enabled=enabled)


async def _event_page(dreams_db, service, scheduling_db, *, story_id=1):
    return await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01", origin_story_id=story_id,
        scheduling_db_getter=_sched_getter(scheduling_db))


async def _switch(service, dreams_db, scheduling_db, *, off: bool) -> int:
    return await set_tracking_suspended(
        service, dreams_db, suspended=off,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)


def _fail_resolution(monkeypatch) -> None:
    from tldw_chatbook.Utils import egress

    async def resolve(_host):
        raise OSError("name resolution failed")

    monkeypatch.setattr(egress, "_resolve_async", resolve)


# --- T2: a left-over record must not make the next switch-off skip --------------


@pytest.mark.asyncio
async def test_switch_off_after_a_held_resume_stops_everything_again(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """The reviewer's reproduction. Off; on while DNS fails (the reminder is
    re-armed, the rule re-enabled, the source held); off again. The record
    left by the held resume made the second switch-off skip the wrapper:
    the event reminder then fired with Dreams off."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]

    await _switch(service, dreams_db, scheduling_db, off=True)
    _fail_resolution(monkeypatch)
    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _reminder_states(scheduling_db) == {str(item_id): True}
    assert [rule["enabled"] for rule in await _change_rules(service, source_id)] == [True]
    assert _record(dreams_db, item_id)["held"] == "dns_failure"

    switched = await _switch(service, dreams_db, scheduling_db, off=True)

    assert switched == 1
    assert _reminder_states(scheduling_db) == {str(item_id): False}
    assert [rule["enabled"] for rule in await _change_rules(service, source_id)] == [False]
    assert _is_active(subs_db, source_id) == 0
    record = _record(dreams_db, item_id)
    assert record["settled"] == "off"
    assert record["subscription"] is True, "the earlier debt is kept, not lost"
    assert len(record["reminders"]) == 1 and len(record["rules"]) == 1
    assert "held" not in record
    assert _boot_needs_switch(dreams_db, enabled=False) is False


# --- T3: a stop that failed or was interrupted is retried -----------------------


@pytest.mark.asyncio
async def test_a_stop_that_failed_once_is_retried_by_the_next_pass(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """``update_source`` raises "database is locked" once. The record said
    the source was stopped, the source was still running, and nothing ever
    tried again."""
    import sqlite3

    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    real_update = service.update_source
    failures = [sqlite3.OperationalError("database is locked")]

    async def flaky_update(source, payload):
        if failures and payload.get("active") is False:
            raise failures.pop()
        return await real_update(source, payload)

    service.update_source = flaky_update

    await _switch(service, dreams_db, scheduling_db, off=True)
    assert _is_active(subs_db, source_id) == 1, "fixture: the stop failed"
    assert _record(dreams_db, item_id).get("settled") != "off"
    assert _boot_needs_switch(dreams_db, enabled=False) is True, (
        "the unfinished stop is visible to the next start"
    )

    assert await _switch(service, dreams_db, scheduling_db, off=True) == 1

    assert _is_active(subs_db, source_id) == 0
    assert _record(dreams_db, item_id)["settled"] == "off"
    assert _boot_needs_switch(dreams_db, enabled=False) is False


@pytest.mark.asyncio
async def test_a_stop_interrupted_after_its_record_is_finished_later(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """A quit or crash between the write-ahead record and the stop: the
    record names what is owed, nothing was stopped yet."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    (reminder,) = _linked_reminders(scheduling_db, item_id)
    dreams_db.set_tracked_suspension(item_id, json.dumps({
        "subscription": True, "reminders": [reminder["id"]], "rules": []}))

    assert _boot_needs_switch(dreams_db, enabled=False) is True
    assert await _switch(service, dreams_db, scheduling_db, off=True) == 1

    assert _is_active(subs_db, source_id) == 0
    assert _reminder_states(scheduling_db) == {str(item_id): False}
    assert [rule["enabled"] for rule in await _change_rules(service, source_id)] == [False]
    assert _record(dreams_db, item_id)["settled"] == "off"


@pytest.mark.asyncio
async def test_a_suspend_without_the_watchlists_service_is_finished_later(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """No service at suspend time: the record used to say "subscription:
    false" for good, and the source was never stopped."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]

    await _switch(None, dreams_db, scheduling_db, off=True)
    assert _reminder_states(scheduling_db) == {str(item_id): False}, (
        "what could be stopped was stopped"
    )
    assert _boot_needs_switch(dreams_db, enabled=False) is True

    await _switch(service, dreams_db, scheduling_db, off=True)

    assert _is_active(subs_db, source_id) == 0
    assert _record(dreams_db, item_id)["subscription"] is True
    assert _boot_needs_switch(dreams_db, enabled=False) is False


@pytest.mark.asyncio
async def test_a_rule_deleted_mid_pass_does_not_strand_the_suspend(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """``update_alert_rule`` raises KeyError for a rule that vanished after
    it was listed: the source must still be stopped and the wrapper
    settled."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]

    async def rule_is_gone(rule_id, **fields):
        raise KeyError(f"Watchlist alert rule not found: {rule_id}")

    service.update_alert_rule = rule_is_gone

    assert await _switch(service, dreams_db, scheduling_db, off=True) == 1

    assert _is_active(subs_db, source_id) == 0
    assert _reminder_states(scheduling_db) == {str(item_id): False}
    assert _record(dreams_db, item_id)["settled"] == "off"
    assert _boot_needs_switch(dreams_db, enabled=False) is False


# --- T13: decide once per subscription, from all its riders --------------------


@pytest.mark.asyncio
async def test_mixed_ownership_riders_stop_and_restore_their_source_once(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """A ``created_by_dreams = 0`` wrapper and a ``= 1`` wrapper on one
    Dreams-created source (the create race). The newer, unowned one was
    processed first and marked the source handled; the owner then recorded
    a stop that never happened, and the source kept running."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    source_id = owned["subscription_id"]
    racer = dreams_db.create_tracked_item(
        origin_story_id=99, mechanism="page", intent="topic",
        subscription_id=source_id, cadence_seconds=43200,
        created_by_dreams=0)

    assert await _switch(service, dreams_db, scheduling_db, off=True) == 2

    assert _is_active(subs_db, source_id) == 0
    for item_id in (owned["tracked_item_id"], racer):
        record = _record(dreams_db, item_id)
        assert record["subscription"] is True and record["settled"] == "off"

    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _is_active(subs_db, source_id) == 1
    assert _record(dreams_db, owned["tracked_item_id"]) is None
    assert _record(dreams_db, racer) is None


# --- T19: a held watch is settled, not retried on every start -------------------


@pytest.mark.asyncio
async def test_a_held_watch_is_settled_and_asked_again_by_a_pass_not_by_a_start(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """Round 3, T19, as TASK-34922 changed it (owner decision 2026-10-13:
    retry private and metadata holds on later cycles). A held watch is
    settled "on", so the boot check (``needs_switch``, the start-time
    question this test asks) does not ask for a pass because of it; a
    start that has other reconcile work runs a pass that looks it up too,
    which this test does not exercise. A resume pass that runs (and the
    daily cycle's retry) asks again, and the page comes back once its name
    resolves publicly."""
    from tldw_chatbook.Utils import egress

    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    lookups: list[str] = []
    answer = ["10.0.0.5"]

    async def resolve(host):
        lookups.append(host)
        return list(answer)

    monkeypatch.setattr(egress, "_resolve_async", resolve)
    await _switch(service, dreams_db, scheduling_db, off=False)
    record = _record(dreams_db, item_id)
    assert (record["held"], record["settled"]) == ("private", "on")
    assert _boot_needs_switch(dreams_db, enabled=True) is False, (
        "a held watch does not make every start import the track chain"
    )

    lookups.clear()
    await _switch(service, dreams_db, scheduling_db, off=False)
    assert lookups, "a resume pass asks again"
    assert _is_active(subs_db, source_id) == 0, "still private: still held"
    assert _record(dreams_db, item_id)["held"] == "private"

    answer[:] = ["93.184.216.34"]
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _is_active(subs_db, source_id) == 1
    assert _record(dreams_db, item_id) is None


# --- T4: the watch's subscription was deleted by hand ---------------------------


@pytest.mark.asyncio
async def test_suspend_carries_on_when_the_subscription_was_deleted(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Deleting the "Dreams: ..." source is the one by-hand action
    Watchlists offers on it. The suspend pass used to abort on the missing
    row before it recorded anything or disabled the reminder -- and the
    start-up check then found work on every start. A really deleted
    source, not a fake that always returns one."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    await service.delete_source(source_id)
    assert subs_db.get_subscription(source_id) is None

    assert await _switch(service, dreams_db, scheduling_db, off=True) == 1

    assert _reminder_states(scheduling_db) == {str(item_id): False}
    record = _record(dreams_db, item_id)
    assert record["settled"] == "off"
    assert record["subscription"] is False, "there is no subscription to stop"
    assert _boot_needs_switch(dreams_db, enabled=False) is False

    # Back on: the reminder returns and nothing is owed to a deleted source.
    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _reminder_states(scheduling_db) == {str(item_id): True}
    assert _record(dreams_db, item_id) is None


# --- T9: an auto-paused Dreams source is stopped too ---------------------------


@pytest.mark.asyncio
async def test_suspend_stops_an_auto_paused_dreams_source(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Watchlists auto-pauses a source after repeated failures (``is_paused``;
    ``is_active`` stays 1). The switch read the NORMALIZED ``active``
    (``is_active and not paused``), took the source for "not running",
    recorded nothing and left ``is_active = 1`` -- so one press of Resume in
    Watchlists had the source fetching again with Dreams off."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    subs_db.update_subscription(source_id, is_paused=True)
    source = await service.get_source(source_id)
    assert (source["active"], source["paused"]) == (False, True)

    await _switch(service, dreams_db, scheduling_db, off=True)

    assert _is_active(subs_db, source_id) == 0, (
        "Resume in Watchlists must not start a Dreams source while Dreams is off"
    )
    assert _record(dreams_db, item_id)["subscription"] is True

    await _switch(service, dreams_db, scheduling_db, off=False)

    row = subs_db.get_subscription(source_id)
    assert int(row["is_active"]) == 1
    assert int(row["is_paused"]) == 1, "its pause is Watchlists' own and stays"
    assert _record(dreams_db, item_id) is None


# --- T12: the record is trimmed as each restore lands ---------------------------


@pytest.mark.asyncio
async def test_a_restored_reminder_leaves_the_record_even_if_resume_then_fails(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Reproduction (a). The reminder is restored, then ``update_source``
    fails. The record was only rewritten at the very end, so it still
    listed the reminder; the user then disabled that reminder by hand, and
    the next resume re-armed it."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    (reminder,) = _linked_reminders(scheduling_db, item_id)
    real_update = service.update_source
    failures = [RuntimeError("database is locked")]

    async def flaky_update(source, payload):
        if failures and payload.get("active") is True:
            raise failures.pop()
        return await real_update(source, payload)

    service.update_source = flaky_update

    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _reminder_states(scheduling_db) == {str(item_id): True}
    record = _record(dreams_db, item_id)
    assert record["reminders"] == [] and record["rules"] == [], (
        "what was restored is no longer owed"
    )
    assert record["subscription"] is True and "settled" not in record

    scheduling_db.update_reminder_task(reminder["id"], enabled=False)  # the user
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _is_active(subs_db, source_id) == 1, "the retry finished the resume"
    assert _reminder_states(scheduling_db) == {str(item_id): False}, (
        "and did not undo the user's own disable"
    )
    assert _record(dreams_db, item_id) is None


@pytest.mark.asyncio
async def test_a_reminder_superseded_by_an_event_move_is_not_rearmed_later(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Reproduction (b). A resume runs with no scheduling DB, so the record
    keeps the reminder id; the event date then moves on a re-track; the
    next resume re-armed the OLD reminder -- two armed reminders for two
    dates."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    await set_tracking_suspended(
        service, dreams_db, suspended=False, scheduling_db_getter=None,
        now=NOW)
    assert len(_record(dreams_db, item_id)["reminders"]) == 1, (
        "fixture: the reminder could not be restored and is still owed"
    )

    await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-11-15", origin_story_id=1,
        scheduling_db_getter=_sched_getter(scheduling_db))
    await _switch(service, dreams_db, scheduling_db, off=False)

    armed = [task for task in _linked_reminders(scheduling_db, item_id)
             if task["enabled"]]
    assert [task["body"] for task in armed] == [
        "Tracked Dreams event on 2026-11-15"]
    assert _record(dreams_db, item_id) is None


@pytest.mark.asyncio
async def test_a_resume_cut_short_after_a_restore_no_longer_owes_it(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """The record is rewritten as each restore lands, not once at the end:
    a pass that is cut off right after re-arming the reminder (here the
    task is cancelled inside the subscription step) must not leave the
    reminder listed as still owed."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)

    async def cut_off(source_id):
        raise asyncio.CancelledError

    service.get_source = cut_off

    with pytest.raises(asyncio.CancelledError):
        await _switch(service, dreams_db, scheduling_db, off=False)

    assert _reminder_states(scheduling_db) == {str(item_id): True}
    record = _record(dreams_db, item_id)
    assert record["reminders"] == [] and record["rules"] == []
    assert record["subscription"] is True, "the source is still owed"
    assert "settled" not in record, "so the next pass finishes the resume"


# --- T1: sources Dreams created before the egress mark existed -----------------
#
# The mark ("nobody typed this URL: no trusted egress origin") was stamped at
# creation and on a re-enable only. A source created by an earlier build had
# none, was never even checked for a public destination, and stayed a trusted
# origin for as long as the user never toggled Dreams.


def _as_an_older_build_left_it(subs_db, dreams_db, subscription_id) -> None:
    with subs_db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET processing_options = NULL WHERE id = ?",
            (subscription_id,))
    with dreams_db.transaction() as conn:
        conn.execute("UPDATE dream_tracked_items SET egress_marked = 0")
    assert subs_db.get_subscription(subscription_id)["processing_options"] is None


@pytest.mark.asyncio
async def test_attaching_to_an_unmarked_dreams_source_marks_it_before_any_fetch(
        tmp_path, dreams_db, settings, subs_stack, monkeypatch, wire):
    """The reviewer's reproduction: a second story tracked at the same URL
    returned ``attached`` and left the options NULL; the host then resolved
    to 192.168.1.1 and ``execute_run`` sent the request."""
    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    source_id = first["subscription_id"]
    _as_an_older_build_left_it(subs_db, dreams_db, source_id)

    second = await track_page(
        service, dreams_db, url=_URL, title="Flights again", intent="deal",
        origin_story_id=8)

    assert second["outcome"] == "attached"
    assert _stored_options(subs_db, source_id) == {"url_provenance": "unknown"}
    assert dreams_db.subscriptions_awaiting_egress_mark() == []
    _resolve_everything_to(monkeypatch, "192.168.1.1")
    run = await service.launch_run(source_id=source_id)
    await service.execute_run(run["run_id"])
    assert wire == [], "no request left for the private address"


@pytest.mark.asyncio
async def test_egress_backfill_marks_dreams_sources_once_and_then_costs_nothing(
        tmp_path, dreams_db, settings, subs_stack, monkeypatch, wire):
    from tldw_chatbook.Dreams.track_service import backfill_egress_marks

    subs_db, service = subs_stack
    live = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    gone = await track_page(
        service, dreams_db, url="https://example.com/gone", title="Gone",
        intent="deal", origin_story_id=8)
    retired = await track_page(
        service, dreams_db, url="https://example.com/old", title="Old",
        intent="deal", origin_story_id=9)
    await untrack(service, dreams_db, retired["tracked_item_id"])
    subs_db.update_subscription(
        retired["subscription_id"], processing_options={"max_items": 5})
    for tracked in (live, gone, retired):
        _as_an_older_build_left_it(subs_db, dreams_db, tracked["subscription_id"])
    subs_db.update_subscription(
        retired["subscription_id"], processing_options={"max_items": 5})
    await service.delete_source(gone["subscription_id"])
    settings["enabled"] = False  # the sources exist whatever the switch says

    assert await backfill_egress_marks(service, dreams_db) == 2

    assert _stored_options(subs_db, live["subscription_id"]) == {
        "url_provenance": "unknown"}
    assert _stored_options(subs_db, retired["subscription_id"]) == {
        "max_items": 5, "url_provenance": "unknown"}, "its own options are kept"
    assert _is_active(subs_db, retired["subscription_id"]) == 0, (
        "marking never turns a source on"
    )
    assert dreams_db.tracking_boot_state()[1] is False, (
        "nothing is pending: the next start does not come back here"
    )

    service.update_source = None  # any further write would raise
    assert await backfill_egress_marks(service, dreams_db) == 0

    # And the mark does its job: the page now resolves somewhere private.
    _resolve_everything_to(monkeypatch, "192.168.1.1")
    run = await service.launch_run(source_id=live["subscription_id"])
    await service.execute_run(run["run_id"])
    assert wire == []


@pytest.mark.asyncio
async def test_egress_backfill_waits_for_the_watchlists_service(
        tmp_path, dreams_db, settings, subs_stack):
    from tldw_chatbook.Dreams.track_service import backfill_egress_marks

    subs_db, service = subs_stack
    tracked = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal")
    _as_an_older_build_left_it(subs_db, dreams_db, tracked["subscription_id"])

    assert await backfill_egress_marks(None, dreams_db) == 0

    assert dreams_db.subscriptions_awaiting_egress_mark() == [
        tracked["subscription_id"]], "still pending for the next start"


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", [None, UrlProvenance.USER_ENTERED])
async def test_a_users_own_source_is_never_marked(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, provenance):
    """Attaching, the backfill and the switch all leave a source the user
    created exactly as trusted as the user made it."""
    from tldw_chatbook.Dreams.track_service import backfill_egress_marks

    subs_db, service = subs_stack
    source_id, attached = await _attach_to_users_source(dreams_db, service, provenance=provenance)
    with dreams_db.transaction() as conn:
        conn.execute("UPDATE dream_tracked_items SET egress_marked = 0")

    assert dreams_db.subscriptions_awaiting_egress_mark() == []
    assert await backfill_egress_marks(service, dreams_db) == 0
    await _switch(service, dreams_db, scheduling_db, off=True)
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _stored_options(subs_db, source_id) == {
        "url_provenance": "user_entered" if provenance is UrlProvenance.USER_ENTERED else "unknown"}


@pytest.mark.asyncio
async def test_suspending_marks_an_unmarked_dreams_source(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    source_id = owned["subscription_id"]
    _as_an_older_build_left_it(subs_db, dreams_db, source_id)

    await _switch(service, dreams_db, scheduling_db, off=True)

    assert _is_active(subs_db, source_id) == 0
    assert _stored_options(subs_db, source_id) == {"url_provenance": "unknown"}
    assert dreams_db.subscriptions_awaiting_egress_mark() == []


# --- T11: Dreams' alert rule is known by its id, never by its name -------------
#
# The rule Dreams pins on a tracked page was recognised by name prefix
# ("Change: ") plus condition. So the user's own "Change: my price alert" on
# their own source was adopted at track time and switched off with Dreams;
# and since ``list_alert_rules(job_id=...)`` also returns GLOBAL rules, a
# matching global rule would have been silenced for every source.


async def _users_rule(service, *, job_id, name="Change: my price alert"):
    return await service.create_alert_rule(
        name=name, condition_type="items_above",
        condition_value={"threshold": 0}, job_id=job_id,
        severity="information")


def _rule_id_on_record(dreams_db, item_id):
    return dreams_db.get_tracked_item(item_id)["alert_rule_id"]


@pytest.mark.asyncio
async def test_a_users_own_change_rule_is_neither_adopted_nor_silenced(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    subs_db, service = subs_stack
    users_url = "https://example.com/users-own"
    existing = await service.create_source(
        {"name": "User's own", "url": users_url, "source_type": "url",
         "url_provenance": UrlProvenance.USER_ENTERED})
    source_id = existing["source_id"]
    theirs = await _users_rule(service, job_id=source_id)
    everywhere = await _users_rule(
        service, job_id=None, name="Change: anything, anywhere")

    tracked = await track_page(
        service, dreams_db, url=users_url, title="Theirs", intent="topic",
        origin_story_id=2)

    rules = {rule["rule_id"]: rule
             for rule in await service.list_alert_rules(job_id=source_id)}
    dreams_rule_id = _rule_id_on_record(dreams_db, tracked["tracked_item_id"])
    assert dreams_rule_id not in (theirs["rule_id"], everywhere["rule_id"]), (
        "Dreams made its own rule instead of adopting one by its name"
    )
    assert rules[dreams_rule_id]["name"] == "Change: Theirs"

    await _switch(service, dreams_db, scheduling_db, off=True)

    rules = {rule["rule_id"]: rule["enabled"]
             for rule in await service.list_alert_rules(job_id=source_id)}
    assert rules == {
        theirs["rule_id"]: True,        # the user's, on their own source
        everywhere["rule_id"]: True,    # a global rule: every source's
        dreams_rule_id: False,          # Dreams' own
    }
    assert _record(dreams_db, tracked["tracked_item_id"])["rules"] == [
        dreams_rule_id]


@pytest.mark.asyncio
async def test_a_second_story_on_the_page_shares_the_recorded_rule(
        tmp_path, dreams_db, settings, subs_stack):
    _subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    second = await track_page(
        service, dreams_db, url=_URL, title="Flights again", intent="deal",
        origin_story_id=8)

    rule_id = _rule_id_on_record(dreams_db, first["tracked_item_id"])
    assert rule_id and rule_id == _rule_id_on_record(
        dreams_db, second["tracked_item_id"])
    assert len(await _change_rules(service, first["subscription_id"])) == 1


def _seed_story(dreams_db, title: str, *, url=None, date="2026-09-20") -> int:
    """A story; ``url`` is the page a page watch of it tracks (a Dreams
    source's URL is its story's, review of #3123, M1). Two stories of one
    page need two dates: a collection holds a URL once."""
    collection = dreams_db.get_collection_by_date(date)
    collection_id = (int(collection["id"]) if collection else
                     dreams_db.create_collection(date, "manual", "d"))
    return dreams_db.insert_story(
        collection_id, title=title,
        url=url or f"https://dreams.test/{title.replace(' ', '-')}", snippet="",
        body="b", status="complete", source="web", kind="content",
        event_date=None, location=None, matched_topics=[], query="q")


@pytest.mark.asyncio
async def test_a_rule_from_before_ids_were_recorded_is_adopted_by_exact_identity(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """A wrapper made by an earlier build records no rule id. Its rule is
    recognised by EXACT identity only -- on this source, the name this
    watch's own story would have given it -- and then by id for good. A
    look-alike with any other name stays the user's."""
    subs_db, service = subs_stack
    story_id = _seed_story(dreams_db, "Festival line-up announced", url=_URL)
    tracked = await track_page(
        service, dreams_db, url=_URL, title="Festival line-up announced",
        intent="event", origin_story_id=story_id)
    item_id, source_id = tracked["tracked_item_id"], tracked["subscription_id"]
    look_alike = await _users_rule(service, job_id=source_id)
    (own,) = [rule for rule in await _change_rules(service, source_id)
              if rule["rule_id"] != look_alike["rule_id"]]
    with dreams_db.transaction() as conn:  # as the earlier build left it
        conn.execute("UPDATE dream_tracked_items SET alert_rule_id = NULL")

    await _switch(service, dreams_db, scheduling_db, off=True)

    assert _rule_id_on_record(dreams_db, item_id) == own["rule_id"]
    enabled = {rule["rule_id"]: rule["enabled"]
               for rule in await _change_rules(service, source_id)}
    assert enabled == {own["rule_id"]: False, look_alike["rule_id"]: True}


# --- T18: reminders are read by their link, not by listing them all ------------


def test_reminders_by_link_returns_only_the_named_links(scheduling_db):
    kept = scheduling_db.create_reminder_task(
        "local", "Dreams: a", schedule_kind="one_time",
        link_type="dream_tracked_item", link_id="7")
    also = scheduling_db.create_reminder_task(
        "local", "Dreams: b", schedule_kind="one_time",
        link_type="dream_tracked_item", link_id="9")
    scheduling_db.create_reminder_task(
        "local", "Dreams: c", schedule_kind="one_time",
        link_type="dream_tracked_item", link_id="70")
    scheduling_db.create_reminder_task(
        "local", "someone else's", schedule_kind="one_time",
        link_type="note", link_id="7")
    scheduling_db.create_reminder_task(
        "local", "plain", schedule_kind="one_time")

    rows = scheduling_db.list_reminder_tasks_by_link(
        "dream_tracked_item", ["7", 9])

    assert {row["id"] for row in rows} == {kept, also}
    assert scheduling_db.list_reminder_tasks_by_link(
        "dream_tracked_item", []) == []


@pytest.mark.asyncio
async def test_the_switch_never_lists_every_reminder_the_user_has(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """Each wrapper used to list ALL enabled reminders and filter them in
    Python -- once per wrapper, per pass."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)

    def no_full_listing(*args, **kwargs):
        raise AssertionError("listed every reminder")

    monkeypatch.setattr(scheduling_db, "list_reminder_tasks", no_full_listing)

    assert await _switch(service, dreams_db, scheduling_db, off=True) == 1
    assert _record(dreams_db, owned["tracked_item_id"])["settled"] == "off"
    assert len(_record(dreams_db, owned["tracked_item_id"])["reminders"]) == 1


# --- T5: "already delivered" is decided per story and event date ---------------
#
# Item 5 looked for a fired reminder among those linked to ONE wrapper, and the
# reuse lookup only finds an ACTIVE wrapper. So untrack-then-track, or a watch
# the sweep had paused, made a new wrapper, the new wrapper had no reminders,
# and the reminder that had already fired was created again -- due at once.


def _all_event_reminders(scheduling_db) -> list[dict]:
    return scheduling_db.list_reminder_tasks(owner_id="local")


@pytest.mark.asyncio
async def test_fired_reminder_is_not_made_again_by_untrack_then_track(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """(i) The reminder fires; the page is untracked; the same story is
    tracked again: a new wrapper, and the same event."""
    _subs_db, service = subs_stack
    first = await _event_page(dreams_db, service, scheduling_db)
    (fired,) = _linked_reminders(scheduling_db, first["tracked_item_id"])
    scheduling_db.mark_reminder_dispatched(fired["id"], NOW)
    await untrack(service, dreams_db, first["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))

    again = await _event_page(dreams_db, service, scheduling_db)

    assert again["tracked_item_id"] != first["tracked_item_id"]
    assert [task["id"] for task in _all_event_reminders(scheduling_db)] == [
        fired["id"]], "delivered once for this story and date: never again"


@pytest.mark.asyncio
async def test_fired_reminder_the_user_deleted_is_not_made_again(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """(ii) The user deletes the fired reminder row. Nothing linked is left
    to find, so the wrapper itself remembers the date its reminder was
    made for -- across a re-track of the live watch and across untrack."""
    _subs_db, service = subs_stack
    first = await _event_page(dreams_db, service, scheduling_db)
    item = dreams_db.get_tracked_item(first["tracked_item_id"])
    assert item["reminder_event_date"] == "2026-10-01"
    (fired,) = _linked_reminders(scheduling_db, first["tracked_item_id"])
    scheduling_db.mark_reminder_dispatched(fired["id"], NOW)
    scheduling_db.delete_reminder_task(fired["id"])

    await _event_page(dreams_db, service, scheduling_db)  # the live watch
    assert _all_event_reminders(scheduling_db) == []

    await untrack(service, dreams_db, first["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))
    await _event_page(dreams_db, service, scheduling_db)  # a new wrapper
    assert _all_event_reminders(scheduling_db) == []


@pytest.mark.asyncio
async def test_retracking_a_paused_watch_reuses_it_and_its_reminder(
        tmp_path, dreams_db, settings, scheduling_db):
    """(iii) The sweep paused the watch (three failed checks). Tracking the
    story again made a SECOND wrapper and a second armed reminder for the
    same event. The paused wrapper is the story's watch: it is reused and
    goes back to work."""
    item_id, task = await _question_with_reminder(dreams_db, scheduling_db)
    dreams_db.set_tracked_status(item_id, "paused")

    assert await _retrack_question(dreams_db, scheduling_db) == item_id

    assert dreams_db.get_tracked_item(item_id)["status"] == "active"
    assert [row["id"] for row in _all_event_reminders(scheduling_db)] == [
        task["id"]]
    assert dreams_db.count_active_tracked() == 1


@pytest.mark.asyncio
async def test_reactivating_a_paused_watch_still_honours_the_cap(
        tmp_path, dreams_db, settings, scheduling_db):
    settings["tracked_item_cap"] = 1
    item_id, _task = await _question_with_reminder(dreams_db, scheduling_db)
    dreams_db.set_tracked_status(item_id, "paused")
    await track_question(
        dreams_db, query_template="another question", intent="topic",
        origin_story_id=8)

    with pytest.raises(TrackCapReached):
        await _retrack_question(dreams_db, scheduling_db)

    assert dreams_db.get_tracked_item(item_id)["status"] == "paused"


@pytest.mark.asyncio
async def test_untrack_then_track_before_the_event_arms_one_reminder_again(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """The other side of (i): the reminder had NOT fired. Untrack disabled
    it -- Dreams did, and its record says so -- so tracking the story again
    leaves exactly one armed reminder for the event."""
    _subs_db, service = subs_stack
    first = await _event_page(dreams_db, service, scheduling_db)
    (old,) = _linked_reminders(scheduling_db, first["tracked_item_id"])
    await untrack(service, dreams_db, first["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))
    assert _record(dreams_db, first["tracked_item_id"])["reminders"] == [
        old["id"]], "the retiring wrapper records what Dreams disabled"

    again = await _event_page(dreams_db, service, scheduling_db)

    armed = [task for task in _all_event_reminders(scheduling_db)
             if task["enabled"]]
    assert len(armed) == 1
    assert armed[0]["link_id"] == str(again["tracked_item_id"])


@pytest.mark.asyncio
async def test_untrack_then_track_leaves_a_reminder_the_user_had_switched_off(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """...but a reminder the USER had disabled before untracking is not in
    that record, and tracking again does not bring the nudge back."""
    _subs_db, service = subs_stack
    first = await _event_page(dreams_db, service, scheduling_db)
    (old,) = _linked_reminders(scheduling_db, first["tracked_item_id"])
    scheduling_db.update_reminder_task(old["id"], enabled=False)  # the user
    await untrack(service, dreams_db, first["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))

    await _event_page(dreams_db, service, scheduling_db)

    assert [(task["id"], task["enabled"])
            for task in _all_event_reminders(scheduling_db)] == [
        (old["id"], False)]


@pytest.mark.asyncio
async def test_a_page_watch_and_a_question_watch_share_one_event_reminder(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """One story, one event date, one nudge -- however many ways the story
    is being watched."""
    _subs_db, service = subs_stack
    await _event_page(dreams_db, service, scheduling_db, story_id=1)
    await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01", origin_story_id=1,
        scheduling_db_getter=_sched_getter(scheduling_db))

    assert len(_all_event_reminders(scheduling_db)) == 1


@pytest.mark.asyncio
async def test_event_moving_away_and_back_before_it_fired_rearms_the_same_one(
        tmp_path, dreams_db, settings, scheduling_db):
    """Dreams disarmed the reminder when the event moved away; when the
    event moves back, that same reminder is armed again -- not a twin."""
    item_id, task = await _question_with_reminder(dreams_db, scheduling_db)

    await _retrack_question(dreams_db, scheduling_db, event_date="2026-11-01")
    await _retrack_question(dreams_db, scheduling_db, event_date="2026-10-01")

    states = {row["body"]: (row["id"], row["enabled"])
              for row in _linked_reminders(scheduling_db, item_id)}
    assert states["Tracked Dreams event on 2026-10-01"] == (task["id"], True)
    assert states["Tracked Dreams event on 2026-11-01"][1] is False
    assert len(states) == 2


# --- T16: an event date that can no longer be read -----------------------------


@pytest.mark.asyncio
async def test_unreadable_new_event_date_disarms_the_old_events_reminder(
        tmp_path, dreams_db, settings, scheduling_db):
    """Owner ruling: an unparsable date means the event is no longer known.
    The pending reminder for the OLD date is disabled -- it used to stay
    armed, because the function returned before looking at it."""
    item_id, task = await _question_with_reminder(dreams_db, scheduling_db)

    await _retrack_question(dreams_db, scheduling_db, event_date="sometime soon")

    (only,) = _linked_reminders(scheduling_db, item_id)
    assert (only["id"], only["enabled"]) == (task["id"], False)


@pytest.mark.asyncio
async def test_unreadable_new_event_date_leaves_a_fired_reminder_alone(
        tmp_path, dreams_db, settings, scheduling_db):
    item_id, task = await _question_with_reminder(dreams_db, scheduling_db)
    scheduling_db.mark_reminder_dispatched(task["id"], NOW)
    before = _linked_reminders(scheduling_db, item_id)

    await _retrack_question(dreams_db, scheduling_db, event_date="sometime soon")

    assert _linked_reminders(scheduling_db, item_id) == before


# --- T10: tracking never switches a source back on behind the user's back -------
#
# ``track_page`` re-enabled ANY disabled source Dreams had created, on the
# assumption that Dreams itself had disabled it at an untrack. It may only do
# so when Dreams can account for the disable.


@pytest.mark.asyncio
async def test_track_refuses_a_dreams_source_the_user_switched_off(
        tmp_path, dreams_db, settings, subs_stack):
    """The reviewer's reproduction: a live wrapper on source S; S disabled
    by hand in Watchlists; a second story tracked at the same URL came back
    ``re-enabled`` with ``is_active = 1``."""
    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    source_id = first["subscription_id"]
    await service.update_source(source_id, {"active": False})  # the user
    rows_before = len(dreams_db.list_tracked_items())

    with pytest.raises(TrackSourceDisabled):
        await track_page(
            service, dreams_db, url=_URL, title="Flights again",
            intent="deal", origin_story_id=8)

    assert _is_active(subs_db, source_id) == 0
    assert len(dreams_db.list_tracked_items()) == rows_before, (
        "refused before any write"
    )


@pytest.mark.asyncio
async def test_track_refuses_a_source_switched_off_before_it_was_untracked(
        tmp_path, dreams_db, settings, subs_stack):
    """The retiring wrapper records whether DREAMS disabled the source. Here
    the user had already switched it off, so untrack disabled nothing and
    tracking again has nothing of its own to undo."""
    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    source_id = first["subscription_id"]
    await service.update_source(source_id, {"active": False})  # the user
    await untrack(service, dreams_db, first["tracked_item_id"])
    assert _record(dreams_db, first["tracked_item_id"])["subscription"] is False

    with pytest.raises(TrackSourceDisabled):
        await track_page(
            service, dreams_db, url=_URL, title="Flights", intent="deal",
            origin_story_id=7)

    assert _is_active(subs_db, source_id) == 0


@pytest.mark.asyncio
async def test_track_reenables_a_source_dreams_disabled_and_settles_the_account(
        tmp_path, dreams_db, settings, subs_stack):
    """Untrack disabled it -- Dreams did, and the record says so -- so
    tracking again switches it back on. Once. If the user then switches it
    off and the watch is untracked again, that old account is spent."""
    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    source_id = first["subscription_id"]
    await untrack(service, dreams_db, first["tracked_item_id"])
    assert _record(dreams_db, first["tracked_item_id"])["subscription"] is True

    again = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)

    assert again["outcome"] == "re-enabled"
    assert _is_active(subs_db, source_id) == 1
    assert _record(dreams_db, first["tracked_item_id"])["subscription"] is False

    await service.update_source(source_id, {"active": False})  # the user
    await untrack(service, dreams_db, again["tracked_item_id"])
    with pytest.raises(TrackSourceDisabled):
        await track_page(
            service, dreams_db, url=_URL, title="Flights", intent="deal",
            origin_story_id=7)


@pytest.mark.asyncio
async def test_track_still_reenables_a_source_retired_before_records_existed(
        tmp_path, dreams_db, settings, subs_stack):
    """A wrapper retired by an earlier build left no record. With no live
    watch on the source, that is the untrack/sweep case as it always was."""
    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    await untrack(service, dreams_db, first["tracked_item_id"])
    dreams_db.set_tracked_suspension(first["tracked_item_id"], None)

    again = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)

    assert again["outcome"] == "re-enabled"
    assert _is_active(subs_db, first["subscription_id"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("same_story", [True, False], ids=["same-story", "new-story"])
async def test_tracking_again_retries_a_held_page(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch,
        same_story):
    """T19's other half: a page left off because its address was not public
    is asked about again when the user tracks it again -- and while it is
    still not public, that is a refusal, not a silent "tracked"."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    _resolve_everything_to(monkeypatch, "10.0.0.5")
    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _record(dreams_db, item_id)["held"] == "private"
    story_id = 1 if same_story else 2

    with pytest.raises(TrackPrivateURL):
        await _event_page(dreams_db, service, scheduling_db, story_id=story_id)
    assert _is_active(subs_db, source_id) == 0

    _resolve_everything_to(monkeypatch, "93.184.216.34")
    await _event_page(dreams_db, service, scheduling_db, story_id=story_id)

    assert _is_active(subs_db, source_id) == 1
    assert _record(dreams_db, item_id) is None, "the hold is over"


@pytest.mark.asyncio
@pytest.mark.parametrize("same_story", [True, False], ids=["reuse", "new-story"])
@pytest.mark.parametrize("failing", ["closing", "unsettling"])
async def test_a_re_track_never_runs_a_source_under_its_hold(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch,
        same_story, failing):
    """Review of #3101, round 2, re-track path (as S1 for the resume pass).
    A re-track whose lookup now passes un-settles the held records just
    before the source is switched on. If the closing write then fails, the
    source runs under an unsettled record the next pass finishes -- not
    under "held: private, settled: on", which the Tracked row calls paused
    and the boot check never looks at. If the un-settling write fails, the
    source stays off and the hold stands."""
    import json

    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    _resolve_everything_to(monkeypatch, "10.0.0.5")
    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _record(dreams_db, item_id)["held"] == "private"
    _resolve_everything_to(monkeypatch, "93.184.216.34")
    real = dreams_db.set_tracked_suspension

    def one_write_fails(tracked_item_id, record, **kwargs):
        if int(tracked_item_id) == item_id:
            closing = record is None
            unsettling = record is not None and "held" not in json.loads(record)
            if (closing, unsettling)[failing == "unsettling"]:
                raise RuntimeError("disk full")
        return real(tracked_item_id, record, **kwargs)

    monkeypatch.setattr(dreams_db, "set_tracked_suspension", one_write_fails)

    if failing == "closing" and same_story:
        # Round 3, T2: the source already runs and the story keeps its
        # wrapper, so the user is told the re-track worked; the next start
        # finishes the record.
        result = await _event_page(dreams_db, service, scheduling_db)
        assert (result["outcome"], result["tracked_item_id"]) == (
            "attached", item_id)
    else:
        with pytest.raises(RuntimeError):
            await _event_page(dreams_db, service, scheduling_db,
                              story_id=1 if same_story else 2)

    record = _record(dreams_db, item_id)
    if failing == "closing":
        assert _is_active(subs_db, source_id) == 1
        assert "held" not in record, "a running source is not reported as held"
        assert record.get("settled") is None, "the next pass finishes it"
        assert _boot_needs_switch(dreams_db, enabled=True) is True
    else:
        assert _is_active(subs_db, source_id) == 0, "the source stays off"
        assert (record["held"], record["settled"]) == ("private", "on")


# --- T15: mutants that used to survive -----------------------------------------
#
# Each test below kills one mutant the reviewer's script left alive.


@pytest.mark.asyncio
async def test_a_check_dispatched_while_dreams_is_off_does_not_even_sweep(
        dreams_db, settings):
    """Kills "delete the start check". The check's first act is the
    lifecycle sweep, which retires and pauses OTHER watches; the pre-search
    gate further down would still stop the search, so only the sweep shows
    whether the start gate is there."""
    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    passed = await track_question(
        dreams_db, query_template="an event long gone", intent="event",
        event_date="2026-01-01")
    settings["enabled"] = False

    result = await run_track_check(
        _track_deps(dreams_db, perform_search=_FakeSearch(_SET_A)), item_id)

    assert result == {"status": "disabled", "notified": False}
    assert dreams_db.get_tracked_item(passed)["status"] == "active", (
        "a disabled Dreams changes nothing, not even by sweeping"
    )


@pytest.mark.asyncio
async def test_a_check_switched_off_after_it_started_makes_no_search(
        dreams_db, settings, monkeypatch):
    """Kills "delete the pre-search check". Dreams is on when the check is
    dispatched and goes off while it is still sweeping: the start gate has
    been passed, so only the gate in front of the search can stop it."""
    from tldw_chatbook.Dreams import track_service

    item_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="deal")
    search = _FakeSearch(_SET_A)

    async def switched_off_during_the_sweep(*args, **kwargs):
        settings["enabled"] = False
        return []

    monkeypatch.setattr(
        track_service, "sweep_track_lifecycle", switched_off_during_the_sweep)

    result = await run_track_check(
        _track_deps(dreams_db, perform_search=search), item_id)

    assert result == {"status": "disabled", "notified": False}
    assert search.calls == []
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}
    assert _run_rows(dreams_db, item_id) == []


@pytest.mark.asyncio
async def test_suspension_is_recorded_before_a_reminder_is_disabled(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Kills "write the record after the reminders are disabled". The
    write-ahead test above only watches the subscription stop; the reminder
    is stopped first, and an interruption right after it must find the
    reminder already on record."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    (reminder,) = _linked_reminders(scheduling_db, item_id)
    seen: list = []
    real_update = scheduling_db.update_reminder_task

    def recording_update(task_id, **fields):
        if fields.get("enabled") is False:
            seen.append(_record(dreams_db, item_id))
        return real_update(task_id, **fields)

    scheduling_db.update_reminder_task = recording_update

    await _switch(service, dreams_db, scheduling_db, off=True)

    assert seen and seen[0] is not None, "no record when the reminder went off"
    assert seen[0]["reminders"] == [reminder["id"]]
    assert "settled" not in seen[0], "and not yet claimed as done"


@pytest.mark.asyncio
async def test_resume_never_rearms_a_reminder_that_fired_meanwhile(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Kills "drop the last_run_at guard". The scheduler had the reminder
    loaded when Dreams went off and delivered it; the record still lists
    it, and resume must not arm a one-time reminder that has fired."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    (reminder,) = _linked_reminders(scheduling_db, item_id)
    assert _record(dreams_db, item_id)["reminders"] == [reminder["id"]]
    scheduling_db.mark_reminder_dispatched(reminder["id"], NOW)

    await _switch(service, dreams_db, scheduling_db, off=False)

    (after,) = _linked_reminders(scheduling_db, item_id)
    assert (after["id"], after["enabled"]) == (reminder["id"], False)
    assert _record(dreams_db, item_id) is None


@pytest.mark.asyncio
async def test_every_rider_of_a_source_records_that_dreams_stopped_it(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """What the old per-subscription cache protected, in the new shape: the
    stop is decided once per source and recorded on EVERY rider. If only
    the first rider carried it, untracking that one while Dreams is off
    would leave the source off for good."""
    subs_db, service = subs_stack
    first, second = await _two_wrappers_on_one_subscription(dreams_db, service)
    source_id = first["subscription_id"]

    await _switch(service, dreams_db, scheduling_db, off=True)

    for tracked in (first, second):
        assert _record(dreams_db, tracked["tracked_item_id"])["subscription"] is True
    for gone, kept in ((first, second), (second, first)):
        dreams_db.set_tracked_status(
            gone["tracked_item_id"], "retired", retired_reason="manual")
        assert _record(dreams_db, kept["tracked_item_id"])["subscription"] is True
        dreams_db.set_tracked_status(gone["tracked_item_id"], "active")
    dreams_db.set_tracked_status(
        second["tracked_item_id"], "retired", retired_reason="manual")

    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _is_active(subs_db, source_id) == 1


# --- T14c: the settle callback runs when everything has settled ----------------


@pytest.mark.asyncio
async def test_overlapping_reconciles_report_settled_once_at_the_end(
        tmp_path, dreams_db, settings, subs_stack):
    """``on_settled`` asks the scheduler queue to reload. It ran after every
    pass, so "on" requested while the "off" pass was still running reloaded
    the queue BETWEEN the two passes... with the first pass's state. It now
    runs once, when no further pass is waiting."""
    from tldw_chatbook.Dreams import track_service

    subs_db, service = subs_stack
    owned = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=7)
    parked = _ParkedSources(service)
    states_at_settle: list[int] = []

    def on_settled():
        states_at_settle.append(int(subs_db.get_subscription(
            owned["subscription_id"])["is_active"]))

    settings["enabled"] = False
    first = asyncio.create_task(track_service.reconcile_tracking(
        parked, dreams_db, on_settled=on_settled))
    await _within_bound(parked.parked.wait(), "the first pass parking on get_source")
    settings["enabled"] = True
    second = asyncio.create_task(track_service.reconcile_tracking(
        parked, dreams_db, now=NOW, on_settled=on_settled))
    await asyncio.sleep(0)
    parked.release.set()
    await _within_bound(asyncio.gather(first, second), "both reconcile requests settling")

    assert states_at_settle == [1], (
        "one reload, after the last pass, seeing the final state"
    )


# --- T14b: resume against a sweep that retires a watch meanwhile ----------------


@pytest.mark.asyncio
async def test_resume_does_not_restore_a_watch_retired_since_the_pass_began(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """The pass works from the list it read at its start. A lifecycle sweep
    (a cycle, a track check) that retired the wrapper in between used to
    have its work undone: the reminder re-armed and the source switched
    back on, with no live watch left on it."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    real_list = dreams_db.list_tracked_items

    def list_then_a_sweep_retires_it(status="active", **kwargs):
        rows = real_list(status, **kwargs)
        if status == "paused":  # the second (last) read of the pass
            dreams_db.set_tracked_status(
                item_id, "retired", retired_reason="event_passed")
        return rows

    dreams_db.list_tracked_items = list_then_a_sweep_retires_it
    try:
        await _switch(service, dreams_db, scheduling_db, off=False)
    finally:
        dreams_db.list_tracked_items = real_list

    assert dreams_db.get_tracked_item(item_id)["status"] == "retired"
    assert _reminder_states(scheduling_db) == {str(item_id): False}
    assert _is_active(subs_db, source_id) == 0, (
        "no live watch rides it: it must not be switched back on"
    )


# === Review of #2921, round 4 ====================================================

# --- A1: a resume that fails part-way must not stay "settled off" ---------------
#
# Resume restored the reminders and the rules and rewrote the record once,
# after both. A failure in between left reminders running under a record
# that still read settled "off" -- which the next switch-off, and the boot
# check, skip.


async def _dreams_work_running(service, subs_db, scheduling_db, owned) -> dict:
    """What of one owned event page's outside work is running right now."""
    rules = await _change_rules(service, owned["subscription_id"])
    return {
        "reminder": _reminder_states(scheduling_db)[str(owned["tracked_item_id"])],
        "rule": any(rule["enabled"] for rule in rules),
        "source": bool(_is_active(subs_db, owned["subscription_id"])),
    }


@pytest.mark.asyncio
async def test_a_resume_that_fails_at_the_rule_step_is_not_left_settled_off(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Reproduction (i): off; on with ``update_alert_rule`` raising once.
    The reminder is re-armed; the record used to still read settled "off"
    with the reminder listed, so "off" again switched nothing and the
    reminder stayed armed with Dreams off."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    assert _record(dreams_db, item_id)["settled"] == "off"
    real_update = service.update_alert_rule
    failures = [RuntimeError("database is locked")]

    async def flaky_rule(rule_id, **fields):
        if failures and fields.get("enabled") is True:
            raise failures.pop()
        return await real_update(rule_id, **fields)

    service.update_alert_rule = flaky_rule

    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _reminder_states(scheduling_db) == {str(item_id): True}, (
        "fixture: the reminder was restored before the failure"
    )
    record = _record(dreams_db, item_id)
    assert record.get("settled") != "off", (
        "a wrapper whose reminder runs again is not suspended"
    )
    assert record["reminders"] == [], "what was restored is no longer owed"
    assert _boot_needs_switch(dreams_db, enabled=False), (
        "a start with Dreams off must find this wrapper"
    )

    assert await _switch(service, dreams_db, scheduling_db, off=True) == 1
    assert await _dreams_work_running(
        service, subs_db, scheduling_db, owned) == {
            "reminder": False, "rule": False, "source": False}
    assert _record(dreams_db, item_id)["settled"] == "off"


@pytest.mark.asyncio
async def test_a_reminder_disabled_by_hand_after_a_failed_resume_stays_off(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Round-3 finding 12(a), reopened by the same failure: the reminder was
    restored, the pass then failed, the user disabled the reminder -- and
    the next resume re-armed it because the record still listed it."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    (reminder,) = _linked_reminders(scheduling_db, item_id)
    real_update = service.update_alert_rule
    failures = [RuntimeError("database is locked")]

    async def flaky_rule(rule_id, **fields):
        if failures and fields.get("enabled") is True:
            raise failures.pop()
        return await real_update(rule_id, **fields)

    service.update_alert_rule = flaky_rule
    await _switch(service, dreams_db, scheduling_db, off=False)
    scheduling_db.update_reminder_task(reminder["id"], enabled=False)  # the user

    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _reminder_states(scheduling_db) == {str(item_id): False}
    assert _record(dreams_db, item_id) is None, "the retry finished the resume"


@pytest.mark.parametrize("failing_write", [1, 2, 3, 4])
@pytest.mark.asyncio
async def test_a_resume_whose_record_write_fails_never_runs_work_under_an_off_record(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db,
        failing_write):
    """Reproduction (ii): off; on with one record write failing. Whichever
    write it is, nothing Dreams restored may be running under a record that
    still says settled "off" -- and the next "off" must stop it all."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    real_write = dreams_db.set_tracked_suspension
    calls = {"n": 0}

    def flaky_write(tracked_id, text, **kwargs):
        calls["n"] += 1
        if calls["n"] == failing_write:
            raise RuntimeError("database is locked")
        return real_write(tracked_id, text, **kwargs)

    dreams_db.set_tracked_suspension = flaky_write
    try:
        await _switch(service, dreams_db, scheduling_db, off=False)
    finally:
        dreams_db.set_tracked_suspension = real_write
    assert calls["n"] >= failing_write, "fixture: that write was reached"

    running = await _dreams_work_running(service, subs_db, scheduling_db, owned)
    record = _record(dreams_db, item_id) or {}
    if record.get("settled") == "off":
        assert running == {"reminder": False, "rule": False, "source": False}

    await _switch(service, dreams_db, scheduling_db, off=True)

    assert await _dreams_work_running(
        service, subs_db, scheduling_db, owned) == {
            "reminder": False, "rule": False, "source": False}
    assert _record(dreams_db, item_id)["settled"] == "off"


# --- A2: no stop without a write-ahead record ------------------------------------
#
# The stop loop sent ``active: False`` for a subscription whenever the plan
# said Dreams owes it a stop -- even when the staging step had failed for
# every rider, so nothing recorded the stop. The next pass then found the
# source not running, recorded "subscription: false" and settled; switching
# Dreams back on restored nothing, and the source stayed off for good.


@pytest.mark.asyncio
async def test_a_source_is_not_stopped_when_its_riders_record_write_failed(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    subs_db, service = subs_stack
    owned = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=1)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    real_write = dreams_db.set_tracked_suspension
    failures = [RuntimeError("database is locked")]

    def flaky_write(tracked_id, text, **kwargs):
        if failures:
            raise failures.pop()
        return real_write(tracked_id, text, **kwargs)

    dreams_db.set_tracked_suspension = flaky_write
    try:
        await _switch(service, dreams_db, scheduling_db, off=True)
    finally:
        dreams_db.set_tracked_suspension = real_write

    assert _record(dreams_db, item_id) is None, "fixture: nothing was recorded"
    assert _is_active(subs_db, source_id) == 1, (
        "a stop nobody recorded is a stop nobody can undo"
    )
    assert _boot_needs_switch(dreams_db, enabled=False), "and it is retried"

    await _switch(service, dreams_db, scheduling_db, off=True)

    assert _is_active(subs_db, source_id) == 0
    record = _record(dreams_db, item_id)
    assert record["subscription"] is True and record["settled"] == "off"

    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _is_active(subs_db, source_id) == 1, "what was recorded is restored"
    assert _record(dreams_db, item_id) is None


@pytest.mark.asyncio
async def test_a_source_is_not_stopped_when_reading_its_riders_reminders_failed(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    real_read = scheduling_db.list_reminder_tasks_by_link
    failures = [RuntimeError("database is locked")]

    def flaky_read(link_type, link_ids):
        if failures:
            raise failures.pop()
        return real_read(link_type, link_ids)

    scheduling_db.list_reminder_tasks_by_link = flaky_read
    try:
        await _switch(service, dreams_db, scheduling_db, off=True)
    finally:
        scheduling_db.list_reminder_tasks_by_link = real_read

    assert _record(dreams_db, item_id) is None, "fixture: nothing was recorded"
    assert _is_active(subs_db, source_id) == 1
    assert _reminder_states(scheduling_db) == {str(item_id): True}

    await _switch(service, dreams_db, scheduling_db, off=True)
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _is_active(subs_db, source_id) == 1, "off and on again is lossless"
    assert _reminder_states(scheduling_db) == {str(item_id): True}


@pytest.mark.asyncio
async def test_one_riders_record_is_enough_to_stop_a_shared_source(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Two riders, the record write fails for one: the other's record says
    the stop happened, so it may -- and the first rider catches up on the
    next pass instead of concluding the source was never running."""
    subs_db, service = subs_stack
    first, second = await _two_wrappers_on_one_subscription(dreams_db, service)
    source_id = first["subscription_id"]
    real_write = dreams_db.set_tracked_suspension

    def flaky_write(tracked_id, text, **kwargs):
        if failures and tracked_id == first["tracked_item_id"]:
            raise failures.pop()
        return real_write(tracked_id, text, **kwargs)

    failures = [RuntimeError("database is locked")]
    dreams_db.set_tracked_suspension = flaky_write
    try:
        await _switch(service, dreams_db, scheduling_db, off=True)
    finally:
        dreams_db.set_tracked_suspension = real_write

    assert _is_active(subs_db, source_id) == 0
    assert _record(dreams_db, second["tracked_item_id"])["subscription"] is True
    assert _record(dreams_db, first["tracked_item_id"]) is None

    await _switch(service, dreams_db, scheduling_db, off=True)

    for tracked in (first, second):
        record = _record(dreams_db, tracked["tracked_item_id"])
        assert record["subscription"] is True and record["settled"] == "off"


# --- A3: the shared event reminder outlives the watch that held it ---------------
#
# One story, one event, one reminder -- linked to whichever of the story's
# watches was made first. Retiring THAT watch disarmed the reminder by its
# link and left the story's other, still-running watch with none.


async def _page_then_question(dreams_db, service, scheduling_db):
    """A story tracked as a page and watched as a question, same event."""
    page = await _event_page(dreams_db, service, scheduling_db, story_id=7)
    question_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01", origin_story_id=7,
        scheduling_db_getter=_sched_getter(scheduling_db))
    (reminder,) = _all_event_reminders(scheduling_db)
    assert reminder["link_id"] == str(page["tracked_item_id"]), (
        "fixture: the one shared reminder is linked to the page watch"
    )
    return page, question_id, reminder


@pytest.mark.asyncio
async def test_untracking_the_watch_that_holds_the_reminder_hands_it_on(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Reproduction (i): track the page, watch the question, press ``u``
    once (the page watch goes first). The question watch is still active
    with its event date -- and its only reminder was disabled."""
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    moved_to = "2026-09-20T07:30:00+00:00"
    scheduling_db.update_reminder_task(
        reminder["id"], run_at=moved_to, next_run_at=moved_to)  # the user

    await untrack(service, dreams_db, page["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))

    assert dreams_db.get_tracked_item(question_id)["status"] == "active"
    (kept,) = _all_event_reminders(scheduling_db)
    assert kept["id"] == reminder["id"], "the same reminder, not a twin"
    assert kept["enabled"], "the story's other watch still has its reminder"
    assert kept["link_id"] == str(question_id)
    assert kept["run_at"] == scheduling_db.get_reminder_task(
        reminder["id"])["run_at"] and "07:30" in str(kept["run_at"]), (
        "and the time the user moved it to"
    )
    assert _record(dreams_db, page["tracked_item_id"])["reminders"] == [], (
        "Dreams did not disarm it, so the retired watch does not claim it"
    )

    # The switch finds it on its new holder: off and on again keeps it.
    await _switch(service, dreams_db, scheduling_db, off=True)
    assert not _all_event_reminders(scheduling_db)[0]["enabled"]
    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _all_event_reminders(scheduling_db)[0]["enabled"]


@pytest.mark.asyncio
async def test_a_watch_retired_as_quiet_hands_its_reminder_to_the_other_watch(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Reproduction (ii): the question is watched first, then the page is
    tracked; the question watch retires as quiet. The page watch ran on to
    the event with no reminder, with no user action at all."""
    _subs_db, service = subs_stack
    question_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-10-01", origin_story_id=7,
        scheduling_db_getter=_sched_getter(scheduling_db))
    page = await _event_page(dreams_db, service, scheduling_db, story_id=7)
    (reminder,) = _all_event_reminders(scheduling_db)
    assert reminder["link_id"] == str(question_id), "fixture"
    _seed_runs(dreams_db, question_id, ["unchanged"] * 14)

    await sweep_track_lifecycle(
        dreams_db, now=NOW, subs_service_getter=lambda: service,
        scheduling_db_getter=_sched_getter(scheduling_db))

    assert dreams_db.get_tracked_item(question_id)["status"] == "retired"
    assert dreams_db.get_tracked_item(
        page["tracked_item_id"])["status"] == "active"
    (kept,) = _all_event_reminders(scheduling_db)
    assert kept["enabled"] and kept["link_id"] == str(page["tracked_item_id"])


@pytest.mark.asyncio
async def test_a_reminder_the_user_switched_off_is_not_handed_on_armed(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    scheduling_db.update_reminder_task(reminder["id"], enabled=False)  # the user

    await untrack(service, dreams_db, page["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))

    (kept,) = _all_event_reminders(scheduling_db)
    assert not kept["enabled"], "their decision stands"


@pytest.mark.asyncio
async def test_untracking_the_holder_while_dreams_is_off_hands_the_debt_on(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """With Dreams off the reminder is disabled and OWED on the holder's
    record. Untracking the holder then must leave that debt with the
    story's other watch, or switching Dreams on restores no reminder."""
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    await _switch(service, dreams_db, scheduling_db, off=True)

    await untrack(service, dreams_db, page["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))

    (kept,) = _all_event_reminders(scheduling_db)
    assert not kept["enabled"], "Dreams is off: nothing is armed"
    assert kept["link_id"] == str(question_id)
    assert _record(dreams_db, question_id)["reminders"] == [reminder["id"]]

    await _switch(service, dreams_db, scheduling_db, off=False)

    (kept,) = _all_event_reminders(scheduling_db)
    assert kept["enabled"] and kept["link_id"] == str(question_id)


@pytest.mark.asyncio
async def test_a_reminder_is_not_handed_to_a_watch_of_another_event_or_story(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Only a live watch of the SAME story and event date inherits."""
    _subs_db, service = subs_stack
    page = await _event_page(dreams_db, service, scheduling_db, story_id=7)
    other_story = await track_question(
        dreams_db, query_template="another question", intent="event",
        event_date="2026-10-01", origin_story_id=8,
        scheduling_db_getter=_sched_getter(scheduling_db))
    other_date = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-12-24", origin_story_id=7,
        scheduling_db_getter=_sched_getter(scheduling_db))
    (reminder,) = _linked_reminders(scheduling_db, page["tracked_item_id"])

    await untrack(service, dreams_db, page["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))

    row = scheduling_db.get_reminder_task(reminder["id"])
    assert not row["enabled"]
    assert row["link_id"] == str(page["tracked_item_id"])
    assert len(_linked_reminders(scheduling_db, other_story)) == 1
    assert len(_linked_reminders(scheduling_db, other_date)) == 1


# --- B1: a reactivated watch gets a fresh start ----------------------------------


@pytest.mark.asyncio
async def test_a_reactivated_watch_is_not_paused_again_before_it_is_checked(
        tmp_path, dreams_db, settings, scheduling_db):
    """Tracking a story whose watch was paused after three failed checks
    puts the watch back to work -- but its three error runs were still its
    newest runs, so the very next sweep paused it again before any check
    could run."""
    item_id, _task = await _question_with_reminder(dreams_db, scheduling_db)
    _seed_runs(dreams_db, item_id, ["error"] * 3)
    await sweep_track_lifecycle(dreams_db, now=NOW)
    assert dreams_db.get_tracked_item(item_id)["status"] == "paused", "fixture"
    checked_before = dreams_db.get_tracked_item(item_id)["last_checked"]

    assert await _retrack_question(dreams_db, scheduling_db) == item_id
    notes = await sweep_track_lifecycle(dreams_db, now=NOW)

    assert dreams_db.get_tracked_item(item_id)["status"] == "active", notes
    assert dreams_db.consecutive_track_dispositions(item_id, "error") == 0
    assert dreams_db.get_tracked_item(item_id)["last_checked"] == (
        checked_before), "the watch is no less due for a check than it was"

    # Three NEW failures still pause it: the streak restarts, it is not off.
    _seed_runs(dreams_db, item_id, ["error"] * 3)
    await sweep_track_lifecycle(dreams_db, now=NOW)
    assert dreams_db.get_tracked_item(item_id)["status"] == "paused"


# --- B2: a page held for a passing reason is asked about again -------------------
#
# Every held source was settled "on", whatever the reason. "The address is
# private" is a fact about the address; "the name could not be looked up" is
# a fact about that minute -- and nothing ever asked again without the user
# switching Dreams off and on or tracking the page again. (TASK-34922 widened
# this: a private or metadata answer is a DNS answer too, so only a non-web
# scheme is lasting; see the TASK-34922 block below.)


async def _held_for_dns(dreams_db, service, scheduling_db, monkeypatch):
    owned = await _event_page(dreams_db, service, scheduling_db)
    await _switch(service, dreams_db, scheduling_db, off=True)
    _fail_resolution(monkeypatch)
    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _record(dreams_db, owned["tracked_item_id"])["held"] == "dns_failure"
    return owned


def _resolve_publicly(monkeypatch) -> list[str]:
    from tldw_chatbook.Utils import egress

    lookups: list[str] = []

    async def resolve(host):
        lookups.append(host)
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", resolve)
    return lookups


@pytest.mark.asyncio
async def test_a_page_held_for_a_failed_lookup_is_retried_by_the_daily_cycle(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    subs_db, service = subs_stack
    owned = await _held_for_dns(dreams_db, service, scheduling_db, monkeypatch)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    assert _boot_needs_switch(dreams_db, enabled=True) is False, (
        "round 3, T19 stands: a start does not look the address up again"
    )
    _resolve_publicly(monkeypatch)

    await retry_unfinished_resume(
        service, dreams_db,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)

    assert _is_active(subs_db, source_id) == 1
    assert _record(dreams_db, item_id) is None


@pytest.mark.asyncio
async def test_a_page_held_for_a_failed_lookup_is_retried_by_any_resume_pass(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """A resume pass that runs for another reason takes it along."""
    subs_db, service = subs_stack
    owned = await _held_for_dns(dreams_db, service, scheduling_db, monkeypatch)
    _resolve_publicly(monkeypatch)

    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _is_active(subs_db, owned["subscription_id"]) == 1
    assert _record(dreams_db, owned["tracked_item_id"]) is None


@pytest.mark.asyncio
async def test_a_lookup_that_still_fails_leaves_the_page_held(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    subs_db, service = subs_stack
    owned = await _held_for_dns(dreams_db, service, scheduling_db, monkeypatch)

    await retry_unfinished_resume(
        service, dreams_db,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)

    assert _is_active(subs_db, owned["subscription_id"]) == 0
    record = _record(dreams_db, owned["tracked_item_id"])
    assert (record["held"], record["settled"]) == ("dns_failure", "on")
    assert _boot_needs_switch(dreams_db, enabled=True) is False


# --- TASK-34922: private and metadata holds are retried too ----------------------
#
# Owner decision 2026-10-13: retry private and metadata holds on later
# cycles. Both are DNS answers, and a hotel, airport or VPN resolver can
# answer every name with a private address; a retry is the same public
# check, so it can never switch on a page that still resolves privately.


def _no_request_may_be_sent(monkeypatch) -> list[str]:
    """Record any HTTP request the code under test tries to send."""
    import httpx

    sent: list[str] = []

    def refuse_sync(self, request, *args, **kwargs):
        sent.append(str(request.url))
        raise AssertionError("a request was sent")

    async def refuse_async(self, request, *args, **kwargs):
        sent.append(str(request.url))
        raise AssertionError("a request was sent")

    monkeypatch.setattr(httpx.Client, "send", refuse_sync)
    monkeypatch.setattr(httpx.AsyncClient, "send", refuse_async)
    return sent


async def _held_at(dreams_db, service, scheduling_db, monkeypatch, address,
                   reason):
    owned = await _event_page(dreams_db, service, scheduling_db)
    await _switch(service, dreams_db, scheduling_db, off=True)
    _resolve_everything_to(monkeypatch, address)
    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _record(dreams_db, owned["tracked_item_id"])["held"] == reason
    return owned


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address, reason",
    [("10.0.0.5", "private"), ("192.168.1.1", "private"),
     ("169.254.169.254", "metadata")])
async def test_a_page_held_for_a_private_answer_clears_on_a_later_cycle(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch,
        address, reason):
    subs_db, service = subs_stack
    owned = await _held_at(
        dreams_db, service, scheduling_db, monkeypatch, address, reason)
    assert _boot_needs_switch(dreams_db, enabled=True) is False, (
        "T19 stands: the hold alone does not make a start run a pass"
    )
    lookups = _resolve_publicly(monkeypatch)

    await retry_unfinished_resume(
        service, dreams_db,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)

    assert lookups, "the daily retry asked again"
    assert _is_active(subs_db, owned["subscription_id"]) == 1
    assert _record(dreams_db, owned["tracked_item_id"]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address, reason",
    [("10.0.0.5", "private"), ("169.254.169.254", "metadata")])
async def test_a_page_that_still_resolves_privately_stays_held_and_is_never_fetched(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch,
        address, reason):
    """A retry is the same public-address check: it can never switch on a
    page that still resolves privately, and nothing is requested from it.

    Neither the retry nor a switch fetches anything, so the page is then
    run through the watcher's real check path (``launch_run`` +
    ``execute_run``, as
    ``test_tracked_page_is_refused_at_fetch_time_once_it_resolves_private``
    does): the fetch-time policy refuses it too, and no request leaves.
    Two layers, each pinned: the hold (the source stays off) and the
    fetch-time refusal (nothing is sent)."""
    subs_db, service = subs_stack
    owned = await _held_at(
        dreams_db, service, scheduling_db, monkeypatch, address, reason)
    lookups: list[str] = []
    from tldw_chatbook.Utils import egress

    async def still_private(host):
        lookups.append(host)
        return [address]

    monkeypatch.setattr(egress, "_resolve_async", still_private)
    sent = _no_request_may_be_sent(monkeypatch)

    await retry_unfinished_resume(
        service, dreams_db,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert lookups, "it was asked about again"
    assert _is_active(subs_db, owned["subscription_id"]) == 0
    record = _record(dreams_db, owned["tracked_item_id"])
    assert (record["held"], record["settled"]) == (reason, "on")

    run = await service.launch_run(source_id=owned["subscription_id"])
    await service.execute_run(run["run_id"])

    assert sent == [], "nothing was requested from it"
    row = subs_db.get_subscription(owned["subscription_id"])
    assert "network safety policy" in str(row["last_error"])


@pytest.mark.asyncio
async def test_a_retry_that_changes_nothing_rewrites_no_record(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """Review of #3101, N1. The retry runs on every cycle trigger, so a
    page that is still private is asked about again each time; its record
    -- held for the same reason, still settled "on" -- is left as it is,
    not written twice with the same content on every retry."""
    subs_db, service = subs_stack
    owned = await _held_at(
        dreams_db, service, scheduling_db, monkeypatch, "10.0.0.5", "private")
    item_id = owned["tracked_item_id"]
    before = dreams_db.get_tracked_item(item_id)["suspension"]
    writes: list[int] = []
    real = dreams_db.set_tracked_suspension

    def spy(tracked_item_id, record, **kwargs):
        writes.append(int(tracked_item_id))
        return real(tracked_item_id, record, **kwargs)

    monkeypatch.setattr(dreams_db, "set_tracked_suspension", spy)

    await retry_unfinished_resume(
        service, dreams_db,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)

    assert item_id not in writes, "an unchanged record is not rewritten"
    assert dreams_db.get_tracked_item(item_id)["suspension"] == before
    assert _is_active(subs_db, owned["subscription_id"]) == 0


@pytest.mark.asyncio
async def test_a_held_page_switched_back_on_never_runs_under_its_hold(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """Review of #3101, round 2, S1. A record settled "on" is not written
    ahead (N1), so once its lookup has passed it must stop saying "held"
    BEFORE its source runs again. Otherwise a failed closing write leaves
    the source running under "held: private, settled: on" -- the Tracked
    row says the page is paused and the boot check sees no work -- where
    an unsettled record is finished by the next pass."""
    subs_db, service = subs_stack
    owned = await _held_at(
        dreams_db, service, scheduling_db, monkeypatch, "10.0.0.5", "private")
    item_id = owned["tracked_item_id"]
    _resolve_publicly(monkeypatch)
    real = dreams_db.set_tracked_suspension

    def closing_write_fails(tracked_item_id, record, **kwargs):
        if int(tracked_item_id) == item_id and record is None:
            raise RuntimeError("disk full")
        return real(tracked_item_id, record, **kwargs)

    monkeypatch.setattr(dreams_db, "set_tracked_suspension", closing_write_fails)

    await retry_unfinished_resume(
        service, dreams_db,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)

    assert _is_active(subs_db, owned["subscription_id"]) == 1
    record = _record(dreams_db, item_id)
    assert "held" not in record, "a running source is not reported as held"
    assert record.get("settled") is None, "the next pass finishes it"
    assert _boot_needs_switch(dreams_db, enabled=True) is True


async def _two_held_riders(dreams_db, service, scheduling_db, monkeypatch):
    """Two stories watching one page, both held for a private answer."""
    first = await _event_page(dreams_db, service, scheduling_db)
    second = await _event_page(dreams_db, service, scheduling_db, story_id=2)
    assert first["subscription_id"] == second["subscription_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    _resolve_everything_to(monkeypatch, "10.0.0.5")
    await _switch(service, dreams_db, scheduling_db, off=False)
    for owned in (first, second):
        record = _record(dreams_db, owned["tracked_item_id"])
        assert (record["held"], record["settled"]) == ("private", "on")
    return first, second


@pytest.mark.asyncio
async def test_a_rider_whose_first_step_failed_is_un_settled_too(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """Review of #3101, round 3, T1. Two riders share one source; the
    first rider's own step fails (its fresh read raises), the second's
    page passes and the source is switched on. The first rider must not
    be left saying "held: private, settled: on" -- paused on its row, and
    no work for the boot check -- under a running source: it is
    un-settled with the others, which is what its skipped write-ahead
    would have written."""
    subs_db, service = subs_stack
    first, second = await _two_held_riders(
        dreams_db, service, scheduling_db, monkeypatch)
    _resolve_publicly(monkeypatch)
    real_read = dreams_db.get_tracked_item
    armed = [True]

    def first_read_fails(tracked_item_id):
        if armed[0] and int(tracked_item_id) == first["tracked_item_id"]:
            raise RuntimeError("read failed")
        return real_read(tracked_item_id)

    monkeypatch.setattr(dreams_db, "get_tracked_item", first_read_fails)
    await retry_unfinished_resume(
        service, dreams_db,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)
    armed[0] = False

    assert _is_active(subs_db, first["subscription_id"]) == 1
    record = _record(dreams_db, first["tracked_item_id"])
    assert "held" not in record, "a running source is not reported as held"
    assert record.get("settled") is None, "the next pass finishes it"
    assert _boot_needs_switch(dreams_db, enabled=True) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("riders, failing_write", [(1, 1), (2, 2)],
                         ids=["only-rider", "second-rider"])
async def test_a_failed_un_settling_keeps_the_hold_while_the_source_is_off(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch,
        riders, failing_write):
    """Review of #3101, round 3, T3. The page passes, but un-settling a
    held record fails, so the source stays off. Every rider then keeps its
    hold as recorded -- its row still reads paused and a start does not
    look the name up -- including a rider un-settled before the failure.
    The closing write must not turn the hold into a bare unsettled debt."""
    import json

    subs_db, service = subs_stack
    if riders == 1:
        owned = [await _held_at(
            dreams_db, service, scheduling_db, monkeypatch, "10.0.0.5",
            "private")]
    else:
        owned = list(await _two_held_riders(
            dreams_db, service, scheduling_db, monkeypatch))
    ids = {o["tracked_item_id"] for o in owned}
    _resolve_publicly(monkeypatch)
    real = dreams_db.set_tracked_suspension
    unsettling: list[int] = []

    def nth_unsettling_fails(tracked_item_id, record, **kwargs):
        if (int(tracked_item_id) in ids and record is not None
                and "held" not in json.loads(record)):
            unsettling.append(int(tracked_item_id))
            if len(unsettling) == failing_write:
                raise RuntimeError("disk full")
        return real(tracked_item_id, record, **kwargs)

    monkeypatch.setattr(dreams_db, "set_tracked_suspension", nth_unsettling_fails)

    await retry_unfinished_resume(
        service, dreams_db,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)

    assert len(unsettling) >= failing_write, "fixture: the n-th write failed"
    assert _is_active(subs_db, owned[0]["subscription_id"]) == 0
    for item_id in ids:
        record = _record(dreams_db, item_id)
        assert (record["held"], record["settled"]) == ("private", "on")
    assert _boot_needs_switch(dreams_db, enabled=True) is False


@pytest.mark.asyncio
async def test_a_settled_record_that_owes_more_than_its_hold_is_written_ahead(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """Review of #3101, round 2, S2. Only a record that owes nothing but
    its held subscription may skip the write-ahead. No writer makes a
    settled "on" record that owes a rule today, so one is written by hand:
    when restoring that rule fails, the record must already have stopped
    saying "settled on", or the boot check would take it as done."""
    from tldw_chatbook.Dreams import track_service

    subs_db, service = subs_stack
    owned = await _held_at(
        dreams_db, service, scheduling_db, monkeypatch, "10.0.0.5", "private")
    item_id = owned["tracked_item_id"]
    dreams_db.set_tracked_suspension(item_id, track_service._record_text(
        subscription=True, reminders=[], rules=[999], held="private",
        settled="on"))

    async def rule_fails(*args, **kwargs):
        raise RuntimeError("rule store unavailable")

    monkeypatch.setattr(service, "update_alert_rule", rule_fails)

    await retry_unfinished_resume(
        service, dreams_db,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)

    record = _record(dreams_db, item_id)
    assert record.get("settled") is None, "written ahead before the restore"
    assert record["rules"] == [999], "the rule is still owed"
    assert _boot_needs_switch(dreams_db, enabled=True) is True


@pytest.mark.asyncio
async def test_a_page_held_for_its_scheme_is_not_asked_about_daily(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """A non-web scheme is a fact about the address, not a DNS answer: the
    only lasting hold. The daily retry makes no lookup for it and does not
    even start a pass.

    In practice the hold is out of reach: a Dreams source is http(s) when
    it is tracked and nothing in the app edits a source's URL, so the test
    puts the scheme in place with a raw database write (review of #3101,
    N3). It pins the rule, not a path a user can take."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    await _switch(service, dreams_db, scheduling_db, off=True)
    subs_db.update_subscription(
        owned["subscription_id"], source="ftp://example.com/flights")
    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _record(dreams_db, owned["tracked_item_id"])["held"] == "scheme"
    lookups = _resolve_publicly(monkeypatch)

    assert await retry_unfinished_resume(
        service, dreams_db,
        scheduling_db_getter=_sched_getter(scheduling_db), now=NOW) == 0

    assert lookups == []
    assert _is_active(subs_db, owned["subscription_id"]) == 0


# --- Round 4 nit: a URL the user changed is the user's, and stays so --------------
#
# Round 3 made two rules that contradicted each other: a user's URL edit
# clears the "nobody typed this" mark (T17), and Dreams stamps every source
# it owns that carries none (T1). So the next switch-off stamped the user's
# own address as a story address again.

_USERS_URL = "https://intranet.example/status"


async def _owned_source_with_a_url_the_user_changed(dreams_db, service):
    owned = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=1)
    # ADR-207 (2026-10-12 amendment): an edit establishes user entry only
    # when the caller passes the app-owned enum, as the UI paths do.
    await service.update_source(
        owned["subscription_id"],
        {"url": _USERS_URL, "url_provenance": UrlProvenance.USER_ENTERED})
    return owned


@pytest.mark.asyncio
async def test_the_switch_does_not_stamp_a_source_whose_url_the_user_changed(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    subs_db, service = subs_stack
    owned = await _owned_source_with_a_url_the_user_changed(dreams_db, service)
    source_id = owned["subscription_id"]
    users = {"url_provenance": "user_entered"}
    assert _stored_options(subs_db, source_id) == users, "fixture"

    await _switch(service, dreams_db, scheduling_db, off=True)
    assert _stored_options(subs_db, source_id) == users, "off: not re-stamped"

    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _stored_options(subs_db, source_id) == users, "on: not re-stamped"
    assert _is_active(subs_db, source_id) == 1


@pytest.mark.asyncio
async def test_the_backfill_does_not_stamp_a_source_whose_url_the_user_changed(
        tmp_path, dreams_db, settings, subs_stack):
    from tldw_chatbook.Dreams.track_service import backfill_egress_marks

    subs_db, service = subs_stack
    owned = await _owned_source_with_a_url_the_user_changed(dreams_db, service)
    with dreams_db.transaction() as conn:
        conn.execute("UPDATE dream_tracked_items SET egress_marked = 0")

    await backfill_egress_marks(service, dreams_db)

    assert _stored_options(subs_db, owned["subscription_id"]) == {
        "url_provenance": "user_entered"}
    assert dreams_db.subscriptions_awaiting_egress_mark() == [], (
        "decided, so it leaves the backlog"
    )


@pytest.mark.asyncio
async def test_resume_switches_on_a_private_address_the_user_entered(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """The public-destination question is asked about a STORY address. One
    the user typed is theirs -- Watchlists fetches it, private or not -- so
    a resume must not hold it as "not public" for good."""
    subs_db, service = subs_stack
    owned = await _owned_source_with_a_url_the_user_changed(dreams_db, service)
    await _switch(service, dreams_db, scheduling_db, off=True)
    _resolve_everything_to(monkeypatch, "10.0.0.5")

    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _is_active(subs_db, owned["subscription_id"]) == 1
    assert _record(dreams_db, owned["tracked_item_id"]) is None


# --- B4: the egress mark is verified from the source, not from the marker ---------


@pytest.mark.asyncio
async def test_a_restored_unmarked_source_is_stamped_by_the_verifying_backfill(
        tmp_path, dreams_db, settings, subs_stack):
    """dreams.sqlite remembers "marked"; the subscriptions store -- the one
    that is backed up -- is restored from before the mark. The marker-only
    backfill never looks again; the verifying one reads the source."""
    from tldw_chatbook.Dreams.track_service import backfill_egress_marks

    subs_db, service = subs_stack
    owned = await track_page(
        service, dreams_db, url=_URL, title="Flights", intent="deal",
        origin_story_id=1)
    source_id = owned["subscription_id"]
    assert dreams_db.subscriptions_awaiting_egress_mark() == [], "fixture"
    subs_db.update_subscription(source_id, processing_options={})  # restored

    assert await backfill_egress_marks(service, dreams_db) == 0
    assert _stored_options(subs_db, source_id) == {}, (
        "the marker alone cannot see it (the documented limit)"
    )

    assert await backfill_egress_marks(service, dreams_db, verify=True) == 1

    assert _stored_options(subs_db, source_id) == {"url_provenance": "unknown"}
    assert await backfill_egress_marks(service, dreams_db, verify=True) == 0, (
        "nothing left to stamp"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provenance", [None, UrlProvenance.USER_ENTERED],
    ids=["no-provenance", "user-entered"])
async def test_the_verifying_backfill_leaves_the_users_own_sources_alone(
        tmp_path, dreams_db, settings, subs_stack, provenance):
    """A source the user created is never stamped by the verifying
    backfill -- whatever provenance it was created with. With none,
    create_source now stores {"url_provenance": "unknown"} itself (ADR-207);
    the backfill must leave that alone too, not just the user-entered case
    (review of #3097, item 2)."""
    from tldw_chatbook.Dreams.track_service import backfill_egress_marks

    subs_db, service = subs_stack
    source_id, _attached = await _attach_to_users_source(
        dreams_db, service, provenance=provenance)
    before = _stored_options(subs_db, source_id)

    assert await backfill_egress_marks(service, dreams_db, verify=True) == 0

    assert _stored_options(subs_db, source_id) == before


# --- Round 4 nit: a held source is not an active source ---------------------------


@pytest.mark.asyncio
async def test_a_source_enabled_by_a_cut_resume_is_stopped_again_when_held(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """A resume enables the source and is cut before it can write the
    record; by the next pass the address is no longer public. The record
    said "held, settled on" -- over a source that was still fetching."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    real_update = service.update_source

    async def enable_then_cut(source, payload):
        result = await real_update(source, payload)
        if payload.get("active") is True:
            raise asyncio.CancelledError
        return result

    service.update_source = enable_then_cut
    with pytest.raises(asyncio.CancelledError):
        await _switch(service, dreams_db, scheduling_db, off=False)
    service.update_source = real_update
    assert _is_active(subs_db, source_id) == 1, "fixture: enabled, then cut"
    assert _record(dreams_db, item_id)["subscription"] is True, "fixture"
    _resolve_everything_to(monkeypatch, "10.0.0.5")

    await _switch(service, dreams_db, scheduling_db, off=False)

    record = _record(dreams_db, item_id)
    assert (record["held"], record["settled"]) == ("private", "on")
    assert _is_active(subs_db, source_id) == 0, "a held source is not active"


# --- Round 4 nit: a switch pass never rewrites a retirement record ----------------


@pytest.mark.parametrize("off", [False, True], ids=["resume", "suspend"])
@pytest.mark.asyncio
async def test_a_switch_pass_leaves_the_record_of_a_watch_retired_under_it(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, off):
    """Both passes work from the list read at their start. A watch that is
    retired after that read carries a retirement record ("what Dreams
    stopped when the watch ended"); the pass wrote its own record over it
    from the stale row -- through the other rider of the same source."""
    subs_db, service = subs_stack
    first, second = await _two_wrappers_on_one_subscription(dreams_db, service)
    retiring = first["tracked_item_id"]
    if not off:
        await _switch(service, dreams_db, scheduling_db, off=True)
    real_list = dreams_db.list_tracked_items
    retirement = json.dumps(
        {"subscription": False, "reminders": [], "rules": [],
         "settled": "retired"})

    def list_then_it_is_retired(status="active", **kwargs):
        rows = real_list(status, **kwargs)
        if status == "paused":  # the second (last) read of the pass
            dreams_db.set_tracked_status(
                retiring, "retired", retired_reason="manual")
            dreams_db.set_tracked_suspension(retiring, retirement)
        return rows

    dreams_db.list_tracked_items = list_then_it_is_retired
    try:
        await _switch(service, dreams_db, scheduling_db, off=off)
    finally:
        dreams_db.list_tracked_items = real_list

    assert _record(dreams_db, retiring) == json.loads(retirement)
    # ...and the pass still did its work for the rider that is left.
    assert _is_active(subs_db, first["subscription_id"]) == (0 if off else 1)


# --- Round 4: two mutants that survived the suite ---------------------------------


@pytest.mark.asyncio
async def test_a_retried_suspend_keeps_the_reminder_debt_of_the_first_attempt(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Mutant: a suspend pass that drops what the earlier record owed and
    records only the reminders armed NOW. The first attempt disabled the
    reminder and then failed at the source; on the retry the reminder is no
    longer armed, so only the merge keeps the debt -- without it switching
    Dreams on re-arms nothing."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    (reminder,) = _linked_reminders(scheduling_db, item_id)
    real_update = service.update_source
    failures = [RuntimeError("database is locked")]

    async def flaky_stop(source, payload):
        if failures and payload.get("active") is False:
            raise failures.pop()
        return await real_update(source, payload)

    service.update_source = flaky_stop
    await _switch(service, dreams_db, scheduling_db, off=True)
    record = _record(dreams_db, item_id)
    assert record["reminders"] == [reminder["id"]] and "settled" not in record
    assert _reminder_states(scheduling_db) == {str(item_id): False}, "fixture"

    await _switch(service, dreams_db, scheduling_db, off=True)  # the retry

    record = _record(dreams_db, item_id)
    assert record["settled"] == "off"
    assert record["reminders"] == [reminder["id"]], (
        "the first attempt's debt is merged, not replaced"
    )

    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _reminder_states(scheduling_db) == {str(item_id): True}


@pytest.mark.asyncio
async def test_a_global_rule_is_never_taken_for_dreams_own_however_it_is_named(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Mutant: ``_dreams_rules_on`` without its "scoped to this very
    subscription" filter. ``list_alert_rules(job_id=...)`` also returns
    GLOBAL rules, and the once-only adoption for wrappers from before rule
    ids were recorded goes by exact name and condition: a global rule the
    user named like Dreams' own would be adopted and silenced -- for every
    source."""
    subs_db, service = subs_stack
    title = "Festival line-up announced"
    story_id = _seed_story(dreams_db, title, url=_URL)
    tracked = await track_page(
        service, dreams_db, url=_URL, title=title, intent="event",
        origin_story_id=story_id)
    item_id, source_id = tracked["tracked_item_id"], tracked["subscription_id"]
    (own,) = await _change_rules(service, source_id)
    await service.delete_alert_rule(own["rule_id"])  # only the global is left
    everywhere = await _users_rule(
        service, job_id=None, name=f"Change: {title}")
    with dreams_db.transaction() as conn:  # as the earlier build left it
        conn.execute("UPDATE dream_tracked_items SET alert_rule_id = NULL")

    await _switch(service, dreams_db, scheduling_db, off=True)

    assert _rule_id_on_record(dreams_db, item_id) == 0, "nothing was adopted"
    (still,) = [rule for rule in await service.list_alert_rules()
                if rule["rule_id"] == everywhere["rule_id"]]
    assert still["enabled"], "a global rule is every source's, never Dreams'"
    assert _record(dreams_db, item_id)["rules"] == []


# --- Round 4, C1: titles Dreams writes into other features' stores ----------------
#
# A story title or a watched query can contain square brackets ("[PDF] Rust
# conference", "see [/r/boardgames] meetup"). Dreams copies them into a
# reminder's title, an alert rule's name and a source's name -- rows that
# Schedules and Watchlists render on screens which read "[...]" as Rich
# markup (pinned rows in scripts/markup_interpolation_census.tsv). An
# unmatched closing tag there is a MarkupError at render time. Dreams
# therefore writes those titles with no square bracket in them.

_BRACKETED_TITLE = "[PDF] Rust [/b] conf [@click=app.quit]x[/]"
_BRACKETED_QUERY = "see [/r/boardgames] meetup in {region}"


@pytest.mark.asyncio
async def test_what_track_page_writes_into_watchlists_has_no_square_brackets(
        tmp_path, dreams_db, settings, subs_stack):
    subs_db, service = subs_stack

    tracked = await track_page(
        service, dreams_db, url=_URL, title=_BRACKETED_TITLE, intent="deal",
        origin_story_id=1)

    (rule,) = await _change_rules(service, tracked["subscription_id"])
    source = subs_db.get_subscription(tracked["subscription_id"])
    expected = "(PDF) Rust (/b) conf (@click=app.quit)x(/)"[:40]
    assert rule["name"] == f"Change: {expected}"
    assert source["name"] == f"Dreams: {expected}"


@pytest.mark.asyncio
async def test_the_event_reminder_dreams_titles_has_no_square_brackets(
        tmp_path, dreams_db, settings, scheduling_db):
    item_id = await track_question(
        dreams_db, query_template=_BRACKETED_QUERY, intent="event",
        event_date="2026-10-01", origin_story_id=7,
        scheduling_db_getter=_sched_getter(scheduling_db))

    (reminder,) = _linked_reminders(scheduling_db, item_id)
    assert reminder["title"] == "Dreams: see (/r/boardgames) meetup in {region}"
    assert dreams_db.get_tracked_item(item_id)["query_template"] == (
        _BRACKETED_QUERY), "the watched query itself is stored as typed"


@pytest.mark.asyncio
async def test_a_rule_an_earlier_build_named_with_brackets_is_still_dreams_own(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """The once-only adoption for wrappers from before rule ids were
    recorded goes by exact name. A rule made before this change carries
    the brackets; it must still be recognised (and silenced with Dreams)."""
    subs_db, service = subs_stack
    title = "[PDF] Rust conference"
    story_id = _seed_story(dreams_db, title, url=_URL)
    tracked = await track_page(
        service, dreams_db, url=_URL, title=title, intent="event",
        origin_story_id=story_id)
    item_id, source_id = tracked["tracked_item_id"], tracked["subscription_id"]
    (own,) = await _change_rules(service, source_id)
    await service.update_alert_rule(own["rule_id"], name=f"Change: {title}")
    with dreams_db.transaction() as conn:  # as the earlier build left it
        conn.execute("UPDATE dream_tracked_items SET alert_rule_id = NULL")

    await _switch(service, dreams_db, scheduling_db, off=True)

    assert _rule_id_on_record(dreams_db, item_id) == own["rule_id"]
    (rule,) = await _change_rules(service, source_id)
    assert not rule["enabled"]


@pytest.mark.asyncio
async def test_a_tracked_update_is_titled_without_square_brackets(
        dreams_db, settings):
    from tldw_chatbook.Dreams.track_service import _dispatch_changed

    sent: list[dict] = []
    dispatcher = type("D", (), {
        "store": None, "dispatch": lambda self, **kwargs: sent.append(kwargs)})()
    deps = _track_deps(
        dreams_db, perform_search=None, dispatch_getter=lambda: dispatcher)

    assert await _dispatch_changed(
        deps, tracked_item_id=3, query_template=_BRACKETED_QUERY,
        note="prices dropped")

    assert sent[0]["title"] == "Tracked update: see (/r/boardgames) meetup in {region}"


# --- TASK-34919: a watch retired while a switch-on pass runs -------------------
#
# The pass checks a watch is live when it starts on it, then re-arms its
# rule and looks its page's name up before switching the source on. A watch
# untracked in that window used to keep its source running with its change
# rule enabled and no live watch behind it; later passes read live watches
# only, so nothing ever stopped it again.


@pytest.mark.asyncio
async def test_a_watch_retired_during_the_switch_on_ends_with_its_source_off(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    from tldw_chatbook.Utils import egress

    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id, source_id = owned["tracked_item_id"], owned["subscription_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    untracked: list[bool] = []

    async def untrack_during_the_lookup(_host):
        if not untracked:
            untracked.append(True)
            await untrack(service, dreams_db, item_id,
                          scheduling_db_getter=_sched_getter(scheduling_db))
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", untrack_during_the_lookup)
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert untracked, "fixture: the watch was untracked during the lookup"
    assert dreams_db.get_tracked_item(item_id)["status"] == "retired"
    assert _is_active(subs_db, source_id) == 0, "no live watch: source off"
    assert [r["enabled"] for r in await _change_rules(service, source_id)] == [
        False], "and its change rule silenced again"

    for off in (True, False):
        await _switch(service, dreams_db, scheduling_db, off=off)
        assert _is_active(subs_db, source_id) == 0
        assert not any(
            r["enabled"] for r in await _change_rules(service, source_id))


# --- TASK-34920: a shared source's restart debt survives its watch -------------
#
# Two watches ride one Dreams source. At switch-off, watch A's record write
# fails and B's lands, so the source stops with only B owing it a restart.
# B is then untracked while Dreams is off: the debt was left on a retired
# watch, and switching on left A active on a source that stayed off.


async def _two_watches_one_owing(dreams_db, service, scheduling_db):
    """A and B ride one source; the switch-off recorded the stop on B only."""
    first = _seed_story(dreams_db, "one", url=_URL)
    second = _seed_story(dreams_db, "two", url=_URL, date="2026-09-21")
    a = await track_page(service, dreams_db, url=_URL, title="Flights",
                         intent="deal", origin_story_id=first)
    b = await track_page(service, dreams_db, url=_URL, title="Flights",
                         intent="deal", origin_story_id=second)
    assert a["subscription_id"] == b["subscription_id"]
    real = dreams_db.set_tracked_suspension

    def a_fails(tracked_item_id, record, **kwargs):
        if int(tracked_item_id) == a["tracked_item_id"]:
            raise RuntimeError("database is locked")
        return real(tracked_item_id, record, **kwargs)

    dreams_db.set_tracked_suspension = a_fails
    try:
        await _switch(service, dreams_db, scheduling_db, off=True)
    finally:
        dreams_db.set_tracked_suspension = real
    assert _record(dreams_db, a["tracked_item_id"]) is None
    assert _record(dreams_db, b["tracked_item_id"])["subscription"] is True
    return first, a, b


@pytest.mark.asyncio
@pytest.mark.parametrize("another_off_pass", [False, True],
                         ids=["on-at-once", "off-pass-first"])
async def test_a_live_watch_gets_its_source_back_after_its_sibling_left(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db,
        another_off_pass):
    subs_db, service = subs_stack
    _story, a, b = await _two_watches_one_owing(
        dreams_db, service, scheduling_db)
    source_id = a["subscription_id"]
    assert _is_active(subs_db, source_id) == 0

    await untrack(service, dreams_db, b["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))
    if another_off_pass:
        await _switch(service, dreams_db, scheduling_db, off=True)
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert dreams_db.get_tracked_item(a["tracked_item_id"])["status"] == "active"
    assert _is_active(subs_db, source_id) == 1, "A's source runs again"
    assert not _record(dreams_db, b["tracked_item_id"])["subscription"], (
        "the retired watch no longer owes it")


@pytest.mark.asyncio
async def test_tracking_a_story_again_restarts_the_source_its_watch_needs(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """AC 2: the debt is left on the retired watch (the hand-over write
    failed, or an earlier build left it there); tracking story one again
    turns its live watch's source back on."""
    subs_db, service = subs_stack
    story, a, b = await _two_watches_one_owing(
        dreams_db, service, scheduling_db)
    source_id = a["subscription_id"]
    real = dreams_db.set_tracked_suspension

    def a_fails(tracked_item_id, record, **kwargs):
        if int(tracked_item_id) == a["tracked_item_id"]:
            raise RuntimeError("database is locked")
        return real(tracked_item_id, record, **kwargs)

    dreams_db.set_tracked_suspension = a_fails
    try:
        await untrack(service, dreams_db, b["tracked_item_id"],
                      scheduling_db_getter=_sched_getter(scheduling_db))
    finally:
        dreams_db.set_tracked_suspension = real
    assert _record(dreams_db, b["tracked_item_id"])["subscription"] is True
    await _switch(service, dreams_db, scheduling_db, off=False)
    assert _is_active(subs_db, source_id) == 0, "fixture: left off"

    again = await track_page(service, dreams_db, url=_URL, title="Flights",
                             intent="deal", origin_story_id=story)

    assert again["tracked_item_id"] == a["tracked_item_id"]
    assert _is_active(subs_db, source_id) == 1
    assert not _record(dreams_db, b["tracked_item_id"])["subscription"]


# --- TASK-34921: a switch-on that stops part-way keeps the hold ----------------
#
# A watch held at switch-on records why, and its Tracked row says "paused"
# with that reason. A later pass that did not finish -- the watchlists
# service failed, or was not there -- wrote its record back without the
# reason: the row lost its label while the source was still off, and the
# next start ran a lookup it is documented not to make.


def _tracked_row(dreams_db, item_id):
    from tldw_chatbook.Dreams.dreams_view import list_tracked_updates

    return next(row for row in list_tracked_updates(dreams_db)
                if row["id"] == item_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("unfinished", ["service-fails", "no-service"])
async def test_a_pass_that_does_not_finish_keeps_the_hold(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch,
        unfinished):
    subs_db, service = subs_stack
    owned = await _held_for_dns(dreams_db, service, scheduling_db, monkeypatch)
    item_id = owned["tracked_item_id"]
    before = _record(dreams_db, item_id)
    _resolve_publicly(monkeypatch)
    if unfinished == "service-fails":
        async def unreachable(source_id):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(service, "get_source", unreachable)
        await retry_unfinished_resume(
            service, dreams_db,
            scheduling_db_getter=_sched_getter(scheduling_db), now=NOW)
    else:
        await _switch(None, dreams_db, scheduling_db, off=False)

    assert _record(dreams_db, item_id) == before, "the hold stands as it was"
    assert _tracked_row(dreams_db, item_id)["held"] == "dns_failure"
    assert _boot_needs_switch(dreams_db, enabled=True) is False
    assert _is_active(subs_db, owned["subscription_id"]) == 0


@pytest.mark.asyncio
async def test_a_hold_with_work_left_keeps_its_reason_through_the_pass(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """A held watch whose reminder is still owed (the switch-on ran without
    the scheduling DB) is written ahead and trimmed as the pass works; it
    keeps its reason all the way, and when the pass cannot decide the
    source it ends held, as before, with the reminder restored."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    _fail_resolution(monkeypatch)
    await set_tracking_suspended(service, dreams_db, suspended=False,
                                 scheduling_db_getter=None, now=NOW)
    record = _record(dreams_db, item_id)
    assert record["held"] == "dns_failure" and record["reminders"], "fixture"
    seen: list[str | None] = []
    real = dreams_db.set_tracked_suspension

    def watch_writes(tracked_item_id, text, **kwargs):
        if int(tracked_item_id) == item_id:
            seen.append((json.loads(text) if text else {}).get("held"))
        return real(tracked_item_id, text, **kwargs)

    async def unreachable(source_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(dreams_db, "set_tracked_suspension", watch_writes)
    monkeypatch.setattr(service, "get_source", unreachable)
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert seen and set(seen) == {"dns_failure"}, "every write keeps it"
    record = _record(dreams_db, item_id)
    assert (record["held"], record.get("settled"), record["reminders"]) == (
        "dns_failure", "on", [])
    assert _tracked_row(dreams_db, item_id)["held"] == "dns_failure"


@pytest.mark.asyncio
async def test_a_watch_whose_restore_failed_keeps_its_last_record(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """AC 3: the closing step leaves a watch whose own restore failed as
    last written. The first watch the pass works on has its reminder
    re-armed (and dropped from its record) before its rule step fails; the
    other then decides the shared source. Writing the failed watch's record
    from the row the pass read at its start would list the re-armed
    reminder as owed again, to be re-armed over the user's head by the
    next pass."""
    subs_db, service = subs_stack
    first = await _event_page(dreams_db, service, scheduling_db, story_id=1)
    second = await _event_page(dreams_db, service, scheduling_db, story_id=2)
    a, b = first["tracked_item_id"], second["tracked_item_id"]
    assert first["subscription_id"] == second["subscription_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    assert _record(dreams_db, a)["reminders"] and _record(dreams_db, a)["rules"]
    real = service.update_alert_rule
    failures = [RuntimeError("database is locked")]

    async def first_rule_step_fails(rule_id, **fields):
        if failures and fields.get("enabled") is True:
            raise failures.pop()
        return await real(rule_id, **fields)

    service.update_alert_rule = first_rule_step_fails
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert not failures, "fixture: one watch's rule step failed"
    assert _is_active(subs_db, first["subscription_id"]) == 1, (
        "the other decided the source")
    (failed,) = [item for item in (a, b) if _record(dreams_db, item)]
    record = _record(dreams_db, failed)
    assert record["reminders"] == [], "the re-armed reminder is not owed again"
    assert record["rules"], "the rule it could not restore is still owed"


# --- TASK-34927: tracking again after the source was deleted -------------------


@pytest.mark.asyncio
async def test_tracking_again_after_deleting_the_source_makes_a_working_watch(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """What the notices and the guide tell the user to do -- delete the
    source in Watchlists and track again -- used to find the story's old
    watch and report it attached, with the deleted source's id, leaving it
    active with nothing behind it."""
    subs_db, service = subs_stack
    old = await _event_page(dreams_db, service, scheduling_db)
    await service.delete_source(old["subscription_id"])

    again = await _event_page(dreams_db, service, scheduling_db)

    assert again["outcome"] == "created"
    assert again["subscription_id"] != old["subscription_id"]
    assert _is_active(subs_db, again["subscription_id"]) == 1
    assert dreams_db.get_tracked_item(old["tracked_item_id"])["status"] == "retired"
    assert dreams_db.get_tracked_item(again["tracked_item_id"])[
        "subscription_id"] == again["subscription_id"]
    assert [r["enabled"] for r in await _change_rules(
        service, again["subscription_id"])] == [True]
    armed = [row for row in scheduling_db.list_reminder_tasks_by_link(
        "dream_tracked_item",
        [str(old["tracked_item_id"]), str(again["tracked_item_id"])])
        if row.get("enabled")]
    assert [row["link_id"] for row in armed] == [str(again["tracked_item_id"])]


# --- TASK-34928: tracking while Dreams is off; the verifying backfill ---------


@pytest.mark.asyncio
async def test_tracking_while_dreams_is_off_changes_nothing_and_says_why(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """The Library hides Track while Dreams is off; a call that reaches the
    service anyway used to switch the page's source back on and arm a
    reminder for a new story of it."""
    from tldw_chatbook.Dreams.track_service import TrackDreamsOff

    subs_db, service = subs_stack
    first = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-10-01",
        origin_story_id=_seed_story(dreams_db, "one", url=_URL),
        scheduling_db_getter=_sched_getter(scheduling_db))
    settings["enabled"] = False
    await _switch(service, dreams_db, scheduling_db, off=True)

    def watches() -> int:
        with dreams_db.connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM dream_tracked_items").fetchone()[0]

    before = (watches(), _reminder_states(scheduling_db),
              _record(dreams_db, first["tracked_item_id"]))

    for track in (
        lambda: track_page(
            service, dreams_db, url=_URL, title="Festival", intent="event",
            event_date="2026-10-01",
            origin_story_id=_seed_story(
                dreams_db, "two", url=_URL, date="2026-09-21"),
            scheduling_db_getter=_sched_getter(scheduling_db)),
        lambda: track_page(
            service, dreams_db, url="https://example.com/other", title="Other",
            intent="topic", origin_story_id=_seed_story(dreams_db, "three")),
        lambda: track_question(
            dreams_db, query_template=_TEMPLATE, intent="event",
            event_date="2026-10-01", origin_story_id=1,
            scheduling_db_getter=_sched_getter(scheduling_db)),
    ):
        with pytest.raises(TrackDreamsOff) as refused:
            await track()
        assert refused.value.reason_code == "track_dreams_off"
        assert "off" in str(refused.value)

    assert _is_active(subs_db, first["subscription_id"]) == 0
    assert _count_rows(subs_db, "subscriptions") == 1
    assert (watches(), _reminder_states(scheduling_db),
            _record(dreams_db, first["tracked_item_id"])) == before


@pytest.mark.asyncio
async def test_a_verifying_backfill_with_nothing_to_mark_writes_nothing(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    from tldw_chatbook.Dreams.track_service import backfill_egress_marks

    subs_db, service = subs_stack
    await _event_page(dreams_db, service, scheduling_db, story_id=1)
    await track_page(service, dreams_db, url="https://example.com/other",
                     title="Other", intent="topic", origin_story_id=2)
    writes: list[str] = []
    real_mark, real_update = dreams_db.set_egress_marked, service.update_source

    def marked(subscription_id):
        writes.append(f"marker {subscription_id}")
        return real_mark(subscription_id)

    async def updated(subscription_id, payload):
        writes.append(f"source {subscription_id}")
        return await real_update(subscription_id, payload)

    dreams_db.set_egress_marked = marked
    service.update_source = updated

    assert await backfill_egress_marks(service, dreams_db, verify=True) == 0
    assert writes == [], "every source is marked: nothing to write"


# --- TASK-34923: the reminder hand-over is durable ------------------------------
#
# One story, one reminder per event date, linked to whichever watch was made
# first; when that watch retires, the story's other live watch of the event
# takes it. A failed record write used to leave it disabled and owed by
# nobody, it kept the old watch's title, and a hand-over during a switch-off
# could leave it recorded on no watch at all.

_QUESTION_TITLE = "Dreams: flights to {region}"


def _owing(dreams_db, reminder_id, *item_ids) -> list[int]:
    """The watches whose record lists the reminder as owed."""
    return [item_id for item_id in item_ids
            if reminder_id in ((_record(dreams_db, item_id) or {})
                               .get("reminders") or [])]


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", [False, True], ids=["writes-land", "write-fails"])
async def test_a_reminder_handed_on_while_off_is_armed_by_the_next_switch_on(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, failing):
    """AC 1 and 2. Dreams is off; the watch that holds the reminder is
    untracked (its retirement record write failing, or not). Switching on
    gives the story's other watch the reminder, armed, under its title."""
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    page_id = page["tracked_item_id"]
    settings["enabled"] = False
    await _switch(service, dreams_db, scheduling_db, off=True)
    real = dreams_db.set_tracked_suspension
    failures = [RuntimeError("database is locked")] if failing else []

    def flaky(tracked_item_id, record, **kwargs):
        if failures:
            raise failures.pop()
        return real(tracked_item_id, record, **kwargs)

    dreams_db.set_tracked_suspension = flaky
    try:
        await untrack(service, dreams_db, page_id,
                      scheduling_db_getter=_sched_getter(scheduling_db))
    finally:
        dreams_db.set_tracked_suspension = real
    assert not failures, "fixture: the write failed"
    assert not scheduling_db.get_reminder_task(reminder["id"])["enabled"]
    assert len(_owing(dreams_db, reminder["id"], page_id, question_id)) == 1

    settings["enabled"] = True
    await _switch(service, dreams_db, scheduling_db, off=False)

    row = scheduling_db.get_reminder_task(reminder["id"])
    assert row["enabled"] and row["link_id"] == str(question_id)
    assert row["title"] == _QUESTION_TITLE


@pytest.mark.asyncio
async def test_a_failed_hand_over_while_dreams_is_on_is_finished_by_the_switch(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """AC 1, Dreams on: the holder still owed its reminder (a switch-on had
    run without the scheduling store), and the heir's record write fails
    during the hand-over. The reminder must not end disabled and owed by
    nobody: the next switch-on re-arms it on the heir."""
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    page_id = page["tracked_item_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    await set_tracking_suspended(service, dreams_db, suspended=False,
                                 scheduling_db_getter=None, now=NOW)
    assert _record(dreams_db, page_id)["reminders"] == [reminder["id"]]
    real = dreams_db.set_tracked_suspension

    def heir_write_fails(tracked_item_id, record, **kwargs):
        if int(tracked_item_id) == question_id:
            raise RuntimeError("database is locked")
        return real(tracked_item_id, record, **kwargs)

    dreams_db.set_tracked_suspension = heir_write_fails
    try:
        await untrack(service, dreams_db, page_id,
                      scheduling_db_getter=_sched_getter(scheduling_db))
    finally:
        dreams_db.set_tracked_suspension = real
    assert _owing(dreams_db, reminder["id"], page_id, question_id), (
        "some watch still owes it")

    await _switch(service, dreams_db, scheduling_db, off=True)
    await _switch(service, dreams_db, scheduling_db, off=False)

    row = scheduling_db.get_reminder_task(reminder["id"])
    assert row["enabled"] and row["link_id"] == str(question_id)


@pytest.mark.asyncio
async def test_a_handed_over_reminder_takes_the_receiving_watchs_title(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """AC 2, Dreams on: the reminder used to keep the page watch's title."""
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    assert reminder["title"] == "Dreams: tracked event", "fixture"

    await untrack(service, dreams_db, page["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))

    row = scheduling_db.get_reminder_task(reminder["id"])
    assert (row["link_id"], row["title"]) == (str(question_id), _QUESTION_TITLE)


@pytest.mark.asyncio
async def test_a_hand_over_during_a_switch_off_leaves_the_reminder_on_one_watch(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """AC 3. The holder is untracked while the switch-off pass runs --
    after the pass recorded both watches and before it settled them. The
    pass's settle write of the heir used to overwrite the debt the
    hand-over had just written, leaving the reminder recorded on nobody."""
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    page_id = page["tracked_item_id"]
    settings["enabled"] = False
    real = service.update_alert_rule
    landed: list[bool] = []

    async def untrack_during_the_stop(rule_id, **fields):
        if not landed:
            landed.append(True)
            await untrack(service, dreams_db, page_id,
                          scheduling_db_getter=_sched_getter(scheduling_db))
        return await real(rule_id, **fields)

    service.update_alert_rule = untrack_during_the_stop
    await _switch(service, dreams_db, scheduling_db, off=True)
    service.update_alert_rule = real

    assert landed, "fixture: the untrack landed inside the pass"
    assert not scheduling_db.get_reminder_task(reminder["id"])["enabled"]
    assert len(_owing(dreams_db, reminder["id"], page_id, question_id)) == 1

    settings["enabled"] = True
    await _switch(service, dreams_db, scheduling_db, off=False)
    row = scheduling_db.get_reminder_task(reminder["id"])
    assert row["enabled"] and row["link_id"] == str(question_id)


# AC 4: the four behaviours a mutation used to survive.


@pytest.mark.asyncio
async def test_a_reminder_the_user_switched_off_stays_off_through_the_switch(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Not handed on as a debt either: a switch-off and switch-on after the
    untrack must not arm it on the other watch over the user's decision."""
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    scheduling_db.update_reminder_task(reminder["id"], enabled=False)  # the user
    await untrack(service, dreams_db, page["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))

    await _switch(service, dreams_db, scheduling_db, off=True)
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert not any(row["enabled"] for row in _all_event_reminders(scheduling_db))


@pytest.mark.asyncio
async def test_a_watch_with_its_own_reminder_is_not_given_a_second(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    own = scheduling_db.create_reminder_task(  # as an older build left it
        "local", "Dreams: flights to {region}",
        body=reminder["body"], schedule_kind="one_time",
        run_at=reminder["run_at"], next_run_at=reminder["run_at"],
        link_type="dream_tracked_item", link_id=str(question_id))

    await untrack(service, dreams_db, page["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))

    handed = scheduling_db.get_reminder_task(reminder["id"])
    assert handed["link_id"] == str(page["tracked_item_id"])
    assert not handed["enabled"]
    assert [row["id"] for row in _linked_reminders(scheduling_db, question_id)
            if row["enabled"]] == [own]


@pytest.mark.asyncio
async def test_the_reminder_of_a_passed_event_is_not_handed_over(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """The sweep retires the page watch because its event is over; the
    question watch of that event, paused after failed checks (the sweep
    never retires a paused watch), must not inherit the nudge."""
    _subs_db, service = subs_stack
    page = await track_page(
        service, dreams_db, url=_URL, title="Festival", intent="event",
        event_date="2026-09-01", origin_story_id=7,
        scheduling_db_getter=_sched_getter(scheduling_db))
    question_id = await track_question(
        dreams_db, query_template=_TEMPLATE, intent="event",
        event_date="2026-09-01", origin_story_id=7,
        scheduling_db_getter=_sched_getter(scheduling_db))
    (reminder,) = _all_event_reminders(scheduling_db)
    assert reminder["link_id"] == str(page["tracked_item_id"]), "fixture"
    dreams_db.set_tracked_status(question_id, "paused")

    await sweep_track_lifecycle(
        dreams_db, now=NOW, subs_service_getter=lambda: service,
        scheduling_db_getter=_sched_getter(scheduling_db))

    assert dreams_db.get_tracked_item(page["tracked_item_id"])["status"] == "retired"
    row = scheduling_db.get_reminder_task(reminder["id"])
    assert row["link_id"] == str(page["tracked_item_id"]) and not row["enabled"]


@pytest.mark.asyncio
async def test_the_retired_watch_does_not_claim_the_reminder_it_handed_on(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """A reminder Dreams disabled and the retiring watch owed, handed on
    to the other watch, is that watch's: the retired record must not list
    it too, or a later re-track could take it for one Dreams disarmed."""
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    page_id = page["tracked_item_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    await set_tracking_suspended(service, dreams_db, suspended=False,
                                 scheduling_db_getter=None, now=NOW)
    assert _record(dreams_db, page_id)["reminders"] == [reminder["id"]], "fixture"

    await untrack(service, dreams_db, page_id,
                  scheduling_db_getter=_sched_getter(scheduling_db))

    assert _owing(dreams_db, reminder["id"], page_id, question_id) == [question_id]


# --- Review of #3108 ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_held_page_with_work_left_loses_its_hold_once_its_page_passes(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch):
    """X1. A held watch with its reminder still owed keeps its reason
    through the pass's writes (TASK-34921). When its page then passes and
    the closing write fails, the source runs -- so the record must not
    still say "held", or the Tracked row calls a running page paused."""
    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    await _switch(service, dreams_db, scheduling_db, off=True)
    _fail_resolution(monkeypatch)
    await set_tracking_suspended(service, dreams_db, suspended=False,
                                 scheduling_db_getter=None, now=NOW)
    record = _record(dreams_db, item_id)
    assert record["held"] == "dns_failure" and record["reminders"], "fixture"
    _resolve_publicly(monkeypatch)
    real = dreams_db.set_tracked_suspension

    def closing_write_fails(tracked_item_id, text, **kwargs):
        if int(tracked_item_id) == item_id and text is None:
            raise RuntimeError("database is locked")
        return real(tracked_item_id, text, **kwargs)

    monkeypatch.setattr(dreams_db, "set_tracked_suspension", closing_write_fails)
    await _switch(service, dreams_db, scheduling_db, off=False)

    assert _is_active(subs_db, owned["subscription_id"]) == 1
    assert "held" not in _record(dreams_db, item_id)
    assert _tracked_row(dreams_db, item_id)["held"] is None
    assert _boot_needs_switch(dreams_db, enabled=True) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["same-story", "new-story"])
async def test_a_switch_off_during_a_re_tracks_lookup_switches_nothing_on(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch,
        path):
    """X2. Dreams was on when the re-track started (TASK-34928 checks at
    entry). It is switched off -- and the switch-off pass finishes --
    while the re-track looks the page's name up; the re-track must not
    then switch the source on with Dreams off."""
    from tldw_chatbook.Dreams.track_service import TrackDreamsOff
    from tldw_chatbook.Utils import egress

    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    source_id = owned["subscription_id"]
    if path == "same-story":
        await _switch(service, dreams_db, scheduling_db, off=True)
        _resolve_everything_to(monkeypatch, "10.0.0.5")
        await _switch(service, dreams_db, scheduling_db, off=False)
        assert _record(dreams_db, owned["tracked_item_id"])["held"] == "private"
    else:
        await untrack(service, dreams_db, owned["tracked_item_id"],
                      scheduling_db_getter=_sched_getter(scheduling_db))
    assert _is_active(subs_db, source_id) == 0, "fixture"
    switched: list[bool] = []

    async def switch_off_during_the_lookup(_host):
        if not switched:
            switched.append(True)
            settings["enabled"] = False
            await _switch(service, dreams_db, scheduling_db, off=True)
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", switch_off_during_the_lookup)

    with pytest.raises(TrackDreamsOff):
        await _event_page(dreams_db, service, scheduling_db,
                          story_id=1 if path == "same-story" else 2)

    assert switched, "fixture: the switch-off ran during the lookup"
    assert _is_active(subs_db, source_id) == 0, "Dreams is off: source off"


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["at-retirement", "at-switch-on"])
async def test_a_reminder_title_the_user_chose_survives_the_hand_over(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, when):
    """X3. Schedules lets the user rename a reminder; the hand-over gives a
    reminder the receiving watch's title only while it still carries the
    old watch's own title."""
    _subs_db, service = subs_stack
    page, question_id, reminder = await _page_then_question(
        dreams_db, service, scheduling_db)
    scheduling_db.update_reminder_task(
        reminder["id"], title="Buy tickets for the festival!")  # the user
    if when == "at-switch-on":
        settings["enabled"] = False
        await _switch(service, dreams_db, scheduling_db, off=True)
    await untrack(service, dreams_db, page["tracked_item_id"],
                  scheduling_db_getter=_sched_getter(scheduling_db))
    if when == "at-switch-on":
        settings["enabled"] = True
        await _switch(service, dreams_db, scheduling_db, off=False)

    row = scheduling_db.get_reminder_task(reminder["id"])
    assert row["link_id"] == str(question_id) and row["enabled"]
    assert row["title"] == "Buy tickets for the festival!"


@pytest.mark.asyncio
async def test_a_reminder_titled_by_an_older_build_takes_the_new_watchs_title(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Builds before round 4 wrote the title with its square brackets; that
    is still the old watch's own title, not one the user chose."""
    _subs_db, service = subs_stack
    question_id = await track_question(
        dreams_db, query_template=_BRACKETED_QUERY, intent="event",
        event_date="2026-10-01", origin_story_id=7,
        scheduling_db_getter=_sched_getter(scheduling_db))
    page = await _event_page(dreams_db, service, scheduling_db, story_id=7)
    (reminder,) = _all_event_reminders(scheduling_db)
    assert reminder["link_id"] == str(question_id), "fixture"
    scheduling_db.update_reminder_task(
        reminder["id"], title=f"Dreams: {_BRACKETED_QUERY}")  # as written then

    await untrack(service, dreams_db, question_id,
                  scheduling_db_getter=_sched_getter(scheduling_db))

    row = scheduling_db.get_reminder_task(reminder["id"])
    assert row["link_id"] == str(page["tracked_item_id"])
    assert row["title"] == "Dreams: tracked event"


@pytest.mark.asyncio
async def test_tracking_again_after_deleting_the_source_keeps_the_users_reminder(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """X4. The new watch is made first and the dead one retired after, so
    the story's reminder -- at the time the user moved it to -- moves to
    the new watch instead of a fresh one at the default time."""
    _subs_db, service = subs_stack
    old = await _event_page(dreams_db, service, scheduling_db)
    (reminder,) = _all_event_reminders(scheduling_db)
    moved_to = "2026-09-29T07:00:00+00:00"
    scheduling_db.update_reminder_task(
        reminder["id"], run_at=moved_to, next_run_at=moved_to)  # the user
    await service.delete_source(old["subscription_id"])

    again = await _event_page(dreams_db, service, scheduling_db)

    assert again["outcome"] == "created"
    assert dreams_db.get_tracked_item(old["tracked_item_id"])["status"] == "retired"
    (kept,) = _all_event_reminders(scheduling_db)
    assert kept["id"] == reminder["id"], "the same reminder, not a new one"
    assert kept["link_id"] == str(again["tracked_item_id"]) and kept["enabled"]
    assert "07:00" in str(kept["run_at"])


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["same-story", "new-story"])
async def test_a_switch_off_after_the_re_tracks_check_never_leaves_the_source_on(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, monkeypatch,
        path):
    """Review of #3108, round 2, Z1. Dreams is switched off, and a
    switch-off pass started, right after the re-track checked the switch
    (at the start of the un-settling) and before it switches the source
    on. The pass may not run in between: it runs after, and stops the
    source the re-track switched on. Without that, the pass finished
    first and the re-track then left the source running with Dreams off."""
    from tldw_chatbook.Dreams import track_service

    subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    source_id = owned["subscription_id"]
    if path == "same-story":
        await _switch(service, dreams_db, scheduling_db, off=True)
        _resolve_everything_to(monkeypatch, "10.0.0.5")
        await _switch(service, dreams_db, scheduling_db, off=False)
        _resolve_everything_to(monkeypatch, "93.184.216.34")
    else:
        await untrack(service, dreams_db, owned["tracked_item_id"],
                      scheduling_db_getter=_sched_getter(scheduling_db))
    assert _is_active(subs_db, source_id) == 0, "fixture"
    real_unsettle = track_service._unsettle_held
    passes: list[asyncio.Task] = []

    async def switch_off_after_the_check(*args, **kwargs):
        if not passes:
            settings["enabled"] = False
            passes.append(asyncio.ensure_future(
                _switch(service, dreams_db, scheduling_db, off=True)))
            try:  # it finishes here unless something holds it back
                await asyncio.wait_for(asyncio.shield(passes[0]), 0.5)
            except TimeoutError:
                pass
        return await real_unsettle(*args, **kwargs)

    monkeypatch.setattr(track_service, "_unsettle_held", switch_off_after_the_check)
    await _event_page(dreams_db, service, scheduling_db,
                      story_id=1 if path == "same-story" else 2)
    await passes[0]

    assert _is_active(subs_db, source_id) == 0, "Dreams is off: source off"
    assert not any(
        r["enabled"] for r in await _change_rules(service, source_id))


@pytest.mark.asyncio
async def test_a_failed_replacement_is_cleaned_up_by_the_next_track(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Review of #3108, round 2, Z2. Retiring the dead watch fails after
    its replacement was made: the user's track still succeeded, and the
    next track of the story retires the dead watch with the live one as
    heir -- one live watch, holding the reminder at the user's time."""
    _subs_db, service = subs_stack
    old = await _event_page(dreams_db, service, scheduling_db)
    (reminder,) = _all_event_reminders(scheduling_db)
    moved_to = "2026-09-29T07:00:00+00:00"
    scheduling_db.update_reminder_task(
        reminder["id"], run_at=moved_to, next_run_at=moved_to)  # the user
    await service.delete_source(old["subscription_id"])
    real = dreams_db.set_tracked_status
    failures = [RuntimeError("database is locked")]

    def retiring_old_fails(tracked_item_id, status, **kwargs):
        if (failures and int(tracked_item_id) == old["tracked_item_id"]
                and status == "retired"):
            raise failures.pop()
        return real(tracked_item_id, status, **kwargs)

    dreams_db.set_tracked_status = retiring_old_fails
    try:
        first = await _event_page(dreams_db, service, scheduling_db)
    finally:
        dreams_db.set_tracked_status = real
    assert not failures, "fixture: the retirement failed"
    assert first["outcome"] == "created", "the user's track succeeded"

    second = await _event_page(dreams_db, service, scheduling_db)

    assert second["tracked_item_id"] == first["tracked_item_id"]
    live = [row["id"] for row in dreams_db.list_tracked_for_story(1)
            if row["status"] in ("active", "paused")]
    assert live == [first["tracked_item_id"]], "one live watch"
    (kept,) = _all_event_reminders(scheduling_db)
    assert kept["id"] == reminder["id"] and kept["enabled"]
    assert kept["link_id"] == str(first["tracked_item_id"])
    assert "07:00" in str(kept["run_at"])


def test_the_switch_lock_is_one_per_event_loop_and_stays_put():
    """Review of #3108, round 3: the lock used to be replaced whenever a
    different loop asked for it, so two loops used alternately would stop
    excluding each other's callers and a holder's lock could be swapped out.
    Each loop now keeps its own lock for as long as it lives."""
    from tldw_chatbook.Dreams.track_service import _switch_lock

    async def ask():
        return _switch_lock()

    first, second = asyncio.new_event_loop(), asyncio.new_event_loop()
    try:
        a1 = first.run_until_complete(ask())
        b1 = second.run_until_complete(ask())
        a2 = first.run_until_complete(ask())
        b2 = second.run_until_complete(ask())
    finally:
        first.close()
        second.close()
    assert a1 is a2 and b1 is b2
    assert a1 is not b1


# --- TASK-34985: a re-track after an untrack while Dreams was off alerts ------


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["dreams-created", "users-own"])
async def test_tracking_again_after_untracking_while_off_alerts_again(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db, source):
    """The switch-off silenced Dreams' change rule and its record said so;
    retiring the watch while Dreams was off dropped that, and tracking the
    page again reused the rule as it stood: a tracked page that never
    alerts. A user's own disabled rule on their own source stays off."""
    _subs_db, service = subs_stack
    users_rule = None
    if source == "dreams-created":
        owned = await _event_page(dreams_db, service, scheduling_db)
        source_id, item_id = owned["subscription_id"], owned["tracked_item_id"]
        track_again = lambda: _event_page(dreams_db, service, scheduling_db)
    else:
        source_id, attached = await _attach_to_users_source(
            dreams_db, service, provenance=UrlProvenance.USER_ENTERED)
        item_id = attached["tracked_item_id"]
        users_rule = await _users_rule(service, job_id=source_id)
        await service.update_alert_rule(users_rule["rule_id"], enabled=False)
        track_again = lambda: track_page(
            service, dreams_db, url="https://example.com/users-own",
            title="Theirs", intent="topic", origin_story_id=2)
    dreams_rule = _rule_id_on_record(dreams_db, item_id)
    settings["enabled"] = False
    await _switch(service, dreams_db, scheduling_db, off=True)
    await untrack(service, dreams_db, item_id,
                  scheduling_db_getter=_sched_getter(scheduling_db))
    settings["enabled"] = True
    await _switch(service, dreams_db, scheduling_db, off=False)

    await track_again()

    rules = {r["rule_id"]: r["enabled"]
             for r in await service.list_alert_rules(job_id=source_id)}
    assert rules[dreams_rule] is True or rules[dreams_rule] == 1, (
        "the tracked page alerts again")
    if users_rule is not None:
        assert not rules[users_rule["rule_id"]], "the user's own rule stays off"


@pytest.mark.asyncio
async def test_tracking_again_leaves_off_a_rule_the_user_switched_off_after(
        tmp_path, dreams_db, settings, subs_stack, scheduling_db):
    """Review of #3123, N4. The rule a retired watch's record owes is the
    one Dreams switched off; switched on and off again by the user after
    the untrack, it is the user's, and tracking the page again leaves it
    off -- and the record no longer owes it."""
    _subs_db, service = subs_stack
    owned = await _event_page(dreams_db, service, scheduling_db)
    item_id = owned["tracked_item_id"]
    dreams_rule = _rule_id_on_record(dreams_db, item_id)
    settings["enabled"] = False
    await _switch(service, dreams_db, scheduling_db, off=True)
    await untrack(service, dreams_db, item_id,
                  scheduling_db_getter=_sched_getter(scheduling_db))
    await service.update_alert_rule(dreams_rule, enabled=True)
    await service.update_alert_rule(dreams_rule, enabled=False)
    settings["enabled"] = True
    await _switch(service, dreams_db, scheduling_db, off=False)

    await _event_page(dreams_db, service, scheduling_db)

    rules = {r["rule_id"]: r["enabled"] for r in await service.list_alert_rules(
        job_id=owned["subscription_id"])}
    assert not rules[dreams_rule], "the user's switch-off stands"
    assert dreams_rule not in (_record(dreams_db, item_id) or {}).get(
        "rules", []), "and is no longer owed"
