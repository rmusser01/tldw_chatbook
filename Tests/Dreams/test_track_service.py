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
import hashlib
import json
from datetime import UTC, datetime

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Dreams.cycle_service import CycleDeps
from tldw_chatbook.Dreams.discovery import normalize_url
from tldw_chatbook.Dreams.track_service import (
    TrackCapReached,
    TrackSourceDisabled,
    rebaseline_track,
    run_track_check,
    track_page,
    track_question,
    untrack,
)
from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    NotificationDispatchService,
)
from tldw_chatbook.Subscriptions import LocalWatchlistsService
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
    return hashlib.sha256("\n".join(
        normalize_url(url) + title for title, url, _snippet in results
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

    assert result == {"status": "changed", "notified": True}
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "changed"
    assert runs[0]["notified"] == 1
    assert "notification unavailable" in runs[0]["verdict_note"], (
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

    assert result == {"status": "changed", "notified": True}
    runs = dreams_db.list_recent_track_runs(item_id)
    assert runs[0]["status"] == "changed"
    assert runs[0]["notified"] == 1
    assert "new dates announced" in runs[0]["verdict_note"]
    assert "degraded" in runs[0]["verdict_note"]


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
