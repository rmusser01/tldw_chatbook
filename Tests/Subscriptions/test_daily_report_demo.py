"""DailyReportDemoService: real DBs and services, faked seams only.

Faked seams per test: the chat callable (DI, the service's own parameter) and
the HTTP fetch (monkeypatched `monitoring_engine.guarded_fetch_httpx_async`,
the convention `test_url_monitor_off_loop.py` set). Everything else -- watchlist
creation, subscription rows, run rows, item upserts, briefing lifecycle -- is
the real production path.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import daily_report_demo
from tldw_chatbook.Subscriptions.daily_report_demo import (
    DEMO_CADENCE_SECONDS,
    DEMO_SOURCES,
    DEMO_WATCHLIST_NAME,
    DailyReportDemoService,
)
from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService
from tldw_chatbook.TTS.profile_types import (
    ProfileStoreResult,
    TTSProfilePage,
)

pytestmark = pytest.mark.unit

_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Demo Feed</title>
<item><title>Demo story</title><link>https://example.com/1</link>
<description>Body of the demo story.</description>
<pubDate>Thu, 28 Aug 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""


def _db(tmp_path) -> SubscriptionsDB:
    return SubscriptionsDB(tmp_path / "subs.db", "test")


class _FakeChat:
    """Stand-in for `Chat_Functions.chat_api_call` (the one faked seam)."""

    def __init__(self, *, error: Exception | None = None):
        self.reply = "## Daily Brief\n\nOne story [item 1].\n"
        self.error = error

    def __call__(self, **kwargs):
        if self.error is not None:
            raise self.error
        return self.reply


class _DispatchSpy:
    def __init__(self):
        self.calls: list[dict] = []

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        return {"persisted": True}


class _ProfileService:
    """Mirrors `TTSProfileService.list_profiles`'s real return shape."""

    def __init__(self, profiles=()):
        self._profiles = tuple(profiles)

    async def list_profiles(self, search=None, limit=50, offset=0):
        return ProfileStoreResult(
            generation=1,
            value=TTSProfilePage(profiles=self._profiles, total=len(self._profiles)),
        )


def _serve_rss(monkeypatch, *, fail_all: bool = False):
    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        if fail_all:
            raise RuntimeError("network unreachable")
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/rss+xml"},
            text=_RSS,
            final_url=url,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )


def _service(tmp_path, monkeypatch, *, chat=None, profiles=(), fail_fetch=False):
    db = _db(tmp_path)
    local = LocalWatchlistsService(db_factory=lambda: db)
    spy = _DispatchSpy()
    _serve_rss(monkeypatch, fail_all=fail_fetch)
    service = DailyReportDemoService(
        subscriptions_db=db,
        local_watchlists_getter=lambda: local,
        dispatch_service=spy,
        app_getter=lambda: None,
        tts_service_getter=lambda: None,
        tts_profile_service_getter=lambda: _ProfileService(profiles),
        chat=chat if chat is not None else _FakeChat(),
    )
    return service, db, spy


def _titles(spy):
    return [c["title"] for c in spy.calls]


@pytest.mark.asyncio
async def test_run_demo_seeds_watchlist_preset_schedule_and_briefs(tmp_path, monkeypatch):
    service, db, spy = _service(tmp_path, monkeypatch)

    outcome = await service.run_demo()

    assert outcome["status"] == "complete"
    watchlist_id = outcome["watchlist_id"]
    assert watchlist_id is not None
    assert outcome["briefing_id"] is not None
    # Seeded setup: sources attached, preset bound, daily cadence set.
    schedules = db.list_briefing_schedules()
    assert len(schedules) == 1
    assert schedules[0]["watchlist_id"] == watchlist_id
    assert schedules[0]["briefing_cadence_seconds"] == DEMO_CADENCE_SECONDS
    with db.transaction() as conn:
        n_sources = conn.execute(
            "SELECT COUNT(*) AS n FROM watchlist_sources WHERE watchlist_id = ?",
            (watchlist_id,),
        ).fetchone()["n"]
    assert n_sources == len(DEMO_SOURCES)
    presets = db.list_briefing_presets()
    assert any(p["name"] == DEMO_WATCHLIST_NAME for p in presets)
    # The live fetch + generation actually ran through the real seams.
    briefing = db.get_briefing(outcome["briefing_id"])
    assert briefing["status"] == "complete"
    assert briefing["item_count"] >= 1
    # Stage trail + one completion dispatch, all under the briefing category.
    assert "Fetching today's stories" in _titles(spy)
    assert "Writing your brief" in _titles(spy)
    assert spy.calls[-1]["category"] == "briefing"


@pytest.mark.asyncio
async def test_run_demo_is_idempotent_when_a_schedule_exists(tmp_path, monkeypatch):
    service, db, _ = _service(tmp_path, monkeypatch)
    await service.run_demo()
    service2, db2, _ = _service(tmp_path, monkeypatch)  # same tmp DB file
    outcome = await service2.run_demo()

    assert outcome["status"] == "complete"
    with db2.transaction() as conn:
        n_watchlists = conn.execute("SELECT COUNT(*) AS n FROM watchlists").fetchone()["n"]
        # Adaptation (disclosed in task-5-report): `resolve_or_create_watchlist`
        # case-insensitively reuses "Daily Brief", so a re-seeding second run
        # still leaves exactly ONE watchlist row -- the watchlist count alone
        # cannot detect re-seeding. The source-membership count can: re-seeding
        # attaches three MORE subscriptions to the same watchlist (6 != 3).
        n_sources = conn.execute(
            "SELECT COUNT(*) AS n FROM watchlist_sources WHERE watchlist_id = ?",
            (outcome["watchlist_id"],),
        ).fetchone()["n"]
    assert n_watchlists == 1, "second run must reuse, not re-seed"
    assert n_sources == len(DEMO_SOURCES), "second run must not re-seed sources"
    assert len(db2.list_briefing_schedules()) == 1
    assert len(db2.list_briefings(outcome["watchlist_id"])) == 2  # ran again, once per demo


@pytest.mark.asyncio
async def test_run_demo_without_local_service_reports_unavailable(tmp_path, monkeypatch):
    db = _db(tmp_path)
    spy = _DispatchSpy()
    service = DailyReportDemoService(
        subscriptions_db=db,
        local_watchlists_getter=lambda: None,
        dispatch_service=spy,
        app_getter=lambda: None,
        chat=_FakeChat(),
    )
    outcome = await service.run_demo()
    assert outcome["status"] == "unavailable"
    with db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM watchlists").fetchone()["n"] == 0


@pytest.mark.asyncio
async def test_run_demo_all_sources_failing_aborts_with_fetch_failed(tmp_path, monkeypatch):
    service, db, spy = _service(tmp_path, monkeypatch, fail_fetch=True)
    outcome = await service.run_demo()
    assert outcome["status"] == "fetch_failed"
    assert db.list_briefings(outcome["watchlist_id"]) == [], "no briefing row on total fetch failure"


@pytest.mark.asyncio
async def test_failed_briefing_dispatches_provider_guidance(tmp_path, monkeypatch):
    service, db, spy = _service(tmp_path, monkeypatch, chat=_FakeChat(error=RuntimeError("401 unauthorized")))
    outcome = await service.run_demo()
    assert outcome["status"] == "briefing_failed"
    last = spy.calls[-1]
    assert last["severity"] == "warning"
    assert "401 unauthorized" in last["message"]
    assert "Settings" in last["message"]
