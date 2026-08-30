"""DailyReportDemoService: real DBs and services, faked seams only.

Faked seams per test: the chat callable (DI, the service's own parameter) and
the HTTP fetch (monkeypatched `monitoring_engine.guarded_fetch_httpx_async`,
the convention `test_url_monitor_off_loop.py` set). Everything else -- watchlist
creation, subscription rows, run rows, item upserts, briefing lifecycle -- is
the real production path.
"""

import asyncio
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
async def test_reused_schedule_with_all_current_runs_failing_is_still_fetch_failed(
    tmp_path, monkeypatch
):
    """Qodo #9: historical items must not mask a total current-fetch failure.

    The old decision read the watchlist's LIFETIME item count, so a reused
    schedule whose earlier runs had persisted items sailed past the gate
    with every current run failing. The verdict is now the runs launched in
    THIS invocation.
    """
    # First run succeeds and persists items (the historical masking data).
    service, db, spy = _service(tmp_path, monkeypatch)
    first = await service.run_demo()
    assert first["status"] == "complete"
    # Second run reuses the schedule but every current fetch fails.
    service2, db2, spy2 = _service(tmp_path, monkeypatch, fail_fetch=True)
    outcome = await service2.run_demo()
    assert outcome["status"] == "fetch_failed"
    assert outcome["watchlist_id"] == first["watchlist_id"]
    # The historical items are still there -- proving the gate ignored them.
    with db2.transaction() as conn:
        n_items = conn.execute(
            "SELECT COUNT(*) AS n FROM subscription_items"
        ).fetchone()["n"]
    assert n_items > 0
    assert len(db2.list_briefings(outcome["watchlist_id"])) == 1, \
        "only the first, successful run's briefing row may exist"


@pytest.mark.asyncio
async def test_run_demo_detached_rejects_a_second_start_and_seeds_once(
    tmp_path, monkeypatch
):
    """Qodo #10/#11: double CTA activation must not double-seed."""
    import threading

    release = threading.Event()

    class _BlockingChat:
        """Fast until first call, then parked until the test releases it."""

        def __call__(self, **kwargs):
            release.wait(timeout=10)
            return "## Daily Brief\n\nOne story [item 1].\n"

    service, db, spy = _service(tmp_path, monkeypatch, chat=_BlockingChat())

    first = service.run_demo_detached()
    assert first is not None
    assert service.demo_in_progress() is True
    second = service.run_demo_detached()
    assert second is None, "second start must be refused while one runs"
    assert any("already running" in t for t in _titles(spy))

    release.set()
    outcome = await asyncio.wait_for(first, timeout=10)
    assert outcome["status"] == "complete"
    assert service.demo_in_progress() is False
    # Exactly one seed happened: one watchlist, one set of sources.
    with db.transaction() as conn:
        n_watchlists = conn.execute(
            "SELECT COUNT(*) AS n FROM watchlists"
        ).fetchone()["n"]
        n_sources = conn.execute(
            "SELECT COUNT(*) AS n FROM watchlist_sources"
        ).fetchone()["n"]
    assert n_watchlists == 1
    assert n_sources == len(DEMO_SOURCES)


@pytest.mark.asyncio
async def test_concurrent_run_demo_calls_cannot_double_seed(tmp_path, monkeypatch):
    """Qodo #11: the module lock serializes discovery-through-fetch for
    DIRECT `run_demo` callers (different service instances, so the pending-
    task guard alone cannot help)."""
    db = _db(tmp_path)
    local = LocalWatchlistsService(db_factory=lambda: db)
    spy = _DispatchSpy()
    _serve_rss(monkeypatch)
    service_a = DailyReportDemoService(
        subscriptions_db=db,
        local_watchlists_getter=lambda: local,
        dispatch_service=spy,
        app_getter=lambda: None,
        chat=_FakeChat(),
    )
    service_b = DailyReportDemoService(
        subscriptions_db=db,
        local_watchlists_getter=lambda: local,
        dispatch_service=spy,
        app_getter=lambda: None,
        chat=_FakeChat(),
    )

    outcomes = await asyncio.gather(service_a.run_demo(), service_b.run_demo())

    assert {o["status"] for o in outcomes} <= {"complete", "in_flight"}
    with db.transaction() as conn:
        n_watchlists = conn.execute(
            "SELECT COUNT(*) AS n FROM watchlists"
        ).fetchone()["n"]
        n_sources = conn.execute(
            "SELECT COUNT(*) AS n FROM watchlist_sources"
        ).fetchone()["n"]
    assert n_watchlists == 1, "the second caller must reuse, not re-seed"
    assert n_sources == len(DEMO_SOURCES), "no duplicate source attachments"


@pytest.mark.asyncio
async def test_run_demo_reused_schedule_without_preset_skips_audio(tmp_path, monkeypatch):
    """Qodo #12: a cleared default preset skips audio accurately.

    The old path fabricated `preset_id=0`, guaranteeing a cast failure. The
    skip must be honest: recorded reason, no script row, calm notification.
    The reused run needs a COMPLETE briefing (a fresh item in the feed), or
    the empty-window skip (Qodo #13) would fire first by design.
    """
    feed = {"body": _RSS}

    async def _variable_guarded(url, *, client, max_bytes, **kwargs):
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/rss+xml"},
            text=feed["body"],
            final_url=url,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        _variable_guarded,
    )
    service, db, spy = _service(tmp_path, monkeypatch, profiles=())
    first = await service.run_demo()
    assert first["status"] == "complete"

    # A second service WITH voice profiles, a fresh item in the feed, and
    # the schedule's preset cleared.
    from tldw_chatbook.TTS.profile_types import TTSGenerationProfile

    def _profile(pid: uuid.UUID) -> TTSGenerationProfile:
        now = datetime.now(timezone.utc)
        return TTSGenerationProfile(
            profile_id=pid, display_name="Host voice", normalized_name="host voice",
            provider_id="openai", model_id="tts-1", voice_id="alloy",
            response_format="wav", speed=1.0, options={}, revision=1,
            created_at=now, updated_at=now,
        )

    service2, db2, spy2 = _service(
        tmp_path, monkeypatch, profiles=(_profile(uuid.uuid4()),)
    )
    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        _variable_guarded,
    )
    db2.set_watchlist_briefing_settings(
        first["watchlist_id"], default_preset_id=None
    )
    feed["body"] = _RSS.replace("example.com/1", "example.com/2")
    outcome = await service2.run_demo()

    assert outcome["status"] == "complete"
    assert outcome["audio"] == "skipped"
    assert "audio:skipped:no-preset" in outcome["reasons"]
    assert db2.get_briefing(outcome["briefing_id"])["status"] == "complete"
    assert db2.list_briefing_scripts(outcome["briefing_id"]) == [], \
        "no cast script may be generated without a preset"
    assert any("no default briefing preset" in c["message"] for c in spy2.calls)


@pytest.mark.asyncio
async def test_run_demo_empty_window_skips_audio_calmly(tmp_path, monkeypatch):
    """Qodo #13: an empty-window briefing is a skip, not a cast failure.

    `generate_script` refuses non-complete rows by contract, so an
    empty-window briefing reaching it could only produce a spurious "could
    not be synthesized" failure.
    """
    from tldw_chatbook.TTS.profile_types import TTSGenerationProfile

    def _profile(pid: uuid.UUID) -> TTSGenerationProfile:
        now = datetime.now(timezone.utc)
        return TTSGenerationProfile(
            profile_id=pid, display_name="Host voice", normalized_name="host voice",
            provider_id="openai", model_id="tts-1", voice_id="alloy",
            response_format="wav", speed=1.0, options={}, revision=1,
            created_at=now, updated_at=now,
        )

    service, db, spy = _service(
        tmp_path, monkeypatch, profiles=(_profile(uuid.uuid4()),)
    )
    first = await service.run_demo()
    assert first["status"] == "complete"  # seeds items and advances watermark

    outcome = await service.run_demo()  # same-day: nothing new above watermark

    assert outcome["status"] == "complete"
    assert outcome["audio"] == "skipped"
    assert "audio:skipped:empty-window" in outcome["reasons"]
    assert db.list_briefing_scripts(outcome["briefing_id"]) == [], \
        "no cast script may be generated for an empty window"
    assert not any(
        "could not be synthesized" in c["title"] for c in spy.calls
    ), "an empty window is not an audio failure"
    assert any("nothing new to read" in c["message"] for c in spy.calls)


@pytest.mark.asyncio
async def test_failed_briefing_dispatches_provider_guidance(tmp_path, monkeypatch):
    service, db, spy = _service(tmp_path, monkeypatch, chat=_FakeChat(error=RuntimeError("401 unauthorized")))
    outcome = await service.run_demo()
    assert outcome["status"] == "briefing_failed"
    last = spy.calls[-1]
    assert last["severity"] == "warning"
    assert "401 unauthorized" in last["message"]
    assert "Settings" in last["message"]


@pytest.mark.asyncio
async def test_run_demo_skips_audio_without_voice_profiles(tmp_path, monkeypatch):
    service, db, spy = _service(tmp_path, monkeypatch, profiles=())
    outcome = await service.run_demo()
    assert outcome["status"] == "complete"
    assert outcome["audio"] == "skipped"
    assert db.list_briefing_scripts(outcome["briefing_id"]) == [], \
        "no cast script should be generated when it could not be voiced"
    assert any("Audio skipped" in t for t in _titles(spy))


@pytest.mark.asyncio
async def test_run_demo_generates_audio_when_ready(tmp_path, monkeypatch):
    from tldw_chatbook.TTS.profile_types import TTSGenerationProfile

    def _profile(pid: uuid.UUID) -> TTSGenerationProfile:
        now = datetime.now(timezone.utc)
        return TTSGenerationProfile(
            profile_id=pid, display_name="Host voice", normalized_name="host voice",
            provider_id="openai", model_id="tts-1", voice_id="alloy",
            response_format="wav", speed=1.0, options={}, revision=1,
            created_at=now, updated_at=now,
        )

    profiles = (_profile(uuid.uuid4()),)
    service, db, spy = _service(tmp_path, monkeypatch, profiles=profiles)

    # Orchestration pin: the demo module's own seams, faked here because
    # `briefing_audio`/`briefing_cast` internals have their own suites.
    scripted: list[tuple[int, int]] = []

    async def _fake_generate_script(db_, briefing_id, *, preset_id, **kwargs):
        script_id = db_.insert_briefing_script(
            briefing_id, preset_id=preset_id, preset_name="Daily Brief",
            roster_snapshot_json="[]",
        )
        db_.update_briefing_script(script_id, status="complete", turns_json="[]")
        scripted.append((briefing_id, script_id))
        return db_.get_briefing_script(script_id)

    audio_calls: list[dict] = []

    async def _fake_generate_script_audio(db_, script_id, **kwargs):
        audio_calls.append({"script_id": script_id, **kwargs})
        return {"id": 1, "script_id": script_id, "status": "complete"}

    monkeypatch.setattr(daily_report_demo, "generate_script", _fake_generate_script)
    monkeypatch.setattr(daily_report_demo, "generate_script_audio", _fake_generate_script_audio)

    outcome = await service.run_demo()

    assert outcome["status"] == "complete"
    assert outcome["audio"] == "complete"
    assert scripted == [(outcome["briefing_id"], audio_calls[0]["script_id"])]
    assert "Recording audio" in _titles(spy)
    assert not any("Audio skipped" in t for t in _titles(spy))


@pytest.mark.asyncio
async def test_run_demo_audio_failure_degrades_to_text_success(tmp_path, monkeypatch):
    from tldw_chatbook.TTS.profile_types import TTSGenerationProfile

    def _profile(pid: uuid.UUID) -> TTSGenerationProfile:
        now = datetime.now(timezone.utc)
        return TTSGenerationProfile(
            profile_id=pid, display_name="Host voice", normalized_name="host voice",
            provider_id="openai", model_id="tts-1", voice_id="alloy",
            response_format="wav", speed=1.0, options={}, revision=1,
            created_at=now, updated_at=now,
        )

    service, db, spy = _service(
        tmp_path, monkeypatch, profiles=(_profile(uuid.uuid4()),)
    )

    async def _fake_generate_script(db_, briefing_id, *, preset_id, **kwargs):
        script_id = db_.insert_briefing_script(
            briefing_id, preset_id=preset_id, preset_name="Daily Brief",
            roster_snapshot_json="[]",
        )
        db_.update_briefing_script(script_id, status="complete", turns_json="[]")
        return db_.get_briefing_script(script_id)

    async def _failing_audio(db_, script_id, **kwargs):
        return {"id": 1, "script_id": script_id, "status": "failed",
                "error": "pydub is not installed"}

    monkeypatch.setattr(daily_report_demo, "generate_script", _fake_generate_script)
    monkeypatch.setattr(daily_report_demo, "generate_script_audio", _failing_audio)

    outcome = await service.run_demo()
    assert outcome["status"] == "complete", "audio failure never fails the demo"
    assert outcome["audio"] == "failed"
    assert any("Audio could not be synthesized" in t for t in _titles(spy))
