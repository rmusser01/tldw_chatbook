"""Artifacts screen Reports slot: empty-state CTA and seeded report rows.

Real app via `_build_test_app`, real `SubscriptionsDB`, real write paths for
seeding; no LLM/fetch involved (no briefing generation here, just rows).
"""

from contextlib import asynccontextmanager

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen

pytestmark = pytest.mark.ui


def _seed_report(app, *, status: str = "complete") -> int:
    db: SubscriptionsDB = app.subscriptions_db
    watchlist_id = int(WatchlistBundleService(db).create("Daily Brief")["id"])
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(
        briefing_id, status=status,
        body_markdown="## Daily Brief\n\nOne story [item 1].", item_count=1,
    )
    return briefing_id


@asynccontextmanager
async def _open_artifacts(app, *, size=(160, 50)):
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        yield screen, pilot


@pytest.mark.asyncio
async def test_empty_state_offers_demo_cta():
    app = _build_test_app(configured_default="artifacts")
    async with _open_artifacts(app) as (screen, pilot):
        cta = screen.query_one("#artifacts-daily-report-demo", Button)
        assert cta.region.height >= 1, "CTA must paint, not just mount"
        assert screen.query_one("#artifacts-list-reports", Static)


@pytest.mark.asyncio
async def test_seeded_reports_list_rows_with_open_button():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    async with _open_artifacts(app) as (screen, pilot):
        screen._start_daily_reports_refresh()
        for _ in range(50):
            await pilot.pause(0.05)
            if screen._daily_reports:
                break
        assert screen._daily_reports, "refresh worker must land rows"
        row = screen.query_one(f"#artifacts-report-row-{briefing_id}", Static)
        assert row.region.height >= 1, "report row must paint"
        assert screen.query_one("#artifacts-open-watchlists", Button)
        # CTA belongs to the empty state only
        assert not screen.query("#artifacts-daily-report-demo")


@pytest.mark.asyncio
async def test_audio_row_shows_play_button():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    db: SubscriptionsDB = app.subscriptions_db
    script_id = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Daily Brief",
        roster_snapshot_json="[]",
    )
    db.update_briefing_script(script_id, status="complete", turns_json="[]")
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    # A lexically-safe path: under the real briefing_audio_dir the guard passes
    # without touching disk (the play handler itself checks existence).
    from tldw_chatbook.Subscriptions.briefing_audio import briefing_audio_dir
    db.update_briefing_audio(
        audio_id, status="complete",
        file_path=str(briefing_audio_dir() / f"script-{script_id}-audio-{audio_id}.wav"),
        duration_seconds=1.0, turn_count=1,
    )
    async with _open_artifacts(app) as (screen, pilot):
        screen._start_daily_reports_refresh()
        for _ in range(50):
            await pilot.pause(0.05)
            if screen._daily_reports:
                break
        play = screen.query_one(f"#artifacts-report-play-{briefing_id}", Button)
        assert play.region.height >= 1


@pytest.mark.asyncio
async def test_demo_cta_runs_the_wired_service():
    app = _build_test_app(configured_default="artifacts")

    class _StubDemo:
        def __init__(self):
            self.calls = 0

        async def run_demo(self):
            self.calls += 1
            return {"status": "complete"}

    stub = _StubDemo()
    app.daily_report_demo_service = stub
    async with _open_artifacts(app) as (screen, pilot):
        screen.query_one("#artifacts-daily-report-demo").press()
        for _ in range(50):
            await pilot.pause(0.05)
            if stub.calls:
                break
        assert stub.calls == 1
