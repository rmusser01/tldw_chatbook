"""Watchlists demo banner: shown only while no briefing schedule exists.

Real app + real SubscriptionsDB via `_build_test_app`; visibility comes from
real `list_briefing_schedules` rows, dismissal from the real config file the
test conftest isolates.
"""

import os
import tomllib
from contextlib import asynccontextmanager

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)

pytestmark = pytest.mark.ui


def _seed_schedule(app) -> None:
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
        WatchlistBundleService,
    )
    db = app.subscriptions_db
    watchlist_id = int(WatchlistBundleService(db).create("Daily Brief")["id"])
    db.set_watchlist_briefing_settings(
        watchlist_id, briefing_cadence_seconds=86_400
    )


@asynccontextmanager
async def _open(app, *, size=(180, 50)):
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        # Give the banner-resolution worker its turn(s).
        for _ in range(50):
            await pilot.pause(0.05)
            if screen.query("#watchlists-daily-report-banner"):
                break
        yield screen, pilot


@pytest.mark.asyncio
async def test_banner_mounts_when_no_schedule_exists():
    app = _build_test_app(configured_default="watchlists_collections")
    async with _open(app) as (screen, pilot):
        assert screen.query_one("#watchlists-daily-report-banner")


@pytest.mark.asyncio
async def test_banner_absent_when_a_schedule_exists():
    app = _build_test_app(configured_default="watchlists_collections")
    _seed_schedule(app)
    async with _open(app) as (screen, pilot):
        for _ in range(20):
            await pilot.pause(0.05)
        assert not screen.query("#watchlists-daily-report-banner")


@pytest.mark.asyncio
async def test_dismiss_persists_and_removes_banner():
    app = _build_test_app(configured_default="watchlists_collections")
    async with _open(app) as (screen, pilot):
        screen.query_one("#watchlists-daily-report-banner-dismiss").press()
        for _ in range(20):
            await pilot.pause(0.05)
        assert not screen.query("#watchlists-daily-report-banner")
        config_path = os.environ["TLDW_CONFIG_PATH"]
        with open(config_path, "rb") as fh:
            data = tomllib.load(fh)
        assert data["scheduling"]["daily_report_demo_banner_dismissed"] is True


@pytest.mark.asyncio
async def test_dismiss_after_worker_removed_banner_does_not_raise():
    """A Dismiss press queued just before demo-worker completion dispatches
    after the worker already took the banner down; the handler must no-op
    (an unhandled NoMatches from query_one panics the app)."""
    app = _build_test_app(configured_default="watchlists_collections")
    async with _open(app) as (screen, pilot):
        dismiss = screen.query_one("#watchlists-daily-report-banner-dismiss")
        # The demo worker's own teardown path removes the banner first.
        screen.query("#watchlists-daily-report-banner").first().remove()
        for _ in range(5):
            await pilot.pause(0.05)
        assert not screen.query("#watchlists-daily-report-banner")
        # The late-arriving press dispatches against a banner-less screen.
        screen.post_message(Button.Pressed(dismiss))
        for _ in range(20):
            await pilot.pause(0.05)
        assert not screen.query("#watchlists-daily-report-banner")
        config_path = os.environ["TLDW_CONFIG_PATH"]
        with open(config_path, "rb") as fh:
            data = tomllib.load(fh)
        assert data["scheduling"]["daily_report_demo_banner_dismissed"] is True


@pytest.mark.asyncio
async def test_demo_cta_starts_the_detached_demo_and_takes_banner_down():
    """Qodo #10: the banner CTA starts the service-owned demo task (never
    the demo itself inside a screen worker) and takes the banner down once
    the demo has started."""
    app = _build_test_app(configured_default="watchlists_collections")

    class _StubDemo:
        def __init__(self):
            self.started = 0

        def run_demo_detached(self):
            self.started += 1
            return object()  # truthy task stand-in

    stub = _StubDemo()
    app.daily_report_demo_service = stub
    async with _open(app) as (screen, pilot):
        screen.query_one("#watchlists-daily-report-demo").press()
        for _ in range(20):
            await pilot.pause(0.05)
            if stub.started:
                break
        assert stub.started == 1
        assert not screen.query("#watchlists-daily-report-banner")


@pytest.mark.asyncio
async def test_demo_cta_refused_start_keeps_banner():
    """Qodo #10/#11: when the service refuses a second start (a demo is
    already running), the banner stays -- the invitation is still live."""
    app = _build_test_app(configured_default="watchlists_collections")

    class _RefusedDemo:
        def run_demo_detached(self):
            return None

    app.daily_report_demo_service = _RefusedDemo()
    async with _open(app) as (screen, pilot):
        screen.query_one("#watchlists-daily-report-demo").press()
        for _ in range(10):
            await pilot.pause(0.05)
        assert screen.query_one("#watchlists-daily-report-banner")
