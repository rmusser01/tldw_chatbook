"""Tests for the new watchlists screen shell structure."""

import asyncio
import itertools
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual.widgets import Button, DataTable, Input, Select, Static, TextArea

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Subscriptions.noise_defaults import default_ignore_selectors_text
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import NotificationsPane
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RulesPane
from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane


def _settings_without_splash(section, key=None, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


class _ScreenWorkerView:
    """Wait only for workers owned by one production screen."""

    def __init__(self, app, screen) -> None:
        self._app = app
        self._screen = screen

    def _owned_workers(self):
        return [
            worker
            for worker in self._app.workers
            if self._screen in worker.node.ancestors_with_self
        ]

    def __iter__(self):
        return iter(self._owned_workers())

    async def wait_for_complete(self) -> None:
        owned_workers = [
            worker
            for worker in self._owned_workers()
            if not worker.is_finished
        ]
        if owned_workers:
            await self._app.workers.wait_for_complete(owned_workers)


class DestinationHarness:
    """Mount the production destination screen inside the full production app."""

    def __init__(self, app, destination: str) -> None:
        assert destination == "watchlists_collections"
        self.app = app
        self.context_screen = WatchlistsCollectionsScreen(app)

    @property
    def screen_stack(self):
        return self.app.screen_stack

    @property
    def workers(self):
        return _ScreenWorkerView(self.app, self.context_screen)

    @asynccontextmanager
    async def run_test(self, **kwargs):
        with patch(
            "tldw_chatbook.app.get_cli_setting",
            side_effect=_settings_without_splash,
        ):
            async with self.app.run_test(**kwargs) as pilot:
                await self.app.push_screen(self.context_screen)
                await pilot.pause()
                yield pilot


class WatchlistsContextHarness:
    """Mount a configured production screen inside its full production app."""

    def __init__(self, screen: WatchlistsCollectionsScreen) -> None:
        self.app = screen.app_instance
        self.context_screen = screen

    @asynccontextmanager
    async def run_test(self, **kwargs):
        with patch(
            "tldw_chatbook.app.get_cli_setting",
            side_effect=_settings_without_splash,
        ):
            async with self.app.run_test(**kwargs) as pilot:
                await self.app.push_screen(self.context_screen)
                await pilot.pause()
                yield pilot


@pytest.mark.asyncio
async def test_watchlists_shell_has_tab_strip_and_panes():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        assert screen.query_one("#wl-tabs")
        # TASK-1344: FEEDS (`#watchlists-list-pane`) is gated to the Read
        # tab, matching CONTENT -- the default section ("overview") shows
        # only ITEMS/the inspector centre-width, with the tab strip and
        # snapshot markers carried by `#wl-centre-status`
        # (`_build_centre_status_header`) instead of FEEDS's own body.
        assert screen.query_one("#wl-centre-status")
        assert not screen.query("#watchlists-list-pane")
        assert screen.query_one("#watchlists-detail-pane")
        assert screen.query_one("#watchlists-inspector-pane")
        assert screen.query_one("#watchlists-backend-select", Select)

        screen.active_section = "items"
        await pilot.pause(0.2)
        assert screen.query_one("#watchlists-list-pane"), (
            "FEEDS's own pane must still exist on the Read tab"
        )


@pytest.mark.asyncio
async def test_watchlists_tab_strip_updates_active_section():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.active_section == "overview"
        screen.query_one("#wl-tab-sources", Button).press()
        await pilot.pause()
        assert screen.active_section == "sources"


def test_watchlists_navigation_context_accepts_only_supported_sections():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)

    screen.apply_navigation_context({"section": "rules"})
    assert screen.active_section == "rules"

    screen.apply_navigation_context({"section": "notifications"})
    assert screen.active_section == "notifications"

    screen.apply_navigation_context({"section": "not-a-section"})
    assert screen.active_section == "notifications"


@pytest.mark.asyncio
async def test_subscriptions_alias_preserves_watchlists_navigation_context(monkeypatch):
    app = _build_test_app()
    applied_contexts = []
    switched_screens = []

    class FakeWatchlistsScreen:
        screen_name = "watchlists_collections"

        def __init__(self, app_instance):
            self.app_instance = app_instance

        def apply_navigation_context(self, context):
            applied_contexts.append(dict(context))

    class FakeOutgoingScreen:
        screen_name = "home"

    async def fake_switch_screen(screen):
        switched_screens.append(screen)

    monkeypatch.setattr(
        app,
        "_resolve_screen_navigation_target",
        lambda target: (
            "watchlists_collections",
            "watchlists_collections",
            FakeWatchlistsScreen,
        ),
    )
    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)
    monkeypatch.setattr(
        type(app), "screen", property(lambda self: FakeOutgoingScreen())
    )

    await app.handle_screen_navigation(
        NavigateToScreen("subscriptions", screen_context={"section": "rules"})
    )

    assert applied_contexts == [{"section": "rules"}]
    assert len(switched_screens) == 1
    assert isinstance(switched_screens[0], FakeWatchlistsScreen)
    assert app.current_tab == "watchlists_collections"


@pytest.mark.asyncio
async def test_watchlists_navigation_context_selects_run_after_initial_load():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen._controller.list_runs = AsyncMock(
        return_value=[
            {
                "id": "local:watchlist_run:4",
                "run_id": 4,
                "source_title": "Earlier run",
                "status": "completed",
            },
            {
                "id": "local:watchlist_run:5",
                "run_id": 5,
                "source_title": "Daily security feed",
                "status": "failed",
            },
        ]
    )
    screen.apply_navigation_context(
        {
            "section": "runs",
            "backend": "local",
            "run_id": "local:watchlist_run:5",
        }
    )
    host = WatchlistsContextHarness(screen)

    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            runs_pane = screen.query_one("#watchlists-runs-pane", RunsPane)
            if runs_pane.selected_run is not None:
                break

        assert screen.active_section == "runs"
        screen._controller.list_runs.assert_awaited_once()
        assert runs_pane.selected_run is not None
        assert runs_pane.selected_run["id"] == "local:watchlist_run:5"
        assert screen.runtime_backend == "local"


@pytest.mark.asyncio
async def test_mounted_watchlists_context_retains_run_selection_across_recompose():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    host = WatchlistsContextHarness(screen)
    record = {
        "id": "local:watchlist_run:5",
        "run_id": 5,
        "source_title": "Daily security feed",
        "status": "failed",
    }

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen._controller.list_runs = AsyncMock(return_value=[record])

        screen.apply_navigation_context(
            {
                "section": "runs",
                "backend": "local",
                "run_id": "local:watchlist_run:5",
            }
        )
        for _ in range(20):
            await pilot.pause()
            runs_pane = screen.query_one("#watchlists-runs-pane", RunsPane)
            if runs_pane.selected_run is not None:
                break

        assert screen._pending_navigation_run_id is None
        assert screen.selected_run == record
        assert screen.selected_entity == record
        assert runs_pane.selected_run == record
        screen._controller.list_runs.assert_awaited_once_with(
            runtime_backend="local",
            limit=100,
        )


@pytest.mark.asyncio
async def test_watchlists_run_deep_link_selects_raw_run_id_record():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen._controller.list_runs = AsyncMock(
        return_value=[
            {"run_id": 5, "source_title": "Legacy-shaped run", "status": "failed"}
        ]
    )
    screen.apply_navigation_context(
        {
            "section": "runs",
            "backend": "local",
            "run_id": "local:watchlist_run:5",
        }
    )
    host = WatchlistsContextHarness(screen)

    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            runs_pane = screen.query_one("#watchlists-runs-pane", RunsPane)
            if runs_pane.selected_run is not None:
                break

        assert runs_pane.selected_run is not None
        assert runs_pane.selected_run["run_id"] == 5
        assert screen.selected_run is runs_pane.selected_run


def test_raw_run_deep_link_matches_canonical_same_backend_record():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    record = {
        "id": "local:watchlist_run:5",
        "run_id": 5,
        "backend": "local",
        "source_title": "Canonical record",
    }
    screen.apply_navigation_context(
        {"section": "runs", "backend": "local", "run_id": "5"}
    )

    assert screen._matching_requested_run([record]) is record


def test_raw_run_deep_link_does_not_match_canonical_other_backend_record():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen.apply_navigation_context(
        {"section": "runs", "backend": "local", "run_id": "5"}
    )

    assert (
        screen._matching_requested_run(
            [
                {
                    "id": "server:watchlist_run:5",
                    "run_id": 5,
                    "backend": "server",
                }
            ]
        )
        is None
    )


@pytest.mark.asyncio
async def test_server_run_deep_link_selects_server_backend_before_loading():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen._controller.list_runs = AsyncMock(
        return_value=[
            {
                "id": "server:watchlist_run:8",
                "run_id": 8,
                "source_title": "Server run",
                "status": "failed",
            }
        ]
    )
    screen.apply_navigation_context(
        {
            "section": "runs",
            "backend": "server",
            "run_id": "server:watchlist_run:8",
        }
    )
    host = WatchlistsContextHarness(screen)

    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            runs_pane = screen.query_one("#watchlists-runs-pane", RunsPane)
            if runs_pane.selected_run is not None:
                break

        assert screen.runtime_backend == "server"
        assert screen.query_one("#watchlists-backend-select", Select).value == "server"
        screen._controller.list_runs.assert_awaited_once_with(
            runtime_backend="server",
            limit=100,
        )
        assert runs_pane.selected_run["id"] == "server:watchlist_run:8"


@pytest.mark.asyncio
async def test_missing_run_deep_link_is_consumed_without_later_stale_selection():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen._controller.list_runs = AsyncMock(
        return_value=[
            {
                "id": "server:watchlist_run:5",
                "run_id": 5,
                "source_title": "Wrong backend",
                "status": "failed",
            }
        ]
    )
    screen.apply_navigation_context(
        {
            "section": "runs",
            "backend": "local",
            "run_id": "local:watchlist_run:5",
        }
    )
    host = WatchlistsContextHarness(screen)

    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if screen._controller.list_runs.await_count:
                break

        runs_pane = screen.query_one("#watchlists-runs-pane", RunsPane)
        assert screen._pending_navigation_run_id is None
        assert runs_pane.selected_run is None

        screen._controller.list_runs.return_value = [
            {
                "id": "local:watchlist_run:5",
                "run_id": 5,
                "source_title": "Now available",
                "status": "failed",
            }
        ]
        await screen._load_runs()

        assert runs_pane.selected_run is None
        assert screen.selected_run is None


def test_leaving_runs_clears_pending_run_deep_link():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen.apply_navigation_context(
        {
            "section": "runs",
            "backend": "server",
            "run_id": "server:watchlist_run:8",
        }
    )

    assert screen.runtime_backend == "server"
    assert screen._pending_navigation_run_id == "server:watchlist_run:8"

    screen.active_section = "sources"

    assert screen._pending_navigation_run_id is None


@pytest.mark.asyncio
async def test_watchlists_notifications_context_loads_and_updates_local_inbox():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    row = {
        "id": 7,
        "title": "Research complete",
        "message": "The synthesis is ready.",
        "category": "research",
        "severity": "info",
        "is_read": False,
    }
    screen._notifications_controller.load_rows = AsyncMock(return_value=[row])
    screen._notifications_controller.mark_read = AsyncMock(return_value=True)
    screen._notifications_controller.dismiss = AsyncMock(return_value=True)
    screen.apply_navigation_context({"section": "notifications"})
    host = WatchlistsContextHarness(screen)

    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            pane = screen.query_one("#watchlists-notifications-pane", NotificationsPane)
            if pane.notifications:
                break

        assert pane.notifications == [row]
        pane.select_notification_by_id("7")
        await pilot.pause()
        pane.query_one("#notifications-mark-read-button", Button).press()
        for _ in range(20):
            await pilot.pause()
            if screen._notifications_controller.mark_read.await_count:
                break
        screen._notifications_controller.mark_read.assert_awaited_once_with(
            7, is_read=True
        )

        pane.select_notification_by_id("7")
        await pilot.pause()
        pane.query_one("#notifications-dismiss-button", Button).press()
        for _ in range(20):
            await pilot.pause()
            if screen._notifications_controller.dismiss.await_count:
                break
        screen._notifications_controller.dismiss.assert_awaited_once_with(
            7, is_dismissed=True
        )


@pytest.mark.asyncio
async def test_switching_to_notifications_does_not_report_recompose_as_load_error():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    host = WatchlistsContextHarness(screen)
    row = {
        "id": 7,
        "title": "Research complete",
        "message": "The synthesis is ready.",
        "category": "research",
        "severity": "info",
        "is_read": False,
    }

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen._notifications_controller.load_rows = AsyncMock(return_value=[row])
        app.notify = Mock()

        screen.active_section = "notifications"
        for _ in range(20):
            await pilot.pause()
            pane = screen.query_one("#watchlists-notifications-pane", NotificationsPane)
            if pane.notifications:
                break

        assert pane.notifications == [row]
        screen._notifications_controller.load_rows.assert_awaited_once()
        app.notify.assert_not_called()


@pytest.mark.asyncio
async def test_notification_selection_survives_screen_recompose():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    host = WatchlistsContextHarness(screen)
    row = {
        "id": 7,
        "title": "Research complete",
        "message": "The synthesis is ready.",
        "category": "research",
        "severity": "info",
        "is_read": False,
    }
    screen._notifications_controller.load_rows = AsyncMock(return_value=[row])
    screen.apply_navigation_context({"section": "notifications"})

    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            pane = screen.query_one("#watchlists-notifications-pane", NotificationsPane)
            if pane.notifications:
                break
        pane.select_notification_by_id("7")
        await pilot.pause()
        original_pane = pane

        screen.refresh(recompose=True)
        for _ in range(20):
            await pilot.pause()
            pane = screen.query_one("#watchlists-notifications-pane", NotificationsPane)
            if pane is not original_pane:
                break

        assert pane.selected_notification == row
        assert screen.selected_entity["id"] == 7
        assert not pane.query_one("#notifications-mark-read-button", Button).disabled
        assert not pane.query_one("#notifications-dismiss-button", Button).disabled


@pytest.mark.asyncio
async def test_watchlists_notifications_section_reads_real_client_inbox():
    app = _build_test_app()
    inserted = app.client_notifications_db.insert(
        category="research",
        title="[b]Research complete[/b]",
        message="The synthesis is ready.",
        severity="info",
    )
    screen = WatchlistsCollectionsScreen(app)
    screen.apply_navigation_context({"section": "notifications"})
    host = WatchlistsContextHarness(screen)

    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            pane = screen.query_one("#watchlists-notifications-pane", NotificationsPane)
            if pane.notifications:
                break

        assert [row["id"] for row in pane.notifications] == [inserted["id"]]
        assert pane.notifications[0]["title"] == "[b]Research complete[/b]"
        assert screen.query_one("#watchlists-backend-select", Select).disabled is True
        assert (
            str(screen.query_one("#watchlists-backend-label").renderable)
            == "Inbox: local"
        )
        assert screen.query_one("#notifications-local-ownership")

        pane.select_notification_by_id(str(inserted["id"]))
        await pilot.pause()
        app.notify = Mock()
        screen.action_delete_selected()
        app.notify.assert_called_once_with(
            "Use Dismiss to remove a notification from the inbox.",
            severity="information",
        )
        pane.query_one("#notifications-mark-read-button", Button).press()
        for _ in range(20):
            await pilot.pause()
            if app.client_notifications_db.get_notification(inserted["id"])["is_read"]:
                break

        assert app.client_notifications_db.get_notification(inserted["id"])["is_read"]


# --- Task 5: re-hosting the existing panes inside the collapsible workbench ---
#
# The file has no `watchlists_app` fixture (none existed before this task), so
# these reuse the same `DestinationHarness` + `_build_test_app()` pattern the
# rest of this file already uses, rather than inventing a second harness.


@pytest.mark.asyncio
async def test_existing_panes_survive_the_workbench_rehost():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.query("#wl-workbench"), "the workbench container should be mounted"
        # The panes that existed before must still be mounted, not replaced.
        # The section navigator was retired for a centre tab strip (Phase C,
        # task 3); #wl-tabs is its direct successor as "a working
        # section-switcher is mounted."
        assert screen.query("#wl-tabs")
        # Default active_section is "overview", so OverviewPane is what's there
        # to start; switch to Sources (as the tab-strip test does) to confirm
        # SourcesPane also still renders inside the re-hosted ITEMS region
        # rather than being dropped.
        assert screen.query("#watchlists-overview-pane")
        screen.query_one("#wl-tab-sources", Button).press()
        await pilot.pause()
        assert screen.query("#watchlists-sources-pane")


@pytest.mark.asyncio
async def test_bracket_keys_toggle_the_rails():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        await pilot.press("[")
        await pilot.pause()
        assert screen.region_layout.is_collapsed(Region.LEFT_RAIL)
        await pilot.press("]")
        await pilot.pause()
        assert screen.region_layout.is_collapsed(Region.RIGHT_RAIL)


@pytest.mark.asyncio
async def test_collapsing_a_region_persists(monkeypatch):
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: saved.append(layout),
    )
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        # task-1344 review, B1: region-layout gestures only apply on the
        # Read tab now -- an ITEMS toggle off Read (the default section is
        # "overview") is refused, not persisted.
        screen.active_section = "items"
        await pilot.pause(0.1)
        screen.focused_region = Region.ITEMS
        await pilot.press("z")
        await pilot.pause()
        assert screen.region_layout.is_collapsed(Region.ITEMS)
        assert saved
        assert Region.ITEMS in saved[-1].collapsed


@pytest.mark.asyncio
async def test_route_and_class_name_are_unchanged():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert type(screen).__name__ == "WatchlistsCollectionsScreen"
        # BaseAppScreen stores the route as `screen_name` (base_app_screen.py:23),
        # not `route_name` — the screen passes "watchlists_collections" to super().
        assert screen.screen_name == "watchlists_collections"


@pytest.mark.asyncio
async def test_focus_drives_which_region_z_collapses():
    """`z` is a lie unless focus tracking actually works: without
    `on_descendant_focus`, every `z` press collapses whatever `focused_region`
    defaults to, regardless of where the user actually is."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        # task-1344 review, B1: region-layout gestures only apply on the
        # Read tab -- the default section is "overview", so switch to Read
        # before exercising the toggle itself (focus-tracking is unaffected
        # by which tab is active and is exercised as-is above).
        screen.active_section = "items"
        await pilot.pause(0.1)
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
        assert screen.focused_region == Region.ITEMS
        await pilot.press("z")
        await pilot.pause()
        assert screen.region_layout.is_collapsed(Region.ITEMS)
        assert not screen.region_layout.is_collapsed(Region.FEEDS)


@pytest.mark.asyncio
async def test_persisted_layout_is_applied_on_mount(monkeypatch):
    """`on_mount` must push the loaded layout into the already-mounted
    workbench, not just this screen's own `region_layout` attribute — compose
    always runs before Mount, so the workbench was already built with the
    reactive's default value by the time `on_mount` fires."""
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.load_region_layout",
        lambda: RegionLayout(collapsed=frozenset({Region.RIGHT_RAIL})),
    )
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.region_layout.is_collapsed(Region.RIGHT_RAIL)
        assert screen.query("#wl-header-right_rail")
        assert not screen.query("#watchlists-inspector-pane")


# --- PR #926 review, Bug 2: `_apply_layout` used to call `save_region_layout`
# (a synchronous whole-file config read-modify-write) unconditionally and on
# the UI thread, including from `on_mount` when nothing had changed. The
# three tests below cover the fix's two halves: skip the write when the
# layout is unchanged from what is already persisted, and move any real
# write off the UI thread via `run_worker(thread=True)`.


@pytest.mark.asyncio
async def test_mounting_with_a_persisted_layout_performs_no_write(monkeypatch):
    """`on_mount` re-applies whatever `load_region_layout` just returned so
    the mounted workbench reflects it (see
    `test_persisted_layout_is_applied_on_mount` above) — but that layout is
    by definition already on disk, so doing so must not itself schedule a
    config write."""
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.load_region_layout",
        lambda: RegionLayout(collapsed=frozenset({Region.RIGHT_RAIL})),
    )
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: saved.append(layout),
    )
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert saved == []


@pytest.mark.asyncio
async def test_a_real_toggle_performs_exactly_one_write(monkeypatch):
    """A single genuine layout change schedules exactly one write, and it
    lands off the UI thread (a plain `run_worker(..., thread=True)` call
    completes here without the test itself spinning up any thread)."""
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: saved.append(layout),
    )
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        await host.workers.wait_for_complete()
        await pilot.pause()
        saved.clear()  # `on_mount`'s own (no-op) apply may have run above.

        screen = host.screen_stack[-1]
        # task-1344 review, B1: region-layout gestures only apply on the
        # Read tab -- the default section is "overview", so switch to Read
        # before the real toggle this test measures.
        screen.active_section = "items"
        await pilot.pause(0.1)
        await host.workers.wait_for_complete()
        await pilot.pause()
        saved.clear()  # the tab switch's own data-load workers, not a layout write.

        screen.focused_region = Region.ITEMS
        await pilot.press("z")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert len(saved) == 1
        assert Region.ITEMS in saved[0].collapsed


@pytest.mark.asyncio
async def test_a_burst_of_toggles_persists_only_the_final_state(monkeypatch):
    """A rapid burst of toggles must not interleave writes out of order —
    once every worker has drained, the persisted value must match the
    layout the screen actually ended up with, not some intermediate state
    from earlier in the burst."""
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: saved.append(layout),
    )
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        await host.workers.wait_for_complete()
        await pilot.pause()
        saved.clear()  # `on_mount`'s own (no-op) apply may have run above.

        screen = host.screen_stack[-1]
        screen.focused_region = Region.ITEMS
        # Fire off several toggles back-to-back with no pause in between, so
        # scheduling for all of them races ahead of any one write completing.
        await pilot.press("z")
        await pilot.press("[")
        await pilot.press("z")
        await pilot.press("]")
        await host.workers.wait_for_complete()
        await pilot.pause()

        final_layout = screen.region_layout
        assert saved, "a real change occurred, so at least one write must have happened"
        assert saved[-1].collapsed == final_layout.collapsed_for_persistence()


# --- Fix round 1, Finding 1: a bracket press must not destroy a half-typed
# create-source form. `region_layout` is `recompose=True`, so ANY region
# toggle — including one on a rail with nothing to do with Sources — rebuilds
# the whole workbench and constructs a fresh SourcesPane. The draft is lifted
# to screen state (`_source_create_draft`/`_source_create_form_open`) the
# same way selected_source/selected_run/active_section already survive pane
# rebuilds; see CreateFormDraftChanged/CreateFormVisibilityChanged in
# sources_pane.py.


@pytest.mark.asyncio
async def test_bracket_toggle_preserves_in_progress_create_form_draft():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.query_one("#wl-tab-sources", Button).press()
        await pilot.pause()
        screen.query_one("#sources-new-button", Button).press()
        await pilot.pause()
        assert screen.query("#sources-create-name"), "the create form should be open"

        # Matches the direct-assignment style Tests/Watchlists/
        # test_watchlists_sources_pane.py already uses to simulate typing:
        # Input.value is a reactive whose own watcher posts Input.Changed
        # regardless of whether the change came from a keystroke or a direct
        # attribute assignment, so this exercises the same code path a real
        # keystroke would.
        screen.query_one("#sources-create-name", Input).value = "Draft Name"
        await pilot.pause()
        screen.query_one("#sources-create-url", Input).value = "https://draft.example"
        await pilot.pause()

        name_before = screen.query_one("#sources-create-name", Input).value
        url_before = screen.query_one("#sources-create-url", Input).value
        assert name_before == "Draft Name"
        assert url_before == "https://draft.example"

        # TASK-1035: move focus off the text field before pressing `[`.
        # The create form now focuses its `Name` Input when it opens (it
        # previously opened with `Screen.focused` at `None`, which is why
        # nothing could be typed into it at all), and a focused Textual
        # `Input` consumes printable keys -- so `[` lands in the field as a
        # bracket instead of reaching the screen's region-collapse binding,
        # exactly as it already does in the Sources search box. Without this
        # the assertion below read `"Draft Name["`, i.e. the workbench was
        # never rebuilt and the test stopped exercising what it names.
        # `#wl-tab-sources` is a Button, so the key bubbles to the screen.
        screen.query_one("#wl-tab-sources", Button).focus()
        await pilot.pause()

        # Toggle a rail that has nothing to do with Sources. This rebuilds
        # the whole workbench, including the SourcesPane living in ITEMS.
        await pilot.press("[")
        await pilot.pause()

        assert screen.query(
            "#sources-create-name"
        ), "the create form must still be open after an unrelated toggle"
        name_after = screen.query_one("#sources-create-name", Input).value
        url_after = screen.query_one("#sources-create-url", Input).value
        assert name_after == "Draft Name", "typed Name text must survive the rebuild"
        assert url_after == "https://draft.example", "typed URL text must survive the rebuild"


@pytest.mark.asyncio
async def test_bracket_toggle_preserves_a_cleared_noise_selector_field():
    """TASK-1362, spec §2: emptiness must survive a workbench rebuild.

    The noise field is the one create-form field whose *untouched* state is
    not empty -- `SourcesPane` prefills it with the shipped selector set. So
    the rebuild the test above covers for Name/URL is strictly more dangerous
    here: a pane rebuilt without the draft does not merely lose what the user
    did, it silently restores the very rules they deleted, and the next
    `Create` stores them. That is the "re-filled behind their back" the spec
    forbids, arriving through a bracket press on an unrelated rail.

    Drives the real path end to end: the pane's `TextArea.Changed` ->
    `CreateFormDraftChanged` -> the screen's `_source_create_draft_selectors`
    -> `_build_detail_pane` seeding the brand new pane.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.query_one("#wl-tab-sources", Button).press()
        await pilot.pause()
        screen.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        field = screen.query_one("#sources-create-ignore-selectors", TextArea)
        assert field.text == default_ignore_selectors_text(), (
            "the field should open prefilled; this test is about clearing it"
        )
        # The real clearing edit -- what select-all-and-delete performs, and
        # what posts `TextArea.Changed`.
        field.clear()
        await pilot.pause()
        assert screen._source_create_draft_selectors == "", (
            "clearing the field must reach the screen's mirror; without that "
            "the rebuild below has nothing to seed from and falls back to the "
            f"default (mirror held {screen._source_create_draft_selectors!r})"
        )

        # Same reason as the test above: move focus off the text field so `[`
        # reaches the screen's region-collapse binding instead of being typed.
        screen.query_one("#wl-tab-sources", Button).focus()
        await pilot.pause()

        await pilot.press("[")
        await pilot.pause()

        assert screen.query("#sources-create-ignore-selectors"), (
            "the create form must still be open after an unrelated toggle"
        )
        rebuilt = screen.query_one("#sources-create-ignore-selectors", TextArea)
        assert rebuilt.text == "", (
            "the rebuilt form re-filled a field the user deliberately emptied "
            f"with {rebuilt.text!r}"
        )


@pytest.mark.asyncio
async def test_submitting_the_create_form_clears_the_draft():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.query_one("#wl-tab-sources", Button).press()
        await pilot.pause()
        screen.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        screen.query_one("#sources-create-name", Input).value = "Draft Name"
        await pilot.pause()
        screen.query_one("#sources-create-url", Input).value = "https://draft.example"
        await pilot.pause()

        screen.query_one("#sources-create-submit", Button).press()
        await pilot.pause()

        assert screen._source_create_draft == {"name": "", "url": "", "tags": ""}
        assert screen._source_create_form_open is False

        # Re-toggle a rail: the rebuilt pane must not resurrect the old draft.
        await pilot.press("[")
        await pilot.press("[")
        await pilot.pause()
        assert not screen.query(
            "#sources-create-name"
        ), "the form should stay closed, not reopen with stale text"


@pytest.mark.asyncio
async def test_cancelling_the_create_form_clears_the_draft():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.query_one("#wl-tab-sources", Button).press()
        await pilot.pause()
        screen.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        screen.query_one("#sources-create-name", Input).value = "Draft Name"
        await pilot.pause()

        screen.query_one("#sources-create-cancel", Button).press()
        await pilot.pause()

        assert screen._source_create_draft == {"name": "", "url": "", "tags": ""}
        assert screen._source_create_form_open is False


# --- Fix round 2 (final whole-branch review): Findings 2, 3, 4. `_build_
# detail_pane`/`_build_inspector_pane` construct a brand new pane on EVERY
# workbench rebuild, not just a section switch -- any region collapse/solo/
# rail toggle recomposes the whole `WatchlistsWorkbench` (`region_layout` is
# `recompose=True`). RunsPane/NotificationsPane/OverviewPane were already
# seeded from screen state; Sources/Items/Rules and the Inspector were not,
# and an in-progress Rules edit had no screen-state mirror at all (unlike the
# Sources create-form draft fixed in round 1).


async def _wait_for_table_rows(pilot, table_id: str, screen, expected: int) -> DataTable:
    table = screen.query_one(table_id, DataTable)
    for _ in range(30):
        if table.row_count >= expected:
            break
        await pilot.pause()
        table = screen.query_one(table_id, DataTable)
    return table


@pytest.mark.asyncio
async def test_bracket_toggle_preserves_loaded_sources_items_and_rules_tables():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen._controller.list_sources = AsyncMock(
            return_value=[{"id": "s1", "name": "Feed One", "source_type": "rss"}]
        )
        screen._controller.list_items = AsyncMock(
            return_value=[{"id": "i1", "title": "Item One", "source_name": "Feed One"}]
        )
        screen._controller.list_alert_rules = AsyncMock(
            return_value=[{"id": "r1", "name": "Rule One", "condition_type": "no_items"}]
        )

        for section, table_id in (
            ("sources", "#sources-table"),
            ("items", "#items-table"),
            ("rules", "#rules-table"),
        ):
            screen.query_one(f"#wl-tab-{section}", Button).press()
            await pilot.pause()

            table = await _wait_for_table_rows(pilot, table_id, screen, 1)
            assert table.row_count == 1, f"{section} table never loaded its one row"

            await pilot.press("[")
            await pilot.pause()

            table_after_toggle = screen.query_one(table_id, DataTable)
            assert table_after_toggle.row_count == 1, (
                f"{section} table was emptied by an unrelated left-rail toggle"
            )

            # Re-expand so the next section's nav button (in the left rail)
            # is reachable again.
            await pilot.press("[")
            await pilot.pause()


@pytest.mark.asyncio
async def test_bracket_toggle_preserves_inspector_selection():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen._controller.list_sources = AsyncMock(
            return_value=[{"id": "s1", "name": "Feed One", "source_type": "rss"}]
        )

        screen.query_one("#wl-tab-sources", Button).press()
        await pilot.pause()
        await _wait_for_table_rows(pilot, "#sources-table", screen, 1)

        sources_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        sources_pane.select_source_by_id("s1")
        await pilot.pause()

        assert screen.selected_entity is not None
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert inspector.selected_entity == screen.selected_entity

        await pilot.press("[")
        await pilot.pause()

        rebuilt_inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert rebuilt_inspector is not inspector, "the inspector should have been rebuilt"
        assert rebuilt_inspector.selected_entity == screen.selected_entity, (
            "the rebuilt inspector lost the screen's selection"
        )
        # The empty state ("Select a source...") must NOT be showing, since a
        # real selection is still in effect after the rebuild.
        assert not rebuilt_inspector.query("#inspector-empty-state")


@pytest.mark.asyncio
async def test_bracket_toggle_preserves_in_progress_rule_edit():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        rule = {
            "id": "r1",
            "name": "Rule One",
            "condition_type": "no_items",
            "severity": "warning",
            "enabled": True,
        }
        screen._controller.list_alert_rules = AsyncMock(return_value=[rule])

        screen.query_one("#wl-tab-rules", Button).press()
        await pilot.pause()
        await _wait_for_table_rows(pilot, "#rules-table", screen, 1)

        rules_pane = screen.query_one("#watchlists-rules-pane", RulesPane)
        rules_pane.edit_rule(rule)
        await pilot.pause()

        assert screen.query("#rules-create-name"), "the edit form should be open"
        assert screen._rule_form_open is True
        assert screen._rule_form_editing == rule

        # Toggle a rail that has nothing to do with Rules. This rebuilds the
        # whole workbench, including the RulesPane living in ITEMS.
        await pilot.press("[")
        await pilot.pause()

        assert screen.query(
            "#rules-create-name"
        ), "the rule edit form must still be open after an unrelated toggle"
        name_input = screen.query_one("#rules-create-name", Input)
        assert name_input.value == "Rule One", (
            "the form must still be pre-filled for the SAME rule being edited"
        )


@pytest.mark.asyncio
async def test_saving_a_rule_edit_does_not_leave_a_phantom_form_open():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        rule = {
            "id": "r1",
            "name": "Rule One",
            "condition_type": "no_items",
            "severity": "warning",
            "enabled": True,
        }
        screen._controller.list_alert_rules = AsyncMock(return_value=[rule])
        # A fast-completing mocked save is exactly the case that lets the
        # screen's overview-data refresh win the race against RulesPane's own
        # RuleFormVisibilityChanged message still bubbling up to the screen
        # (see `handle_save_rule_requested`).
        screen._controller.save_alert_rule = AsyncMock(return_value=dict(rule))
        # A distinct payload on every call, so `overview_data` really changes
        # value and its watcher really fires: the real `get_overview_data()`
        # return is otherwise byte-for-byte identical before and after this
        # mocked save (nothing in the backing store actually changed), which
        # would mask the interleaving this test targets. TASK-2200 took
        # `recompose=True` off that reactive -- so this now also pins that the
        # background refresh no longer rebuilds the pane at all (see the
        # identity assertion below), which is the whole point of that change.
        overview_call_count = itertools.count(1)

        async def _fake_overview_data(**_kwargs: Any) -> dict[str, Any]:
            return {
                "total_sources": 0,
                "active_sources": 0,
                "sources_in_error": 0,
                "total_items": 0,
                "new_items": 0,
                "latest_run_status": "unavailable",
                "failed_runs": [],
                "active_alert_rules": next(overview_call_count),
            }

        screen._controller.get_overview_data = _fake_overview_data

        screen.query_one("#wl-tab-rules", Button).press()
        await pilot.pause()
        await _wait_for_table_rows(pilot, "#rules-table", screen, 1)

        rules_pane = screen.query_one("#watchlists-rules-pane", RulesPane)
        rules_pane.edit_rule(rule)
        await pilot.pause()

        assert screen.query("#rules-create-name"), "the edit form should be open"

        screen.query_one("#rules-create-submit", Button).press()

        # Drive enough ticks for the save worker to finish, its overview-data
        # refresh to land, and the pane's own form-close to settle.
        settled_rules_pane = rules_pane
        for _ in range(30):
            await asyncio.sleep(0.02)
            await pilot.pause()
            settled_rules_pane = screen.query_one("#watchlists-rules-pane", RulesPane)
            if not settled_rules_pane.show_rule_form and not screen.query(
                "#rules-create-name"
            ):
                break

        assert screen._controller.save_alert_rule.await_count == 1, (
            "the precondition: pressing Save really did reach the controller"
        )
        assert next(overview_call_count) > 1, (
            "the precondition: the save's `_refresh_overview_data()` really "
            "did run, so `overview_data` really did change value"
        )
        # TASK-2200: the background overview refresh must NOT rebuild the
        # pane. This used to assert the opposite (`is not rules_pane`) and
        # relied on that rebuild to close the form; a pane rebuilt out from
        # under an in-flight pane recompose is the destroyer TASK-1960
        # confirmed.
        assert settled_rules_pane is rules_pane, (
            "a background overview refresh must not replace the mounted pane"
        )
        assert settled_rules_pane.show_rule_form is False, (
            "the rule edit form must be closed after a successful save, not "
            "reopened pre-filled with the just-submitted rule"
        )
        assert settled_rules_pane.query("#rules-table"), (
            "the pane must still have its table -- a form that 'closed' by "
            "leaving an empty pane behind is the masked defect, not a fix"
        )
        assert not screen.query("#rules-create-name"), (
            "no rule edit form fields should remain in the DOM after a "
            "successful save"
        )


# --- Task 4: wire the watchlist tree and tab strip into the screen ---
#
# As with the Task 5 section above, there is no `watchlists_app` fixture in
# this file; the brief's snippets assumed one, but none existed before this
# task either. These reuse the same `DestinationHarness` + `_build_test_app()`
# pattern already established here.


@pytest.mark.asyncio
async def test_left_rail_hosts_the_tree_not_the_navigator():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.query("#wl-tree")
        assert not screen.query("#watchlists-navigator")


@pytest.mark.asyncio
async def test_centre_hosts_the_tab_strip():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        assert host.screen_stack[-1].query("#wl-tabs")


@pytest.mark.asyncio
async def test_clicking_a_tab_switches_the_active_section():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        await pilot.click("#wl-tab-runs")
        await pilot.pause()
        assert screen.active_section == "runs"


@pytest.mark.asyncio
async def test_tree_selection_sets_the_screen_scope():
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
        TreeScope,
        TreeScopeChanged,
    )

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=7)))
        await pilot.pause()
        assert screen.selected_scope.kind == "watchlist"
        assert screen.selected_scope.watchlist_id == 7


@pytest.mark.asyncio
async def test_scope_survives_a_region_toggle():
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
        TreeScope,
        TreeScopeChanged,
    )

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=7)))
        await pilot.pause()
        await pilot.press("[")
        await pilot.pause()
        assert screen.selected_scope.watchlist_id == 7, (
            "scope lives on the screen, so a workbench recompose must not lose it"
        )


@pytest.mark.asyncio
async def test_feeds_region_follows_the_tree_scope():
    """Task 7: narrowing the tree scope must change what Feeds covers.

    Uses `_build_test_app()` + `DestinationHarness`, this file's own
    established pattern (see every other test above) rather than a
    `watchlists_app` fixture -- no such fixture exists in this file.
    """
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged

    app = _build_test_app()
    # Seeded (fix round 1): against this harness's empty subscriptions DB
    # the assertion below passes through its own `or all_rows == []` escape
    # hatch, so it also passed against a `scoped_source_rows` stubbed to
    # `return []`. `_build_test_app()` wires the bundle service to an
    # isolated temp-dir SQLite file, never the developer's database.
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    assert watchlist["id"] == 1, "a fresh temp DB numbers the first watchlist 1"
    arxiv = service._db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/f"
    )
    loose = service._db.add_subscription(
        name="Loose", type="rss", source="https://c.example/f"
    )
    service.add_source(watchlist["id"], arxiv)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.post_message(TreeScopeChanged(TreeScope(kind="all")))
        await pilot.pause()
        all_rows = screen.scoped_source_rows()

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1))
        )
        await pilot.pause()
        scoped_rows = screen.scoped_source_rows()

        assert scoped_rows != all_rows or all_rows == [], (
            "narrowing the scope to one watchlist must change what Feeds covers"
        )
        assert {row["id"] for row in all_rows} == {arxiv, loose}
        assert [row["id"] for row in scoped_rows] == [arxiv]
        assert all_rows != [], (
            "the escape hatch above must not be what is carrying this test"
        )


@pytest.mark.asyncio
async def test_source_scope_narrows_to_exactly_one():
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged

    app = _build_test_app()
    # Seeded (fix round 1): `len(rows) <= 1` is trivially true against an
    # empty DB. Ten sources make the id-10 narrowing real, with nine
    # siblings in the same watchlist that must NOT come back.
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    assert watchlist["id"] == 1
    for index in range(10):
        source_id = service._db.add_subscription(
            name=f"source-{index:02d}",
            type="rss",
            source=f"https://feed-{index:02d}.example/rss",
        )
        service.add_source(watchlist["id"], source_id)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.post_message(
            TreeScopeChanged(TreeScope(kind="source", watchlist_id=1, source_id=10))
        )
        await pilot.pause()
        rows = screen.scoped_source_rows()
        assert len(rows) <= 1
        assert all(int(r["id"]) == 10 for r in rows)
        assert len(rows) == 1, (
            "the source exists, so `<= 1` must be carried by a real row"
        )
        assert rows[0]["name"] == "source-09"
        assert len(screen._watchlist_bundle_service().list_source_rows(1)) == 10, (
            "...and its nine siblings are present to be narrowed away"
        )


# --- Whole-branch review, Finding 1 + Finding 4: a tree move must reach the
# panes' own selection state, not only the screen's mirrors of it ---
#
# Every pre-existing mirror-clear test writes `screen.selected_*` directly and
# never drives a pane into a selected state, which is exactly why the pane-side
# resurrection below shipped. These drive the pane.


def _running_run_row() -> dict[str, Any]:
    return {
        "id": "run-1",
        "source_title": "ArXiv",
        "source_id": 1,
        "status": "running",
        "started_at": "2026-07-26T09:00:00Z",
    }


@pytest.mark.asyncio
async def test_a_running_run_poll_cannot_resurrect_a_cleared_tree_scope():
    """`RunsPane.run_poll` re-posts `RunSelected` once a second for 60 ticks.

    If a tree move clears only the screen's `selected_run` mirror and leaves
    the pane's own copy standing, the very next poll tick re-posts the
    pre-move run, `_select_entity` snaps `selected_scope` back to "all" and
    empties `_breadcrumb_labels` -- with no user action, about a second after
    the user navigated somewhere else.
    """
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
        TreeScope,
        TreeScopeChanged,
    )

    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    assert watchlist["id"] == 1, "a fresh temp DB numbers the first watchlist 1"

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.active_section = "runs"
        await pilot.pause(0.2)

        runs_pane = screen.query_one("#watchlists-runs-pane", RunsPane)
        runs_pane.runs = [_running_run_row()]
        await pilot.pause()
        # Drive the PANE, not `screen.selected_run` -- the pane holding its own
        # copy is the whole point.
        runs_pane.select_run_by_id("run-1")
        await pilot.pause()

        assert runs_pane.selected_run is not None
        assert screen.selected_run is not None, (
            "the pane's selection must have reached the screen"
        )
        assert any(
            worker.node is runs_pane and worker.name == "run_poll"
            for worker in host.workers
        ), "the running-run poll must be live, or this test proves nothing"

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1))
        )
        await pilot.pause()

        assert screen.selected_run is None
        assert screen.selected_entity is None
        assert screen.selected_scope.kind == "watchlist"
        assert screen._breadcrumb_labels == ["Morning AI Brief"]
        assert runs_pane.selected_run is None, (
            "the tree move must reach the pane's own copy, not only the mirror"
        )

        # One full poll interval, plus slack. No user action in between.
        await pilot.pause(1.4)

        assert screen.selected_run is None, (
            "a poll tick resurrected a run the tree navigated away from"
        )
        assert screen.selected_entity is None
        assert screen.selected_scope.kind == "watchlist"
        assert screen.selected_scope.watchlist_id == 1
        assert screen.tree_scope.kind == "watchlist"
        assert screen._breadcrumb_labels == ["Morning AI Brief"], (
            "the Inspector's breadcrumb must survive the poll interval"
        )


@pytest.mark.asyncio
async def test_moving_the_tree_disarms_run_actions_selected_before_the_move():
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
        TreeScope,
        TreeScopeChanged,
    )

    app = _build_test_app()
    app.watchlist_bundle_service.create("Morning AI Brief")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.active_section = "runs"
        await pilot.pause(0.2)

        runs_pane = screen.query_one("#watchlists-runs-pane", RunsPane)
        runs_pane.runs = [_running_run_row()]
        await pilot.pause()
        runs_pane.select_run_by_id("run-1")
        await pilot.pause()

        cancel = screen.query_one("#runs-cancel-button", Button)
        rerun = screen.query_one("#runs-rerun-button", Button)
        assert not cancel.disabled, "precondition: a running run arms Cancel"
        assert not rerun.disabled, "precondition: a selected run arms Re-run"

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1))
        )
        await pilot.pause()

        cancel = screen.query_one("#runs-cancel-button", Button)
        rerun = screen.query_one("#runs-rerun-button", Button)
        assert cancel.disabled, (
            "Cancel must not stay armed on a run the tree navigated away from"
        )
        assert rerun.disabled


@pytest.mark.asyncio
async def test_moving_the_tree_disarms_source_actions_selected_before_the_move():
    """`SourcesPane`'s own Preview/Check-now post against `self.selected_source`.

    Left standing after a tree move, they act on a source the screen believes
    is deselected.
    """
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
        TreeScope,
        TreeScopeChanged,
    )

    app = _build_test_app()
    app.watchlist_bundle_service.create("Morning AI Brief")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause(0.2)

        sources_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        sources_pane.sources = [
            {"id": 1, "name": "ArXiv", "source_type": "rss", "active": True}
        ]
        await pilot.pause()
        sources_pane.select_source_by_id("1")
        await pilot.pause()

        assert screen.selected_source is not None
        preview = screen.query_one("#sources-preview-button", Button)
        check_now = screen.query_one("#sources-check-now-button", Button)
        assert not preview.disabled, "precondition: a selected source arms Preview"
        assert not check_now.disabled

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1))
        )
        await pilot.pause()

        assert sources_pane.selected_source is None, (
            "the tree move must reach the pane's own copy, not only the mirror"
        )
        preview = screen.query_one("#sources-preview-button", Button)
        check_now = screen.query_one("#sources-check-now-button", Button)
        assert preview.disabled, (
            "Preview must not stay armed on a source the screen deselected"
        )
        assert check_now.disabled


@pytest.mark.asyncio
async def test_moving_the_tree_clears_a_notification_selected_in_its_pane():
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
        TreeScope,
        TreeScopeChanged,
    )

    app = _build_test_app()
    app.watchlist_bundle_service.create("Morning AI Brief")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.active_section = "notifications"
        await pilot.pause(0.2)

        pane = screen.query_one(
            "#watchlists-notifications-pane", NotificationsPane
        )
        pane.notifications = [
            {"id": 1, "title": "Feed failed", "message": "boom", "is_read": False}
        ]
        await pilot.pause()
        pane.select_notification_by_id("1")
        await pilot.pause()
        assert screen.selected_notification is not None

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1))
        )
        await pilot.pause()

        pane = screen.query_one(
            "#watchlists-notifications-pane", NotificationsPane
        )
        assert pane.selected_notification is None, (
            "the tree move must reach the pane's own copy, not only the mirror"
        )
        assert screen.selected_notification is None


@pytest.mark.asyncio
async def test_clearing_pane_selections_degrades_quietly_when_unmounted():
    """The panes are only mounted for their own section, and the workbench
    recomposes; `_apply_tree_scope` must not depend on any of them existing.
    """
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope

    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    # Never mounted: every pane query raises.
    screen._apply_tree_scope(TreeScope(kind="watchlist", watchlist_id=3))
    assert screen.tree_scope.watchlist_id == 3
    assert screen.selected_entity is None


# --- Whole-branch review, Finding 2: tree expansion and tag filter are screen
# state, because every section switch fully recomposes the workbench ---


@pytest.mark.asyncio
async def test_tree_expansion_survives_a_section_switch():
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
        TreeScope,
        TreeScopeChanged,
        WatchlistTree,
    )

    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    assert watchlist["id"] == 1
    arxiv = service._db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/f"
    )
    service.add_source(watchlist["id"], arxiv)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.query_one("#wl-tree-expand-1", Button).press()
        await pilot.pause()
        source_node = f"#wl-tree-node-source-1-{arxiv}"
        assert screen.query(source_node), "precondition: the watchlist expanded"

        screen.post_message(
            TreeScopeChanged(
                TreeScope(kind="source", watchlist_id=1, source_id=arxiv)
            )
        )
        await pilot.pause()

        screen.query_one("#wl-tab-runs", Button).press()
        await pilot.pause(0.2)

        assert screen.active_section == "runs"
        assert screen.query(source_node), (
            "a section switch recomposes the workbench; the node the centre "
            "is scoped to must still be in the rail"
        )
        tree = screen.query_one("#wl-tree", WatchlistTree)
        assert tree.expanded == frozenset({1})


@pytest.mark.asyncio
async def test_tree_tag_filter_survives_a_section_switch():
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import WatchlistTree

    app = _build_test_app()
    service = app.watchlist_bundle_service
    service.create("Morning AI Brief", tags=["ai"])
    service.create("Weekend Reads", tags=["fun"])

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.query_one("#wl-tree-tag-0", Button).press()
        await pilot.pause()
        tree = screen.query_one("#wl-tree", WatchlistTree)
        assert tree.active_tag == "ai", "precondition: the tag filter applied"
        assert screen.query("#wl-tree-node-watchlist-1")
        assert not screen.query("#wl-tree-node-watchlist-2")

        screen.query_one("#wl-tab-runs", Button).press()
        await pilot.pause(0.2)

        tree = screen.query_one("#wl-tree", WatchlistTree)
        assert tree.active_tag == "ai", (
            "a tag filter set in the rail must not be dropped by a section switch"
        )
        assert not screen.query("#wl-tree-node-watchlist-2")


@pytest.mark.asyncio
async def test_tree_expansion_survives_a_tree_data_reload():
    """`_load_tree_data` and `_apply_local_wc_snapshot` both full-recompose."""
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import WatchlistTree

    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    arxiv = service._db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/f"
    )
    service.add_source(watchlist["id"], arxiv)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.query_one("#wl-tree-expand-1", Button).press()
        await pilot.pause()
        assert screen.query(f"#wl-tree-node-source-1-{arxiv}")

        screen._load_tree_data()
        await pilot.pause(0.2)

        assert screen.query(f"#wl-tree-node-source-1-{arxiv}"), (
            "reloading tree data must not collapse what the user expanded"
        )
        assert screen.query_one("#wl-tree", WatchlistTree).expanded == frozenset({1})


@pytest.mark.asyncio
async def test_seeded_tree_expansion_takes_effect_on_the_first_render():
    """Seeding a `recompose=True` reactive must not cost a second compose.

    `WatchlistTree.expanded` recomposes on write, so seeding it by plain
    assignment would render the collapsed tree first and rebuild it a tick
    later. The constructor uses `set_reactive` precisely to avoid that; this
    pins the behaviour rather than assuming it.
    """
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import WatchlistTree

    composes: list[int] = []

    class CountingTree(WatchlistTree):
        def compose(self):
            composes.append(1)
            yield from super().compose()

    app = _build_test_app()
    tree = CountingTree(
        watchlists=[{"id": 1, "name": "Morning AI Brief", "tags": ["ai"]}],
        counts={1: {"unread": 2}},
        source_rows_loader=lambda _wl: [{"id": 5, "name": "ArXiv"}],
        expanded=frozenset({1}),
        active_tag="ai",
        id="wl-tree",
    )

    with patch(
        "tldw_chatbook.app.get_cli_setting",
        side_effect=_settings_without_splash,
    ):
        async with app.run_test(size=(60, 30)) as pilot:
            await app.screen.mount(tree)
            await pilot.pause()
            assert tree.expanded == frozenset({1})
            assert tree.active_tag == "ai"
            assert tree.query("#wl-tree-node-source-1-5"), (
                "the seeded expansion must be visible on the first render"
            )
            assert composes == [1], (
                f"seeding queued an extra recompose ({len(composes)} composes)"
            )


# --- TASK-1344: FEEDS gated to the Read tab, like CONTENT (AC#1); solo/toggle
# refused on any region hidden on the active tab (AC#2); no sequence of tab
# switches and region gestures may leave the centre with nothing expanded
# (AC#3). Mirrors the CONTENT-only tests Task 4 added in
# `Tests/UI/test_watchlists_content_pane.py`, generalized to cover FEEDS too.


@pytest.mark.asyncio
async def test_feeds_region_is_gated_to_the_items_read_tab():
    """AC#1: FEEDS occupies the centre only on the Read tab, matching the
    CONTENT gating Task 4 added (see
    `test_content_region_is_gated_to_the_items_read_tab` in
    `test_watchlists_content_pane.py`).

    AC#4: gated regions UNMOUNT rather than keep a one-row header, so "every
    other tab" means FEEDS has no DOM presence at all there -- no
    `#wl-header-feeds`, not just no `#wl-region-feeds`.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        # Default section is "overview" -- not Read.
        assert not screen.region_layout.is_collapsed(Region.FEEDS)
        assert not screen.query("#wl-header-feeds")
        assert not screen.query("#wl-region-feeds")

        screen.active_section = "items"
        await pilot.pause(0.2)
        assert screen.query("#wl-region-feeds")
        assert not screen.query("#wl-header-feeds")

        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert not screen.query("#wl-header-feeds")
        assert not screen.query("#wl-region-feeds")


@pytest.mark.asyncio
async def test_the_feeds_toggle_off_the_read_tab_neither_collapses_nor_persists():
    """AC#1/#2, FEEDS's half of
    `test_the_content_chevron_off_the_read_tab_neither_collapses_nor_persists`
    (`test_watchlists_content_pane.py`).

    FEEDS is unmounted off the Read tab, so there is no chevron to click --
    but a stale `focused_region` (left over from a prior visit to Read) can
    still be pointed at FEEDS when `z`/`Z` fires, and that must be refused
    rather than silently flipping the user's real preference and persisting
    it (dead forever, honoured on every future Read visit).
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause(0.3)
        assert not screen.query("#wl-header-feeds")
        assert not screen.query("#wl-region-feeds")
        assert not screen.region_layout.is_collapsed(Region.FEEDS), (
            "the real preference is still expanded -- the precondition"
        )

        screen.focused_region = Region.FEEDS
        screen.action_toggle_region()
        await pilot.pause(0.3)

        assert not screen.region_layout.is_collapsed(Region.FEEDS), (
            "the gesture must be refused, not run against the real preference"
        )
        assert Region.FEEDS not in (screen._last_persisted_collapsed or frozenset()), (
            "and it must never reach the persisted collapse set"
        )

        screen.action_solo_region()
        await pilot.pause(0.3)
        assert screen.region_layout.solo_region is None
        assert not screen.region_layout.is_collapsed(Region.FEEDS)

        # And back on Read, FEEDS is still there, untouched.
        screen.active_section = "items"
        await pilot.pause(0.3)
        assert screen.query("#wl-region-feeds")


@pytest.mark.asyncio
async def test_solo_on_feeds_off_the_read_tab_is_refused():
    """AC#2, FEEDS's half of `test_solo_on_content_off_the_read_tab_is_refused`
    (`test_watchlists_content_pane.py`): the single `_refuse_region_gesture_
    off_read_tab` source of truth must cover FEEDS exactly as it covers
    CONTENT, not special-case CONTENT alone.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "runs"
        await pilot.pause(0.3)
        before = screen.region_layout

        screen.focused_region = Region.FEEDS
        screen.action_solo_region()
        await pilot.pause(0.3)

        assert screen.region_layout is before or screen.region_layout == before, (
            "solo on the gated FEEDS region must not touch the real layout"
        )
        assert screen.region_layout.solo_region is None
        assert screen.query("#wl-region-items"), (
            "and the centre must still have something in it"
        )


@pytest.mark.asyncio
async def test_the_items_toggle_off_the_read_tab_neither_collapses_nor_persists():
    """task-1344 whole-branch review, B1 -- ITEMS's half of
    `test_the_feeds_toggle_off_the_read_tab_neither_collapses_nor_persists`,
    the one leg the original AC#3/#4 work never covered.

    Unlike FEEDS/CONTENT, ITEMS is force-shown off the Read tab
    (`_rendered_region_layout`) -- it is the section's own full-width pane,
    never a member of `_hidden_centre_regions()`. But a stale
    `focused_region == ITEMS` (set by `on_descendant_focus` any time the
    user's focus lands inside that pane, e.g. simply using Sources) let
    `z`/`Z` reach `_apply_layout(region_layout.toggle(ITEMS))` against the
    REAL, persisted layout with zero visible feedback -- the render already
    forces ITEMS back out of `collapsed`, so the collapse only bit the next
    time the user visited Read, at which point it (and FEEDS/CONTENT, if
    already collapsed there) could leave the centre with nothing expanded
    at all. Must be refused exactly like FEEDS/CONTENT, with copy that is
    actually true for ITEMS (it IS shown off Read, just not collapsible
    from here).
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause(0.3)
        assert screen.query("#wl-region-items"), (
            "unlike FEEDS/CONTENT, ITEMS IS rendered off Read -- the "
            "section's own full-width pane"
        )
        assert not screen.region_layout.is_collapsed(Region.ITEMS), (
            "the real preference is still expanded -- the precondition"
        )

        screen.notify = Mock()
        screen.focused_region = Region.ITEMS
        screen.action_toggle_region()
        await pilot.pause(0.3)

        assert not screen.region_layout.is_collapsed(Region.ITEMS), (
            "the gesture must be refused, not run against the real preference"
        )
        assert Region.ITEMS not in (screen._last_persisted_collapsed or frozenset()), (
            "and it must never reach the persisted collapse set"
        )
        screen.notify.assert_called_once()
        message = screen.notify.call_args.args[0]
        assert "only shown on the Read tab" not in message, (
            "ITEMS IS shown off Read -- claiming otherwise would be false"
        )
        assert screen.notify.call_args.kwargs.get("markup") is False

        # And back on Read, ITEMS is still there, untouched.
        screen.active_section = "items"
        await pilot.pause(0.3)
        assert screen.query("#wl-region-items")
        assert not screen.region_layout.is_collapsed(Region.ITEMS)


@pytest.mark.asyncio
async def test_solo_on_items_off_the_read_tab_is_refused():
    """AC#2, ITEMS's half of `test_solo_on_feeds_off_the_read_tab_is_refused`
    (task-1344 whole-branch review, B1): the same generalized
    `_refuse_region_gesture_off_read_tab` must refuse ITEMS's solo gesture
    off Read too -- ITEMS is not hidden there, but region-layout gestures
    still do not apply to it outside the Read tab.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "runs"
        await pilot.pause(0.3)
        before = screen.region_layout

        screen.notify = Mock()
        screen.focused_region = Region.ITEMS
        screen.action_solo_region()
        await pilot.pause(0.3)

        assert screen.region_layout is before or screen.region_layout == before, (
            "solo on ITEMS off Read must not touch the real layout"
        )
        assert screen.region_layout.solo_region is None
        assert screen.query("#wl-region-items"), (
            "and the centre must still have something in it"
        )
        screen.notify.assert_called_once()
        message = screen.notify.call_args.args[0]
        assert "only shown on the Read tab" not in message, (
            "ITEMS IS shown off Read -- claiming otherwise would be false"
        )


@pytest.mark.asyncio
async def test_no_sequence_of_tab_switches_and_region_gestures_leaves_the_centre_empty():
    """AC#3: no sequence of tab switches and region toggles/solos may leave
    the workbench with zero expanded centre regions -- recoverable only by
    clicking a header the user has no reason to suspect (PR #1091 review,
    F2's original report, now widened past the single CONTENT-solo path
    Task 4 fixed: FEEDS is gated the same way, and both FEEDS's and
    CONTENT's solo/toggle gestures are refused off the Read tab).

    Drives real gestures (tab switches, `z`, `Z`, `[`) through the full
    production shell (`DestinationHarness`, the same harness every other
    test in this file uses), asserting after EACH one that at least one
    centre region is genuinely mounted and expanded -- not merely that the
    real `region_layout` looks fine, which is exactly the gap a purely
    layout-level (non-DOM) assertion could miss.
    """

    def _any_centre_region_expanded(screen) -> bool:
        return bool(
            screen.query("#wl-region-feeds")
            or screen.query("#wl-region-items")
            or screen.query("#wl-region-content")
        )

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        async def step(label: str) -> None:
            await pilot.pause(0.2)
            assert _any_centre_region_expanded(screen), (
                f"the centre has nothing expanded after {label!r}: "
                f"active_section={screen.active_section!r} "
                f"region_layout={screen.region_layout!r}"
            )

        await step("mount (Overview, default)")

        screen.active_section = "items"
        await step("switch to Read")

        # The specific reported path: solo CONTENT on Read, then leave --
        # now refused on return, but must never have emptied the centre
        # even before this fix's refusal existed (Task 4's own regression).
        screen.focused_region = Region.CONTENT
        screen.action_solo_region()
        await step("solo CONTENT on Read")

        screen.active_section = "sources"
        await step("leave Read with CONTENT soloed")

        # A stale `focused_region` still pointed at a hidden region: both
        # gestures must be refused, not just one.
        screen.action_toggle_region()
        await step("toggle refused on Sources (focused_region=CONTENT)")
        screen.action_solo_region()
        await step("solo refused on Sources (focused_region=CONTENT)")

        screen.active_section = "items"
        await step("back to Read (CONTENT solo must have survived)")

        # Un-solo, then manually collapse ITEMS itself on Read -- this is
        # the OTHER route `_rendered_region_layout` has to guard (a manual
        # `z` on ITEMS while soloing CONTENT, or a plain ITEMS collapse,
        # both leave `region_layout.collapsed` containing ITEMS).
        screen.focused_region = Region.CONTENT
        screen.action_solo_region()
        await step("un-solo CONTENT on Read")

        screen.focused_region = Region.ITEMS
        screen.action_toggle_region()
        await step("collapse ITEMS on Read")

        screen.active_section = "runs"
        await step("switch to Runs with ITEMS collapsed on Read")

        # Rail toggles are orthogonal to the centre and must not interact.
        await pilot.press("[")
        await step("collapse the left rail")
        await pilot.press("]")
        await step("expand the left rail")

        screen.active_section = "items"
        await step("back to Read with ITEMS still collapsed from before")


@pytest.mark.asyncio
async def test_off_read_items_toggle_never_empties_the_read_centre_or_persists():
    """AC#3 gap the whole-branch review found (task-1344 review, B1): the
    sequence test above never drove the ONE path that actually breaks
    AC#3 -- collapse FEEDS and CONTENT on Read (both legitimate,
    persistable states), leave for a non-Read tab where ITEMS is
    force-shown, then `z` with `focused_region == ITEMS` (set by
    `on_descendant_focus` for anything inside the section pane, so simply
    using Sources/Runs/... sets it up).

    Before this fix that toggle was ACCEPTED: `region_layout.toggle(ITEMS)`
    mutated and PERSISTED the real layout to `{feeds, items, content}` with
    no visible change on the current tab (ITEMS is forced back out of
    `collapsed` for the render), so returning to Read rendered three
    headers over an empty centre -- on disk, surviving a restart.
    """

    def _any_centre_region_expanded(screen) -> bool:
        return bool(
            screen.query("#wl-region-feeds")
            or screen.query("#wl-region-items")
            or screen.query("#wl-region-content")
        )

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "items"
        await pilot.pause(0.3)

        # Collapse FEEDS and CONTENT on Read -- the report's exact
        # precondition, both legitimate and persistable on their own.
        screen.focused_region = Region.FEEDS
        screen.action_toggle_region()
        await pilot.pause(0.2)
        screen.focused_region = Region.CONTENT
        screen.action_toggle_region()
        await pilot.pause(0.2)

        assert screen.region_layout.is_collapsed(Region.FEEDS)
        assert screen.region_layout.is_collapsed(Region.CONTENT)
        assert not screen.region_layout.is_collapsed(Region.ITEMS)
        collapsed_before = screen.region_layout.collapsed
        persisted_before = screen._last_persisted_collapsed

        screen.active_section = "sources"
        await pilot.pause(0.3)
        assert _any_centre_region_expanded(screen), (
            "ITEMS is force-shown off Read even with FEEDS/CONTENT collapsed"
        )

        screen.notify = Mock()
        screen.focused_region = Region.ITEMS
        screen.action_toggle_region()
        await pilot.pause(0.3)

        assert screen.region_layout.collapsed == collapsed_before, (
            "an off-Read ITEMS toggle must be refused, not mutate the real, "
            "persisted layout"
        )
        assert screen._last_persisted_collapsed == persisted_before, (
            "and it must never reach the persisted collapse set"
        )
        screen.notify.assert_called_once()

        screen.notify.reset_mock()
        screen.action_solo_region()
        await pilot.pause(0.3)
        assert screen.region_layout.collapsed == collapsed_before, (
            "solo on ITEMS off Read must be refused too"
        )
        assert screen._last_persisted_collapsed == persisted_before
        screen.notify.assert_called_once()

        screen.active_section = "items"
        await pilot.pause(0.3)
        assert _any_centre_region_expanded(screen), (
            "returning to Read must not land on a dead end -- ITEMS was "
            "never actually collapsed by the refused off-Read gesture"
        )
        assert screen.query("#wl-region-items"), (
            "ITEMS specifically must still be the expanded centre region"
        )


# --- task-1344 fix wave (Qodo correctness): `z`/`Z` while focus is in the --
# centre header/tab strip -----------------------------------------------
#
# `#wl-centre-status`/`#wl-tabs` (`_build_centre_status_header`) are mounted
# directly under `#wl-centre`, outside every `wl-region-*`/`wl-header-*`
# wrapper -- so `on_descendant_focus` never updates `focused_region` while
# focus sits there, leaving it naming whatever region the user last actually
# visited.


@pytest.mark.asyncio
async def test_z_with_focus_in_the_centre_header_does_not_toggle_a_stale_region():
    """Before this fix, a user who last focused the left rail (setting
    `focused_region = LEFT_RAIL`), then tabbed into the tab strip and
    pressed `z`, collapsed -- and PERSISTED -- the rail anyway:
    `_refuse_region_gesture_off_read_tab` only gates `CENTRE_REGIONS`, so a
    rail's toggle was never refused there regardless of where real focus
    was. This is the one gesture the header-focus guard actually prevents
    from mutating anything (unlike solo below, whose CENTRE-region path was
    already refused by the existing off-Read gate regardless of focus).
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        # Default section is "overview" -- not Read, so the header exists
        # (`_build_centre_status_header`) and FEEDS/CONTENT are hidden.
        screen.query_one("#wl-region-left_rail").focus()
        await pilot.pause()
        assert screen.focused_region == Region.LEFT_RAIL, (
            "precondition: focused_region names a REAL prior focus"
        )
        assert not screen._focus_in_centre_header
        before = screen.region_layout

        screen.query_one("#wl-tab-runs").focus()
        await pilot.pause()
        assert screen._focus_in_centre_header, (
            "precondition: the tab strip must be recognized as the centre "
            "header"
        )

        await pilot.press("z")
        await pilot.pause(0.2)

        assert screen.region_layout == before, (
            "z with focus in the tab strip must not act on the stale "
            "focused_region left over from the rail"
        )
        assert not screen.region_layout.is_collapsed(Region.LEFT_RAIL)
        assert screen._last_persisted_collapsed == before.collapsed_for_persistence()


@pytest.mark.asyncio
async def test_capital_z_with_focus_in_the_centre_header_does_not_solo_a_stale_region():
    """The solo half of the test above.

    Unlike the toggle case, ITEMS's solo off the Read tab is ALREADY
    refused by `_refuse_region_gesture_off_read_tab` regardless of focus
    (task-1344 whole-branch review, B1), so `region_layout` never had a
    path to actually mutate here either way. What this fix changes instead:
    without it, that refusal still fires its `self.notify(...)`, keyed to
    a region (`focused_region`, stale) the user is not looking at and has
    no reason to associate with the tab strip they actually pressed `Z`
    in. With the fix, focus-in-header short-circuits before that notify.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause(0.2)
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
        assert screen.focused_region == Region.ITEMS, (
            "precondition: focused_region names a REAL prior focus"
        )
        before = screen.region_layout

        screen.query_one("#wl-tab-runs").focus()
        await pilot.pause()
        assert screen._focus_in_centre_header, (
            "precondition: the tab strip must be recognized as the centre "
            "header"
        )

        screen.notify = Mock()
        await pilot.press("Z")
        await pilot.pause(0.2)

        assert screen.region_layout == before, (
            "Z with focus in the tab strip must not solo the stale "
            "focused_region left over from ITEMS"
        )
        assert screen.region_layout.solo_region is None
        screen.notify.assert_not_called()


@pytest.mark.asyncio
async def test_focus_leaving_the_header_for_a_non_region_widget_clears_the_flag():
    """Re-review follow-up (task-1344): the `_focus_in_centre_header` sentinel
    must be True ONLY while focus is genuinely in the status header. Focus
    that moves from the header to a widget in NEITHER zone (here the
    top-level backend picker `#watchlists-backend-select`, a sibling of the
    workbench) has to clear the flag -- `on_descendant_focus`'s ancestor walk
    otherwise falls off the top without touching it, leaving a stale `True`
    that a later `z`/`Z` would wrongly consult. Reds without the explicit
    else-reset: the flag stays True after focus leaves the header.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.query_one("#wl-tab-runs").focus()
        await pilot.pause()
        assert screen._focus_in_centre_header, "precondition: focus is in the header"

        # `#watchlists-backend-select` sits in `#watchlists-header-bar`, a
        # top-level bar outside both `wl-region-*`/`wl-header-*` and the
        # status header -- the exact "neither zone" case.
        screen.query_one("#watchlists-backend-select").focus()
        await pilot.pause()
        assert not screen._focus_in_centre_header, (
            "focus outside the status header must clear the sentinel; a stale "
            "True would wrongly refuse a later z/Z"
        )


# --- TASK-2200: background loaders patch in place, never recompose the screen -
#
# `_apply_local_wc_snapshot`, `_load_tree_data` and the `overview_data` reactive
# all used to end in a screen-level `refresh(recompose=True)`, which tore down
# and rebuilt every region -- including whichever detail pane was mid-recompose
# of its own. That is the confirmed destroyer behind TASK-1960's crash class.
# Each test below drives one of the three loaders, asserts the surface it feeds
# really did change, and asserts the ITEMS pane was NOT replaced while it did.


async def _sources_tab(pilot, host):
    """Put the screen on the Sources tab with its pane mounted."""
    screen = host.screen_stack[-1]
    screen.active_section = "sources"
    for _ in range(200):
        await pilot.pause(0.01)
        if screen.query("#watchlists-sources-pane"):
            break
    return screen, screen.query_one("#watchlists-sources-pane", SourcesPane)


@pytest.mark.asyncio
async def test_a_background_snapshot_lands_in_place_without_rebuilding_the_pane():
    """AC#1/AC#2: the snapshot's four surfaces update, the detail pane does not.

    The snapshot feeds the loading/error/empty/summary marker (rendered in the
    centre header off the Read tab), the Inspector's `State:` line, and the
    Console attach button's `disabled`/tooltip pair. Every one of those used to
    be repainted only as a side effect of rebuilding the entire screen.
    """
    app = _build_test_app()
    # A real source, so the pre-change state is "loaded, populated, attach
    # ENABLED" -- otherwise `disabled` is True before and after and proves
    # nothing.
    app.watchlist_bundle_service._db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, pane = await _sources_tab(pilot, host)

        attach = screen.query_one("#wc-attach-to-console", Button)
        for _ in range(200):
            await pilot.pause(0.01)
            if screen._wc_loaded and not attach.disabled:
                break
        assert not attach.disabled, (
            "precondition: the real snapshot resolved populated, so Stage is "
            "armed before the failing snapshot below lands"
        )
        assert not screen.query("#wc-service-error")

        screen._apply_local_wc_snapshot((), 0, True, "Watchlists services unavailable; retry Watchlists later.", None)
        for _ in range(200):
            await pilot.pause(0.01)
            if screen.query("#wc-service-error"):
                break

        assert screen.query("#wc-service-error"), (
            "the snapshot's own error marker must reach the centre header "
            "without a screen recompose"
        )
        assert str(screen.query_one("#watchlists-state-summary").renderable) == (
            "State: unavailable"
        ), "the Inspector's State line must follow the snapshot"
        attach = screen.query_one("#wc-attach-to-console", Button)
        assert attach.disabled is True, (
            "Stage must be disarmed once the snapshot reports the service is "
            "unavailable"
        )
        assert "Watchlists services are unavailable" in str(attach.tooltip)
        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is pane, (
            "a background snapshot must not replace the mounted detail pane"
        )


@pytest.mark.asyncio
async def test_a_background_tree_reload_lands_in_the_rail_without_rebuilding_the_pane():
    """AC#1/AC#2: newly created watchlists appear in the rail in place."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, pane = await _sources_tab(pilot, host)

        created = app.watchlist_bundle_service.create("Morning AI Brief")
        node_id = f"#wl-tree-node-watchlist-{created['id']}"
        assert not screen.query(node_id), (
            "precondition: the rail has not heard about the new watchlist yet"
        )

        screen._load_tree_data()
        for _ in range(200):
            await pilot.pause(0.01)
            if screen.query(node_id):
                break

        assert screen.query(node_id), (
            "a tree reload must repaint the rail without a screen recompose"
        )
        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is pane, (
            "a background tree reload must not replace the mounted detail pane"
        )


@pytest.mark.asyncio
async def test_a_background_overview_refresh_repaints_the_inspector_in_place():
    """AC#1/AC#2: `overview_data`'s two Inspector counts follow the payload.

    `overview_data` was `reactive({}, recompose=True)`, so these two lines were
    only ever repainted by rebuilding the whole screen. `watch_overview_data`
    now updates them directly; the Inspector instance must survive.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, pane = await _sources_tab(pilot, host)
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        async def _fake_overview_data(**_kwargs: Any) -> dict[str, Any]:
            return {
                "total_sources": 3,
                "active_sources": 3,
                "sources_in_error": 0,
                "total_items": 9,
                "new_items": 2,
                "latest_run_status": "completed",
                "failed_runs": [],
                "active_alert_rules": 7,
            }

        screen._controller.get_overview_data = _fake_overview_data
        screen._refresh_overview_data()
        for _ in range(200):
            await pilot.pause(0.01)
            if "7" in str(screen.query_one("#watchlists-alerts-summary").renderable):
                break

        assert str(screen.query_one("#watchlists-alerts-summary").renderable) == (
            "Alert rules active: 7"
        )
        assert str(screen.query_one("#watchlists-latest-run-summary").renderable) == (
            "Latest run status: completed"
        )
        assert screen.query_one("#watchlists-entity-inspector", InspectorPane) is (
            inspector
        ), "a background overview refresh must not replace the Inspector"
        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is pane, (
            "a background overview refresh must not replace the detail pane"
        )


@pytest.mark.asyncio
async def test_a_background_tree_reload_updates_the_first_run_copy_in_place():
    """AC#2: TASK-998's watchlist-count-dependent first-run copy still follows.

    `OverviewPane.watchlist_count` is the one thing the ITEMS region reads out
    of `_load_tree_data`, and it decides which of two first-run paragraphs a
    brand-new profile is shown ("create a watchlist" vs "your watchlists have
    no sources yet"). The full-screen recompose used to carry it via
    `_build_detail_pane`; the tree reload now pushes it into the live pane.
    """
    from tldw_chatbook.UI.Watchlists_Modules.overview_pane import OverviewPane

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        assert screen.active_section == "overview"

        overview = None
        for _ in range(300):
            await pilot.pause(0.01)
            overview = screen.query_one("#watchlists-overview-pane", OverviewPane)
            if overview.query("#overview-first-run-body"):
                break
        body = str(overview.query_one("#overview-first-run-body").renderable)
        assert "a watchlist is a folder of feeds" in body.lower(), (
            f"precondition: a profile with no watchlists gets the 'make one' "
            f"copy; it renders {body!r}"
        )

        app.watchlist_bundle_service.create("Morning AI Brief")
        screen._load_tree_data()
        for _ in range(300):
            await pilot.pause(0.01)
            overview = screen.query_one("#watchlists-overview-pane", OverviewPane)
            if overview.watchlist_count:
                break

        assert overview.watchlist_count == 1, (
            "the tree reload must push the new count into the live pane"
        )
        body = str(overview.query_one("#overview-first-run-body").renderable)
        assert "no sources yet" in body.lower(), (
            f"a user who already has a watchlist must be told their next step "
            f"is a SOURCE; it renders {body!r}"
        )


@pytest.mark.asyncio
async def test_switching_backend_clears_the_mounted_panes_selection():
    """AC#2: the backend switch's own state reset still reaches the pane.

    `watch_runtime_backend` clears `selected_source`/`selected_run`/
    `selected_notification` on the SCREEN, and until TASK-2200 the only thing
    that carried those clears into the mounted pane was the full-screen
    recompose its snapshot refresh triggered (`_build_detail_pane` re-seeds
    every pane from exactly those attributes). Without an explicit push, the
    Sources pane keeps its old selection -- so Preview / Check now stay armed
    on a source from a backend the user just left.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen._controller.list_sources = AsyncMock(
            return_value=[{"id": "s1", "name": "Feed One", "source_type": "rss"}]
        )

        screen.query_one("#wl-tab-sources", Button).press()
        await pilot.pause()
        await _wait_for_table_rows(pilot, "#sources-table", screen, 1)

        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        pane.select_source_by_id("s1")
        await pilot.pause()
        assert pane.selected_source is not None, "precondition: a row is selected"

        screen.runtime_backend = "server"
        await pilot.pause(0.2)

        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is pane, (
            "the backend switch must not rebuild the pane either"
        )
        assert pane.selected_source is None, (
            "the mounted pane kept a selection the screen had already cleared"
        )


@pytest.mark.asyncio
async def test_a_surface_refresh_requested_mid_drain_is_served_by_the_same_drainer():
    """TASK-2200: record intent, drain serially, never cancel.

    `refresh_region_content`/`refresh_header_content` are remove-then-mount
    pairs with an `await` between the halves, so a worker cancelled in that
    window (what `exclusive=True` does to its predecessor) leaves the region
    stripped and never refilled. Three call sites now want those swaps, so the
    screen queues surfaces and runs at most one drainer. This drives the exact
    interleaving: a second request arrives WHILE the drainer is awaiting the
    first swap, and must be picked up by that same drainer rather than
    starting -- or cancelling -- anything.
    """
    from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
        WatchlistsWorkbench,
    )

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        workbench = screen.query_one(WatchlistsWorkbench)

        drains: list[int] = []
        real_drain = screen._drain_surface_refresh

        def _counting_drain():
            drains.append(1)
            return real_drain()

        screen._drain_surface_refresh = _counting_drain

        rebuilt: list[Region] = []
        real_region = workbench.refresh_region_content

        async def _tracking_region(region):
            rebuilt.append(region)
            if len(rebuilt) == 1:
                # Lands while the drainer is inside its first swap.
                screen._request_surface_refresh(screen._SURFACE_RAIL)
            await real_region(region)

        workbench.refresh_region_content = _tracking_region

        screen._request_surface_refresh(screen._SURFACE_FEEDS)
        for _ in range(300):
            await pilot.pause(0.01)
            if Region.LEFT_RAIL in rebuilt and not screen._surface_refresh_draining:
                break

        assert Region.LEFT_RAIL in rebuilt, (
            "a request made mid-drain was dropped -- the drainer stopped "
            "without serving it"
        )
        assert drains == [1], (
            f"exactly one drainer must run for a burst; started {len(drains)}"
        )
        assert not screen._surface_refresh_draining, "the drainer must clear its flag"
        assert screen.query_one("#wl-tree"), (
            "the rail must come back populated, not stripped by an "
            "interrupted remove/mount pair"
        )


@pytest.mark.asyncio
async def test_a_background_tree_reload_repaints_the_artifacts_scope_note():
    """Review wave I1: a rename must reach EVERY surface that names the scope.

    `ArtifactsPane.scope_label` resolves the scoped watchlist's display name
    from `_tree_watchlists` (`_briefing_scope_label` ->
    `_watchlist_display_name`), so it is tree data like the rail, the FEEDS
    heading and the Inspector breadcrumb -- but it lives on an ITEMS-region
    pane, which the first pass of TASK-2200 left alone wholesale. The result
    was two surfaces on one screen naming the same watchlist differently until
    the user changed tab or scope.

    Drives exactly what `_rename_watchlist_flow` does once its dialog returns:
    `service.rename(...)` then `_load_tree_data()`.
    """
    from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import ArtifactsPane
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
        TreeScope,
        TreeScopeChanged,
    )

    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Mroning AI Brief")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()
        screen.active_section = "artifacts"
        for _ in range(300):
            await pilot.pause(0.01)
            if screen.query("#artifacts-scope-note"):
                break

        note = screen.query_one("#artifacts-scope-note", Static)
        assert "Mroning AI Brief" in str(note.renderable), (
            f"precondition: the pane names the scoped watchlist; it renders "
            f"{str(note.renderable)!r}"
        )
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)

        service.rename(watchlist["id"], "Morning AI Brief")
        screen._load_tree_data()
        for _ in range(300):
            await pilot.pause(0.01)
            note = screen.query_one("#artifacts-scope-note", Static)
            if "Morning AI Brief" in str(note.renderable):
                break

        note = screen.query_one("#artifacts-scope-note", Static)
        assert "Morning AI Brief" in str(note.renderable), (
            f"the Artifacts pane still names the pre-rename watchlist: "
            f"{str(note.renderable)!r}"
        )
        assert screen.query_one("#watchlists-artifacts-pane", ArtifactsPane) is pane, (
            "the scope-label push must patch the mounted pane, not have the "
            "screen rebuild the region around it"
        )


@pytest.mark.asyncio
async def test_a_failed_surface_refresh_start_does_not_wedge_the_queue():
    """Review wave M1, the sibling of `test_a_failed_tree_write_start_...`.

    `_surface_refresh_draining` is lowered only by `_drain_surface_refresh`'s
    `finally`, which never runs if `run_worker` raises synchronously. Arming
    before scheduling would leave the flag stuck True for the life of the
    screen: every later request would queue and return, and the rail, FEEDS
    and centre header would silently stop following every background loader.
    """
    import inspect

    app = _build_test_app()
    service = app.watchlist_bundle_service
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        exploded: list[Any] = []
        real_run_worker = screen.run_worker

        def _exploding_run_worker(work, **kwargs):
            if kwargs.get("group") == "wc_surface_refresh" and not exploded:
                exploded.append(work)
                raise RuntimeError("worker could not be scheduled")
            return real_run_worker(work, **kwargs)

        screen.run_worker = _exploding_run_worker
        screen._request_surface_refresh(screen._SURFACE_HEADER)
        await pilot.pause()

        assert exploded, "precondition: scheduling really did raise"
        assert screen._surface_refresh_draining is False, (
            "a drain that never started must leave the guard down, or every "
            "later background refresh is silently swallowed"
        )
        assert inspect.getcoroutinestate(exploded[0]) == "CORO_CLOSED", (
            "the un-awaited drain coroutine must be closed, not left to leak "
            "a RuntimeWarning at collection time"
        )

        # ...and the next request really drains.
        created = service.create("Morning AI Brief")
        node_id = f"#wl-tree-node-watchlist-{created['id']}"
        screen._load_tree_data()
        for _ in range(300):
            await pilot.pause(0.01)
            if screen.query(node_id):
                break
        assert screen.query(node_id), (
            "the queue stayed wedged: a later background loader never reached "
            "the rail"
        )


@pytest.mark.asyncio
async def test_a_raising_console_follow_poll_does_not_take_the_app_down():
    """Review wave M2: every step of the drain loop must be non-fatal.

    `run_worker` defaults to `exit_on_error=True`, so an exception escaping
    the drainer reaches `App._handle_exception` and the app goes down. The
    Console-follow poll was the one step outside `_rebuild_surface`'s
    `except Exception`.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        polls: list[int] = []

        def _exploding_drift() -> bool:
            polls.append(1)
            raise RuntimeError("the active-work adapter exploded")

        screen._resolve_console_follow_drift = _exploding_drift
        screen._request_surface_refresh(screen._SURFACE_INSPECTOR)
        for _ in range(300):
            await pilot.pause(0.01)
            if polls and not screen._surface_refresh_draining:
                break

        assert polls, "precondition: the drain really did reach the poll"
        assert screen._surface_refresh_draining is False, (
            "the drainer must clear its flag even when a step raises"
        )
        assert app.is_running, "a raising poll must not take the app down"
        assert screen.query("#watchlists-follow-in-console"), (
            "the Inspector must be left standing, not half-swapped"
        )
