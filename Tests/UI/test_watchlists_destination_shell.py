"""Tests for the new watchlists screen shell structure."""

import asyncio
import itertools
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select, TextArea

from textual.widgets import DataTable

from tldw_chatbook.Subscriptions.noise_defaults import default_ignore_selectors_text

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.app_factory import _build_test_app
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


class WatchlistsContextHarness(App):
    def __init__(self, screen: WatchlistsCollectionsScreen) -> None:
        super().__init__()
        self.context_screen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.context_screen)


@pytest.mark.asyncio
async def test_watchlists_shell_has_tab_strip_and_panes():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        assert screen.query_one("#wl-tabs")
        assert screen.query_one("#watchlists-list-pane")
        assert screen.query_one("#watchlists-detail-pane")
        assert screen.query_one("#watchlists-inspector-pane")
        assert screen.query_one("#watchlists-backend-select", Select)


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
        # screen's overview-data-triggered recompose win the race against
        # RulesPane's own RuleFormVisibilityChanged message still bubbling
        # up to the screen (see `handle_save_rule_requested`).
        screen._controller.save_alert_rule = AsyncMock(return_value=dict(rule))
        # `overview_data` is a `recompose=True` reactive that only rebuilds
        # the screen when the *value* actually changes; the real
        # `get_overview_data()` return value is otherwise byte-for-byte
        # identical before and after this mocked save (nothing in the
        # backing store actually changed), which would mask the race this
        # test targets. Returning a distinct dict on every call reproduces
        # the real-world case where a save legitimately changes a count
        # (e.g. `active_alert_rules`), which is what triggers the recompose.
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

        # Drive enough ticks for the save worker to finish and its
        # overview-data refresh to trigger the screen-level recompose.
        rebuilt_rules_pane = rules_pane
        for _ in range(30):
            await asyncio.sleep(0.02)
            await pilot.pause()
            rebuilt_rules_pane = screen.query_one("#watchlists-rules-pane", RulesPane)
            if rebuilt_rules_pane is not rules_pane:
                break

        assert rebuilt_rules_pane is not rules_pane, (
            "the recompose triggered by the save should have rebuilt the pane"
        )
        assert rebuilt_rules_pane.show_rule_form is False, (
            "the rule edit form must be closed after a successful save, not "
            "reopened pre-filled with the just-submitted rule"
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

    class TreeHarness(App):
        def compose(self):
            yield CountingTree(
                watchlists=[{"id": 1, "name": "Morning AI Brief", "tags": ["ai"]}],
                counts={1: {"unread": 2}},
                source_rows_loader=lambda _wl: [{"id": 5, "name": "ArXiv"}],
                expanded=frozenset({1}),
                active_tag="ai",
                id="wl-tree",
            )

    async with TreeHarness().run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        tree = pilot.app.query_one("#wl-tree", CountingTree)
        assert tree.expanded == frozenset({1})
        assert tree.active_tag == "ai"
        assert pilot.app.query("#wl-tree-node-source-1-5"), (
            "the seeded expansion must be visible on the first render"
        )
        assert composes == [1], (
            f"seeding queued an extra recompose ({len(composes)} composes)"
        )
