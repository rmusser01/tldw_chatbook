"""Tests for the new watchlists screen shell structure."""

from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Select

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import NotificationsPane
from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane


class WatchlistsContextHarness(App):
    def __init__(self, screen: WatchlistsCollectionsScreen) -> None:
        super().__init__()
        self.context_screen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.context_screen)


@pytest.mark.asyncio
async def test_watchlists_shell_has_navigator_and_panes():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        assert screen.query_one("#watchlists-navigator")
        assert screen.query_one("#watchlists-list-pane")
        assert screen.query_one("#watchlists-detail-pane")
        assert screen.query_one("#watchlists-inspector-pane")
        assert screen.query_one("#watchlists-backend-select", Select)


@pytest.mark.asyncio
async def test_watchlists_navigator_updates_active_section():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.active_section == "overview"
        screen.query_one("#nav-sources", Button).press()
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
