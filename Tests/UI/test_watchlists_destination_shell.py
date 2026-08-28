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
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    CheckNowRequested,
    InspectorPane,
)
from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import NotificationsPane
from tldw_chatbook.UI.Watchlists_Modules.pane_grip import RegionToggled
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RulesPane
from tldw_chatbook.UI.Watchlists_Modules.runs_pane import (
    RerunRunRequested,
    RunsPane,
)
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlists_backend_controller import (
    WatchlistsBackendController,
)
from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
    WatchlistsWorkbench,
)


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
        # task-2513: the FEEDS region (`#watchlists-list-pane`) is gone
        # entirely; the default section is now Read ("items"), and the tab
        # strip plus snapshot markers are carried by `#wl-centre-status`
        # (`_build_centre_status_header`) on every tab, Read included.
        assert screen.active_section == "items"
        assert screen.query_one("#wl-centre-status")
        assert not screen.query("#watchlists-list-pane")
        assert screen.query_one("#watchlists-detail-pane")
        assert screen.query_one("#watchlists-inspector-pane")
        assert screen.query_one("#watchlists-backend-select", Select)

        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert not screen.query("#watchlists-list-pane"), (
            "the list pane must not come back on any tab"
        )


@pytest.mark.asyncio
async def test_the_backend_value_is_stated_exactly_once_on_a_normal_section():
    """TASK-2313, AC#3: on a section where the Select is a live choice
    (Sources, not a `_LOCAL_ONLY_SECTIONS` member), the Select's own
    current value ("Local"/"Server") is the ONLY place that fact appears
    -- the old always-present `#watchlists-backend-label` restated it a
    second time as "Backend: local". A `Static("Backend", ...)` label
    ahead of the Select (the 2310 idiom) still names the axis."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause(0.2)

        select = screen.query_one("#watchlists-backend-select", Select)
        assert select.disabled is False, "precondition: a real, live choice"
        assert not screen.query("#watchlists-backend-label"), (
            "the value-restating label must not exist when the Select "
            "already states it"
        )
        header_bar = screen.query_one("#watchlists-header-bar")
        children = list(header_bar.children)
        select_index = children.index(select)
        label = children[select_index - 1]
        assert isinstance(label, Static)
        assert str(label.renderable) == "Backend"


@pytest.mark.asyncio
async def test_the_backend_reason_still_shows_on_a_local_only_section():
    """The counterpart: `#watchlists-backend-label` is NOT pure duplication
    on a `_LOCAL_ONLY_SECTIONS` member -- it explains WHY the (disabled)
    Select's value does not matter, which the Select cannot say on its
    own. That copy must survive TASK-2313's dedup."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.active_section = "artifacts"
        await pilot.pause(0.2)

        assert screen.query_one("#watchlists-backend-select", Select).disabled is True
        label = screen.query_one("#watchlists-backend-label", Static)
        assert str(label.renderable) == "Artifacts: local"


@pytest.mark.asyncio
async def test_the_inspector_first_run_hint_does_not_repeat_overviews_own_walkthrough():
    """TASK-2313, AC#1: UAT -- three stacked "nothing yet" messages on one
    screen (header + Overview pane + Inspector), the Inspector's own hint
    fully re-teaching the SAME two-step walkthrough ("New" in the rail,
    then "New source" under Sources) Overview's numbered first-run body
    already gives in full. The Inspector's hint is now shorter and
    Inspector-specific (what appears HERE), and must not re-teach the
    watchlist-creation step that is Overview's job."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = host.screen_stack[-1]
        for _ in range(60):
            await pilot.pause(0.02)
            if screen.query("#inspector-first-run-hint"):
                break
        hint = str(
            screen.query_one("#inspector-first-run-hint", Static).renderable
        )
        assert "New source" in hint, (
            "must still name the one action relevant to this pane"
        )
        assert "New in the rail" not in hint, (
            "must not re-teach Overview's own watchlist-creation step -- "
            f"the whole walkthrough duplicated there: {hint!r}"
        )
        # task-2513: Overview is no longer the DEFAULT section (Read/"items"
        # is), so its pane is not mounted until the tab is selected -- the
        # walkthrough body only exists once Overview has been shown.
        screen.active_section = "overview"
        await pilot.pause(0.2)
        overview_body = str(
            screen.query_one("#overview-first-run-body", Static).renderable
        )
        assert "New" in overview_body, (
            "precondition: Overview is still the one place that teaches "
            "the rail step"
        )


@pytest.mark.asyncio
async def test_the_selected_object_block_sits_above_console_actions():
    """TASK-2313, AC#5: UAT -- the Inspector's Console block permanently
    outranked the selected object's own actions. `#watchlists-entity-
    inspector` (the selected-object block: title/type/action buttons) must
    now come BEFORE the "Console actions" heading in the right rail's DOM
    order, not after."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        rail = screen.query_one("#watchlists-inspector-pane")
        children = list(rail.children)
        inspector_index = next(
            i for i, child in enumerate(children) if child.id == "watchlists-entity-inspector"
        )
        console_heading_index = next(
            i
            for i, child in enumerate(children)
            if isinstance(child, Static) and str(child.renderable) == "Console actions"
        )
        assert inspector_index < console_heading_index, (
            f"the selected-object block (index {inspector_index}) must sit "
            f"above Console actions (index {console_heading_index})"
        )


@pytest.mark.asyncio
async def test_import_opml_appears_once_on_the_sources_tab():
    """TASK-2313, AC#3: UAT -- "Import OPML" twice on one screen. The
    header's bootstrap actions (New source/Import OPML) are the only
    entry point from every OTHER section, but on Sources itself its own
    toolbar already offers the identical pair one row below -- so the
    header's copy is omitted there specifically, while every other
    section keeps its one bootstrap path."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        assert screen.query_one("#wc-empty-import-opml"), (
            "precondition: Overview (the default section) keeps the "
            "header's bootstrap actions"
        )

        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert not screen.query("#wc-empty-import-opml"), (
            "the header's Import OPML must not duplicate the Sources "
            "toolbar's own copy"
        )
        assert not screen.query("#wc-empty-create-source")
        assert screen.query_one("#sources-import-opml-button"), (
            "the Sources toolbar's own Import OPML must still be there"
        )
        # The header's STATUS text is not an "action" and stays -- only
        # the duplicate buttons are gone.
        assert screen.query_one("#wc-empty-state")

        screen.active_section = "runs"
        await pilot.pause(0.2)
        assert screen.query_one("#wc-empty-import-opml"), (
            "every OTHER section keeps its one bootstrap path"
        )
        assert screen.query_one("#wl-centre-status")


@pytest.mark.asyncio
async def test_watchlists_tab_strip_updates_active_section():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.active_section == "items"
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
    # `_handle_screen_navigation_locked` swallows every navigation until the
    # initial screen exists (splash still up / startup push in flight). This
    # harness never mounts a screen, so mark the app interactive explicitly.
    app._initial_screen_pushed = True
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
        # The default active_section is "items" (Read -- task-2513), so
        # ItemsPane is what's there to start; switch to Sources (as the
        # tab-strip test does) to confirm SourcesPane also still renders
        # inside the re-hosted ITEMS region rather than being dropped.
        assert screen.query("#watchlists-items-pane")
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
        # Read tab -- which is now the default section (task-2513), so no
        # section switch is needed before the real toggle.
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
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
        # Read tab -- the default section since task-2513, so no switch is
        # needed before exercising the toggle itself (focus-tracking is
        # unaffected by which tab is active and is exercised as-is above).
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
        assert screen.focused_region == Region.ITEMS
        await pilot.press("z")
        await pilot.pause()
        assert screen.region_layout.is_collapsed(Region.ITEMS)
        assert not screen.region_layout.is_collapsed(Region.CONTENT)


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
        assert screen.query("#wl-grip-right_rail")
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
        # Read tab -- the default section since task-2513, so no switch is
        # needed before the real toggle this test measures.
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
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
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
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
        assert saved[-1].collapsed == final_layout.collapsed


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
        # TASK-2302 renders the noise field only for the url family (an RSS
        # feed has no elements for a CSS rule to match), so this test has to
        # put the form in the one state the field exists in. `Select.value`
        # is the same change a click through the overlay makes.
        screen.query_one("#sources-create-type", Select).value = "url"
        for _ in range(200):
            await pilot.pause(0.02)
            if screen.query("#sources-create-ignore-selectors"):
                break

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
        deadline = asyncio.get_running_loop().time() + 5.0
        while screen._source_create_draft["name"] or screen._source_create_form_open:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)

        assert screen._source_create_draft == {"name": "", "url": "", "tags": ""}
        assert screen._source_create_form_open is False


# --- Fix round 2 (final whole-branch review): Findings 2, 3, 4. `_build_
# detail_pane`/`_build_inspector_pane` construct a brand new pane on EVERY
# workbench rebuild, not just a section switch. RunsPane/NotificationsPane/
# OverviewPane were already
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
            return_value=[
                {"id": "i1", "title": "Item One", "source_name": "Feed One",
                 # The real backend always carries a status; without one the
                 # reader-set filter (TASK-3072) legitimately hides the row.
                 "status": "new"}
            ]
        )
        screen._controller.list_alert_rules = AsyncMock(
            return_value=[{"id": "r1", "name": "Rule One", "condition_type": "no_items"}]
        )

        for section, table_id in (
            ("sources", "#sources-table"),
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

        # The items list is a ListView (TASK-3072), so its row count lives in
        # the pane's rendered items, not a DataTable: same "an unrelated rail
        # toggle must not empty it" assertion, one level up.
        from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane

        screen.query_one("#wl-tab-items", Button).press()
        await pilot.pause()
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(40):
            await pilot.pause()
            if pane.displayed_items():
                break
        assert len(pane.displayed_items()) == 1, (
            "items list never loaded its one row"
        )

        await pilot.press("[")
        await pilot.pause()

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert len(pane.displayed_items()) == 1, (
            "items list was emptied by an unrelated left-rail toggle"
        )


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
        # task-15461 strengthened this from "rebuilt, and re-seeded correctly"
        # to "never rebuilt at all". A left-rail toggle used to recompose the
        # whole workbench, so the Inspector was torn down and rebuilt for a
        # change on the other side of the screen and the seeding was the only
        # thing standing between the user and a blank rail. Layout changes are
        # now scoped to the region whose form moved (AC#3), so the same
        # gesture leaves this instance -- and therefore the selection -- alone
        # by construction. `test_a_rail_toggle_rebuilds_only_the_toggled_
        # region` pins the scoping itself; this keeps pinning the outcome the
        # user sees.
        assert rebuilt_inspector is inspector, (
            "a left-rail toggle must not rebuild the Inspector at all"
        )
        assert rebuilt_inspector.selected_entity == screen.selected_entity, (
            "the Inspector lost the screen's selection across a rail toggle"
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
async def test_read_mode_class_is_set_before_each_section_layout_swap():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        workbench = screen.query_one(WatchlistsWorkbench)
        left = screen.query_one("#wl-region-left_rail")
        right = screen.query_one("#wl-region-right_rail")

        assert screen.active_section == "items"
        assert workbench.has_class("watchlists-read-mode")

        observed: list[tuple[str, bool]] = []
        reconcile_body = workbench._reconcile_body

        async def record_mode_before_layout(**kwargs):
            observed.append(
                (
                    screen.active_section,
                    workbench.has_class("watchlists-read-mode"),
                )
            )
            await reconcile_body(**kwargs)

        workbench._reconcile_body = record_mode_before_layout

        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert observed[-1] == ("sources", False)
        assert not workbench.has_class("watchlists-read-mode")

        screen.active_section = "items"
        await pilot.pause(0.2)
        assert observed[-1] == ("items", True)
        assert workbench.has_class("watchlists-read-mode")
        assert screen.query_one(WatchlistsWorkbench) is workbench
        assert screen.query_one("#wl-region-left_rail") is left
        assert screen.query_one("#wl-region-right_rail") is right


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
async def test_scoped_rows_follow_the_tree_scope():
    """Task 7: narrowing the tree scope must change what the scoped queries
    cover (the readout the centre header's summary line renders; the FEEDS
    region it originally drove was removed in task-2513).

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
            "narrowing the scope to one watchlist must change what the scope covers"
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
    """`RunsPane.run_poll` ticks once a second for 60 ticks.

    If a tree move clears only the screen's `selected_run` mirror and leaves
    the pane's own copy standing, the very next poll tick re-announces the
    pre-move run, `_select_entity` snaps `selected_scope` back to "all" and
    empties `_breadcrumb_labels` -- with no user action, about a second after
    the user navigated somewhere else.

    The tick now posts `RunProgressTick` rather than `RunSelected` (Qodo, PR
    #1348), whose handler never touches `_select_entity` at all -- so this
    invariant is held by two things now, and the assertions below are
    unchanged because they were always about the OUTCOME, not the message.
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
async def test_rerun_busy_state_survives_a_mounted_runs_pane_rebuild():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    host = WatchlistsContextHarness(screen)
    started = asyncio.Event()
    release = asyncio.Event()
    app.notify = Mock()

    async def launch(**kwargs):
        started.set()
        await release.wait()
        return {"status": "completed", "found_count": 7, "processed_count": 3}

    async with host.run_test(size=(180, 50)) as pilot:
        screen.active_section = "runs"
        for _ in range(40):
            await pilot.pause()
            if screen.query("#watchlists-runs-pane"):
                break

        run = {
            "id": "local:watchlist_run:5",
            "run_id": 5,
            "backend": "local",
            "source_id": 5,
            "source_title": "Feed [five]",
            "status": "completed",
        }
        pane = screen.query_one("#watchlists-runs-pane", RunsPane)
        pane.runs = [run]
        await pilot.pause()
        pane.select_run_by_id(run["id"])
        for _ in range(40):
            await pilot.pause()
            if screen.selected_run is not None:
                break

        screen._controller.launch_run = AsyncMock(side_effect=launch)
        screen._request_runs_refresh = Mock()
        app.notify.reset_mock()
        pane.query_one("#runs-rerun-button", Button).press()
        for _ in range(40):
            await pilot.pause()
            if started.is_set():
                break

        assert started.is_set(), "the gated Re-run must have reached launch_run"
        button = pane.query_one("#runs-rerun-button", Button)
        assert str(button.label) == "Re-running..."
        assert button.disabled
        app.notify.assert_any_call(
            "Re-running Feed [five]...",
            severity="information",
            markup=False,
        )

        old_pane = pane
        screen.active_section = "sources"
        for _ in range(40):
            await pilot.pause()
            if not screen.query("#watchlists-runs-pane"):
                break
        screen.active_section = "runs"
        for _ in range(40):
            await pilot.pause()
            if screen.query("#watchlists-runs-pane"):
                pane = screen.query_one("#watchlists-runs-pane", RunsPane)
                if pane is not old_pane:
                    break

        assert pane is not old_pane, "the section swap must replace RunsPane"
        rebuilt_button = pane.query_one("#runs-rerun-button", Button)
        assert str(rebuilt_button.label) == "Re-running..."
        assert rebuilt_button.disabled

        release.set()
        for _ in range(40):
            await pilot.pause()
            if screen._request_runs_refresh.call_count:
                break

        assert screen._checks_in_flight == set()
        assert screen._reruns_in_flight == set()
        assert str(rebuilt_button.label) == "Re-run source"
        assert not rebuilt_button.disabled
        screen._controller.launch_run.assert_awaited_once_with(
            runtime_backend="local",
            source_id=5,
            job_id=None,
        )
        screen._request_runs_refresh.assert_called_once_with()
        app.notify.assert_any_call(
            "Re-run complete: Feed [five] — 7 found, 3 new.",
            severity="information",
            markup=False,
        )


@pytest.mark.asyncio
async def test_external_local_check_now_paints_the_selected_run_as_checking():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    host = WatchlistsContextHarness(screen)
    started = asyncio.Event()
    release = asyncio.Event()

    async def check_now(**kwargs):
        started.set()
        await release.wait()
        return {"status": "completed"}

    async with host.run_test(size=(180, 50)) as pilot:
        screen.active_section = "runs"
        for _ in range(40):
            await pilot.pause()
            if screen.query("#watchlists-runs-pane"):
                break

        run = {
            "id": "local:watchlist_run:5",
            "run_id": 5,
            "backend": "local",
            "source_id": 5,
            "source_title": "Feed five",
            "status": "completed",
        }
        pane = screen.query_one("#watchlists-runs-pane", RunsPane)
        pane.runs = [run]
        await pilot.pause()
        pane.select_run_by_id(run["id"])
        for _ in range(40):
            await pilot.pause()
            if screen.selected_run is not None:
                break

        screen._controller.check_now = AsyncMock(side_effect=check_now)
        screen.post_message(
            CheckNowRequested(
                {
                    "id": "local:subscription:5",
                    "source_id": 5,
                    "name": "Feed five",
                }
            )
        )
        for _ in range(40):
            await pilot.pause()
            if started.is_set():
                break

        assert started.is_set(), "the gated Check now must reach the controller"
        button = pane.query_one("#runs-rerun-button", Button)
        assert str(button.label) == "Checking..."
        assert button.disabled

        release.set()
        for _ in range(40):
            await pilot.pause()
            if not screen._checks_in_flight:
                break
        assert str(button.label) == "Re-run source"
        assert not button.disabled


@pytest.mark.asyncio
async def test_server_rerun_deduplicates_one_job_without_blocking_another():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen.runtime_backend = "server"
    host = WatchlistsContextHarness(screen)
    started = {"job-5": asyncio.Event(), "job-6": asyncio.Event()}
    release = asyncio.Event()
    app.notify = Mock()

    async def launch(*, runtime_backend, source_id, job_id):
        started[job_id].set()
        await release.wait()
        return {"status": "queued"}

    async with host.run_test(size=(180, 50)) as pilot:
        screen._controller.launch_run = AsyncMock(side_effect=launch)
        screen._request_runs_refresh = Mock()
        app.notify.reset_mock()

        screen.post_message(
            RerunRunRequested(
                runtime_backend="server", target_id="job-5", name="Job [five]"
            )
        )
        for _ in range(40):
            await pilot.pause()
            if started["job-5"].is_set():
                break

        screen.post_message(
            RerunRunRequested(
                runtime_backend="server", target_id="job-5", name="Job [five]"
            )
        )
        screen.post_message(
            RerunRunRequested(
                runtime_backend="server", target_id="job-6", name="Job six"
            )
        )
        for _ in range(40):
            await pilot.pause()
            if started["job-6"].is_set():
                break

        assert started["job-6"].is_set(), "a different job must launch independently"
        assert screen._controller.launch_run.await_count == 2
        app.notify.assert_any_call(
            "Already checking Job [five].", severity="warning", markup=False
        )
        assert screen._checks_in_flight == {
            screen._rerun_operation_key("server", "job-5"),
            screen._rerun_operation_key("server", "job-6"),
        }

        release.set()
        for _ in range(40):
            await pilot.pause()
            if screen._request_runs_refresh.call_count == 2:
                break
        assert screen._checks_in_flight == set()
        assert screen._reruns_in_flight == set()
        assert screen._request_runs_refresh.call_count == 2


def test_rerun_rejects_an_old_backend_request_before_launch_or_busy_state():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen.runtime_backend = "local"
    screen._controller.launch_run = AsyncMock()
    screen._set_check_now_busy = Mock()
    screen.run_worker = Mock()

    screen.handle_rerun_run_requested(
        RerunRunRequested(
            runtime_backend="server", target_id="job-5", name="Old server job"
        )
    )

    screen._controller.launch_run.assert_not_awaited()
    screen._set_check_now_busy.assert_not_called()
    screen.run_worker.assert_not_called()
    assert screen._checks_in_flight == set()
    assert screen._reruns_in_flight == set()


@pytest.mark.asyncio
async def test_rerun_completion_after_backend_switch_does_not_repaint_old_state():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen.runtime_backend = "server"
    screen._controller.launch_run = AsyncMock(return_value={"status": "completed"})
    screen._set_check_now_busy = Mock()
    screen._request_runs_refresh = Mock()
    operation_key = screen._rerun_operation_key("local", 5)
    screen._checks_in_flight.add(operation_key)
    screen._reruns_in_flight.add(operation_key)

    await screen._rerun_run(
        runtime_backend="local",
        target_id=5,
        operation_key=operation_key,
        name="Old local source",
    )

    screen._set_check_now_busy.assert_not_called()
    screen._request_runs_refresh.assert_called_once_with()
    assert screen._checks_in_flight == set()
    assert screen._reruns_in_flight == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "running"])
async def test_server_rerun_reports_started_and_launches_by_job_id(status):
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen.runtime_backend = "server"
    app.notify = Mock()
    screen._controller.launch_run = AsyncMock(return_value={"status": status})
    screen._request_runs_refresh = Mock()
    operation_key = screen._rerun_operation_key("server", "job-7")
    screen._checks_in_flight.add(operation_key)
    screen._reruns_in_flight.add(operation_key)

    await screen._rerun_run(
        runtime_backend="server",
        target_id="job-7",
        operation_key=operation_key,
        name="Feed [seven]",
    )

    screen._controller.launch_run.assert_awaited_once_with(
        runtime_backend="server",
        source_id=None,
        job_id="job-7",
    )
    app.notify.assert_called_once_with(
        "Re-run started: Feed [seven].",
        severity="information",
        markup=False,
    )
    assert screen._checks_in_flight == set()
    assert screen._reruns_in_flight == set()
    screen._request_runs_refresh.assert_called_once_with()


@pytest.mark.asyncio
async def test_local_rerun_reports_an_entirely_skipped_run():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    app.notify = Mock()
    screen._controller.launch_run = AsyncMock(
        return_value={
            "status": "completed",
            "stats": {"dispositions": {"skipped": 1}},
        }
    )
    screen._request_runs_refresh = Mock()
    operation_key = screen._rerun_operation_key("local", 5)
    screen._checks_in_flight.add(operation_key)
    screen._reruns_in_flight.add(operation_key)

    await screen._rerun_run(
        runtime_backend="local",
        target_id=5,
        operation_key=operation_key,
        name="Feed [five]",
    )

    app.notify.assert_called_once_with(
        "Re-run skipped: Feed [five] — a check of this source is already running.",
        severity="warning",
        markup=False,
    )
    assert screen._checks_in_flight == set()
    assert screen._reruns_in_flight == set()
    screen._request_runs_refresh.assert_called_once_with()


@pytest.mark.asyncio
async def test_rerun_reports_a_returned_failed_status():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    app.notify = Mock()
    screen._controller.launch_run = AsyncMock(
        return_value={"status": "failed", "error_msg": "source denied"}
    )
    screen._request_runs_refresh = Mock()
    operation_key = screen._rerun_operation_key("local", 5)
    screen._checks_in_flight.add(operation_key)
    screen._reruns_in_flight.add(operation_key)

    await screen._rerun_run(
        runtime_backend="local",
        target_id=5,
        operation_key=operation_key,
        name="Feed [five]",
    )

    app.notify.assert_called_once_with(
        "Re-run failed: Feed [five] — source denied",
        severity="error",
        markup=False,
    )
    assert screen._checks_in_flight == set()
    assert screen._reruns_in_flight == set()
    screen._request_runs_refresh.assert_called_once_with()


@pytest.mark.asyncio
async def test_rerun_raises_with_a_safe_stated_error_and_warning_log():
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    app.notify = Mock()
    screen._controller.launch_run = AsyncMock(
        side_effect=RuntimeError(
            "unexpected /Users/private/feed.xml?token=secret"
        )
    )
    screen._request_runs_refresh = Mock()
    operation_key = screen._rerun_operation_key("local", 5)
    screen._checks_in_flight.add(operation_key)
    screen._reruns_in_flight.add(operation_key)

    with patch(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.logger"
    ) as logger:
        await screen._rerun_run(
            runtime_backend="local",
            target_id=5,
            operation_key=operation_key,
            name="Feed [five]",
        )

    logger.opt.assert_called_once_with(exception=True)
    logger.opt.return_value.warning.assert_called_once()
    app.notify.assert_called_once_with(
        "Re-run failed: Feed [five].",
        severity="error",
        markup=False,
    )
    assert "/Users/private" not in app.notify.call_args.args[0]
    assert "token=secret" not in app.notify.call_args.args[0]
    assert screen._checks_in_flight == set()
    assert screen._reruns_in_flight == set()
    screen._request_runs_refresh.assert_called_once_with()


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


# --- Permanent Reader and management-canvas section contracts ------------


@pytest.mark.asyncio
async def test_the_feeds_region_is_gone_and_content_stays_gated_to_read():
    """task-2513: the FEEDS region was removed outright -- it must have no
    DOM presence on ANY tab (no `#wl-region-feeds`, no `#wl-header-feeds`),
    and its old pane (`#watchlists-list-pane`) with it.

    CONTENT is the permanent Reader on Read and unmounted elsewhere.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        # The default section is Read ("items") since task-2513.
        assert screen.active_section == "items"
        assert not screen.query("#wl-header-feeds")
        assert not screen.query("#wl-region-feeds")
        assert not screen.query("#watchlists-list-pane")
        assert screen.query("#wl-region-items")
        assert screen.query("#wl-region-content")

        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert not screen.query("#wl-header-feeds")
        assert not screen.query("#wl-region-feeds")
        assert not screen.query("#watchlists-list-pane")
        assert screen.query("#wl-region-items"), (
            "ITEMS is force-shown off Read -- the section's own full-width pane"
        )
        assert not screen.query("#wl-region-content")


@pytest.mark.asyncio
async def test_the_items_toggle_off_the_read_tab_neither_collapses_nor_persists():
    """A stale Feed Items message cannot change preference off Read."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause(0.3)
        assert screen.query("#wl-region-items"), (
            "the management section owns the permanent centre canvas"
        )
        assert not screen.query("#wl-grip-items")
        preferred_before = screen.region_layout
        effective_before = screen._effective_region_layout
        persisted_before = screen._last_persisted_collapsed

        screen.notify = Mock()
        screen.post_message(RegionToggled(Region.ITEMS))
        await pilot.pause(0.3)

        assert screen.region_layout == preferred_before
        assert screen._effective_region_layout == effective_before
        assert screen._last_persisted_collapsed == persisted_before
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
        assert screen.query("#wl-grip-items")


@pytest.mark.asyncio
async def test_article_focus_on_a_management_tab_is_refused():
    """Article Focus is a Read-only effective layout, never a preference."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "runs"
        await pilot.pause(0.3)
        preferred_before = screen.region_layout
        effective_before = screen._effective_region_layout

        screen.notify = Mock()
        screen.action_article_focus()
        await pilot.pause(0.3)

        assert screen.region_layout == preferred_before
        assert screen._effective_region_layout == effective_before
        assert screen._article_focus_active is False
        assert screen.query("#wl-region-items")
        screen.notify.assert_called_once()
        assert screen.notify.call_args.kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_tab_switches_grips_and_article_focus_never_remove_the_centre():
    """Read keeps Reader mounted; management tabs keep their canvas mounted."""

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        async def step(label: str) -> None:
            await pilot.pause(0.2)
            assert screen.query("#wl-region-content") or screen.query(
                "#wl-region-items"
            ), (
                f"the centre is empty after {label!r}: "
                f"active_section={screen.active_section!r} "
                f"preferred={screen.region_layout!r} "
                f"effective={screen._effective_region_layout!r}"
            )

        await step("mount")
        preferred_before = screen.region_layout
        screen.action_article_focus()
        await step("Article Focus")
        assert screen.query("#wl-region-content")
        assert not screen.query("#wl-region-items")
        assert screen.region_layout == preferred_before

        screen.active_section = "sources"
        await step("management section")
        assert screen._article_focus_active is False
        assert not screen.query("#wl-region-content")

        screen.active_section = "items"
        await step("return to Read")
        screen.query_one("#wl-grip-items", Button).press()
        await step("collapse Feed Items grip")
        assert screen.query("#wl-region-content")
        assert screen.region_layout.is_collapsed(Region.ITEMS)

        screen.active_section = "runs"
        await step("Runs with Feed Items preference collapsed")
        await pilot.press("[")
        await step("toggle Navigation")
        await pilot.press("]")
        await step("toggle Inspector")

        screen.active_section = "items"
        await step("return to Reader")
        assert screen.query("#wl-region-content")


@pytest.mark.asyncio
async def test_management_canvas_does_not_override_feed_items_preference():
    """Management uses ITEMS as its canvas without changing Read preference."""

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "items"
        await pilot.pause(0.3)

        screen.query_one("#wl-grip-items", Button).press()
        await pilot.pause(0.3)
        assert screen.region_layout.is_collapsed(Region.ITEMS)
        assert screen._effective_region_layout.is_collapsed(Region.ITEMS)
        preferred_before = screen.region_layout

        screen.active_section = "sources"
        await pilot.pause(0.3)
        assert screen.query("#wl-region-items")
        assert not screen.query("#wl-grip-items")
        assert not screen._effective_region_layout.is_collapsed(Region.ITEMS)

        screen.notify = Mock()
        screen.post_message(RegionToggled(Region.ITEMS))
        await pilot.pause(0.3)
        assert screen.region_layout == preferred_before
        screen.notify.assert_called_once()

        screen.active_section = "items"
        await pilot.pause(0.3)
        assert screen.query("#wl-region-content")
        assert not screen.query("#wl-region-items")
        assert screen._effective_region_layout.is_collapsed(Region.ITEMS)


# --- `z` / Article Focus while focus is in the status header ------------
#
# `#wl-centre-status`/`#wl-tabs` (`_build_centre_status_header`) are mounted
# above `#wl-workbench-body`, outside every `wl-region-*`/`wl-grip-*` wrapper.


@pytest.mark.asyncio
async def test_z_with_focus_in_the_centre_header_does_not_toggle_a_stale_region():
    """Before this fix, a user who last focused the left rail (setting
    `focused_region = LEFT_RAIL`), then tabbed into the tab strip and
    pressed `z`, collapsed -- and PERSISTED -- the rail anyway:
    `_refuse_region_gesture_off_read_tab` only gates `CENTRE_REGIONS`, so a
    rail's toggle was never refused there regardless of where real focus
    was. This is the one gesture the header-focus guard actually prevents
    from mutating anything.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        # The default section is Read ("items") since task-2513, and the
        # header (`_build_centre_status_header`) exists on every tab now.
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
        assert screen._last_persisted_collapsed == before.collapsed


@pytest.mark.asyncio
async def test_the_tab_strip_is_recognized_as_the_centre_header_on_the_read_tab_too():
    """TASK-2312: before this task, the Read ("items") tab was the ONE
    section where the tab strip did NOT live in `#wl-centre-status` -- it
    was mounted INSIDE FEEDS's own bordered body instead, so focusing it
    there matched `on_descendant_focus`'s `wl-region-`/`wl-header-` prefix
    check first and set `focused_region = FEEDS`, never reaching the
    header-tracking branch at all. No prior test exercised that
    combination (focus in the Items-tab tab strip) in either direction, so
    this pins the corrected, now-uniform behaviour: focusing the tab strip
    on Read sets `_focus_in_centre_header`, exactly like every other
    section, and leaves `focused_region` at whatever it was before (a
    stale FEEDS reference must not be manufactured by this focus move)."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.2)

        screen.query_one("#wl-region-left_rail").focus()
        await pilot.pause()
        assert screen.focused_region == Region.LEFT_RAIL, (
            "precondition: focused_region names a REAL prior focus"
        )
        assert not screen._focus_in_centre_header

        screen.query_one("#wl-tab-runs").focus()
        await pilot.pause()

        assert screen._focus_in_centre_header, (
            "the tab strip must be recognized as the centre header on the "
            "Read tab too, not just every other section"
        )
        assert screen.focused_region == Region.LEFT_RAIL, (
            "focus moving into the header must not silently reassign "
            "focused_region to FEEDS (the region the tab strip used to "
            "live inside, pre-TASK-2312)"
        )


@pytest.mark.asyncio
async def test_capital_z_in_the_header_activates_article_focus_only_effectively():
    """Article Focus is independent of stale region focus and not persisted."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.query_one("#wl-region-left_rail").focus()
        await pilot.pause()
        assert screen.focused_region == Region.LEFT_RAIL, (
            "precondition: focused_region names a REAL prior focus"
        )
        preferred_before = screen.region_layout

        screen.query_one("#wl-tab-runs").focus()
        await pilot.pause()
        assert screen._focus_in_centre_header, (
            "precondition: the tab strip must be recognized as the centre "
            "header"
        )

        await pilot.press("Z")
        await pilot.pause(0.2)

        assert screen._article_focus_active is True
        assert screen.region_layout == preferred_before
        assert screen.query("#wl-region-content")
        assert not screen.query("#wl-region-items")
        for region in (Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL):
            assert screen._effective_region_layout.is_collapsed(region)


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

        screen.active_section = "runs"
        await pilot.pause(0.2)

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
            "Watchlists: error"
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


def test_watchlists_state_summary_text_loading_branch():
    """UAT batch-5 review, m2: `_watchlists_state_summary_text()`'s three
    branches (loading/error/loaded) had only the LAST two under test
    anywhere in the suite -- "Watchlists: loading…", the transient before
    the first snapshot resolves, was real, shipped, untested behaviour.

    A bare, unmounted screen instance is enough: both inputs
    (`_wc_loaded`/`_wc_lookup_error`) are plain instance attributes, not
    Textual reactives, and the method touches no DOM -- no need to race an
    async snapshot load to exercise this branch deterministically.
    """
    screen = object.__new__(WatchlistsCollectionsScreen)
    screen._wc_loaded = False
    screen._wc_lookup_error = None

    assert screen._watchlists_state_summary_text() == "Watchlists: loading…"


def test_watchlists_state_summary_text_error_and_loaded_branches_still_agree():
    """Companion to the loading-branch test above: pins the other two
    branches at the same unmounted-instance granularity so all three read
    from one place."""
    screen = object.__new__(WatchlistsCollectionsScreen)
    screen._wc_loaded = True
    screen._wc_lookup_error = "boom"
    assert screen._watchlists_state_summary_text() == "Watchlists: error"

    screen._wc_lookup_error = None
    assert screen._watchlists_state_summary_text() == "Watchlists: loaded"


def test_latest_run_status_text_distinguishes_not_configured_from_no_runs_yet():
    """UAT batch-5 review, finding I1: `scope_service` being entirely
    unwired (`WatchlistsBackendController.NOT_CONFIGURED_STATUS`) must
    render distinctly from a healthy watchlist that genuinely has zero
    runs (`None` -> "no runs yet") -- collapsing the two back into the
    same text is the exact regression this test pins. Reverting either
    sentinel check in `_latest_run_status_text` back to a bare `if not
    status` turns this red.
    """
    # `overview_data` is `reactive({}, recompose=True)`; a bare, unmounted
    # instance has no `_id` (Textual's `DOMNode.id` property, which the
    # reactive descriptor's own `__get__`/`__set__` both gate on), so
    # writing/reading it through the normal `self.overview_data = ...`
    # attribute path raises `ReactiveError`. `_latest_run_status_text`
    # only ever reads through a plain `.get()`, so a stand-in `_id` plus
    # writing straight to the reactive's internal storage slot
    # (`_reactive_<name>`, Textual's own convention) is enough and avoids
    # mounting a whole screen for what is otherwise a pure-function test.
    screen = object.__new__(WatchlistsCollectionsScreen)
    screen._id = 1

    screen._reactive_overview_data = {"latest_run_status": WatchlistsBackendController.NOT_CONFIGURED_STATUS}
    not_configured_text = screen._latest_run_status_text()

    screen._reactive_overview_data = {"latest_run_status": None}
    no_runs_text = screen._latest_run_status_text()

    assert not_configured_text != no_runs_text, (
        "an unwired scope_service must not read identically to a healthy, "
        "simply-unrun watchlist"
    )
    assert not_configured_text == "Latest run status: not connected"
    assert no_runs_text == "Latest run status: no runs yet"


def test_latest_run_status_text_reports_a_real_lookup_failure_distinctly():
    """UAT batch-5 review, finding I1: the screen's own except-handler
    fallback (`_refresh_overview_data`, a REAL exception fetching the
    profile) must read as a fault, distinct from both "no runs yet" and
    "not connected" -- and distinct from the old bare "unavailable"
    literal, which this test also guards against reappearing verbatim.
    """
    screen = object.__new__(WatchlistsCollectionsScreen)
    screen._id = 1
    screen._reactive_overview_data = {"latest_run_status": WatchlistsBackendController.LOOKUP_FAILED_STATUS}

    text = screen._latest_run_status_text()

    assert text == "Latest run status: couldn't check"
    assert text != "Latest run status: unavailable"
    assert text != "Latest run status: no runs yet"


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
        # The default section is Read since task-2513; the overview first-run
        # panel lives behind its own tab now.
        screen.active_section = "overview"
        await pilot.pause(0.2)

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

        screen._request_surface_refresh(screen._SURFACE_RAIL)
        for _ in range(300):
            await pilot.pause(0.01)
            if len(rebuilt) >= 2 and not screen._surface_refresh_draining:
                break

        assert len(rebuilt) == 2, (
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
    `_watchlist_display_name`), so it is tree data like the rail, the centre
    header's summary line and the Inspector breadcrumb -- but it lives on an
    ITEMS-region pane, which the first pass of TASK-2200 left alone
    wholesale. The result was two surfaces on one screen naming the same
    watchlist differently until the user changed tab or scope.

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
        for _ in range(1000):
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
        await screen._load_tree_data().wait()
        for _ in range(300):
            await pilot.pause(0.01)
            notes = screen.query("#artifacts-scope-note")
            if notes and "Morning AI Brief" in str(notes.first(Static).renderable):
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
    `finally`, which never runs if scheduling raises synchronously. Arming
    before scheduling would leave the flag stuck True for the life of the
    screen: every later request would queue and return, and the rail and
    centre header would silently stop following every background loader.

    task-15461 moved the scheduling from `run_worker(group=
    "wc_surface_refresh")` to `call_next` -- so that anything waiting for the
    message pump to go quiet also waits for the DOM swap, which the section
    switch now rides on. The failure mode this test pins is unchanged, so the
    seam it explodes moves with it. The "the un-awaited coroutine was closed"
    half is gone with the coroutine: `call_next` is handed the bound method,
    so nothing is constructed ahead of scheduling and there is nothing left
    to leak.
    """
    app = _build_test_app()
    service = app.watchlist_bundle_service
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        exploded: list[Any] = []
        real_call_next = screen.call_next

        def _exploding_call_next(callback, *args, **kwargs):
            if callback == screen._drain_surface_refresh and not exploded:
                exploded.append(callback)
                raise RuntimeError("callback could not be scheduled")
            return real_call_next(callback, *args, **kwargs)

        screen.call_next = _exploding_call_next
        screen._request_surface_refresh(screen._SURFACE_HEADER)
        await pilot.pause()

        assert exploded, "precondition: scheduling really did raise"
        assert screen._surface_refresh_draining is False, (
            "a drain that never started must leave the guard down, or every "
            "later background refresh is silently swallowed"
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


@pytest.mark.asyncio
async def test_loader_results_landing_before_textual_flips_is_mounted_still_paint():
    """Live-verification wave: the in-place updates must not use `is_mounted`.

    `Widget.is_mounted` returns `_is_mounted`, which `MessagePump._pre_process`
    sets in its `finally` -- AFTER dispatching both `Compose` and `Mount`. So
    for the whole of `on_mount`, and for anything `on_mount` starts that
    finishes before that `finally` runs, `is_mounted` is False while the entire
    subtree is already registered and queryable. On a cold local database the
    Watchlists loaders finish inside exactly that window; instrumented on a real
    terminal at 235x52:

        OVERVIEW watcher is_mounted=False keys=[]  pane=0 inspector=0
        ON_MOUNT         is_mounted=False wb=1 centre=1 status=1
        SNAPSHOT applied is_mounted=False loaded=True wb=1 centre=1 status=1
        OVERVIEW watcher is_mounted=False keys=[...] pane=1 inspector=1

    Every update was dropped by an `is_mounted` guard and nothing re-requested
    them: the screen sat on "Loading local Watchlists snapshot..." /
    "Loading watchlist activity..." / "State: unavailable" for 100+ seconds
    until an unrelated tab switch recomposed it.

    `run_test`'s pilot settles the DOM before loader results are applied, so
    the ordering cannot be raced for here. This RECONSTRUCTS the captured state
    instead -- a live, fully-attached screen with `_is_mounted` forced back to
    False -- the same technique TASK-1960 used for its own captured DOM state.
    """
    from tldw_chatbook.UI.Watchlists_Modules.overview_pane import OverviewPane

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        for _ in range(300):
            await pilot.pause(0.01)
            if screen._wc_loaded and screen.query("#wc-empty-state"):
                break
        assert screen.query("#wc-empty-state"), "precondition: the normal path paints"

        # The default section is Read since task-2513; this test exercises
        # the Overview pane, so move to its tab first.
        screen.active_section = "overview"
        for _ in range(300):
            await pilot.pause(0.01)
            if screen.query("#watchlists-overview-pane"):
                break

        # Rewind to what a cold screen looks like a millisecond into `on_mount`:
        # nothing loaded, the loading markers on screen, the DOM fully attached.
        screen._wc_loaded = False
        screen._wc_lookup_error = None
        screen._local_watchlist_records = ()
        screen._local_watchlist_count = 0
        screen.overview_data = {}
        screen._request_surface_refresh(screen._SURFACE_HEADER)
        for _ in range(300):
            await pilot.pause(0.01)
            if screen.query("#wc-loading-state"):
                break
        assert screen.query("#wc-loading-state"), "precondition: rewound to loading"
        overview = screen.query_one("#watchlists-overview-pane", OverviewPane)
        assert overview.query("#overview-loading"), "precondition: overview loading"
        assert screen.is_attached, "precondition: the DOM is live throughout"

        # THE WINDOW. Everything below -- including the drain worker's own
        # loop, which the live log shows running at `is_mounted=False` -- runs
        # with `is_mounted` False and every widget present, byte-for-byte the
        # state the log above captured. The assertions are made INSIDE the
        # window too, so nothing can be satisfied by the restore below.
        screen._is_mounted = False
        try:
            assert not screen.is_mounted
            assert screen.query_one("#wl-centre-status"), (
                "precondition: the header really is queryable in this window"
            )
            screen._apply_local_wc_snapshot((), 0, True, None, None)
            screen.overview_data = {
                "total_sources": 0,
                "active_sources": 0,
                "sources_in_error": 0,
                "total_items": 0,
                "new_items": 0,
                "latest_run_status": "unavailable",
                "failed_runs": [],
                "active_alert_rules": 0,
            }

            for _ in range(300):
                await pilot.pause(0.01)
                if screen.query("#wc-empty-state") and not screen.query(
                    "#overview-loading"
                ):
                    break

            assert not screen.query("#wc-loading-state"), (
                "the snapshot marker is still 'Loading local Watchlists "
                "snapshot...' after the load landed -- the update was dropped "
                "and nothing re-requests it"
            )
            assert screen.query("#wc-empty-state"), (
                "the loaded snapshot state must paint"
            )
            overview = screen.query_one("#watchlists-overview-pane", OverviewPane)
            assert not overview.query("#overview-loading"), (
                "the Overview pane is still 'Loading watchlist activity...' "
                "after `overview_data` landed"
            )
            assert overview.query("#overview-first-run"), (
                "the loaded (empty-profile) Overview state must paint"
            )
            assert str(screen.query_one("#watchlists-state-summary").renderable) == (
                "Watchlists: loaded"
            ), "the Inspector's State line must follow the snapshot here too"

            # The tree loader lands in the same window (`on_mount` starts it
            # too), so its own updater has to reach the rail from here as well.
            created = app.watchlist_bundle_service.create("Morning AI Brief")
            node_id = f"#wl-tree-node-watchlist-{created['id']}"
            assert not screen.query(node_id), "precondition: the rail is stale"
            screen._load_tree_data()
            for _ in range(300):
                await pilot.pause(0.01)
                if screen.query(node_id):
                    break
            assert screen.query(node_id), (
                "the rail never picked up the tree reload that landed before "
                "Textual flipped `is_mounted`"
            )
        finally:
            screen._is_mounted = True


@pytest.mark.asyncio
async def test_section_loader_results_landing_in_the_mount_window_still_paint():
    """Re-review L1: the six section loaders share the `is_mounted` trap.

    `on_mount` starts exactly one of them (`_load_active_section_data`), and
    the section it starts is attacker-chosen in the sense that matters here:
    `apply_navigation_context` sets `active_section` on an UNMOUNTED screen
    (`app.py` calls it before `switch_screen`), which is how the "open this run
    in Watchlists" deep link works. On a cold database that loader lands inside
    the mount window -- `is_mounted` False, every widget present -- so its
    `if self.is_mounted:` push was dropped and the section's table rendered
    blank until the user clicked something. The full-screen recompose this task
    removed used to cover it.

    Set up through the real deep-link entry point (`apply_navigation_context`
    on the unmounted screen), then reconstructs the mount window for each of
    the six sections in turn -- the guard is identical at all six sites, and a
    test that only covered the deep-link's own section would let the other five
    rot.
    """
    from tldw_chatbook.Subscriptions.watchlist_item_page import WatchlistItemPage
    from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import ArtifactsPane
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
        TreeScope,
        TreeScopeChanged,
    )

    app = _build_test_app()
    watchlist = app.watchlist_bundle_service.create("Morning AI Brief")
    host = DestinationHarness(app, "watchlists_collections")

    # THE DEEP LINK: applied while the screen is unmounted, exactly as
    # `app.py` does before `switch_screen`.
    host.context_screen.apply_navigation_context({"section": "runs"})
    assert host.context_screen.active_section == "runs", (
        "precondition: the deep link really did set the section pre-mount"
    )

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen._controller.list_sources = AsyncMock(
            return_value=[{"id": "s1", "name": "Feed One", "source_type": "rss"}]
        )
        screen._controller.list_runs = AsyncMock(
            return_value=[{"id": "r1", "source_title": "Feed One", "status": "ok"}]
        )
        screen._controller.list_reader_items_page = AsyncMock(
            return_value=WatchlistItemPage(
                items=(
                    {
                        "id": "i1",
                        "title": "Item One",
                        "source_name": "Feed One",
                    },
                ),
                has_more=False,
                snapshot_max_item_id=1,
                snapshot_count=1,
                next_cursor=None,
            )
        )
        screen._controller.list_alert_rules = AsyncMock(
            return_value=[{"id": "a1", "name": "Rule One", "condition_type": "no_items"}]
        )
        screen._notifications_controller.load_rows = AsyncMock(
            return_value=[
                {
                    "id": 7,
                    "title": "Research complete",
                    "message": "The synthesis is ready.",
                    "category": "research",
                    "severity": "info",
                    "is_read": False,
                }
            ]
        )

        cases = [
            ("runs", "#watchlists-runs-pane", "runs", lambda: screen._load_runs()),
            ("sources", "#watchlists-sources-pane", "sources", lambda: screen._load_sources()),
            (
                "items",
                "#watchlists-items-pane",
                "items",
                lambda: screen._replace_items_snapshot(reason="refresh"),
            ),
            ("rules", "#watchlists-rules-pane", "rules", lambda: screen._load_rules()),
            (
                "notifications",
                "#watchlists-notifications-pane",
                "notifications",
                lambda: screen._load_notifications(),
            ),
        ]

        for section, selector, attribute, loader in cases:
            screen.active_section = section
            pane = None
            for _ in range(300):
                await pilot.pause(0.01)
                found = screen.query(selector)
                if found:
                    pane = found.first()
                    break
            assert pane is not None, f"precondition: the {section} pane mounted"
            await host.workers.wait_for_complete()

            # Rewind the pane to "nothing loaded", so only an in-window push
            # can satisfy the assertion below.
            setattr(pane, attribute, [])
            await pilot.pause()
            assert not getattr(pane, attribute), f"precondition: {section} rewound"

            screen._is_mounted = False
            try:
                await loader()
                assert getattr(pane, attribute), (
                    f"the {section} loader's rows never reached the mounted pane: "
                    f"they landed while Textual still reported is_mounted=False, "
                    f"and nothing re-pushes them"
                )
            finally:
                screen._is_mounted = True

        # Artifacts: the same guard, but the push is a bundle of pane state
        # rather than a row list, so it is asserted on its own terms.
        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()
        screen.active_section = "artifacts"
        artifacts = None
        for _ in range(300):
            await pilot.pause(0.01)
            found = screen.query("#watchlists-artifacts-pane")
            if found:
                artifacts = found.first(ArtifactsPane)
                break
        assert artifacts is not None, "precondition: the artifacts pane mounted"

        artifacts.scope_label = ""
        artifacts.can_generate = False
        await pilot.pause()

        screen._is_mounted = False
        try:
            await screen._load_briefings()
            assert artifacts.scope_label, (
                "the briefings loader never repainted the mounted Artifacts pane "
                "from inside the mount window"
            )
            assert artifacts.can_generate is True
        finally:
            screen._is_mounted = True


@pytest.mark.asyncio
async def test_a_deep_linked_run_seeds_the_inspector_from_the_mount_window():
    """Qodo, PR #1331: the selection watchers are in the mount-window class too.

    The deep link that carries a run id is the reachable path.
    `apply_navigation_context` sets `active_section = "runs"` **and**
    `_pending_navigation_run_id` on an unmounted screen; `on_mount` then starts
    `_load_runs`, which on a cold database finishes inside the mount window and
    calls `_select_entity(requested_run)` (`_load_runs`'s `had_pending_target`
    branch). That writes `selected_entity`, `watch_selected_entity` fires -- and
    an `is_mounted` guard drops the push while the Inspector is mounted and
    queryable.

    Nothing recovers it afterwards: `_build_inspector_pane` re-seeds only on a
    REBUILD, and the one rebuild this screen still schedules for the right rail
    is gated on `_resolve_console_follow_drift()`, which is False on a normal
    cold start. So the user follows a run deep link and the Inspector shows
    "Nothing to inspect yet." over a run the screen believes is selected.
    """
    run = {"id": "r1", "source_title": "Feed One", "status": "ok", "backend": "local"}
    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)
    screen._controller.list_runs = AsyncMock(return_value=[run])
    screen.apply_navigation_context({"section": "runs", "run_id": "r1"})
    assert screen.active_section == "runs"
    assert screen._pending_navigation_run_id == "r1", (
        "precondition: the deep link armed a run target pre-mount"
    )

    host = WatchlistsContextHarness(screen)
    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(300):
            await pilot.pause(0.01)
            if screen.query("#watchlists-runs-pane"):
                break
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        # Rewind to the instant `on_mount` fired: the deep-link target armed,
        # nothing selected yet, the Inspector mounted and empty.
        screen._pending_navigation_run_id = "r1"
        screen._pending_navigation_run_backend = "local"
        screen.selected_entity = None
        screen.selected_run = None
        inspector.selected_entity = None
        await pilot.pause()
        assert inspector.selected_entity is None, "precondition: Inspector rewound"

        screen._is_mounted = False
        try:
            await screen._load_runs()
            assert screen.selected_entity is not None, (
                "precondition: the deep link really did resolve to a run"
            )
            assert inspector.selected_entity == screen.selected_entity, (
                "the deep-linked run never reached the mounted Inspector: the "
                "selection landed while Textual still reported is_mounted=False "
                "and nothing re-seeds it"
            )
        finally:
            screen._is_mounted = True
