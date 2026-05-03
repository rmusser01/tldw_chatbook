"""Master shell destination wrapper tests."""

import pytest
from textual.app import App
from textual.widgets import Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.schedules_screen import SchedulesScreen
from tldw_chatbook.UI.Screens.watchlists_collections_screen import WatchlistsCollectionsScreen
from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen


SCREEN_BY_ROUTE = {
    "library": LibraryScreen,
    "artifacts": ArtifactsScreen,
    "personas": PersonasScreen,
    "watchlists_collections": WatchlistsCollectionsScreen,
    "schedules": SchedulesScreen,
    "workflows": WorkflowsScreen,
}


class DestinationHarness(App):
    def __init__(self, app_instance, route, seen_routes=None):
        super().__init__()
        self.app_instance = app_instance
        self.route = route
        self.seen_routes = seen_routes if seen_routes is not None else []

    async def on_mount(self) -> None:
        await self.push_screen(SCREEN_BY_ROUTE[self.route](self.app_instance))

    def on_navigate_to_screen(self, message) -> None:
        self.seen_routes.append(message.screen_name)


def _active_destination_screen(host: DestinationHarness):
    return host.screen_stack[-1]


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


@pytest.mark.parametrize(
    ("route", "title_id", "purpose_text"),
    [
        ("library", "#library-title", "source material"),
        ("artifacts", "#artifacts-title", "generated"),
        ("personas", "#personas-title", "behavior"),
    ],
)
@pytest.mark.asyncio
async def test_primary_destination_wrappers_mount(route, title_id, purpose_text):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        title = screen.query_one(title_id)
        assert title
        assert title.has_class("ds-destination-header")
        assert purpose_text in _static_text(screen.query_one(".destination-purpose", Static)).lower()
        assert screen.query_one(".ds-panel")


@pytest.mark.asyncio
async def test_library_exposes_source_sections_and_import_export_boundary():
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        for selector in [
            "#library-open-notes",
            "#library-open-media",
            "#library-open-conversations",
            "#library-open-import-export",
            "#library-open-search",
        ]:
            assert screen.query_one(selector)


@pytest.mark.parametrize(
    ("route", "selector", "target_route"),
    [
        ("library", "#library-open-notes", "notes"),
        ("library", "#library-open-media", "media"),
        ("library", "#library-open-conversations", "conversation"),
        ("library", "#library-open-import-export", "ingest"),
        ("library", "#library-open-search", "search"),
        ("artifacts", "#artifacts-open-chatbooks", "chatbooks"),
        ("personas", "#personas-open-profiles", "ccp"),
        ("watchlists_collections", "#wc-open-watchlists", "subscriptions"),
        ("schedules", "#schedules-open-console", "chat"),
        ("workflows", "#workflows-launch-console", "chat"),
    ],
)
@pytest.mark.asyncio
async def test_destination_action_buttons_emit_compatibility_routes(route, selector, target_route):
    app = _build_test_app()
    seen_routes = []
    host = DestinationHarness(app, route, seen_routes)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click(selector)
        await pilot.pause(0.1)

    assert seen_routes[-1] == target_route


@pytest.mark.parametrize(
    ("route", "expected_sections"),
    [
        ("watchlists_collections", ["Watchlists", "Collections"]),
        ("schedules", ["Next Run", "Paused", "Failed"]),
        ("workflows", ["Recipes", "Dry Run", "Launch in Console"]),
    ],
)
@pytest.mark.asyncio
async def test_automation_destination_wrappers_explain_ownership(route, expected_sections):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        visible_text = " ".join(
            _static_text(widget)
            for widget in screen.query(Static)
            if hasattr(widget, "renderable")
        )
        for section in expected_sections:
            assert section in visible_text
