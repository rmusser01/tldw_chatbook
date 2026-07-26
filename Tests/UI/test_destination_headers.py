"""DestinationHeader identity coverage for folded/orphan shell screens.

Every screen folded under a shell destination (or seated in Lab/Settings)
mounts the shared DestinationHeader component with its own identity: plain
screen-name title, short purpose subtitle, and a text-labeled status badge.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.UI.Workbench.workbench_widgets import DestinationHeader
from Tests.UI.test_screen_navigation import _build_test_app


# route -> (screen import path pieces resolved lazily in the test), title
#
# "evals" is deliberately absent (PR3a Task 3): it moved from the flat
# compose_content() pattern this list assumes -- DestinationHeader as
# widgets[0], callable via list(screen.compose_content()) with no running
# app -- to the three-pane workbench shell, which wraps everything in
# with Vertical(id="evals-shell"): ..., and entering that context manager
# requires an active Textual app. This mirrors why skills/mcp/personas/
# watchlists_collections (already on the workbench shell pattern) were
# never in this list either. See
# test_evals_screen_composes_destination_header_in_the_workbench_shell
# below for the equivalent header-identity coverage through a real
# running app.
_SIMPLE_SCREEN_ROUTES = (
    ("search", "Search"),
    ("media", "Media"),
    ("writing", "Writing"),
    ("llm", "Models"),
    ("stts", "Speech"),
    ("logs", "Logs"),
    ("stats", "Stats"),
)


def _screen_for_route(route: str, app):
    if route == "search":
        from tldw_chatbook.UI.Screens.search_screen import SearchScreen

        return SearchScreen(app)
    if route == "media":
        from tldw_chatbook.UI.Screens.media_screen import MediaScreen

        return MediaScreen(app)
    if route == "writing":
        from tldw_chatbook.UI.Screens.writing_screen import WritingScreen

        return WritingScreen(app)
    if route == "llm":
        from tldw_chatbook.UI.Screens.llm_screen import LLMScreen

        return LLMScreen(app)
    if route == "stts":
        from tldw_chatbook.UI.Screens.stts_screen import STTSScreen

        return STTSScreen(app)
    if route == "logs":
        from tldw_chatbook.UI.Screens.logs_screen import LogsScreen

        return LogsScreen(app)
    if route == "stats":
        from tldw_chatbook.UI.Screens.stats_screen import StatsScreen

        return StatsScreen(app)
    raise AssertionError(f"unmapped route: {route}")


@pytest.mark.parametrize(("route", "expected_title"), _SIMPLE_SCREEN_ROUTES)
def test_folded_screen_composes_destination_header_first(route, expected_title):
    app = _build_test_app()
    screen = _screen_for_route(route, app)

    widgets = list(screen.compose_content())

    header = widgets[0]
    assert isinstance(header, DestinationHeader), route
    assert header.id == f"{route}-destination-header"
    assert header.has_class("workbench-header")
    assert header.has_class("ds-destination-header")
    assert header.state.title == expected_title
    # Plain purpose copy, kept short, with no em dashes.
    assert header.state.subtitle
    assert len(header.state.subtitle) <= 60
    assert "—" not in header.state.subtitle
    assert "--" not in header.state.subtitle
    # States are text-labeled, never color-only.
    assert header.state.status == "ready"


@pytest.mark.asyncio
async def test_evals_screen_composes_destination_header_in_the_workbench_shell():
    """Equivalent of test_folded_screen_composes_destination_header_first
    for Evals, which no longer fits that test's flat-compose_content()
    assumption (see the comment on _SIMPLE_SCREEN_ROUTES above) -- driven
    through a real running app instead, mirroring
    test_study_screen_mounts_destination_header_and_boxes_library below.
    """
    from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen

    class _EvalsHarness(App):
        def __init__(self, app_instance):
            super().__init__()
            self._app_instance = app_instance

        async def on_mount(self) -> None:
            await self.push_screen(EvalsScreen(self._app_instance))

    app_instance = SimpleNamespace(
        evaluation_orchestrator=None,
        notify=lambda *args, **kwargs: None,
    )
    app = _EvalsHarness(app_instance)

    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.1)
        screen = app.screen_stack[-1]

        header = screen.query_one("#evals-destination-header", DestinationHeader)
        assert header.has_class("workbench-header")
        assert header.has_class("ds-destination-header")
        title = screen.query_one(
            "#evals-destination-header #workbench-header-title", Static
        )
        assert str(title.renderable) == "Evals"
        subtitle = screen.query_one(
            "#evals-destination-header #workbench-header-subtitle", Static
        )
        subtitle_text = str(subtitle.renderable)
        # Plain purpose copy, kept short, with no em dashes.
        assert subtitle_text
        assert len(subtitle_text) <= 60
        assert "—" not in subtitle_text
        assert "--" not in subtitle_text
        # States are text-labeled, never color-only.
        status = screen.query_one(
            "#evals-destination-header #workbench-header-status", Static
        )
        assert str(status.renderable) == "Ready"


class _StudyHarness(App):
    def __init__(self, app_instance):
        super().__init__()
        self._screen = self._build_screen(app_instance)

    @staticmethod
    def _build_screen(app_instance):
        from tldw_chatbook.UI.Screens.study_screen import StudyScreen

        return StudyScreen(app_instance=app_instance)

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


@pytest.mark.asyncio
async def test_study_screen_mounts_destination_header_and_boxes_library():
    app_instance = SimpleNamespace(
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = _StudyHarness(app_instance)

    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = app.screen_stack[-1]

        header = screen.query_one("#study-destination-header", DestinationHeader)
        assert header.has_class("ds-destination-header")
        title = screen.query_one(
            "#study-destination-header #workbench-header-title", Static
        )
        assert str(title.renderable) == "Study"
        status = screen.query_one(
            "#study-destination-header #workbench-header-status", Static
        )
        assert str(status.renderable) == "Ready"

        # Folded under Library: the nav boxes Library while the header names
        # the actual screen.
        library_button = screen.query_one("#nav-library")
        assert library_button.has_class("is-active")


@pytest.mark.asyncio
async def test_folded_screens_box_owning_destination_in_nav():
    from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar

    class _Harness(App):
        def __init__(self, route):
            super().__init__()
            self._route = route

        def compose(self):
            yield MainNavigationBar(active=self._route, active_route=self._route)

    expectations = {
        "search": "nav-library",
        "media": "nav-library",
        "writing": "nav-library",
        "research": "nav-library",
        "llm": "nav-lab",
        "stts": "nav-lab",
        "evals": "nav-lab",
        "logs": "nav-logs",
        "stats": "nav-settings",
        "coding": "nav-console",
    }

    for route, expected_button_id in expectations.items():
        app = _Harness(route)
        async with app.run_test(size=(180, 20)) as pilot:
            await pilot.pause(0.1)
            assert app.query_one(f"#{expected_button_id}").has_class("is-active"), route
