"""Visual parity geometry tests for destination correction pass."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button
from textual.widgets import Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticWatchlistsScopeService,
    StaticPersonasScopeService,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _wait_for_selector,
)
from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.MCP_Modules import unified_mcp_panel as unified_mcp_panel_module
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Screens import (
    artifacts_screen as artifacts_screen_module,
    library_screen as library_screen_module,
    personas_screen as personas_screen_module,
    schedules_screen as schedules_screen_module,
    skills_screen as skills_screen_module,
    watchlists_collections_screen as wc_screen_module,
    workflows_screen as workflows_screen_module,
)
from tldw_chatbook.Widgets.destination_workbench import DestinationWorkbench, WorkbenchPane


def _region(widget):
    region = widget.region
    return region.x, region.y, region.width, region.height


def _assert_no_horizontal_overlap(left, right, *, context: str) -> None:
    lx, ly, lw, lh = _region(left)
    rx, ry, rw, rh = _region(right)
    if ly + lh <= ry or ry + rh <= ly:
        return
    assert lx + lw <= rx or rx + rw <= lx, context


def _assert_visible_in_viewport(
    widget,
    *,
    height: int,
    context: str,
    viewport_width: int | None = None,
) -> None:
    x, y, widget_width, widget_height = _region(widget)
    assert x >= 0, context
    if viewport_width is not None:
        assert x < viewport_width, context
        assert x + widget_width <= viewport_width, context
    assert y >= 0, context
    assert y < height, context
    assert y + widget_height <= height, context


def _assert_strip_compact(screen, selector: str, *, max_height: int = 2) -> None:
    strip = screen.query_one(selector)
    assert strip.region.height <= max_height, f"{selector} is too tall: {strip.region}"


def _assert_horizontal_panes(screen, selectors: tuple[str, str, str]) -> None:
    panes = [screen.query_one(selector) for selector in selectors]
    assert panes[0].region.x < panes[1].region.x < panes[2].region.x
    assert panes[0].region.y == panes[1].region.y == panes[2].region.y
    for selector, pane in zip(selectors, panes):
        assert pane.region.width > 0, f"{selector} has no width"
        assert pane.region.height > 0, f"{selector} has no height"


def _assert_any_action_visible(
    screen,
    selectors: tuple[str, ...],
    *,
    height: int,
    context: str,
    viewport_width: int | None = None,
) -> None:
    for selector in selectors:
        matches = list(screen.query(selector))
        if not matches:
            continue
        try:
            _assert_visible_in_viewport(
                matches[0],
                height=height,
                context=f"{context}:{selector}",
                viewport_width=viewport_width,
            )
            return
        except AssertionError:
            continue
    raise AssertionError(f"{context} has no visible action/recovery path from {selectors!r}")


def _assert_marker_inside_container(screen, marker: str, container: str, *, context: str) -> None:
    marker_widget = screen.query_one(marker)
    container_region = screen.query_one(container).region
    assert marker_widget.region.x >= container_region.x, context
    assert marker_widget.region.y >= container_region.y, context
    assert marker_widget.region.x < container_region.x + container_region.width, context
    assert marker_widget.region.y < container_region.y + container_region.height, context


def _assert_any_marker_inside_container(
    screen,
    markers: tuple[str, ...],
    container: str,
    *,
    context: str,
) -> None:
    for marker in markers:
        if list(screen.query(marker)):
            _assert_marker_inside_container(screen, marker, container, context=context)
            return
    raise AssertionError(f"{context} missing expected marker from {markers!r}")


def _assert_ascii_workbench_contract(
    screen,
    *,
    workbench: str,
    panes: tuple[str, str, str],
    strip: str | None = None,
    actions: tuple[str, ...] = (),
    height: int = 42,
    start_by: int = 12,
    min_pane_rows: int = 20,
) -> None:
    """Assert the rendered layout matches the ASCII list/detail/inspector contract."""
    if strip is not None:
        _assert_strip_compact(screen, strip)
    workbench_widget = screen.query_one(workbench)
    assert workbench_widget.region.y <= start_by, f"{workbench} starts too low: {workbench_widget.region}"
    _assert_visible_in_viewport(workbench_widget, height=height, context=workbench)
    _assert_horizontal_panes(screen, panes)
    for selector in panes:
        pane = screen.query_one(selector)
        assert pane.region.height >= min_pane_rows, f"{selector} is too short: {pane.region}"
        _assert_visible_in_viewport(pane, height=height, context=selector)
    if actions:
        _assert_any_action_visible(screen, actions, height=height, context=workbench)


def _visible_static_text(screen) -> str:
    return " ".join(
        getattr(widget.renderable, "plain", str(widget.renderable))
        for widget in screen.query(Static)
        if widget.display and hasattr(widget, "renderable")
    )


def _visible_button_labels(screen) -> set[str]:
    return {str(button.label) for button in screen.query(Button) if button.display}


class StaticArtifactsChatbookService:
    def __init__(self, chatbooks):
        self.chatbooks = tuple(chatbooks)

    async def list_chatbooks(self, *, q=None, limit=100, offset=0, **kwargs):
        return list(self.chatbooks)[int(offset) : int(offset) + int(limit)]


@pytest.mark.asyncio
async def test_main_navigation_overflow_hint_does_not_overlap_settings_at_default_size():
    app = _build_test_app()
    host = HomeHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-dashboard")
        nav = home.query_one(MainNavigationBar)
        settings = nav.query_one("#nav-settings", Button)
        more = nav.query_one("#nav-overflow-hint")
        _assert_no_horizontal_overlap(settings, more, context="More hint overlaps Settings nav item")


@pytest.mark.asyncio
async def test_destination_content_starts_immediately_below_nav():
    app = _build_test_app()
    host = HomeHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-dashboard")
        content = home.query_one("#screen-content")
        dashboard = home.query_one("#home-dashboard")
        assert content.region.y == 3
        assert dashboard.region.y <= 4


class WorkbenchHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield DestinationWorkbench(
            WorkbenchPane("List", Static("left"), id="test-list-pane"),
            WorkbenchPane("Detail", Static("center"), id="test-detail-pane"),
            WorkbenchPane("Inspector", Static("right"), id="test-inspector-pane"),
            id="test-workbench",
        )


@pytest.mark.asyncio
async def test_destination_workbench_renders_three_horizontal_panes():
    app = WorkbenchHarness()
    async with app.run_test(size=(100, 20)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#test-workbench")
        left = app.query_one("#test-list-pane")
        center = app.query_one("#test-detail-pane")
        right = app.query_one("#test-inspector-pane")
        assert left.region.x < center.region.x < right.region.x
        assert left.region.y == center.region.y == right.region.y


@pytest.mark.asyncio
async def test_home_dashboard_regions_fit_default_viewport():
    app = _build_test_app()
    host = HomeHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-dashboard-grid")
        assert home.query_one("#home-dashboard-grid").region.y <= 12
        _assert_horizontal_panes(
            home,
            ("#home-attention-queue", "#home-active-work-region", "#home-inspector"),
        )
        for selector in (
            "#home-dashboard-grid",
            "#home-next-actions-region",
            "#home-recent-work-region",
        ):
            _assert_visible_in_viewport(home.query_one(selector), height=42, context=selector)
        _assert_any_action_visible(
            home,
            (
                "#home-primary-action",
                "#home-open-details",
                "#home-open-in-console",
                "#home-open-chatbook-details",
                "#home-open-chatbook-in-console",
            ),
            height=42,
            context="home",
        )


@pytest.mark.asyncio
async def test_console_uses_three_pane_workbench_and_visible_composer():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")
        _assert_strip_compact(console, "#console-control-bar", max_height=3)
        _assert_ascii_workbench_contract(
            console,
            workbench="#console-workspace-grid",
            panes=("#console-staged-context-tray", "#console-main-column", "#console-run-inspector"),
            actions=("#console-send-message", "#console-attach-context", "#console-save-chatbook"),
            height=42,
        )
        transcript = console.query_one("#console-session-surface")
        composer = console.query_one("#console-native-composer")
        _assert_visible_in_viewport(transcript, height=42, context="Console transcript")
        _assert_visible_in_viewport(composer, height=42, context="Console composer")


@pytest.mark.asyncio
async def test_library_mode_strip_is_compact_and_workbench_visible():
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-contract-grid")
        _assert_ascii_workbench_contract(
            library,
            workbench="#library-contract-grid",
            panes=("#library-source-browser", "#library-source-detail", "#library-source-inspector"),
            strip="#library-mode-bar",
            actions=("#library-open-notes", "#library-open-media", "#library-open-search", "#library-use-in-console"),
            height=42,
        )


@pytest.mark.asyncio
async def test_library_mode_strip_keeps_all_mode_chips_visible():
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-contract-grid")
        mode_bar = library.query_one("#library-mode-bar")
        mode_label = library.query_one("#library-mode-label")
        assert mode_bar.region.height <= 2
        assert mode_label.region.width <= 8
        for button in library.query(".library-mode-chip"):
            assert button.region.x >= mode_bar.region.x
            assert button.region.x + button.region.width <= mode_bar.region.x + mode_bar.region.width
            assert button.region.y >= mode_bar.region.y
            assert button.region.y + button.region.height <= mode_bar.region.y + mode_bar.region.height


@pytest.mark.asyncio
async def test_library_workbench_renders_terminal_borders():
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-contract-grid")
        for selector in (
            "#library-contract-grid",
            "#library-source-browser",
            "#library-source-detail",
            "#library-source-inspector",
        ):
            widget = library.query_one(selector)
            assert widget.styles.border_top[0], f"{selector} has no top border"
            assert widget.styles.padding.top >= 1, f"{selector} needs readable pane padding"


@pytest.mark.asyncio
async def test_library_empty_state_reports_empty_with_next_action():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-source-empty")

        status_row = str(library.query_one("#library-status-row", Static).renderable)
        visible_text = " ".join(str(widget.renderable) for widget in library.query(Static))

    assert "Empty" in status_row
    assert "Ready" not in status_row
    assert "Import/Export Sources or Open Notes/Media to add content." in visible_text


@pytest.mark.asyncio
async def test_library_inspector_uses_empty_state_until_item_selected():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-source-empty")

        inspector_title = str(library.query_one("#library-inspector-title", Static).renderable)
        inspector_text = " ".join(
            str(widget.renderable)
            for widget in library.query("#library-source-inspector Static")
        )
        has_source_authority = bool(list(library.query("#library-source-authority")))

    assert inspector_title == "Inspector"
    assert "No source selected." in inspector_text
    assert "Select a note, media item, conversation, collection, or RAG result to inspect." in inspector_text
    assert "Source Inspector" not in inspector_text
    assert not has_source_authority


@pytest.mark.asyncio
async def test_library_source_browser_collections_action_switches_to_collections_mode():
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-open-collections")

        library.query_one("#library-open-collections", Button).press()
        await _wait_for_selector(library, pilot, "#library-collections-panel")

        active_mode_title = str(library.query_one("#library-active-mode-title", Static).renderable)
        active_chip = library.query_one("#library-mode-collections", Button)

    assert active_mode_title == "Collections mode"
    assert active_chip.has_class("is-active")


@pytest.mark.asyncio
async def test_library_source_browser_search_action_switches_to_search_mode():
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-open-search")

        library.query_one("#library-open-search", Button).press()
        await _wait_for_selector(library, pilot, "#library-search-rag-panel")

        active_mode_title = str(library.query_one("#library-active-mode-title", Static).renderable)
        active_chip = library.query_one("#library-mode-search", Button)
        inspector_title = str(library.query_one("#library-rag-inspector-title", Static).renderable)
        assert not list(library.query("#library-inspector-title"))

    assert active_mode_title == "Search/RAG mode"
    assert active_chip.has_class("is-active")
    assert inspector_title == "Retrieval Inspector"


@pytest.mark.asyncio
async def test_library_source_snapshot_times_out_to_stable_error(monkeypatch):
    class SlowNotesService:
        async def list_notes(self, **_kwargs):
            await asyncio.sleep(0.2)

    class SlowMediaService:
        async def list_media_items(self, **_kwargs):
            await asyncio.sleep(0.2)

    class SlowConversationService:
        async def list_conversations(self, **_kwargs):
            await asyncio.sleep(0.2)

    monkeypatch.setattr(
        library_screen_module,
        "LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS",
        0.01,
    )
    screen = library_screen_module.LibraryScreen(
        SimpleNamespace(
            notes_scope_service=SlowNotesService(),
            media_reading_scope_service=SlowMediaService(),
            chat_conversation_scope_service=SlowConversationService(),
            notes_user_id="default_user",
        )
    )

    start = time.perf_counter()
    records, counts, total_known, error, recovery_state = await screen._list_local_source_snapshot()
    elapsed = time.perf_counter() - start

    assert records == {"notes": (), "media": (), "conversations": ()}
    assert counts == {"notes": 0, "media": 0, "conversations": 0}
    assert total_known == {"notes": True, "media": True, "conversations": True}
    assert error == library_screen_module.LIBRARY_SERVICE_ERROR_COPY
    assert recovery_state is None
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_library_source_snapshot_timeout_handles_blocking_async_services(monkeypatch):
    class BlockingAsyncNotesService:
        async def list_notes(self, **_kwargs):
            time.sleep(0.2)
            return {"items": []}

    class BlockingAsyncMediaService:
        async def list_media_items(self, **_kwargs):
            time.sleep(0.2)
            return {"items": []}

    class BlockingAsyncConversationService:
        async def list_conversations(self, **_kwargs):
            time.sleep(0.2)
            return {"items": []}

    monkeypatch.setattr(
        library_screen_module,
        "LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS",
        0.01,
    )
    screen = library_screen_module.LibraryScreen(
        SimpleNamespace(
            notes_scope_service=BlockingAsyncNotesService(),
            media_reading_scope_service=BlockingAsyncMediaService(),
            chat_conversation_scope_service=BlockingAsyncConversationService(),
            notes_user_id="default_user",
        )
    )

    start = time.perf_counter()
    records, counts, total_known, error, recovery_state = await screen._list_local_source_snapshot()
    elapsed = time.perf_counter() - start

    assert records == {"notes": (), "media": (), "conversations": ()}
    assert counts == {"notes": 0, "media": 0, "conversations": 0}
    assert total_known == {"notes": True, "media": True, "conversations": True}
    assert error == library_screen_module.LIBRARY_SERVICE_ERROR_COPY
    assert recovery_state is None
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_library_service_call_awaits_coroutine_functions_without_worker(monkeypatch):
    async def async_service_call():
        return "direct-result"

    async def fail_to_thread(*_args, **_kwargs):  # pragma: no cover - failure path
        raise AssertionError("direct coroutine service calls should not use to_thread")

    monkeypatch.setattr(library_screen_module.asyncio, "to_thread", fail_to_thread)

    result = await library_screen_module.LibraryScreen._run_library_service_call(
        async_service_call
    )

    assert result == "direct-result"


@pytest.mark.parametrize(
    "route,host_factory,workbench,panes,actions,markers,marker_container",
    [
        (
            "chat",
            ConsoleHarness,
            "#console-workspace-grid",
            ("#console-staged-context-tray", "#console-main-column", "#console-run-inspector"),
            ("#console-send-message", "#console-attach-context", "#console-save-chatbook"),
            ("#console-run-inspector-state",),
            "#console-run-inspector",
        ),
        (
            "library",
            lambda app: DestinationHarness(app, "library"),
            "#library-contract-grid",
            ("#library-source-browser", "#library-source-detail", "#library-source-inspector"),
            ("#library-open-notes", "#library-open-media", "#library-open-search", "#library-use-in-console"),
            ("#library-source-empty", "#library-source-error", "#library-source-loading"),
            "#library-source-detail",
        ),
    ],
)
@pytest.mark.asyncio
async def test_core_default_empty_or_blocked_states_keep_workbench_geometry(
    route, host_factory, workbench, panes, actions, markers, marker_container
):
    app = _build_test_app()
    host = host_factory(app)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, workbench)
        _assert_ascii_workbench_contract(
            screen,
            workbench=workbench,
            panes=panes,
            actions=actions,
            height=42,
        )
        _assert_any_marker_inside_container(
            screen,
            markers,
            marker_container,
            context=f"{route} non-happy marker escaped workbench pane",
        )


@pytest.mark.asyncio
async def test_library_loading_state_preserves_workbench_geometry(monkeypatch):
    monkeypatch.setattr(
        library_screen_module.LibraryScreen,
        "_refresh_local_source_snapshot",
        lambda self: None,
    )
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-source-loading")
        _assert_ascii_workbench_contract(
            library,
            workbench="#library-contract-grid",
            panes=("#library-source-browser", "#library-source-detail", "#library-source-inspector"),
            strip="#library-mode-bar",
            actions=("#library-open-notes", "#library-open-media", "#library-open-search"),
            height=42,
        )
        _assert_marker_inside_container(
            library,
            "#library-source-loading",
            "#library-source-detail",
            context="Library loading state escaped source detail pane",
        )


@pytest.mark.asyncio
async def test_library_loading_state_fails_safe_when_snapshot_never_applies(monkeypatch):
    monkeypatch.setattr(
        library_screen_module,
        "LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        library_screen_module.LibraryScreen,
        "_refresh_local_source_snapshot",
        lambda self: None,
    )
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-source-error", timeout=1.0)
        assert not list(library.query("#library-source-loading"))
        _assert_marker_inside_container(
            library,
            "#library-source-error",
            "#library-source-detail",
            context="Library fallback error escaped source detail pane",
        )


SOURCE_PREP_WORKBENCHES = {
    "artifacts": {
        "workbench": "#artifacts-workbench",
        "strip": "#artifacts-mode-strip",
        "panes": ("#artifacts-list-pane", "#artifacts-detail-pane", "#artifacts-inspector-pane"),
        "actions": (
            "#artifacts-open-chatbooks",
            "#artifacts-open-console",
            "#artifacts-open-library",
            "#artifacts-import-artifact",
            "#artifacts-use-in-console",
        ),
        "markers": ("#artifacts-console-unavailable",),
        "marker_container": "#artifacts-inspector-pane",
    },
    "personas": {
        "workbench": "#personas-workbench",
        "strip": "#personas-mode-strip",
        "panes": ("#personas-list-pane", "#personas-detail-pane", "#personas-inspector-pane"),
        "actions": ("#personas-open-profiles", "#personas-attach-to-console"),
        "markers": ("#personas-empty-state", "#personas-service-error", "#personas-loading-state"),
        "marker_container": "#personas-detail-pane",
    },
    "watchlists_collections": {
        "workbench": "#watchlists-workbench",
        "strip": "#watchlists-filter-strip",
        "panes": ("#watchlists-list-pane", "#watchlists-detail-pane", "#watchlists-inspector-pane"),
        "actions": ("#wc-open-watchlists", "#wc-attach-to-console", "#watchlists-follow-in-console"),
        "markers": ("#wc-empty-state", "#wc-service-error", "#wc-loading-state"),
        "marker_container": "#watchlists-detail-pane",
    },
    "skills": {
        "workbench": "#skills-workbench",
        "strip": "#skills-mode-strip",
        "panes": ("#skills-list-pane", "#skills-detail-pane", "#skills-inspector-pane"),
        "actions": ("#skills-import-skill", "#skills-attach-to-console"),
        "markers": ("#skills-empty-state", "#skills-service-error", "#skills-loading-state"),
        "marker_container": "#skills-detail-pane",
    },
}


@pytest.mark.parametrize("route,contract", SOURCE_PREP_WORKBENCHES.items())
@pytest.mark.asyncio
async def test_source_prep_destinations_use_list_detail_inspector_workbench(route, contract):
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, contract["workbench"])
        _assert_ascii_workbench_contract(
            screen,
            workbench=contract["workbench"],
            strip=contract["strip"],
            panes=contract["panes"],
            actions=contract["actions"],
            height=42,
        )
        _assert_any_marker_inside_container(
            screen,
            contract["markers"],
            contract["marker_container"],
            context=f"{route} non-happy marker escaped workbench pane",
        )


@pytest.mark.parametrize("route,contract", SOURCE_PREP_WORKBENCHES.items())
@pytest.mark.asyncio
async def test_source_prep_default_empty_or_unavailable_states_preserve_workbench_geometry(route, contract):
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, contract["workbench"])
        _assert_ascii_workbench_contract(
            screen,
            workbench=contract["workbench"],
            strip=contract["strip"],
            panes=contract["panes"],
            actions=contract["actions"],
            height=42,
        )


@pytest.mark.asyncio
async def test_watchlists_screen_matches_approved_control_plane_columns():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wc-empty-state")

        assert (
            _visible_static_text(screen).find(
                "Watchlists | Monitored sources, runs, alerts, recovery | Mixed | Local/Server"
            )
            >= 0
        )
        visible_text = _visible_static_text(screen)
        assert "Filters: Running Failed Recent Alerts Sources Feeds" in visible_text
        assert "Column 1: Watchlist List" in visible_text
        assert "Column 2: Detail / Items / Runs" in visible_text
        assert "Column 3: Status Inspector" in visible_text
        assert "State:" in visible_text
        assert "Retry/backoff:" in visible_text
        assert "Collections" not in visible_text

        for selector in (
            "#watchlists-list-detail-divider",
            "#watchlists-detail-inspector-divider",
        ):
            divider = screen.query_one(selector)
            assert divider.has_class("destination-pane-divider")
            assert divider.region.width == 1


@pytest.mark.asyncio
async def test_schedules_screen_matches_approved_control_plane_columns():
    app = _build_test_app()
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#schedules-empty-state")

        visible_text = _visible_static_text(screen)
        for expected in (
            "Schedules | Jobs, digests, timers, retries | Local | Console handoff",
            "Filters: Next run Paused Failed Retry History",
            "Column 1: Schedule Queue",
            "Column 2: Run Detail / Output",
            "Column 3: Status Inspector",
            "State:",
            "Retry/backoff:",
            "Next action:",
            "Console: blocked",
        ):
            assert expected in visible_text

        for selector in (
            "#schedules-list-detail-divider",
            "#schedules-detail-inspector-divider",
        ):
            divider = screen.query_one(selector)
            assert divider.has_class("destination-pane-divider")
            assert divider.region.width == 1


@pytest.mark.asyncio
async def test_workflows_screen_matches_approved_procedure_columns():
    app = _build_test_app()
    host = DestinationHarness(app, "workflows")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#workflows-console-unavailable")

        visible_text = _visible_static_text(screen)
        for expected in (
            "Workflows | Procedures, runs, dry-runs, approvals | Local | Console handoff",
            "Modes: Recipes Inputs Steps Dry run Approvals Outputs",
            "Column 1: Procedure Library",
            "Column 2: Run Detail / Output",
            "Column 3: Run Inspector",
            "State: blocked",
            "Console: blocked",
            "Next action: start or select a workflow run",
        ):
            assert expected in visible_text

        for selector in (
            "#workflows-list-detail-divider",
            "#workflows-detail-inspector-divider",
        ):
            divider = screen.query_one(selector)
            assert divider.has_class("destination-pane-divider")
            assert divider.region.width == 1


@pytest.mark.asyncio
async def test_artifacts_empty_state_exposes_full_artifact_workbench_taxonomy():
    app = _build_test_app()
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-workbench")
        visible_text = _visible_static_text(screen)
        for expected in (
            "Types: All",
            "Chatbooks",
            "Reports",
            "Datasets",
            "Drafts",
            "Exports",
            "Sort: Recent",
            "Artifact List",
            "Artifact Preview",
            "Provenance",
        ):
            assert expected in visible_text


@pytest.mark.asyncio
async def test_personas_workbench_exposes_approved_three_column_ia():
    app = _build_test_app()
    app.character_persona_scope_service = StaticPersonasScopeService(
        characters=[
            {"name": "Research Analyst", "id": 1, "description": "Synthesizes evidence."},
        ],
        profiles=[
            {"name": "Fiction Character", "id": "profile-1", "description": "Roleplay voice."},
        ],
    )
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#personas-characters-summary")
        visible_text = _visible_static_text(screen)
        buttons = _visible_button_labels(screen)

        for expected in (
            "Personas | Behavior, characters, prompts, lore | Ready | Local/Server",
            "Modes: Personas | Characters | Prompts | Dictionaries | Lore | Import/Export",
            "Column 1: Persona List",
            "Column 2: Behavior Profile Detail",
            "Column 3: Attachments",
            "Research Analyst",
            "Fiction Character",
            "Console: ready",
            "Workflows: ready",
        ):
            assert expected in visible_text
        assert {"Open Personas", "Attach to Console"}.issubset(buttons)


@pytest.mark.asyncio
async def test_personas_workbench_has_explicit_column_dividers_for_future_resize():
    app = _build_test_app()
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#personas-workbench")
        left_pane = screen.query_one("#personas-list-pane")
        detail_pane = screen.query_one("#personas-detail-pane")
        inspector_pane = screen.query_one("#personas-inspector-pane")
        left_divider = screen.query_one("#personas-list-detail-divider")
        right_divider = screen.query_one("#personas-detail-inspector-divider")

        assert "destination-pane-divider" in left_divider.classes
        assert "destination-pane-divider" in right_divider.classes
        assert left_pane.region.x < left_divider.region.x < detail_pane.region.x
        assert detail_pane.region.x < right_divider.region.x < inspector_pane.region.x
        assert left_divider.region.width == 1
        assert right_divider.region.width == 1


@pytest.mark.asyncio
async def test_artifacts_empty_state_labels_three_clear_columns():
    app = _build_test_app()
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-workbench")
        visible_text = _visible_static_text(screen)
        for expected in (
            "Column 1: Artifact List",
            "Column 2: Artifact Preview / Detail",
            "Column 3: Provenance",
        ):
            assert expected in visible_text


@pytest.mark.asyncio
async def test_artifacts_empty_state_keeps_console_library_import_recovery_visible():
    app = _build_test_app()
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-workbench")
        labels = _visible_button_labels(screen)
        assert "Open Console" in labels
        assert "Open Library" in labels
        assert "Import Artifact" in labels
        assert list(screen.query("#artifacts-open-console"))


@pytest.mark.asyncio
async def test_artifacts_dynamic_metadata_renders_markup_as_literal_text():
    app = _build_test_app()
    app.local_chatbook_service = StaticArtifactsChatbookService(
        (
            {
                "chatbook_id": 9,
                "id": "9",
                "name": "[red]Markup Title[/red]",
                "description": "[bold]Description[/bold]",
                "updated_at": "2026-05-09T20:00:00Z",
                "metadata": {
                    "artifact_source": "console",
                    "artifact_kind": "assistant-response",
                    "content": "[green]Preview[/green]",
                },
            },
        )
    )
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-console-available")
        visible_text = _visible_static_text(screen)
        assert "Title: [red]Markup Title[/red]" in visible_text
        assert "[bold]Description[/bold]" in visible_text
        assert "Transcript preview: [green]Preview[/green]" in visible_text


SOURCE_PREP_LOADING_CONTRACTS = [
    (
        "artifacts",
        artifacts_screen_module.ArtifactsScreen,
        "_refresh_latest_chatbook_context",
        "#artifacts-loading-state",
        SOURCE_PREP_WORKBENCHES["artifacts"],
        "#artifacts-detail-pane",
    ),
    (
        "personas",
        personas_screen_module.PersonasScreen,
        "_refresh_local_behavior_snapshot",
        "#personas-loading-state",
        SOURCE_PREP_WORKBENCHES["personas"],
        "#personas-detail-pane",
    ),
    (
        "watchlists_collections",
        wc_screen_module.WatchlistsCollectionsScreen,
        "_refresh_local_wc_snapshot",
        "#wc-loading-state",
        SOURCE_PREP_WORKBENCHES["watchlists_collections"],
        "#watchlists-detail-pane",
    ),
    (
        "skills",
        skills_screen_module.SkillsScreen,
        "_refresh_local_skills_context",
        "#skills-loading-state",
        SOURCE_PREP_WORKBENCHES["skills"],
        "#skills-detail-pane",
    ),
]


@pytest.mark.parametrize(
    "route,screen_cls,refresh_method,loading_marker,contract,loading_container",
    SOURCE_PREP_LOADING_CONTRACTS,
)
@pytest.mark.asyncio
async def test_source_prep_loading_states_preserve_workbench_geometry(
    monkeypatch,
    route,
    screen_cls,
    refresh_method,
    loading_marker,
    contract,
    loading_container,
):
    monkeypatch.setattr(screen_cls, refresh_method, lambda self: None)
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, loading_marker)
        _assert_ascii_workbench_contract(
            screen,
            workbench=contract["workbench"],
            strip=contract["strip"],
            panes=contract["panes"],
            actions=contract["actions"],
            height=42,
        )
        _assert_marker_inside_container(
            screen,
            loading_marker,
            loading_container,
            context=f"{route} loading state escaped workbench geometry",
        )


@pytest.mark.parametrize(
    "route,strip,workbench,panes,actions",
    [
        (
            "schedules",
            "#schedules-filter-strip",
            "#schedules-workbench",
            ("#schedules-list-pane", "#schedules-detail-pane", "#schedules-inspector-pane"),
            ("#schedules-follow-in-console",),
        ),
        (
            "workflows",
            "#workflows-mode-strip",
            "#workflows-workbench",
            ("#workflows-list-pane", "#workflows-detail-pane", "#workflows-inspector-pane"),
            ("#workflows-launch-in-console",),
        ),
    ],
)
@pytest.mark.asyncio
async def test_operational_destinations_use_timing_or_procedure_workbench(
    route, strip, workbench, panes, actions
):
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, workbench)
        _assert_ascii_workbench_contract(
            screen,
            workbench=workbench,
            strip=strip,
            panes=panes,
            actions=actions,
            height=42,
        )


@pytest.mark.parametrize(
    "route,strip,workbench,panes,actions,markers,marker_container",
    [
        (
            "schedules",
            "#schedules-filter-strip",
            "#schedules-workbench",
            ("#schedules-list-pane", "#schedules-detail-pane", "#schedules-inspector-pane"),
            ("#schedules-follow-in-console",),
            ("#schedules-empty-state", "#schedules-console-unavailable"),
            "#schedules-detail-pane",
        ),
        (
            "workflows",
            "#workflows-mode-strip",
            "#workflows-workbench",
            ("#workflows-list-pane", "#workflows-detail-pane", "#workflows-inspector-pane"),
            ("#workflows-launch-in-console",),
            ("#workflows-console-unavailable",),
            "#workflows-detail-pane",
        ),
    ],
)
@pytest.mark.asyncio
async def test_operational_empty_or_blocked_states_preserve_workbench_geometry(
    route, strip, workbench, panes, actions, markers, marker_container
):
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, workbench)
        _assert_ascii_workbench_contract(
            screen,
            workbench=workbench,
            strip=strip,
            panes=panes,
            actions=actions,
            height=42,
        )
        _assert_any_marker_inside_container(
            screen,
            markers,
            marker_container,
            context=f"{route} non-happy marker escaped workbench pane",
        )


OPERATIONAL_LOADING_CONTRACTS = [
    (
        "schedules",
        schedules_screen_module.SchedulesScreen,
        "_refresh_latest_console_context",
        "#schedules-loading-state",
        "#schedules-detail-pane",
        "#schedules-filter-strip",
        "#schedules-workbench",
        ("#schedules-list-pane", "#schedules-detail-pane", "#schedules-inspector-pane"),
        ("#schedules-follow-in-console",),
    ),
    (
        "workflows",
        workflows_screen_module.WorkflowsScreen,
        "_refresh_latest_console_context",
        "#workflows-loading-state",
        "#workflows-detail-pane",
        "#workflows-mode-strip",
        "#workflows-workbench",
        ("#workflows-list-pane", "#workflows-detail-pane", "#workflows-inspector-pane"),
        ("#workflows-launch-in-console",),
    ),
]


@pytest.mark.parametrize(
    "route,screen_cls,refresh_method,loading_marker,loading_container,strip,workbench,panes,actions",
    OPERATIONAL_LOADING_CONTRACTS,
)
@pytest.mark.asyncio
async def test_operational_loading_states_preserve_workbench_geometry(
    monkeypatch,
    route,
    screen_cls,
    refresh_method,
    loading_marker,
    loading_container,
    strip,
    workbench,
    panes,
    actions,
):
    monkeypatch.setattr(screen_cls, refresh_method, lambda self: None)
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, loading_marker)
        _assert_ascii_workbench_contract(
            screen,
            workbench=workbench,
            strip=strip,
            panes=panes,
            actions=actions,
            height=42,
        )
        _assert_marker_inside_container(
            screen,
            loading_marker,
            loading_container,
            context=f"{route} loading state escaped workbench geometry",
        )


@pytest.mark.asyncio
async def test_mcp_uses_visible_server_detail_readiness_layout_without_overflow():
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#mcp-workbench")
        _assert_strip_compact(screen, "#mcp-title", max_height=1)
        _assert_strip_compact(screen, "#mcp-purpose", max_height=1)
        _assert_ascii_workbench_contract(
            screen,
            workbench="#mcp-workbench",
            strip="#mcp-mode-strip",
            panes=("#mcp-server-tree-pane", "#mcp-detail-pane", "#mcp-readiness-pane"),
            actions=("#unified-mcp-action-run",),
            height=42,
            min_pane_rows=30,
        )


@pytest.mark.asyncio
async def test_mcp_unavailable_or_local_default_state_keeps_workbench_geometry():
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#mcp-workbench")
        _assert_ascii_workbench_contract(
            screen,
            workbench="#mcp-workbench",
            strip="#mcp-mode-strip",
            panes=("#mcp-server-tree-pane", "#mcp-detail-pane", "#mcp-readiness-pane"),
            actions=("#unified-mcp-action-run",),
            height=42,
        )
        _assert_marker_inside_container(
            screen,
            "#unified-mcp-content",
            "#mcp-detail-pane",
            context="MCP loading/status content escaped detail pane",
        )


@pytest.mark.asyncio
async def test_mcp_forced_loading_state_stays_inside_workbench(monkeypatch):
    async def keep_initial_loading_state(self):
        return self.context

    monkeypatch.setattr(
        unified_mcp_panel_module.UnifiedMCPPanel,
        "load_context",
        keep_initial_loading_state,
    )
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#unified-mcp-content")
        _assert_ascii_workbench_contract(
            screen,
            workbench="#mcp-workbench",
            strip="#mcp-mode-strip",
            panes=("#mcp-server-tree-pane", "#mcp-detail-pane", "#mcp-readiness-pane"),
            actions=("#unified-mcp-action-run",),
            height=42,
        )
        _assert_marker_inside_container(
            screen,
            "#unified-mcp-content",
            "#mcp-detail-pane",
            context="MCP forced loading state escaped detail pane",
        )


@pytest.mark.parametrize(
    "route,strip,workbench,panes,actions",
    [
        (
            "acp",
            "#acp-mode-strip",
            "#acp-workbench",
            ("#acp-list-pane", "#acp-detail-pane", "#acp-inspector-pane"),
            ("#acp-follow-in-console", "#acp-launch-agent"),
        ),
        (
            "settings",
            "#settings-category-strip",
            "#settings-workbench",
            ("#settings-category-pane", "#settings-detail-pane", "#settings-impact-pane"),
            ("#settings-open-appearance",),
        ),
    ],
)
@pytest.mark.asyncio
async def test_runtime_and_settings_destinations_use_pane_layouts(
    route, strip, workbench, panes, actions
):
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, workbench)
        _assert_ascii_workbench_contract(
            screen,
            workbench=workbench,
            strip=strip,
            panes=panes,
            actions=actions,
            height=42,
        )


@pytest.mark.parametrize(
    "route,strip,workbench,panes,actions,markers,marker_container",
    [
        (
            "acp",
            "#acp-mode-strip",
            "#acp-workbench",
            ("#acp-list-pane", "#acp-detail-pane", "#acp-inspector-pane"),
            ("#acp-follow-in-console", "#acp-launch-agent"),
            ("#acp-empty-state", "#acp-console-unavailable"),
            "#acp-detail-pane",
        ),
        (
            "settings",
            "#settings-category-strip",
            "#settings-workbench",
            ("#settings-category-pane", "#settings-detail-pane", "#settings-impact-pane"),
            ("#settings-open-appearance",),
            ("#settings-boundary-note",),
            "#settings-impact-pane",
        ),
    ],
)
@pytest.mark.asyncio
async def test_runtime_and_settings_default_states_preserve_workbench_geometry(
    route, strip, workbench, panes, actions, markers, marker_container
):
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, workbench)
        _assert_ascii_workbench_contract(
            screen,
            workbench=workbench,
            strip=strip,
            panes=panes,
            actions=actions,
            height=42,
        )
        _assert_any_marker_inside_container(
            screen,
            markers,
            marker_container,
            context=f"{route} non-happy marker escaped workbench pane",
        )


@pytest.mark.asyncio
async def test_acp_runtime_blocked_state_uses_setup_and_compatibility_columns():
    app = _build_test_app()
    host = DestinationHarness(app, "acp")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#acp-workbench")
        _assert_ascii_workbench_contract(
            screen,
            workbench="#acp-workbench",
            strip="#acp-mode-strip",
            panes=("#acp-list-pane", "#acp-detail-pane", "#acp-inspector-pane"),
            actions=("#acp-follow-in-console", "#acp-launch-agent"),
            height=42,
            start_by=8,
            min_pane_rows=26,
        )
        visible_text = _visible_static_text(screen)
        assert "Agents / Sessions" in visible_text
        assert "Session Detail / Runtime Setup" in visible_text
        assert "Compatibility / Actions" in visible_text
        assert "Runtime owner: ACP" in visible_text
        assert "ACP version: n/a" in visible_text
        runtime_copy = str(screen.query_one("#acp-empty-state").renderable)
        assert "Settings" not in runtime_copy
        assert "Configure ACP runtime setup in ACP" in runtime_copy


COMPACT_DESTINATION_CONTRACTS = {
    "home": {
        "identity": "#home-title",
        "workbench": "#home-dashboard-grid",
        "object": "#home-attention-queue",
        "detail": "#home-active-work-region",
        "actions": ("#home-primary-action", "#home-open-details", "#home-open-chatbook-details"),
    },
    "chat": {
        "identity": "#console-title",
        "workbench": "#console-workspace-grid",
        "object": "#console-staged-context-tray",
        "detail": "#console-session-surface",
        "actions": ("#console-send-message", "#console-attach-context", "#console-save-chatbook"),
    },
    "library": {
        "identity": "#library-title",
        "workbench": "#library-contract-grid",
        "object": "#library-source-browser",
        "detail": "#library-source-detail",
        "actions": ("#library-open-search", "#library-use-in-console", "#library-open-notes"),
    },
    "artifacts": {
        "identity": "#artifacts-title",
        "workbench": "#artifacts-workbench",
        "object": "#artifacts-list-pane",
        "detail": "#artifacts-detail-pane",
        "actions": (
            "#artifacts-open-chatbooks",
            "#artifacts-open-console",
            "#artifacts-open-library",
            "#artifacts-import-artifact",
            "#artifacts-use-in-console",
        ),
    },
    "personas": {
        "identity": "#personas-title",
        "workbench": "#personas-workbench",
        "object": "#personas-list-pane",
        "detail": "#personas-detail-pane",
        "actions": ("#personas-open-profiles", "#personas-attach-to-console"),
    },
    "watchlists_collections": {
        "identity": "#watchlists-collections-title",
        "workbench": "#watchlists-workbench",
        "object": "#watchlists-list-pane",
        "detail": "#watchlists-detail-pane",
        "actions": ("#wc-open-watchlists", "#watchlists-follow-in-console"),
    },
    "schedules": {
        "identity": "#schedules-title",
        "workbench": "#schedules-workbench",
        "object": "#schedules-list-pane",
        "detail": "#schedules-detail-pane",
        "actions": ("#schedules-follow-in-console",),
    },
    "workflows": {
        "identity": "#workflows-title",
        "workbench": "#workflows-workbench",
        "object": "#workflows-list-pane",
        "detail": "#workflows-detail-pane",
        "actions": ("#workflows-launch-in-console",),
    },
    "mcp": {
        "identity": "#mcp-title",
        "workbench": "#mcp-workbench",
        "object": "#mcp-server-tree-pane",
        "detail": "#mcp-detail-pane",
        "actions": ("#unified-mcp-action-run",),
    },
    "acp": {
        "identity": "#acp-title",
        "workbench": "#acp-workbench",
        "object": "#acp-list-pane",
        "detail": "#acp-detail-pane",
        "actions": ("#acp-follow-in-console", "#acp-launch-agent"),
    },
    "skills": {
        "identity": "#skills-title",
        "workbench": "#skills-workbench",
        "object": "#skills-list-pane",
        "detail": "#skills-detail-pane",
        "actions": ("#skills-import-skill", "#skills-attach-to-console"),
    },
    "settings": {
        "identity": "#settings-title",
        "workbench": "#settings-workbench",
        "object": "#settings-category-pane",
        "detail": "#settings-detail-pane",
        "actions": ("#settings-open-appearance",),
    },
}


TOP_LEVEL_WORKBENCH_SELECTORS = {
    route: contract["workbench"] for route, contract in COMPACT_DESTINATION_CONTRACTS.items()
}


@pytest.mark.parametrize("route,contract", COMPACT_DESTINATION_CONTRACTS.items())
@pytest.mark.asyncio
async def test_top_level_destinations_keep_primary_workbench_visible_at_compact_size(route, contract):
    app = _build_test_app()
    if route == "home":
        host = HomeHarness(app)
    elif route == "chat":
        host = ConsoleHarness(app)
    else:
        host = DestinationHarness(app, route)
    async with host.run_test(size=(100, 32)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, contract["workbench"])
        nav = screen.query_one(MainNavigationBar)
        assert nav.region.y == 0, f"{route}: global nav is not docked at top: {nav.region}"
        assert nav.region.height <= 3, f"{route}: global nav is too tall: {nav.region}"
        _assert_visible_in_viewport(nav, height=32, context=f"{route}:global-nav", viewport_width=100)
        assert list(nav.query(Button)), f"{route}: global nav has no visible destination buttons"
        for required in ("identity", "workbench", "object", "detail"):
            _assert_visible_in_viewport(
                screen.query_one(contract[required]),
                height=32,
                context=f"{route}:{required}:{contract[required]}",
                viewport_width=100,
            )
        _assert_any_action_visible(
            screen,
            contract["actions"],
            height=32,
            context=f"{route}:compact-action",
            viewport_width=100,
        )


VISIBLE_FOCUS_TARGETS = {
    "home": {"home-primary-action", "home-open-details", "home-open-in-console", "home-open-chatbook-details"},
    "chat": {"console-send-message", "console-attach-context", "console-save-chatbook", "console-run-library-rag"},
    "library": {"library-open-notes", "library-open-media", "library-open-search", "library-use-in-console"},
    "artifacts": {
        "artifacts-open-chatbooks",
        "artifacts-open-console",
        "artifacts-open-library",
        "artifacts-import-artifact",
        "artifacts-use-in-console",
    },
    "personas": {"personas-open-profiles", "personas-attach-to-console"},
    "watchlists_collections": {"wc-open-watchlists", "wc-attach-to-console", "watchlists-follow-in-console"},
    "schedules": {"schedules-follow-in-console"},
    "workflows": {"workflows-launch-in-console"},
    "mcp": {"unified-mcp-action-run"},
    "acp": {"acp-follow-in-console", "acp-launch-agent"},
    "skills": {"skills-import-skill", "skills-attach-to-console"},
    "settings": {"settings-open-appearance"},
}


@pytest.mark.parametrize("route,targets", VISIBLE_FOCUS_TARGETS.items())
@pytest.mark.asyncio
async def test_tab_order_reaches_visible_primary_action(route, targets):
    app = _build_test_app()
    if route == "home":
        host = HomeHarness(app)
    elif route == "chat":
        host = ConsoleHarness(app)
    else:
        host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = host.screen_stack[-1]
        workbench = TOP_LEVEL_WORKBENCH_SELECTORS[route]
        await _wait_for_selector(screen, pilot, workbench)
        target_buttons = [
            screen.query_one(f"#{target}", Button)
            for target in targets
            if list(screen.query(f"#{target}"))
        ]
        enabled_targets = {button.id for button in target_buttons if button.id and not button.disabled}
        if not enabled_targets:
            _assert_any_action_visible(
                screen,
                tuple(f"#{target}" for target in targets),
                height=42,
                context=f"{route}:disabled-recovery-action",
                viewport_width=140,
            )
            return
        for _ in range(24):
            await pilot.press("tab")
            focused = host.focused
            if focused is not None and focused.id in enabled_targets:
                _assert_visible_in_viewport(
                    focused,
                    height=42,
                    context=f"{route}:{focused.id} focused below viewport",
                    viewport_width=140,
                )
                return
        pytest.fail(f"{route} did not focus a visible primary action from {sorted(enabled_targets)}")
