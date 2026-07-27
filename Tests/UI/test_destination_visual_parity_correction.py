"""Visual parity geometry tests for destination correction pass."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Button
from textual.widgets import Collapsible
from textual.widgets import DataTable
from textual.widgets import Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticHomeActiveWorkAdapter,
    StaticWatchlistsScopeService,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _wait_for_selector,
)
from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _wait_for_library_shell,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Screens import (
    artifacts_screen as artifacts_screen_module,
    library_screen as library_screen_module,
    skills_screen as skills_screen_module,
    watchlists_collections_screen as wc_screen_module,
    workflows_screen as workflows_screen_module,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import NotificationsPane
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout
from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged
from tldw_chatbook.Widgets.destination_workbench import (
    DestinationWorkbench,
    WorkbenchPane,
)


class ProductionCSSDestinationHarness(DestinationHarness):
    """Mount one destination with the production stylesheet."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


class WatchlistsVisualHarness(ProductionCSSDestinationHarness):
    """Mount Watchlists with the production stylesheet for geometry checks."""


class SchedulesVisualHarness(ProductionCSSDestinationHarness):
    """Mount Schedules with the production stylesheet for geometry checks."""


def _visual_destination_harness(app, route: str) -> DestinationHarness:
    harness_type = {
        "watchlists_collections": WatchlistsVisualHarness,
        "schedules": SchedulesVisualHarness,
    }.get(route, DestinationHarness)
    return harness_type(app, route)


@pytest.fixture(autouse=True)
def _default_advanced_open(monkeypatch):
    """Task 5 (MCP Hub Phase 6): same rationale as test_mcp_workbench.py's
    fixture of the same name -- the MCP destination mounts a nested
    `MCPInspector`, whose `compose()` reads `mcp.hub_state.advanced_open`
    AND the new `mcp.hub_state.advanced_visible` opt-in via
    `mcp_inspector.get_cli_setting` at mount time. Without this, every MCP
    parity test here would hit the developer's real config
    (non-deterministic) and -- with `advanced_visible` defaulting False --
    `_assert_advanced_run_reachable()`'s `#mcp-adv-*` queries would find the
    opt-in reveal Button instead of the composed Advanced pane. The blanket
    True answers both keys (visible + expanded), matching the pre-Task-5
    layout these geometry assertions were written against.
    """
    import tldw_chatbook.UI.MCP_Modules.mcp_inspector as mcp_inspector_module

    monkeypatch.setattr(mcp_inspector_module, "get_cli_setting", lambda *a, **k: True)
    monkeypatch.setattr(
        mcp_inspector_module, "save_setting_to_cli_config", lambda *a, **k: True
    )


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


def _assert_horizontal_panes(screen, selectors: tuple[str, ...]) -> None:
    panes = [screen.query_one(selector) for selector in selectors]
    assert len(panes) >= 2
    for left, right in zip(panes, panes[1:]):
        assert left.region.x < right.region.x
    assert len({pane.region.y for pane in panes}) == 1
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
    raise AssertionError(
        f"{context} has no visible action/recovery path from {selectors!r}"
    )


def _assert_marker_inside_container(
    screen, marker: str, container: str, *, context: str
) -> None:
    marker_widget = screen.query_one(marker)
    container_region = screen.query_one(container).region
    assert marker_widget.region.x >= container_region.x, context
    assert marker_widget.region.y >= container_region.y, context
    assert marker_widget.region.x < container_region.x + container_region.width, context
    assert marker_widget.region.y < container_region.y + container_region.height, (
        context
    )


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
    strip_max_height: int = 2,
    actions: tuple[str, ...] = (),
    height: int = 42,
    start_by: int = 12,
    min_pane_rows: int = 20,
) -> None:
    """Assert the rendered layout matches the ASCII list/detail/inspector contract."""
    if strip is not None:
        _assert_strip_compact(screen, strip, max_height=strip_max_height)
    workbench_widget = screen.query_one(workbench)
    assert workbench_widget.region.y <= start_by, (
        f"{workbench} starts too low: {workbench_widget.region}"
    )
    _assert_visible_in_viewport(workbench_widget, height=height, context=workbench)
    _assert_horizontal_panes(screen, panes)
    for selector in panes:
        pane = screen.query_one(selector)
        assert pane.region.height >= min_pane_rows, (
            f"{selector} is too short: {pane.region}"
        )
        _assert_visible_in_viewport(pane, height=height, context=selector)
    if actions:
        _assert_any_action_visible(screen, actions, height=height, context=workbench)


def _visible_static_text(screen) -> str:
    return " ".join(
        getattr(widget.renderable, "plain", str(widget.renderable))
        for widget in screen.query(Static)
        if widget.display and hasattr(widget, "renderable")
    )


def _visible_workbench_pane_titles(screen, workbench: str) -> list[str]:
    workbench_widget = screen.query_one(workbench)
    titles = []
    for widget in workbench_widget.query(Static):
        if not widget.display or not hasattr(widget, "renderable"):
            continue
        if not any(
            str(class_name).endswith("-column-title") for class_name in widget.classes
        ):
            continue
        renderable = widget.renderable
        titles.append(getattr(renderable, "plain", str(renderable)))
    return titles


def _visible_button_labels(screen) -> set[str]:
    return {str(button.label) for button in screen.query(Button) if button.display}


def _composited_rows(container) -> list[str]:
    """`container`'s own row-span exactly as the compositor painted it.

    `Widget.render_line()` is NOT ground truth for what is actually visible:
    it returns a widget's own strip at whatever size the widget computed for
    ITSELF (e.g. `width: auto` sizes to fit the label), which can be wider
    than the space its container/viewport actually has -- Textual does not
    clamp an overflowing child down to its parent's box before rendering it.
    Verified empirically while writing this test: with the pre-fix 28-wide
    rail (no `width: 100%`/`text-wrap: wrap`), `Button.render_line(0)`
    happily returned the FULL, untruncated label even though the button's
    own `.region` (x=210, width=37) already ran 12 columns past the rail's
    right edge at x=235 -- a `render_line`-based assertion would have passed
    against the very CSS this test exists to catch, the same false-negative
    shape the task brief warned a bare-`App` unit test would produce, just
    one layer further down. The COMPOSITOR (`Screen._compositor`) is what
    actually clips overlapping/overflowing widgets down to what a real
    terminal shows; `render_strips()` returns that final, already-clipped
    output, confirmed against a live capture reproducing the exact
    truncated strings from the task brief ("Stage Watchlists Cont", "Open
    current Watchlis", "Console follow unavai") before the CSS fix, and
    their full, wrapped text after it.
    """
    strips = container.screen._compositor.render_strips()
    region = container.region
    rows = []
    for y in range(region.y, region.y + region.height):
        if 0 <= y < len(strips):
            row_text = "".join(segment.text for segment in strips[y])
            rows.append(row_text[region.x : region.x + region.width])
    return rows


_BORDER_GLYPHS = "─│┌┐└┘╭╮╯╰═║╔╗╚╝├┤┬┴┼"


def _assert_label_intact_on_screen(container, label: str, *, context: str) -> None:
    """Assert `label` appears whole (possibly wrapped, never clipped) in the
    compositor's actual painted output for `container`.

    Border-drawing glyphs are stripped before joining rows: the label is
    read back by squeezing whitespace and concatenating rows in order (a
    wrapped label reads correctly this way since Textual wraps at word
    boundaries without reordering), and a literal border character
    surviving between two words a wrap split apart -- e.g. "Context ││ in"
    from the panel's own frame sitting inside the column slice -- would
    otherwise break a substring match that has nothing to do with clipping.
    """
    rows = _composited_rows(container)
    cleaned_rows = [
        "".join(ch for ch in row if ch not in _BORDER_GLYPHS) for row in rows
    ]
    combined = " ".join(" ".join(row.split()) for row in cleaned_rows if row.strip())
    assert "…" not in combined, (
        f"{context}: composited output shows an ellipsis (clipped text): {combined!r}"
    )
    normalized = " ".join(label.split())
    assert normalized in combined, (
        f"{context}: {label!r} does not appear intact on screen -- composited "
        f"rail reads {combined!r}"
    )


def _mark_console_onboarding_complete(app) -> None:
    app.app_config = getattr(app, "app_config", {}) or {}
    console_config = app.app_config.setdefault("console", {})
    onboarding = console_config.setdefault("onboarding", {})
    onboarding["first_send_completed"] = True


def _is_effectively_displayed(widget) -> bool:
    current = widget
    while current is not None:
        if current.display is False or current.styles.display == "none":
            return False
        current = getattr(current, "parent", None)
    return True


async def _click_settings_category(screen, pilot, category_id: str) -> None:
    """Click a settings sidebar category, scrolling it into view first."""
    selector = f"#settings-category-{category_id}"
    await _wait_for_selector(screen, pilot, selector)
    try:
        category_list = screen.query_one("#settings-category-list")
        category_list.scroll_to_widget(
            screen.query_one(selector), animate=False, immediate=True
        )
        await pilot.pause()
    except Exception:
        pass
    await pilot.click(selector)


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
        await _wait_for_selector(home, pilot, "#home-triage-grid")
        nav = home.query_one(MainNavigationBar)
        strip = nav.query_one("#nav-destination-strip")
        settings = nav.query_one("#nav-settings", Button)
        more = nav.query_one("#nav-overflow-hint")
        _assert_no_horizontal_overlap(
            strip, more, context="More hint overlaps the destination scroll strip"
        )
        visible_settings = settings.region.intersection(strip.content_region)
        assert visible_settings.x + visible_settings.width <= more.region.x, (
            "Visible Settings nav content overlaps the More hint"
        )


@pytest.mark.asyncio
async def test_destination_content_starts_immediately_below_nav():
    app = _build_test_app()
    host = HomeHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-triage-grid")
        content = home.query_one("#screen-content")
        dashboard = home.query_one("#home-triage-grid")
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
        await _wait_for_selector(home, pilot, "#home-triage-grid")
        assert home.query_one("#home-triage-grid").region.y <= 12
        _assert_horizontal_panes(
            home,
            ("#home-rail", "#home-canvas"),
        )
        for selector in (
            "#home-triage-grid",
            "#home-rail",
            "#home-canvas",
            "#home-details-body",
        ):
            _assert_visible_in_viewport(
                home.query_one(selector), height=42, context=selector
            )
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
async def test_console_first_start_shows_left_rail_main_and_right_handle():
    app = _build_test_app()
    _mark_console_onboarding_complete(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")
        _assert_strip_compact(console, "#console-control-bar", max_height=3)
        workbench = console.query_one("#console-workspace-grid")
        left_rail = console.query_one("#console-left-rail")
        main_column = console.query_one("#console-main-column")
        right_rail = console.query_one("#console-right-rail")
        right_handle = console.query_one("#console-inspector-rail-handle")
        transcript = console.query_one("#console-session-surface")
        composer = console.query_one("#console-native-composer")

        assert workbench.region.y <= 12, (
            f"Console workbench starts too low: {workbench.region}"
        )
        _assert_visible_in_viewport(workbench, height=42, context="Console workbench")
        assert _is_effectively_displayed(left_rail)
        assert _is_effectively_displayed(main_column)
        assert not _is_effectively_displayed(right_rail)
        assert _is_effectively_displayed(right_handle)
        assert left_rail.region.x < main_column.region.x < right_handle.region.x
        assert left_rail.region.height >= 20
        assert main_column.region.height >= 20
        assert right_handle.region.height >= 20
        _assert_visible_in_viewport(left_rail, height=42, context="Console left rail")
        _assert_visible_in_viewport(
            main_column, height=42, context="Console main column"
        )
        _assert_visible_in_viewport(
            right_handle, height=42, context="Console right handle"
        )
        assert console.query_one("#console-staged-context-tray")
        assert console.query_one("#console-workspace-context")
        _assert_visible_in_viewport(transcript, height=42, context="Console transcript")
        _assert_visible_in_viewport(composer, height=42, context="Console composer")


@pytest.mark.asyncio
async def test_library_shell_grid_is_visible_in_viewport():
    # The retired horizontal mode-chip strip (#library-mode-bar,
    # .library-mode-chip) and the list/detail/inspector 3-pane contract grid
    # (#library-contract-grid) are both gone with no analog: LibraryRail
    # renders vertical Browse/Create/Ingest/Details sections instead of a
    # chip strip, and the canvas is a single pane, not three. The surviving
    # geometry contract is that the rail + canvas shell itself renders near
    # the top of the viewport with both panes fully visible.
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-shell-grid")
        shell_grid = library.query_one("#library-shell-grid")
        rail = library.query_one("#library-rail")
        canvas = library.query_one("#library-canvas")
        assert shell_grid.region.y <= 12, (
            f"#library-shell-grid starts too low: {shell_grid.region}"
        )
        _assert_visible_in_viewport(
            shell_grid, height=42, context="#library-shell-grid"
        )
        _assert_visible_in_viewport(rail, height=42, context="#library-rail")
        _assert_visible_in_viewport(canvas, height=42, context="#library-canvas")
        assert rail.region.x < canvas.region.x
        assert rail.region.y == canvas.region.y


@pytest.mark.asyncio
async def test_library_workbench_prioritizes_canvas_width():
    # The retired browser/detail/inspector 3-pane width contract is gone
    # (only 2 areas exist now: rail, canvas); the same "give most of the
    # horizontal space to content, not navigation" intent survives via the
    # rail (3fr, min 24) vs. canvas (13fr, min 40) width ratio.
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-shell-grid")

        rail = library.query_one("#library-rail")
        canvas = library.query_one("#library-canvas")

        assert rail.region.width < canvas.region.width
        assert canvas.region.width >= rail.region.width * 1.35


@pytest.mark.asyncio
async def test_library_canvas_stays_content_fit_at_wide_viewport():
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(212, 64)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-shell-grid")

        rail = library.query_one("#library-rail")
        canvas = library.query_one("#library-canvas")

        assert canvas.region.width >= rail.region.width * 2.5


@pytest.mark.asyncio
async def test_library_workbench_renders_terminal_borders():
    # The retired browser/detail/inspector panes each carried a border plus
    # top padding via a DEFAULT_CSS fallback (so the check held even
    # without the app bundle stylesheet); the rail + canvas panes have no
    # such fallback and only get their border from the bundle CSS, so this
    # check needs the CSS-loading LibraryHarness instead of the bundle-less
    # DestinationHarness. Their padding is horizontal-only (padding: 0 1)
    # by design -- rows supply their own vertical spacing -- so only the
    # border half of the original contract survives.
    app = _build_test_app()
    host = LibraryHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_library_screen(host)
        await _wait_for_library_shell(library, pilot)
        for selector in ("#library-shell-grid", "#library-rail", "#library-canvas"):
            widget = library.query_one(selector)
            assert widget.styles.border_top[0], f"{selector} has no top border"


@pytest.mark.asyncio
async def test_library_empty_state_reports_empty_with_next_action():
    # The retired #library-status-row / #library-source-empty marker and its
    # LIBRARY_EMPTY_COPY / LIBRARY_EMPTY_NEXT_ACTION_COPY text lived only in
    # the never-mounted #library-action-region and never render in the rail
    # + canvas shell (see test_destination_shells.py). The rail's zero
    # counts plus the landing canvas purpose line are the surviving "there
    # is nothing here yet, here's what to do" signal.
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-canvas-landing")

        visible_text = " ".join(
            [
                *(str(widget.renderable) for widget in library.query(Static)),
                *(str(button.label) for button in library.query(Button)),
            ]
        )

    assert "Notes (0)" in visible_text
    assert "Media (0)" in visible_text
    assert "Conversations (0)" in visible_text
    assert "Search, pick a content type, or ingest something new." in visible_text


@pytest.mark.asyncio
async def test_library_source_browser_collections_action_switches_to_collections_mode():
    # The retired #library-open-collections button and #library-mode-*
    # chip strip are dead (they lived only in the never-mounted
    # #library-source-browser); the Browse > Collections rail row is the
    # surviving trigger, and its own selected-row styling plus the
    # selection-state field are the successors of "the detail title changed
    # and the mode chip went active".
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-row-browse-collections")

        library.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(library, pilot, "#library-collections-panel")

        active_row = library.query_one("#library-row-browse-collections", Button)
        selected_row_id = getattr(library, "_library_selected_row_id")

    assert selected_row_id == "browse-collections"
    assert active_row.has_class("library-rail-row-selected")


@pytest.mark.asyncio
async def test_library_source_browser_search_action_switches_to_search_mode():
    # The retired #library-open-search button, #library-mode-* chip strip,
    # and #library-rag-inspector-title / #library-inspector-title Inspector
    # column are all dead (the Inspector column lived only in the
    # never-mounted #library-inspector-mode-region); the Browse > Search /
    # RAG rail row is the surviving trigger, and its selected-row styling
    # plus the selection-state field are the successors of "the detail
    # title changed and the mode chip went active".
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-row-browse-search")

        library.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(library, pilot, "#library-search-rag-panel")

        active_row = library.query_one("#library-row-browse-search", Button)
        selected_row_id = getattr(library, "_library_selected_row_id")

    assert selected_row_id == "browse-search"
    assert active_row.has_class("library-rail-row-selected")


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
    (
        records,
        counts,
        total_known,
        error,
        recovery_state,
        study_counts,
    ) = await screen._list_local_source_snapshot()
    elapsed = time.perf_counter() - start

    assert records == {
        "notes": (),
        "media": (),
        "conversations": (),
        # The prompts and skills seams carry (count, payload) placeholders,
        # not bare records tuples (see LibraryScreen.__init__).
        "prompts": (None, ()),
        "skills": (None, {"available_skills": [], "blocked_skills": []}),
    }
    assert counts == {"notes": 0, "media": 0, "conversations": 0}
    assert total_known == {"notes": True, "media": True, "conversations": True}
    assert error == library_screen_module.LIBRARY_SERVICE_ERROR_COPY
    assert recovery_state is None
    assert study_counts == {
        "study_decks": None,
        "flashcards_due": None,
        "quizzes": None,
    }
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_library_source_snapshot_timeout_handles_blocking_async_services(
    monkeypatch,
):
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
    (
        records,
        counts,
        total_known,
        error,
        recovery_state,
        study_counts,
    ) = await screen._list_local_source_snapshot()
    elapsed = time.perf_counter() - start

    assert records == {
        "notes": (),
        "media": (),
        "conversations": (),
        # The prompts and skills seams carry (count, payload) placeholders,
        # not bare records tuples (see LibraryScreen.__init__).
        "prompts": (None, ()),
        "skills": (None, {"available_skills": [], "blocked_skills": []}),
    }
    assert counts == {"notes": 0, "media": 0, "conversations": 0}
    assert total_known == {"notes": True, "media": True, "conversations": True}
    assert error == library_screen_module.LIBRARY_SERVICE_ERROR_COPY
    assert recovery_state is None
    assert study_counts == {
        "study_decks": None,
        "flashcards_due": None,
        "quizzes": None,
    }
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_library_service_call_awaits_coroutine_functions_without_worker(
    monkeypatch,
):
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
            # The run inspector is a section inside the right rail; the rail
            # itself is the third workbench pane.
            ("#console-left-rail", "#console-main-column", "#console-right-rail"),
            (
                "#console-send-message",
                "#console-attach-context",
                "#console-save-chatbook",
            ),
            ("#console-run-inspector-state",),
            "#console-run-inspector",
        ),
        # The "library" case was retired here: #library-contract-grid and its
        # 3 panes/markers never mount in the rail + canvas shell (the
        # #library-source-empty/-error/-loading markers lived only in the
        # never-mounted #library-local-snapshot-region). Library's
        # non-happy-state geometry coverage now lives in the dedicated
        # test_library_empty_state_reports_empty_with_next_action below;
        # the "loading" and "timeout falls back to a stable error" marker
        # geometry checks have no successor since neither state has a
        # distinct positioned marker anymore (see
        # test_library_source_snapshot_times_out_to_stable_error for the
        # underlying business-logic coverage that survives).
    ],
)
@pytest.mark.asyncio
async def test_core_default_empty_or_blocked_states_keep_workbench_geometry(
    route, host_factory, workbench, panes, actions, markers, marker_container
):
    app = _build_test_app()
    if route == "chat":
        _mark_console_onboarding_complete(app)
    host = host_factory(app)
    # 160 wide: the Console force-collapses its inspector rail below 150
    # columns (CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS), and this
    # contract covers the full three-pane workbench.
    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, workbench)
        if route == "chat":
            # The inspector rail composes collapsed; open it via its handle
            # (only honored at >=150 columns) and wait out the recompose.
            screen.query_one("#console-inspector-rail-open", Button).press()
            for _ in range(40):
                await pilot.pause(0.05)
                if screen.query_one(panes[2]).display:
                    break
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


SOURCE_PREP_WORKBENCHES = {
    "artifacts": {
        "workbench": "#artifacts-workbench",
        "strip": "#artifacts-mode-strip",
        "panes": (
            "#artifacts-list-pane",
            "#artifacts-detail-pane",
            "#artifacts-inspector-pane",
        ),
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
    # Personas is now a destination-native workbench (library / work area /
    # inspector). Its empty/count state renders inside the library pane; the
    # legacy thin-shell empty/error/loading markers were retired with the
    # snapshot worker.
    "personas": {
        "workbench": "#personas-workbench",
        "strip": "#personas-mode-strip",
        "panes": (
            "#personas-library-pane",
            "#personas-work-area",
            "#personas-inspector-pane",
        ),
        "actions": ("#personas-library-new", "#personas-attach-to-console"),
        "markers": ("#personas-library-empty", "#personas-library-count"),
        "marker_container": "#personas-library-pane",
    },
    # Watchlists moved from the placeholder 3-column shell onto
    # `WatchlistsWorkbench` (rails around a VERTICALLY stacked centre — see
    # `watchlists_workbench.py`'s docstring on why the shared
    # `DestinationWorkbench`/3-column-ASCII contract can't express this
    # layout). `#watchlists-list-pane` and `#watchlists-detail-pane` are now
    # stacked one above the other inside `#wl-centre`, not side by side, so
    # this entry is kept for the `markers`/`marker_container`/`actions` keys
    # (still valid — those panes still exist, just not horizontally arranged)
    # but excluded from the two horizontal-geometry parametrizations below via
    # `SOURCE_PREP_WORKBENCHES_HORIZONTAL`, the same way "personas" above
    # excluded itself from the retired snapshot-worker markers.
    "watchlists_collections": {
        "workbench": "#wl-workbench",
        "strip": "#watchlists-header-bar",
        "strip_max_height": 3,
        "panes": (
            "#watchlists-list-pane",
            "#watchlists-detail-pane",
            "#watchlists-inspector-pane",
        ),
        # `#nav-overview` was the retired left-rail navigator's Overview
        # button; the rail now hosts the watchlist tree and the section
        # buttons live in the centre tab strip as `#wl-tab-*`.
        # `_assert_any_action_visible` skips selectors with no matches, so
        # the dead id silently shrank this guard by one action rather than
        # failing.
        "actions": (
            "#wl-tab-overview",
            "#wc-empty-create-source",
            "#wc-open-watchlists",
            "#wc-attach-to-console",
            "#watchlists-follow-in-console",
        ),
        "markers": ("#wc-empty-state", "#wc-service-error", "#wc-loading-state"),
        "marker_container": "#watchlists-list-pane",
    },
    "skills": {
        "workbench": "#skills-workbench",
        "strip": "#skills-mode-strip",
        "panes": ("#skills-list-pane", "#skills-detail-pane", "#skills-inspector-pane"),
        "actions": ("#skills-import-skill", "#skills-attach-to-console"),
        "markers": (
            "#skills-empty-state",
            "#skills-service-error",
            "#skills-loading-state",
        ),
        "marker_container": "#skills-detail-pane",
    },
}

#: `SOURCE_PREP_WORKBENCHES` minus destinations that no longer lay their
#: list/detail/inspector panes out horizontally. Watchlists' rehost onto
#: `WatchlistsWorkbench` stacks FEEDS/ITEMS/CONTENT vertically inside a
#: centre column by deliberate design (see `watchlists_workbench.py`), so it
#: cannot satisfy `_assert_horizontal_panes` — that is a real, intentional
#: geometry change, not a regression to paper over.
SOURCE_PREP_WORKBENCHES_HORIZONTAL = {
    route: contract
    for route, contract in SOURCE_PREP_WORKBENCHES.items()
    if route != "watchlists_collections"
}


@pytest.mark.parametrize("route,contract", SOURCE_PREP_WORKBENCHES_HORIZONTAL.items())
@pytest.mark.asyncio
async def test_source_prep_destinations_use_list_detail_inspector_workbench(
    route, contract
):
    app = _build_test_app()
    host = _visual_destination_harness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, contract["workbench"])
        _assert_ascii_workbench_contract(
            screen,
            workbench=contract["workbench"],
            strip=contract["strip"],
            strip_max_height=contract.get("strip_max_height", 2),
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


@pytest.mark.parametrize("route,contract", SOURCE_PREP_WORKBENCHES_HORIZONTAL.items())
@pytest.mark.asyncio
async def test_source_prep_default_empty_or_unavailable_states_preserve_workbench_geometry(
    route, contract
):
    app = _build_test_app()
    host = _visual_destination_harness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, contract["workbench"])
        _assert_ascii_workbench_contract(
            screen,
            workbench=contract["workbench"],
            strip=contract["strip"],
            strip_max_height=contract.get("strip_max_height", 2),
            panes=contract["panes"],
            actions=contract["actions"],
            height=42,
        )


@pytest.mark.asyncio
async def test_watchlists_screen_matches_approved_control_plane_columns():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

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
        assert screen.query_one("#watchlists-header-bar").region.height == 3
        assert "Backend: local" in visible_text
        assert "Sources" in visible_text
        assert "Overview" in visible_text
        assert "Inspector" in visible_text
        assert "State: ready" in visible_text
        assert "Alert rules active:" in visible_text
        assert "Latest run status:" in visible_text
        assert "Collections" not in visible_text
        assert "Column 1:" not in visible_text
        assert "Column 2:" not in visible_text
        assert "Column 3:" not in visible_text

        # The Rule-divided three-column body was replaced by the collapsible
        # WatchlistsWorkbench (rails around a vertically stacked centre; see
        # watchlists_workbench.py) — region borders replace Rule dividers, so
        # the old #watchlists-*-divider ids no longer exist. Assert the new
        # structural landmarks instead: the workbench container, and each
        # region wrapper.
        #
        # CONTENT starts collapsed on a genuinely fresh config (as here): its
        # reader is a Phase D stub, and `region_layout_store.load_region_layout`
        # distinguishes a never-saved key (`None`, applies the first-run
        # default with CONTENT collapsed) from an explicitly-saved empty
        # layout (`[]`, honored exactly) — see that module for why the two
        # can't be collapsed into one. So the header, not the body, is
        # asserted here.
        assert screen.query_one("#wl-workbench")
        for region_id in (
            "wl-region-left_rail",
            "wl-region-feeds",
            "wl-region-items",
            "wl-region-right_rail",
        ):
            assert screen.query_one(f"#{region_id}")
        assert screen.query_one("#wl-header-content")
        assert not screen.query("#wl-region-content")


@pytest.mark.asyncio
async def test_watchlists_centre_regions_stack_vertically_in_order():
    """Vertical-geometry replacement for the horizontal contract Watchlists
    was excluded from (see `SOURCE_PREP_WORKBENCHES_HORIZONTAL` above): its
    rehost onto `WatchlistsWorkbench` stacks FEEDS/ITEMS/CONTENT vertically
    inside the centre column by deliberate design, so `_assert_horizontal_panes`
    (which asserts `left.region.x < right.region.x`) can never apply — but
    that must not mean Watchlists has zero automated geometry coverage.
    """
    app = _build_test_app()
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-workbench")

        # Force every region open so all three centre regions have real
        # geometry to compare — independent of whatever CONTENT's first-run
        # collapse default happens to be (region_layout_store.load_region_layout).
        screen._apply_layout(RegionLayout())
        await pilot.pause()

        feeds = screen.query_one("#wl-region-feeds")
        items = screen.query_one("#wl-region-items")
        content = screen.query_one("#wl-region-content")

        assert feeds.region.y < items.region.y < content.region.y, (
            f"centre regions are not stacked top-to-bottom: "
            f"feeds={feeds.region} items={items.region} content={content.region}"
        )
        for selector, pane in (
            ("#wl-region-feeds", feeds),
            ("#wl-region-items", items),
            ("#wl-region-content", content),
        ):
            assert pane.region.width > 0, f"{selector} has no width"
            assert pane.region.height > 0, f"{selector} has no height"
            _assert_visible_in_viewport(pane, height=42, context=selector)

        # No two centre regions may overlap.
        for top, bottom in ((feeds, items), (items, content)):
            assert top.region.y + top.region.height <= bottom.region.y, (
                f"centre regions overlap: {top.region} vs {bottom.region}"
            )


@pytest.mark.asyncio
async def test_watchlists_collapsing_both_rails_keeps_every_region_in_viewport():
    """Fix round 2, Finding 1 (CRITICAL): `.watchlists-region-header`'s
    `width: 100%` rule was unscoped, so a RAIL header (a direct child of the
    workbench `Horizontal`, not the centre `Vertical`) took the ENTIRE
    workbench width instead of a narrow collapsed-handle width -- pushing
    every region after it off-screen. Measured pre-fix at 160x42 after
    collapsing both rails: the right rail's header landed at
    x=161 with the workbench only 160 wide, unreachable by mouse. This forces
    the exact reproduction and asserts every remaining region/header stays
    inside the 160-wide viewport.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-workbench")

        screen._apply_layout(
            RegionLayout(collapsed=frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL}))
        )
        await pilot.pause()

        for selector in (
            "#wl-header-left_rail",
            "#wl-region-feeds",
            "#wl-region-items",
            "#wl-header-right_rail",
        ):
            _assert_visible_in_viewport(
                screen.query_one(selector),
                height=42,
                context=selector,
                viewport_width=160,
            )

        left_header = screen.query_one("#wl-header-left_rail")
        right_header = screen.query_one("#wl-header-right_rail")
        assert left_header.region.width < 160, (
            f"left rail header claimed (close to) the full workbench width: "
            f"{left_header.region}"
        )
        assert right_header.region.x + right_header.region.width <= 160, (
            f"right rail header fell outside the viewport: {right_header.region}"
        )


@pytest.mark.asyncio
async def test_watchlists_items_region_is_taller_than_feeds_region_when_expanded():
    """Fix round 2, Finding 5 (human-ruled, CSS-only rebalance): FEEDS only
    ever hosts a short (<=5-line) source-count summary but shared an equal
    `1fr` with ITEMS -- which hosts all six section panes and their
    DataTables -- and CONTENT, before this rebalance. `height: auto` on
    FEEDS lets it take only what its content needs so ITEMS (still `1fr`)
    gets the space FEEDS no longer claims.
    """
    app = _build_test_app()
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-workbench")

        screen._apply_layout(RegionLayout())
        await pilot.pause()

        feeds = screen.query_one("#wl-region-feeds")
        items = screen.query_one("#wl-region-items")

        assert items.region.height > feeds.region.height, (
            f"ITEMS should be taller than FEEDS once FEEDS is content-fit: "
            f"feeds={feeds.region} items={items.region}"
        )


@pytest.mark.parametrize("size", [(160, 42), (235, 52)])
@pytest.mark.asyncio
async def test_watchlists_feeds_empty_state_fits_without_scrolling(size):
    """Fix round 2, regression pin: round 1 shipped `max-height: 10`, one row
    BELOW the real empty state's measured 11-row need (tab strip, "Sources"
    title, "No sources yet.", Create/Import button row, inside the pane's
    own border) -- so a watchlist with zero sources rendered a scrollbar.
    That was a ceiling below its own minimum content, not the intended
    "grows to fit, caps, then scrolls" behaviour.

    `max-height` is now 12 (see `.watchlists-region-feeds` in
    `_watchlists.tcss` for the full derivation), which clears the 11-row
    need with headroom: the pane's real content must sit entirely inside
    FEEDS -- the cap never engages, nothing is clipped -- and there must be
    nothing left to scroll, at both the guard suite's 160x42 and the app's
    real 235x52.

    Fix round 3 restated the containment check. It was
    `feeds.region.height == pane.region.height`, which silently also encoded
    "the PANE draws FEEDS's only border" -- true only while round 1's border
    decision stood. Round 3 inverted that decision (the region draws the box,
    `#watchlists-list-pane` is stripped by ID; see `_watchlists.tcss`), so
    the region is now exactly its own 2 border rows taller than the pane:
    measured 160x42 feeds=11 pane=9, 235x52 feeds=11 pane=9. `contains_region`
    asserts the same property -- the pane fits, unclipped -- without pinning
    which widget happens to own the border rows.
    """
    app = _build_test_app()
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=size) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-workbench")

        screen._apply_layout(RegionLayout())
        await pilot.pause()

        feeds = screen.query_one("#wl-region-feeds")
        pane = screen.query_one("#watchlists-list-pane")

        assert feeds.region.contains_region(pane.region), (
            f"the empty state's real content should fit inside FEEDS's cap "
            f"without being clipped: feeds={feeds.region} pane={pane.region}"
        )
        assert feeds.max_scroll_y == 0, (
            f"the empty state should have nothing left to scroll: "
            f"max_scroll_y={feeds.max_scroll_y} feeds={feeds.region}"
        )


#: Watchlists' 40-source stand-in for a populated FEEDS list. Enough rows to
#: force the region past any plausible `max-height`.
_WL_OVERFLOW_RECORDS = tuple(
    {"name": f"source-{index:02d}", "title": f"source-{index:02d}"}
    for index in range(40)
)


def _seed_overflow_sources(app, count: int = 40) -> None:
    """Put `count` real sources in FEEDS's own list.

    Task 7 fix round 1, Finding 1 moved the long list in FEEDS: the
    Console-staging block used to render one row per
    `_local_watchlist_records` entry and now collapses to a single line, so
    `_apply_local_wc_snapshot(_WL_OVERFLOW_RECORDS, ...)` alone no longer
    produces enough rows to reach the cap. The rows that remain are
    `scoped_source_rows()`, which reads the bundle service's own
    `subscriptions` table -- an isolated temp file per `_build_test_app()`,
    never the developer's database.

    Setup only: every assertion in the tests below is unchanged.
    """
    db = app.watchlist_bundle_service._db
    for index in range(count):
        db.add_subscription(
            name=f"source-{index:02d}",
            type="rss",
            source=f"https://feed-{index:02d}.example/rss",
        )

_ROUND_CORNERS = "╭╮╰╯"
_SQUARE_CORNERS = "┌┐└┘"


@pytest.mark.asyncio
async def test_watchlists_every_region_draws_exactly_one_round_border():
    """Task 6 fix round 3, Finding 2: one box per region, all the same shape.

    Round 1 kept the *pane's* border and dropped the *region's* in three of
    the five regions. Region borders are `round` and the shared destination
    pane's is `solid`, so the screen drew round corners on LEFT_RAIL/CONTENT
    and square ones on FEEDS/ITEMS/RIGHT_RAIL. Round 3 inverted it: the region
    wrapper draws the box everywhere, and `#watchlists-list-pane`/
    `#watchlists-detail-pane`/`#watchlists-inspector-pane` are stripped by ID
    in `features/_watchlists.tcss` (an ID rule that beats the shared block in
    `components/_agentic_terminal.tcss` on source order, touching no other
    destination).

    Counting corners in the compositor's output catches both failure modes at
    once: a doubled border shows more than four corners inside a region, and a
    mixed style shows square ones.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wc-empty-state")

        screen._apply_layout(RegionLayout())
        await pilot.pause()
        # Regions are focusable, and the app-wide focus affordance is
        # `*:focus { outline: solid $ds-focus-accent }` (core reset, top of
        # the bundle) -- a SQUARE outline painted over whichever box has
        # keyboard focus, on every screen. That is a deliberate focus signal,
        # not a border-style inconsistency, so this test measures the resting
        # state and blurs first rather than depending on where focus landed.
        screen.set_focus(None)
        await pilot.pause()

        # 1. Every region draws its own box, and draws it `round`.
        for region in Region:
            widget = screen.query_one(f"#wl-region-{region.value}")
            rows = _composited_rows(widget)
            assert rows[0].startswith("╭") and rows[0].endswith("╮"), (
                f"{region.value} has no round top border: {rows[0]!r}"
            )
            assert rows[-1].startswith("╰") and rows[-1].endswith("╯"), (
                f"{region.value} has no round bottom border: {rows[-1]!r}"
            )

        # 2. The pane inside draws none -- that is the doubling this task
        #    exists to remove. (Inner content may still draw its own cards,
        #    e.g. the Overview grid; only the pane's own frame is checked.)
        for pane_id in (
            "watchlists-list-pane",
            "watchlists-detail-pane",
            "watchlists-inspector-pane",
        ):
            rows = _composited_rows(screen.query_one(f"#{pane_id}"))
            edges = (rows[0][0], rows[0][-1], rows[-1][0], rows[-1][-1])
            assert not any(ch in _ROUND_CORNERS + _SQUARE_CORNERS for ch in edges), (
                f"#{pane_id} still draws its own frame inside the region's: "
                f"{rows[0]!r} ... {rows[-1]!r}"
            )

        # 3. Nothing in the workbench draws a square-cornered box, so the
        #    five outer boxes and everything nested in them read as one
        #    family. Round 1's split left FEEDS/ITEMS/RIGHT_RAIL square while
        #    LEFT_RAIL and CONTENT stayed round.
        workbench_rows = _composited_rows(screen.query_one("#wl-workbench"))
        squares = {ch for row in workbench_rows for ch in row if ch in _SQUARE_CORNERS}
        assert not squares, (
            f"the workbench mixes border styles ({sorted(squares)}); every box "
            f"here should be `round`"
        )


@pytest.mark.asyncio
async def test_watchlists_left_rail_is_labelled_when_expanded():
    """Task 6 fix round 3, Finding 1: title suppression keyed on
    factory-presence rather than on "the pane supplies its own heading", and
    LEFT_RAIL is where the two diverge -- `WatchlistTree` composes navigation
    buttons and no heading. The expanded rail rendered as an unlabelled box
    while its collapsed header still read "▸ Watchlists".
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wc-empty-state")

        screen._apply_layout(RegionLayout())
        await pilot.pause()

        rail = screen.query_one("#wl-region-left_rail")
        _assert_label_intact_on_screen(
            rail, "Watchlists", context="expanded left rail heading"
        )


@pytest.mark.asyncio
async def test_watchlists_active_section_tab_label_is_visible():
    """Task 3 defect, folded into Task 6 fix round 3 by the reviewer
    (Finding 5): `WatchlistsTabStrip` pins its strip to `height: 1`, and
    `.watchlists-tab` had no styling, so the active tab inherited the global
    `.is-active { border: round $ds-action-focus }`. A `round` border needs
    two rows before it has a content row at all, so the active button painted
    as its own top border and nothing else -- the user could not see which
    section was selected. Measured pre-fix at 160x42: the strip's only row
    read `╭──────────────╮    Sources    Items ...`.

    The compositor is the instrument, not `render_line()`: `render_line()`
    returns a widget's own strip at the size the widget computed for itself,
    which false-negatives on exactly this class of assertion (see
    `_composited_rows`).
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wc-empty-state")

        screen._apply_layout(RegionLayout())
        await pilot.pause()

        strip = screen.query_one("#wl-tabs")
        active = screen.query_one("#wl-tab-overview", Button)
        assert active.has_class("is-active"), sorted(active.classes)

        rows = _composited_rows(strip)
        assert len(rows) == 1, f"the section strip must stay one row: {rows!r}"
        assert not any(ch in rows[0] for ch in _ROUND_CORNERS + _SQUARE_CORNERS), (
            f"a border inside the one-row strip eats the row the labels need: "
            f"{rows[0]!r}"
        )
        for label in ("Overview", "Sources", "Items", "Runs", "Rules", "Notifications"):
            _assert_label_intact_on_screen(
                strip, label, context="watchlists section tab strip"
            )


@pytest.mark.parametrize("size", [(160, 42), (100, 40)])
@pytest.mark.asyncio
async def test_watchlists_soloed_feeds_fills_the_centre(size):
    """Task 6 fix round 3, Finding 3: `Z` soloed FEEDS into a degenerate view.

    Solo (`action_solo_region` -> `RegionLayout.solo`) collapses the other two
    centre regions to one-line headers. FEEDS is the only capped region, so it
    stayed pinned at `max-height: 13` while the rest of the centre went blank
    -- measured pre-fix at 100x40 with 40 sources: `feeds=13@y0, items=1,
    content=1`, 17 of the centre's 32 rows empty while FEEDS scrolled a 13-row
    window over 44 rows of content. Solo-ITEMS and solo-CONTENT filled
    correctly; only the capped region was broken.

    Nothing in the DOM distinguished the state, so `_region_widget` now adds
    `.watchlists-region-sole-centre` and the stylesheet lifts the cap there.
    No solo geometry was tested anywhere before this.
    """
    app = _build_test_app()
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=size) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-workbench")

        screen._apply_layout(RegionLayout())
        await pilot.pause()
        screen._apply_local_wc_snapshot(_WL_OVERFLOW_RECORDS, 40, True)
        await pilot.pause()

        for soloed in (Region.FEEDS, Region.ITEMS, Region.CONTENT):
            screen._apply_layout(RegionLayout().solo(soloed))
            await pilot.pause()
            await pilot.pause()

            # Re-query every widget after each layout change: the workbench
            # reactive is `recompose=True`, so the previous iteration's
            # references are unmounted and report a zero-sized region.
            centre = screen.query_one("#wl-centre")
            body = screen.query_one(f"#wl-region-{soloed.value}")
            headers = [
                screen.query_one(f"#wl-header-{other.value}")
                for other in (Region.FEEDS, Region.ITEMS, Region.CONTENT)
                if other is not soloed
            ]
            covered = body.region.height + sum(h.region.height for h in headers)
            assert covered == centre.region.height, (
                f"solo({soloed.value}) leaves "
                f"{centre.region.height - covered} of the centre's "
                f"{centre.region.height} rows blank: body={body.region} "
                f"headers={[h.region for h in headers]}"
            )


@pytest.mark.parametrize("height,budget", [(42, 34), (41, 33)])
@pytest.mark.asyncio
async def test_watchlists_feeds_cap_keeps_items_taller_when_it_actually_binds(
    height: int, budget: int
):
    """Task 6 fix round 3, Finding 4: the cap's derivation, made executable.

    `test_watchlists_items_region_is_taller_than_feeds_region_when_expanded`
    (above) runs the EMPTY state, where FEEDS's natural 11 rows never reach
    the 12-row cap -- so the constraint that actually chose the cap has never
    been exercised by a test. This forces FEEDS past its cap inside the real
    chrome-wrapped screen and pins the resulting split.

    The numbers, measured at 160x42 (the tightest viewport this app ships):
    the three centre regions share a fixed 34-row budget, so with
    ITEMS/CONTENT at `2fr`/`1fr` a capped FEEDS leaves `items ~= (34 - cap) *
    2/3`. Swept: cap=12 -> items=14, cap=13 -> items=14, cap=14 -> items=13,
    which inverts the invariant. 13 is the maximum that holds here, but it
    holds ONLY at height >= 42 -- at 160x41 the budget drops to 33 and 13
    ties at items=13. The shipped cap is 12, which gives items=14 at both
    budgets, so the invariant survives a 41-row terminal and one more row
    of chrome above the workbench. Anything beyond that trips this test --
    which is the point of pinning it.
    """
    app = _build_test_app()
    # Setup change only (Task 7 fix round 1, Finding 1) -- see
    # `_seed_overflow_sources`. Every assertion below is unchanged.
    _seed_overflow_sources(app)
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(160, height)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-workbench")

        screen._apply_layout(RegionLayout())
        await pilot.pause()
        screen._apply_local_wc_snapshot(_WL_OVERFLOW_RECORDS, 40, True)
        await pilot.pause()
        await pilot.pause()

        feeds = screen.query_one("#wl-region-feeds")
        items = screen.query_one("#wl-region-items")
        content = screen.query_one("#wl-region-content")

        assert feeds.max_scroll_y > 0, (
            "40 sources should overflow FEEDS -- if they do not, this test is "
            f"no longer exercising the cap: feeds={feeds.region}"
        )
        assert feeds.region.height == 12, (
            f"FEEDS should sit exactly at its `max-height: 12` at 160x{height}: "
            f"{feeds.region}"
        )
        assert items.region.height == 14, (
            f"the 2fr/1fr split of the {budget}-row budget should leave ITEMS "
            f"14 rows -- two more than the cap. This is the whole reason the "
            f"cap is 12 and not the 13 the 160x42 maximum would allow: 13 "
            f"gives items=14 at budget 34 but ties at items=13 when the "
            f"budget drops to 33. feeds={feeds.region} items={items.region} "
            f"content={content.region}"
        )
        assert items.region.height > feeds.region.height, (
            f"ITEMS must stay the taller reading area even when FEEDS is "
            f"pinned at its cap: feeds={feeds.region} items={items.region}"
        )
        assert (
            feeds.region.height + items.region.height + content.region.height
            == budget
        ), (
            f"the centre budget moved; the cap's derivation needs redoing: "
            f"feeds={feeds.region} items={items.region} content={content.region}"
        )


@pytest.mark.parametrize("size", [(235, 52), (160, 42)])
@pytest.mark.asyncio
async def test_watchlists_right_rail_does_not_clip_action_labels(size):
    """Task 5, defect 1: at the pre-fix 28-wide rail (26 usable columns),
    every action label on the right rail truncated -- measured live at
    235x52 as "Stage Watchlists Cont", "Open current Watchlis", "Console
    follow unavai". Reproduced here byte-for-byte against the unfixed CSS
    while writing this test (see `_composited_rows`'s docstring for why a
    per-widget assertion is not enough to catch it), then confirmed fixed.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=size) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-workbench")

        right_rail = screen.query_one("#wl-region-right_rail")
        context = f"watchlists right rail at {size}"
        for label in (
            "Stage Watchlists Context in Console",
            "Open current Watchlists",
            "Console follow unavailable",
        ):
            _assert_label_intact_on_screen(right_rail, label, context=context)

        # Also drive the entity Inspector itself through every action set
        # it can show, including the new watchlist-level (scope-only) one,
        # so a future action label long enough to threaten wrapping is
        # caught here too, not just the console-handoff buttons above it.
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        inspector.selected_entity = {
            "id": "source-1",
            "name": "AI News RSS",
            "source_type": "rss",
            "url": "http://example.com/feed",
        }
        await pilot.pause()
        for label in ("Preview", "Check now", "Stage in Console", "Delete"):
            _assert_label_intact_on_screen(
                right_rail, label, context=f"{context} (source actions)"
            )

        inspector.selected_entity = None
        inspector.scope = TreeScope(kind="watchlist", watchlist_id=1)
        await pilot.pause()
        for label in ("Check now", "Delete"):
            _assert_label_intact_on_screen(
                right_rail, label, context=f"{context} (watchlist-scope actions)"
            )


@pytest.mark.parametrize(
    ("route", "workbench", "expected_titles"),
    (
        (
            "artifacts",
            "#artifacts-workbench",
            ("Artifact List", "Artifact Preview", "Provenance"),
        ),
        # The personas work area renders mode-driven section headers
        # (Character / Character Editor / Persona Profile) instead of a fixed
        # column title, so only the library and inspector panes carry
        # *-column-title statics.
        ("personas", "#personas-workbench", ("Library", "Inspector")),
        (
            "schedules",
            "#scheduling-workbench",
            ("Schedule Queue", "Task Detail", "Inspector"),
        ),
        (
            "workflows",
            "#workflows-workbench",
            ("Procedure Library", "Run Detail", "Run Inspector"),
        ),
        # With no runtime configured (the harness default), ACP's middle pane
        # is the Runtime Setup column rather than Session Detail.
        (
            "acp",
            "#acp-workbench",
            ("Agents / Sessions", "Runtime Setup", "Compatibility / Actions"),
        ),
        (
            "skills",
            "#skills-workbench",
            ("Skill Library", "Skill Detail", "Skill Inspector"),
        ),
        # Settings' Overview card title carries the column-title class.
        (
            "settings",
            "#settings-workbench",
            ("Settings Sections", "Preference Detail", "Overview", "Scope Inspector"),
        ),
        # The legacy ccp route/screen was retired; its workbench is the Personas
        # destination, covered by Tests/UI/test_personas_workbench.py.
    ),
)
@pytest.mark.asyncio
async def test_destination_pane_titles_are_user_facing_not_ordinal(
    route, workbench, expected_titles
):
    app = _build_test_app()
    host = _visual_destination_harness(app, route)

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, workbench)
        visible_text = _visible_static_text(screen)

        assert _visible_workbench_pane_titles(screen, workbench) == list(
            expected_titles
        )
        assert "Column 1:" not in visible_text
        assert "Column 2:" not in visible_text
        assert "Column 3:" not in visible_text


@pytest.mark.asyncio
async def test_schedules_screen_matches_approved_control_plane_columns():
    app = _build_test_app()
    host = _visual_destination_harness(app, "schedules")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#scheduling-task-detail-empty-state")
        await pilot.pause()

        visible_text = _visible_static_text(screen)
        for expected in (
            "Last pull: —",
            "Last push: —",
            "Schedule Queue",
            "Task Detail",
            "No scheduled tasks yet",
            "Inspector",
            "No conflict",
        ):
            assert expected in visible_text
        assert {"Local", "Server (unavailable)", "Follow in Console"}.issubset(
            _visible_button_labels(screen)
        )
        assert screen.query_one("#scheduling-owner-server", Button).disabled
        assert screen.query_one("#schedules-follow-in-console", Button).disabled
        assert "Column 1:" not in visible_text
        assert "Column 2:" not in visible_text
        assert "Column 3:" not in visible_text

        _assert_horizontal_panes(
            screen,
            (
                "#scheduling-list-pane",
                "#scheduling-detail-pane",
                "#scheduling-inspector-pane",
            ),
        )


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
            "Modes: Recipes Inputs Steps Dry Run Approvals Outputs",
            "Procedure Library",
            "Run Detail",
            "Run Inspector",
            "State: blocked",
            "Console: blocked",
            "Next action: start or select a workflow run",
        ):
            assert expected in visible_text
        assert "Column 1:" not in visible_text
        assert "Column 2:" not in visible_text
        assert "Column 3:" not in visible_text

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
    """The destination-native workbench renders library / work area / inspector.

    The legacy thin-shell snapshot summary (#personas-characters-summary) was
    retired with the workbench rebuild; the deep behavior contract lives in
    Tests/UI/test_personas_workbench.py.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#personas-workbench")
        visible_text = _visible_static_text(screen)
        buttons = _visible_button_labels(screen)

        _assert_horizontal_panes(
            screen,
            (
                "#personas-library-pane",
                "#personas-work-area",
                "#personas-inspector-pane",
            ),
        )
        assert _visible_workbench_pane_titles(screen, "#personas-workbench") == [
            "Library",
            "Inspector",
        ]
        assert "Modes:" in visible_text
        assert "Column 1:" not in visible_text
        assert "Column 2:" not in visible_text
        assert "Column 3:" not in visible_text
        assert {"Characters", "User Profiles", "New", "Attach to Console"}.issubset(
            buttons
        )


@pytest.mark.asyncio
async def test_personas_workbench_separates_columns_without_legacy_dividers():
    """The workbench retired the 1-cell divider widgets: column separation now
    comes from the bordered destination-workbench-pane containers, so resize
    affordances no longer need standalone divider widgets."""
    app = _build_test_app()
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#personas-workbench")
        library_pane = screen.query_one("#personas-library-pane")
        work_area = screen.query_one("#personas-work-area")
        inspector_pane = screen.query_one("#personas-inspector-pane")

        for pane in (library_pane, work_area, inspector_pane):
            assert "destination-workbench-pane" in pane.classes
        assert library_pane.region.x + library_pane.region.width <= work_area.region.x
        assert work_area.region.x + work_area.region.width <= inspector_pane.region.x
        workbench = screen.query_one("#personas-workbench")
        assert not list(workbench.query(".destination-pane-divider"))


@pytest.mark.asyncio
async def test_artifacts_empty_state_labels_three_clear_columns():
    app = _build_test_app()
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-workbench")
        visible_text = _visible_static_text(screen)
        for expected in (
            "Artifact List",
            "Artifact Preview",
            "Provenance",
        ):
            assert expected in visible_text
        assert "Column 1:" not in visible_text
        assert "Column 2:" not in visible_text
        assert "Column 3:" not in visible_text


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
    # The Personas thin shell's snapshot worker (and its loading marker) was
    # retired with the workbench rebuild; loading/empty behavior is covered by
    # Tests/UI/test_personas_workbench.py.
    #
    # Watchlists' loading marker (#wc-loading-state) still exists and is
    # still checked (Tests/UI/test_destination_shells.py::
    # test_watchlists_collections_initial_load_uses_distinct_loading_copy),
    # but `_assert_ascii_workbench_contract` requires the list/detail/
    # inspector panes to be laid out horizontally, which Watchlists' rehost
    # onto the collapsible WatchlistsWorkbench deliberately no longer does
    # (FEEDS/ITEMS/CONTENT stack vertically inside a centre column — see
    # watchlists_workbench.py). See `SOURCE_PREP_WORKBENCHES_HORIZONTAL`
    # above for the same exclusion applied to the two sibling tests.
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
    host = _visual_destination_harness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, loading_marker)
        _assert_ascii_workbench_contract(
            screen,
            workbench=contract["workbench"],
            strip=contract["strip"],
            strip_max_height=contract.get("strip_max_height", 2),
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
            "#scheduling-sync-status",
            "#scheduling-workbench",
            (
                "#scheduling-list-pane",
                "#scheduling-detail-pane",
                "#scheduling-inspector-pane",
            ),
            ("#schedules-follow-in-console",),
        ),
        (
            "workflows",
            "#workflows-mode-strip",
            "#workflows-workbench",
            (
                "#workflows-list-pane",
                "#workflows-detail-pane",
                "#workflows-inspector-pane",
            ),
            ("#workflows-launch-in-console",),
        ),
    ],
)
@pytest.mark.asyncio
async def test_operational_destinations_use_timing_or_procedure_workbench(
    route, strip, workbench, panes, actions
):
    app = _build_test_app()
    host = _visual_destination_harness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, workbench)
        _assert_ascii_workbench_contract(
            screen,
            workbench=workbench,
            strip=strip,
            strip_max_height=5 if route == "schedules" else 2,
            panes=panes,
            actions=actions,
            height=42,
        )


@pytest.mark.parametrize(
    "route,strip,workbench,panes,actions,markers,marker_container",
    [
        (
            "schedules",
            "#scheduling-sync-status",
            "#scheduling-workbench",
            (
                "#scheduling-list-pane",
                "#scheduling-detail-pane",
                "#scheduling-inspector-pane",
            ),
            ("#schedules-follow-in-console",),
            ("#scheduling-task-detail-empty-state",),
            "#scheduling-detail-pane",
        ),
        (
            "workflows",
            "#workflows-mode-strip",
            "#workflows-workbench",
            (
                "#workflows-list-pane",
                "#workflows-detail-pane",
                "#workflows-inspector-pane",
            ),
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
    host = _visual_destination_harness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, workbench)
        await pilot.pause()
        _assert_ascii_workbench_contract(
            screen,
            workbench=workbench,
            strip=strip,
            strip_max_height=5 if route == "schedules" else 2,
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
        if route == "schedules":
            assert screen.query_one("#schedules-follow-in-console", Button).disabled


OPERATIONAL_LOADING_CONTRACTS = [
    (
        "schedules",
        SchedulesWorkbench,
        "load_tasks",
        "#scheduling-task-table",
        "#scheduling-list-pane",
        "#scheduling-sync-status",
        "#scheduling-workbench",
        (
            "#scheduling-list-pane",
            "#scheduling-detail-pane",
            "#scheduling-inspector-pane",
        ),
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
    load_started: asyncio.Event | None = None
    load_cancelled: asyncio.Event | None = None
    if route == "schedules":
        load_started = asyncio.Event()
        load_cancelled = asyncio.Event()

        async def hold_initial_load(self):
            load_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                load_cancelled.set()

        monkeypatch.setattr(screen_cls, refresh_method, hold_initial_load)
    else:
        monkeypatch.setattr(screen_cls, refresh_method, lambda self: None)
    app = _build_test_app()
    host = _visual_destination_harness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, loading_marker)
        if load_started is not None:
            await asyncio.wait_for(load_started.wait(), timeout=1)
        _assert_ascii_workbench_contract(
            screen,
            workbench=workbench,
            strip=strip,
            strip_max_height=5 if route == "schedules" else 2,
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
        if route == "schedules":
            assert screen.query_one("#schedules-follow-in-console", Button).disabled
    if load_cancelled is not None:
        await asyncio.wait_for(load_cancelled.wait(), timeout=1)


async def _assert_advanced_run_reachable(screen, pilot) -> None:
    """The Advanced "Run Action" button is mounted and (when actions exist) focusable.

    #mcp-adv-run is the direct successor of the retired
    #unified-mcp-action-run, but unlike its predecessor it is NOT an
    always-visible column action: the inspector's Advanced escape hatch is a
    scrollable section, and in the default loaded state the rendered section
    content pushes the run button below the scroll fold (verified — an
    in-viewport assertion on it fails at both 140x42 and 100x32). The
    genuinely visible primary action is the rail row; this asserts the
    Advanced runner's surviving contract instead: mounted, not scrolled out
    via `display: none`, and focusable whenever the current section actually
    has actions.

    The button is legitimately *disabled* when the current section has zero
    actions (mirrors the legacy panel's `_sync_action_controls()`, which
    also disables `#unified-mcp-action-run` for a zero-descriptor section —
    see unified_mcp_panel.py). Phase 1's default "Overview" section has no
    actions, so `#mcp-adv-run` starts disabled by default; that is not a
    regression, it is `MCPInspector._refresh_advanced_actions()` correctly
    reflecting the loaded section instead of the stale action set the mount
    happened to compute before the section finished loading.

    T12: the whole Advanced block now lives inside a `Collapsible` that
    defaults to collapsed (`display: none` on its contents), so this expands
    it first when needed -- this check's actual subject (does the button get
    lost to overflow/CSS once the pane IS open) is unchanged by that; it
    would otherwise trivially fail on every run against the new default.
    """
    try:
        collapsible = screen.query_one("#mcp-adv-collapsible", Collapsible)
    except NoMatches:
        collapsible = None
    if collapsible is not None and collapsible.collapsed:
        collapsible.collapsed = False
        await pilot.pause()
    adv_run = screen.query_one("#mcp-adv-run", Button)
    assert _is_effectively_displayed(adv_run), "#mcp-adv-run is not displayed"
    if not adv_run.disabled:
        assert adv_run.can_focus, "#mcp-adv-run is not focusable"


@pytest.mark.asyncio
async def test_mcp_uses_visible_server_detail_readiness_layout_without_overflow():
    """Realigned from the retired `UnifiedMCPPanel` 3-column embed.

    Task 8 replaced the embedded panel with the rail/canvas/inspector
    `MCPWorkbench` triad. Same product intent -- server list, server detail,
    and readiness/actions each get a fully visible, non-overflowing pane --
    verified against the new `#mcp-hub-rail` / `#mcp-hub-canvas` /
    `#mcp-hub-inspector` landmarks.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#mcp-hub-workbench")
        _assert_strip_compact(screen, "#mcp-title", max_height=1)
        _assert_strip_compact(screen, "#mcp-purpose", max_height=1)
        _assert_ascii_workbench_contract(
            screen,
            workbench="#mcp-hub-workbench",
            strip="#mcp-mode-strip",
            panes=("#mcp-hub-rail", "#mcp-hub-canvas", "#mcp-hub-inspector"),
            actions=("#mcp-rail-row-0",),
            height=42,
            min_pane_rows=30,
        )
        await _assert_advanced_run_reachable(screen, pilot)


@pytest.mark.asyncio
async def test_mcp_unavailable_or_local_default_state_keeps_workbench_geometry():
    """Realigned from the retired `UnifiedMCPPanel` embed (same intent as above).

    Verifies the default local-source state (no server selected, only the
    built-in server present) does not break the workbench triad's geometry,
    and that the servers-mode overview content stays confined to the canvas
    pane rather than escaping it (successor to the old
    `#unified-mcp-content` inside `#mcp-detail-pane` check).
    """
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#mcp-hub-workbench")
        _assert_ascii_workbench_contract(
            screen,
            workbench="#mcp-hub-workbench",
            strip="#mcp-mode-strip",
            panes=("#mcp-hub-rail", "#mcp-hub-canvas", "#mcp-hub-inspector"),
            actions=("#mcp-rail-row-0",),
            height=42,
        )
        await _assert_advanced_run_reachable(screen, pilot)
        _assert_marker_inside_container(
            screen,
            "#mcp-overview-summary",
            "#mcp-hub-canvas",
            context="MCP servers-mode overview content escaped canvas pane",
        )


@pytest.mark.asyncio
async def test_mcp_forced_loading_state_stays_inside_workbench(monkeypatch):
    """Realigned from the retired `UnifiedMCPPanel.load_context` monkeypatch.

    `UnifiedMCPPanel` is no longer mounted anywhere in the MCP screen's
    component chain (Task 8 replaced it with `MCPWorkbench`), so forcing
    *its* `load_context()` to hang no longer has any effect. Same product
    intent -- force the surface to sit in a still-loading state and verify
    nothing overflows its pane geometry -- reproduced by no-op'ing
    `MCPWorkbench.reload()`, which pins the rail/canvas/inspector to their
    just-mounted, pre-reload compose shape (empty rail besides "All servers";
    inspector's default "Select a server" prompt) for the whole test. A no-op
    rather than a hung await: `MCPWorkbench.on_mount` awaits `reload()`
    inline, so an await that never resolves would deadlock app teardown.
    """

    async def keep_loading(self):
        return None

    monkeypatch.setattr(MCPWorkbench, "reload", keep_loading)
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#mcp-hub-inspector")
        _assert_ascii_workbench_contract(
            screen,
            workbench="#mcp-hub-workbench",
            strip="#mcp-mode-strip",
            panes=("#mcp-hub-rail", "#mcp-hub-canvas", "#mcp-hub-inspector"),
            actions=("#mcp-adv-run",),
            height=42,
        )
        _assert_marker_inside_container(
            screen,
            "#mcp-inspector-state",
            "#mcp-hub-inspector",
            context="MCP forced loading state escaped inspector pane",
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
            (
                "#settings-category-pane",
                "#settings-detail-pane",
                "#settings-impact-pane",
            ),
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
            (
                "#settings-category-pane",
                "#settings-detail-pane",
                "#settings-impact-pane",
            ),
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
async def test_settings_dirty_category_status_has_visual_marker_class():
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _click_settings_category(screen, pilot, "console-behavior")
        await pilot.click("#settings-console-collapse-large-pastes-toggle")
        status = screen.query_one("#settings-category-console-behavior-status")

        assert "Status: Unsaved" in str(status.renderable)
        assert status.has_class("settings-dirty-category")


@pytest.mark.asyncio
async def test_settings_advanced_config_controls_use_action_and_status_rows():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _click_settings_category(screen, pilot, "advanced-config")
        await _wait_for_selector(screen, pilot, "#settings-advanced-config-editor")

        actions = screen.query_one("#settings-advanced-config-actions")
        result = screen.query_one("#settings-advanced-config-result")

        assert actions.has_class("settings-action-row")
        assert result.has_class("settings-status-row")
        for selector in (
            "#settings-advanced-config-editor",
            "#settings-advanced-validate-config",
            "#settings-advanced-save-config",
            "#settings-advanced-config-result",
        ):
            _assert_marker_inside_container(
                screen,
                selector,
                "#settings-detail-pane",
                context=f"Advanced config control escaped Settings detail pane: {selector}",
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
        # With no runtime, the middle column reads "Runtime Setup" (not the
        # old combined "Session Detail / Runtime Setup" label).
        assert "Runtime Setup" in visible_text
        assert "Session Detail / Runtime Setup" not in visible_text
        assert "Compatibility / Actions" in visible_text
        assert "Runtime owner: ACP" in visible_text
        assert "ACP version: n/a" in visible_text
        assert "Column 1:" not in visible_text
        assert "Column 2:" not in visible_text
        assert "Column 3:" not in visible_text
        runtime_copy = str(screen.query_one("#acp-empty-state").renderable)
        assert "Settings" not in runtime_copy
        assert "Configure ACP runtime setup in ACP" in runtime_copy


COMPACT_DESTINATION_CONTRACTS = {
    "home": {
        "identity": "#home-header-line",
        "workbench": "#home-triage-grid",
        "object": "#home-rail",
        "detail": "#home-canvas",
        "actions": (
            "#home-primary-action",
            "#home-open-details",
            "#home-open-chatbook-details",
        ),
    },
    "chat": {
        "identity": "#console-workbench-header",
        "workbench": "#console-workspace-grid",
        "object": "#console-left-rail",
        "detail": "#console-session-surface",
        "actions": (
            "#console-send-message",
            "#console-attach-context",
            "#console-save-chatbook",
        ),
    },
    "library": {
        # #library-title / #library-contract-grid / #library-source-browser
        # / #library-source-detail / #library-open-* are all retired (see
        # test_destination_shells.py); the rail + canvas shell is the
        # surviving 2-pane workbench, with the header line as identity and
        # always-visible rail rows as the compact-viewport actions.
        "identity": "#library-header-line",
        "workbench": "#library-shell-grid",
        "object": "#library-rail",
        "detail": "#library-canvas",
        "actions": (
            "#library-row-browse-search",
            "#library-row-browse-media",
            "#library-row-ingest-import-media",
        ),
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
        "identity": "#personas-header",
        "workbench": "#personas-workbench",
        "object": "#personas-library-pane",
        "detail": "#personas-work-area",
        "actions": ("#personas-library-new", "#personas-attach-to-console"),
    },
    "watchlists_collections": {
        "identity": "#watchlists-collections-title",
        "workbench": "#wl-workbench",
        "object": "#watchlists-list-pane",
        "detail": "#watchlists-detail-pane",
        # `#nav-overview` retired with the left-rail navigator -- see the
        # note on the same key in SOURCE_PREP_WORKBENCHES above.
        "actions": (
            "#wl-tab-overview",
            "#wc-empty-create-source",
            "#wc-open-watchlists",
            "#watchlists-follow-in-console",
        ),
    },
    "schedules": {
        "identity": "#scheduling-sync-status",
        "workbench": "#scheduling-workbench",
        "object": "#scheduling-list-pane",
        "detail": "#scheduling-detail-pane",
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
        # #mcp-workbench / #mcp-server-tree-pane / #mcp-detail-pane /
        # #mcp-readiness-pane are all retired (see test_destination_shells.py):
        # Task 8 replaced the embedded `UnifiedMCPPanel` with the
        # rail/canvas/inspector `MCPWorkbench` triad, mounted as
        # #mcp-hub-workbench. Same intent -- rail (server list) as "object",
        # canvas (server detail/overview) as "detail" -- verified against the
        # new landmarks. The compact-viewport visible action is the
        # always-visible rail row: #mcp-adv-run (successor of the retired
        # #unified-mcp-action-run) sits below the inspector's Advanced scroll
        # fold at 100x32 (verified), and is covered by
        # _assert_advanced_run_reachable + the tab-order test instead.
        "identity": "#mcp-title",
        "workbench": "#mcp-hub-workbench",
        "object": "#mcp-hub-rail",
        "detail": "#mcp-hub-canvas",
        "actions": ("#mcp-rail-row-0",),
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
        # The Overview card grew; #settings-open-appearance now sits below the
        # compact fold. Any of these visible actions satisfies the contract.
        "actions": (
            "#settings-manual-sync-preview",
            "#settings-save-category",
            "#settings-open-appearance",
        ),
    },
}


TOP_LEVEL_WORKBENCH_SELECTORS = {
    route: contract["workbench"]
    for route, contract in COMPACT_DESTINATION_CONTRACTS.items()
}


@pytest.mark.parametrize("route,contract", COMPACT_DESTINATION_CONTRACTS.items())
@pytest.mark.asyncio
async def test_top_level_destinations_keep_primary_workbench_visible_at_compact_size(
    route, contract
):
    app = _build_test_app()
    if route == "home":
        host = HomeHarness(app)
    elif route == "chat":
        host = ConsoleHarness(app)
    else:
        host = _visual_destination_harness(app, route)
    async with host.run_test(size=(100, 32)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, contract["workbench"])
        nav = screen.query_one(MainNavigationBar)
        assert nav.region.y == 0, (
            f"{route}: global nav is not docked at top: {nav.region}"
        )
        assert nav.region.height <= 3, f"{route}: global nav is too tall: {nav.region}"
        _assert_visible_in_viewport(
            nav, height=32, context=f"{route}:global-nav", viewport_width=100
        )
        assert list(nav.query(Button)), (
            f"{route}: global nav has no visible destination buttons"
        )
        for required in ("identity", "workbench", "object", "detail"):
            _assert_visible_in_viewport(
                screen.query_one(contract[required]),
                height=32,
                context=f"{route}:{required}:{contract[required]}",
                viewport_width=100,
            )
        if route == "schedules":
            assert screen.has_class("schedules-workbench-compact")
            compact_inspector = screen.query_one("#scheduling-inspector-pane")
            assert _is_effectively_displayed(compact_inspector)
            assert compact_inspector.region.width > 0
            _assert_visible_in_viewport(
                compact_inspector,
                height=32,
                context="schedules:compact-inspector",
                viewport_width=100,
            )
            for width, compact in ((121, False), (120, True)):
                await pilot.resize_terminal(width, 32)
                await pilot.pause()
                assert screen.has_class("schedules-workbench-compact") is compact
                for selector in (
                    "#scheduling-list-pane",
                    "#scheduling-detail-pane",
                    "#scheduling-inspector-pane",
                ):
                    pane = screen.query_one(selector)
                    assert _is_effectively_displayed(pane)
                    assert pane.region.width > 0
                    _assert_visible_in_viewport(
                        pane,
                        height=32,
                        context=f"schedules:{width}:{selector}",
                        viewport_width=width,
                    )
            await pilot.resize_terminal(100, 32)
            await pilot.pause()
            assert screen.has_class("schedules-workbench-compact")
        _assert_any_action_visible(
            screen,
            contract["actions"],
            height=32,
            context=f"{route}:compact-action",
            viewport_width=100,
        )


VISIBLE_FOCUS_TARGETS = {
    "home": {
        "home-primary-action",
        "home-open-details",
        "home-open-in-console",
        "home-open-chatbook-details",
    },
    "chat": {
        "console-send-message",
        "console-attach-context",
        "console-save-chatbook",
        "console-run-library-rag",
    },
    # The retired #library-open-* buttons lived only in the never-mounted
    # #library-source-browser; the always-visible rail rows are the
    # surviving tab-focusable primary actions.
    "library": {
        "library-row-browse-search",
        "library-row-browse-media",
        "library-row-ingest-import-media",
    },
    "artifacts": {
        "artifacts-open-chatbooks",
        "artifacts-open-console",
        "artifacts-open-library",
        "artifacts-import-artifact",
        "artifacts-use-in-console",
    },
    "personas": {"personas-library-new", "personas-attach-to-console"},
    "watchlists_collections": {
        "wc-open-watchlists",
        "wc-attach-to-console",
        "watchlists-follow-in-console",
    },
    "schedules": {"schedules-follow-in-console"},
    "workflows": {"workflows-launch-in-console"},
    # #unified-mcp-action-run is retired with the `UnifiedMCPPanel` embed
    # (Task 8); its direct successor is #mcp-adv-run, the workbench
    # inspector's Advanced "Run Action" button. It is legitimately disabled
    # by default (Phase 1's default "Overview" section has no actions --
    # see _assert_advanced_run_reachable), so #mcp-rail-row-0 ("All
    # servers", always enabled) is listed alongside it as the genuinely
    # reachable default primary action.
    "mcp": {"mcp-rail-row-0", "mcp-adv-run"},
    "acp": {"acp-follow-in-console", "acp-launch-agent"},
    "skills": {"skills-import-skill", "skills-attach-to-console"},
    # Theme and Splash Screen are now first-class sidebar categories, so the
    # primary tab-focusable actions on Settings are the category rail buttons.
    "settings": {"settings-category-overview", "settings-category-providers-models"},
}

#: Tab-order search budget per destination, default 24. Watchlists needs a
#: few more presses: `WatchlistsWorkbench`'s five regions are each
#: individually focusable (`can_focus = True`, so `z` can target whichever
#: one has focus — see `watchlists_workbench.py`), adding five stops on top
#: of everything the pre-rehost tree already had. Measured empirically at 29
#: presses to `wc-open-watchlists` with the default `_build_test_app()`
#: empty-state fixture; 32 leaves a small margin.
TAB_ORDER_ATTEMPTS = {
    "watchlists_collections": 32,
}


@pytest.mark.parametrize("route,targets", VISIBLE_FOCUS_TARGETS.items())
@pytest.mark.asyncio
async def test_tab_order_reaches_visible_primary_action(route, targets):
    app = _build_test_app()
    if route == "schedules":
        app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
            HomeActiveWorkItem(
                item_id="local:schedule_run:visual-parity",
                title="Visual parity schedule",
                source="Schedules",
                status="running",
                detail_route="schedules",
                console_available=True,
            )
        )
    if route == "home":
        host = HomeHarness(app)
    elif route == "chat":
        _mark_console_onboarding_complete(app)
        host = ConsoleHarness(app)
    else:
        host = _visual_destination_harness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = host.screen_stack[-1]
        workbench = TOP_LEVEL_WORKBENCH_SELECTORS[route]
        await _wait_for_selector(screen, pilot, workbench)
        if route == "schedules":
            for _ in range(20):
                follow_button = screen.query_one("#schedules-follow-in-console", Button)
                if not follow_button.disabled:
                    break
                await pilot.pause()
            assert not follow_button.disabled
        target_buttons = [
            screen.query_one(f"#{target}", Button)
            for target in targets
            if list(screen.query(f"#{target}"))
        ]
        enabled_targets = {
            button.id for button in target_buttons if button.id and not button.disabled
        }
        if not enabled_targets:
            _assert_any_action_visible(
                screen,
                tuple(f"#{target}" for target in targets),
                height=42,
                context=f"{route}:disabled-recovery-action",
                viewport_width=140,
            )
            return
        for _ in range(TAB_ORDER_ATTEMPTS.get(route, 24)):
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
        pytest.fail(
            f"{route} did not focus a visible primary action from {sorted(enabled_targets)}"
        )


@pytest.mark.asyncio
async def test_watchlists_feed_source_row_stays_one_row_however_long_the_name():
    """Task 7 fix round 1, Finding 1 (CSS half): `watchlist-feed-source-row`
    shipped as a class name with no rule behind it.

    It now has one, and `height: 1` is the load-bearing declaration. FEEDS is
    capped at `max-height: 12` and that cap was derived against one-row
    children; a bare `Static` sizes to `auto`, so a long feed title -- these
    arrive from remote feeds and OPML imports, not from us -- would wrap and
    silently eat the region's budget. Measured through the production
    stylesheet, since a bare `App` with no CSS cannot see this at all.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    long_name = " ".join(["A remote feed title that runs on well past the feeds column"] * 6)
    source_id = service._db.add_subscription(
        name=long_name, type="rss", source="https://long.example/f"
    )
    service.add_source(watchlist["id"], source_id)

    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-workbench")
        screen._tree_watchlists = [{"id": watchlist["id"], "name": "Morning AI Brief"}]
        screen._apply_layout(RegionLayout())
        await pilot.pause()

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        for _ in range(20):
            await pilot.pause()
            if list(screen.query(f"#wl-feeds-source-{source_id}")):
                break

        row = screen.query_one(f"#wl-feeds-source-{source_id}", Static)
        assert row.region.height == 1, (
            f"a long feed name must not wrap the row to {row.region.height} rows; "
            "FEEDS's max-height was derived against one-row children"
        )
        feeds = screen.query_one("#wl-region-feeds")
        assert feeds.region.contains_region(row.region), (
            f"the row should sit inside FEEDS: feeds={feeds.region} row={row.region}"
        )


@pytest.mark.asyncio
async def test_watchlists_tree_selection_is_visually_distinct_against_the_bundle():
    """task-876: the tree node matching `tree_scope` must be visually
    distinguished from its siblings under the REAL stylesheet.

    A bare `App` would pass this even if `.watchlist-tree-watchlist.is-active`
    carried no rule at all -- the exact LabModeStrip/Watchlists-tab-strip
    failure mode this program has already hit twice. Loads the production
    bundle (`_visual_destination_harness`) and reads the resolved styles
    Textual actually computed, not a bare class-presence check.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")

    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(
            screen, pilot, f"#wl-tree-node-watchlist-{watchlist['id']}"
        )

        all_button = screen.query_one("#wl-tree-node-all", Button)
        watchlist_button = screen.query_one(
            f"#wl-tree-node-watchlist-{watchlist['id']}", Button
        )

        # Resting state: `tree_scope` defaults to "all", so the "All
        # sources" root starts active and must already read differently
        # from its as-yet-unselected "Morning AI Brief" sibling. `text_style`
        # is NOT part of this comparison: Textual's own default Button
        # variant (`-style-default`) is unconditionally bold, so both nodes
        # already agree on that regardless of `is-active` -- background and
        # foreground colour are what this CSS actually changes.
        assert all_button.has_class("is-active")
        assert not watchlist_button.has_class("is-active")
        assert all_button.styles.background != watchlist_button.styles.background, (
            "the active root must not share its background with an inactive "
            "sibling under the production stylesheet"
        )
        assert all_button.styles.color != watchlist_button.styles.color

        # Click the watchlist node: the highlight must MOVE, not merely
        # duplicate onto a second node. `active_scope` is `recompose=True`
        # on `WatchlistTree` itself, so the click swaps in brand new button
        # instances -- the `all_button`/`watchlist_button` references above
        # are now stale and must be re-queried, not reused.
        await pilot.click(f"#wl-tree-node-watchlist-{watchlist['id']}")
        await pilot.pause()

        all_button = screen.query_one("#wl-tree-node-all", Button)
        watchlist_button = screen.query_one(
            f"#wl-tree-node-watchlist-{watchlist['id']}", Button
        )
        assert watchlist_button.has_class("is-active")
        assert not all_button.has_class("is-active")
        assert all_button.styles.background != watchlist_button.styles.background
        assert all_button.styles.color != watchlist_button.styles.color

        # And the label itself must actually be painted, not clipped away by
        # the border-round-in-a-one-row-strip defect this program has
        # already hit twice (task-875) -- `render_line()` would not catch a
        # regression of that shape; the compositor is ground truth.
        _assert_label_intact_on_screen(
            watchlist_button, "Morning AI Brief", context="tree watchlist node"
        )


def _row_reverse_video(strips, region, needle: str) -> bool | None:
    """Whether ANY segment in the row containing `needle` renders reverse
    video, or `None` if `needle` is not on screen inside `region` at all.
    """
    for y in range(region.y, region.y + region.height):
        if not (0 <= y < len(strips)):
            continue
        row_segments = strips[y]
        row_text = "".join(segment.text for segment in row_segments)
        if needle in row_text:
            return any(
                bool(getattr(segment.style, "reverse", False))
                for segment in row_segments
            )
    return None


@pytest.mark.asyncio
async def test_sources_pane_selected_row_renders_reverse_video_under_the_bundle():
    """task-876, AC #6: confirm the Sources selection highlight is real
    painted output, not just a Python-side style attribute, under the
    production stylesheet + theme (`SourcesPane`/`RunsPane`/
    `NotificationsPane` share the identical mechanism -- a `reverse bold`
    Rich `Text` style baked into the selected row's cells, since a
    DataTable cell cannot reference Textual CSS variables; see
    `SourcesPane._SELECTED_ROW_STYLE`'s docstring).

    Does not also assert against a second, merely-focused row: this
    destination's `#sources-toolbar` currently claims nearly all of
    `SourcesPane`'s vertical budget in the full shell (measured: 33 of 34
    rows at 160x60), leaving the table only 1 visible row regardless of
    terminal size -- a real, pre-existing layout defect unrelated to this
    task's CSS-vs-bare-App concern. The per-pane unit tests in
    Tests/Watchlists/test_watchlists_*_pane.py already cover "does the
    highlight move / does an unselected row stay unstyled" with two rows
    both on screen; this test only needs to confirm the SAME mechanism
    still renders as reverse video once real CSS/theme are in the loop.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "sources"
        await pilot.pause(0.2)

        sources_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        sources_pane.sources = [
            {"id": 1, "name": "AI News RSS", "source_type": "rss", "active": True},
        ]
        await pilot.pause()
        sources_pane.select_source_by_id("1")
        await pilot.pause()

        table = sources_pane.query_one("#sources-table", DataTable)
        strips = screen._compositor.render_strips()
        assert _row_reverse_video(strips, table.region, "AI News RSS") is True, (
            "the selected row must actually paint as reverse video under the "
            "production stylesheet, not just carry the style in Python"
        )


@pytest.mark.asyncio
async def test_runs_pane_selected_row_renders_reverse_video_under_the_bundle():
    """Same confirmation as the Sources pane test above, for `RunsPane`."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "runs"
        await pilot.pause(0.2)

        runs_pane = screen.query_one("#watchlists-runs-pane", RunsPane)
        runs_pane.runs = [
            {"id": "run-1", "source_title": "AI News RSS", "status": "completed"},
        ]
        await pilot.pause()
        runs_pane.select_run_by_id("run-1")
        await pilot.pause()

        table = runs_pane.query_one("#runs-table", DataTable)
        strips = screen._compositor.render_strips()
        assert _row_reverse_video(strips, table.region, "AI News RSS") is True, (
            "the selected run row must actually paint as reverse video under "
            "the production stylesheet"
        )


@pytest.mark.asyncio
async def test_notifications_pane_selected_row_renders_reverse_video_under_the_bundle():
    """Same confirmation as the Sources pane test above, for
    `NotificationsPane` -- whose `selected_notification` is `recompose=True`
    (unlike Sources/Runs), so the highlight is applied entirely in
    `compose()` rather than via a targeted `update_cell`.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "notifications"
        await pilot.pause(0.2)

        notifications_pane = screen.query_one(
            "#watchlists-notifications-pane", NotificationsPane
        )
        notifications_pane.notifications = [
            {"id": 1, "title": "Research complete", "category": "research"},
        ]
        await pilot.pause()
        notifications_pane.select_notification_by_id("1")
        await pilot.pause()

        table = notifications_pane.query_one("#notifications-table", DataTable)
        strips = screen._compositor.render_strips()
        assert _row_reverse_video(strips, table.region, "Research complete") is True, (
            "the selected notification row must actually paint as reverse "
            "video under the production stylesheet"
        )


@pytest.mark.asyncio
async def test_watchlists_right_rail_says_inspector_exactly_once():
    """Post-branch live-capture finding: the RIGHT_RAIL rendered "Inspector"
    twice in one box.

    `_build_inspector_pane` opens the region with
    `Static("Inspector", classes="destination-section watchlists-column-title")`
    -- the rail's heading, left-aligned, covering the state summaries and
    Console actions as well as the entity inspector. `InspectorPane.compose`
    then yielded its own `Static("Inspector", classes="pane-title")`, centred,
    directly below the Console action buttons.

    Task 6 ("one border, one title per region") did not catch this because it
    compared each REGION's title against its content's, and both of these
    live inside the content. The screenshot caught it in seconds -- which is
    the whole argument for looking at the assembled app before shipping.

    Asserts on the rendered text rather than on widget identity so it fails
    for any future re-introduction, whichever widget emits the duplicate.
    """
    app = _build_test_app()
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-workbench")
        screen._apply_layout(RegionLayout())
        await pilot.pause()

        rail = screen.query_one("#wl-region-right_rail")
        headings = [
            str(node.renderable).strip()
            for node in rail.query(Static)
            if str(node.renderable).strip() == "Inspector"
        ]
        assert len(headings) == 1, (
            f"the right rail must carry exactly one 'Inspector' heading, "
            f"found {len(headings)}"
        )
        # The one that survives is the rail's, not the nested pane's: the
        # region holds more than the entity inspector.
        assert "Inspector" in _visible_static_text(screen), (
            "dropping the duplicate must not drop the heading entirely"
        )


@pytest.mark.parametrize("size", [(160, 42), (235, 52)])
@pytest.mark.asyncio
async def test_watchlists_sources_toolbar_does_not_starve_its_table(size):
    """TASK-897: `#sources-toolbar` took every row its pane had.

    It is a bare `Vertical` with no height rule anywhere in the stylesheet,
    so it inherited Textual's `height: 1fr` default and claimed all the
    space in `SourcesPane`, leaving `#sources-table` a single visible row --
    at any terminal size, because `1fr` grows with the pane. The Sources
    section is the screen's main list of what a user is monitoring, so it
    showed one source at a time.

    Same shape as the FEEDS clipping bug: a height that is fine in
    isolation and wrong once the widget is nested in the real layout. Which
    is why this runs in the full shell under the production stylesheet -- a
    bare `App` with no CSS cannot see it, and on this screen that blind spot
    has now shipped three separate defects.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=size) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "sources"
        await pilot.pause(0.2)

        sources_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        # The empty state is where this bites, and it is the first thing a
        # new user sees: with no rows the table collapses to 1 and the
        # elastic toolbar swallows the rest of the pane. Measured pre-fix at
        # 160x42: toolbar=15, table=1 inside a 16-row pane.
        toolbar = sources_pane.query_one("#sources-toolbar")
        table = sources_pane.query_one("#sources-table", DataTable)
        assert toolbar.region.height <= 6, (
            f"the toolbar must take only what its own controls need, not "
            f"whatever the table is not using: toolbar={toolbar.region} "
            f"table={table.region} pane={sources_pane.region}"
        )

        sources_pane.sources = [
            {"id": i, "name": f"feed-{i:02d}", "source_type": "rss", "active": True}
            for i in range(1, 13)
        ]
        await pilot.pause()

        toolbar = sources_pane.query_one("#sources-toolbar")
        table = sources_pane.query_one("#sources-table", DataTable)

        assert table.region.height > 1, (
            f"the sources table must show more than one row; the toolbar is "
            f"eating the pane: toolbar={toolbar.region} table={table.region} "
            f"pane={sources_pane.region}"
        )
        assert table.region.height >= toolbar.region.height, (
            f"the table is the point of this pane and must not be shorter "
            f"than its own toolbar: toolbar={toolbar.region} "
            f"table={table.region}"
        )



@pytest.mark.asyncio
async def test_watchlists_tree_action_labels_fit_the_rail_intact():
    """TASK-895: the tree's five write verbs must be readable in the real
    28-column rail, not clipped to an ellipsis.

    A bare `App` cannot see this at all. Textual's own Button CSS pins
    `min-width: 16` and `compact=True` only drops the border, so three
    action buttons in a `Horizontal` claim 48 columns inside a 26-column
    interior unless `features/_watchlists.tcss` overrides it -- measured
    pre-rule as `New Rena… Dele…`. The compositor is the instrument
    (`render_strips()` via `_composited_rows`): `render_line()` returns each
    button's own self-computed strip and would report the full label even
    while the button overflowed its rail.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-tree-new")
        screen._apply_layout(RegionLayout())
        await pilot.pause()

        rail = screen.query_one("#wl-region-left_rail")
        for label in ("New", "Rename", "Delete", "Add source", "Remove"):
            _assert_label_intact_on_screen(
                rail, label, context=f"tree action {label!r}"
            )

        # And every action button sits inside the rail's own box rather than
        # spilling past its right edge, which is what the `min-width: 16`
        # default does when it wins.
        for action_id in (
            "#wl-tree-new",
            "#wl-tree-rename",
            "#wl-tree-delete",
            "#wl-tree-add-source",
            "#wl-tree-remove-source",
        ):
            button = screen.query_one(action_id, Button)
            assert rail.region.contains_region(button.region), (
                f"{action_id} escapes the left rail: rail={rail.region} "
                f"button={button.region}"
            )


@pytest.mark.asyncio
async def test_watchlists_tree_blocked_verbs_render_as_disabled_under_the_bundle():
    """TASK-895 / AC #5's rendering half: a disabled action must not paint
    like a live one.

    "A disabled button that looks enabled" is a defect this program has
    already fixed once, and `disabled=True` alone is a Python attribute --
    whether it reaches the screen depends on the theme and the bundle
    winning over `.watchlist-tree-action`'s own `border: none`. Compared
    against a sibling in the same strip that IS live, so the assertion
    cannot pass by both being styled identically.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wl-tree-new")

        live = screen.query_one("#wl-tree-new", Button)
        blocked = screen.query_one("#wl-tree-rename", Button)
        assert not live.disabled and blocked.disabled

        strips = screen._compositor.render_strips()

        def _cells(button):
            row = button.region.y
            column = 0
            out = []
            for segment in strips[row]:
                for char in segment.text:
                    if button.region.x <= column < button.region.x + button.region.width:
                        out.append((char, segment.style))
                    column += 1
            return out

        live_cells = [cell for cell in _cells(live) if cell[0].strip()]
        blocked_cells = [cell for cell in _cells(blocked) if cell[0].strip()]
        assert live_cells and blocked_cells, (
            "both buttons must actually be painted for this comparison to mean "
            f"anything: live={live_cells!r} blocked={blocked_cells!r}"
        )
        assert {str(style) for _, style in live_cells} != {
            str(style) for _, style in blocked_cells
        }, (
            "the disabled action paints identically to the live one under the "
            "production stylesheet"
        )
