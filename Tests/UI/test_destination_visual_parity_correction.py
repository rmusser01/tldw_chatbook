"""Visual parity geometry tests for destination correction pass."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button
from textual.widgets import Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _wait_for_selector,
)
from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from Tests.UI.test_screen_navigation import _build_test_app
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


def _assert_visible_in_viewport(widget, *, height: int, context: str) -> None:
    x, y, width, widget_height = _region(widget)
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


def _assert_any_action_visible(screen, selectors: tuple[str, ...], *, height: int, context: str) -> None:
    for selector in selectors:
        matches = list(screen.query(selector))
        if not matches:
            continue
        try:
            _assert_visible_in_viewport(matches[0], height=height, context=f"{context}:{selector}")
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


SOURCE_PREP_WORKBENCHES = {
    "artifacts": {
        "workbench": "#artifacts-workbench",
        "strip": "#artifacts-mode-strip",
        "panes": ("#artifacts-list-pane", "#artifacts-detail-pane", "#artifacts-inspector-pane"),
        "actions": (
            "#artifacts-open-chatbooks",
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
