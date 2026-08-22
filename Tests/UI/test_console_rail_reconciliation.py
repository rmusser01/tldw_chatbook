"""Atomic allocation contracts for the mounted Console Context rail."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from types import MethodType

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.events import MouseDown, MouseScrollDown, MouseUp
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_display_state import (
    ConversationFileEntry,
    ConsoleDisplayRow,
)
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.console_rail_state import ConsoleRailState
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsSummaryState
from tldw_chatbook.UI.Console_Modules import left_rail as left_rail_module
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
from tldw_chatbook.UI.Console_Modules.right_rail import ConsoleInspectorRail
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)
from tldw_chatbook.Widgets.Console.console_changed_files_section import (
    ConsoleChangedFilesSection,
    ConsoleChangedFilesState,
)
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector
from tldw_chatbook.Widgets.Console.console_settings_summary import (
    ConsoleSettingsSummary,
)
from tldw_chatbook.Widgets.Console.console_staged_context import (
    ConsoleStagedContextTray,
)
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSectionState,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    build_console_conversation_browser_state,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState


SECTION_IDS = (
    "session",
    "workspace",
    "conversations",
    "model",
    "agent",
    "details",
    "character",
)
LOCAL_HINT = "▼ more — scroll"
OUTER_HINT = "▼ more sections — scroll"


def _workspace_state() -> ConsoleWorkspaceContextState:
    return ConsoleWorkspaceContextState(
        heading="Context",
        workspace_label="Workspace: Default",
        authority_label="Authority: local",
        sync_label="Sync: ready",
        runtime_label="Runtime: local",
        conversation_rows=(),
        conversation_empty_copy="No conversations yet.",
        conversation_browser=build_console_conversation_browser_state(
            rows=(), active_workspace_id=None
        ),
        change_workspace_enabled=False,
        change_workspace_recovery="",
        new_conversation_enabled=False,
        new_conversation_recovery="",
        recovery_copy="",
    )


def _all_open_rail_state() -> ConsoleRailState:
    return ConsoleRailState(
        left_open=True,
        right_open=False,
        preferred_left_open=True,
        preferred_right_open=False,
        session_open=True,
        workspace_open=True,
        conversations_open=True,
        model_open=True,
        agent_open=True,
        details_open=True,
        character_open=True,
    )


class _RailHarness(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    ConsoleLeftRail {
        width: 40;
        height: 100%;
        min-height: 0;
    }

    .console-rail-header {
        height: 1;
        min-height: 1;
    }

    #console-left-rail-body {
        height: 1fr;
        min-height: 0;
        overflow-y: auto;
    }

    .console-rail-section-header {
        height: 2;
        min-height: 2;
    }

    .console-rail-section-body,
    ConsoleBoundedSection,
    .console-bounded-section-viewport {
        height: auto;
        min-height: 0;
    }

    .console-bounded-section-viewport {
        overflow-y: auto;
    }

    .console-bounded-section-hint,
    #console-left-rail-outer-hint {
        display: none;
        height: 1;
        min-height: 1;
    }
    """

    def __init__(self, *, show_character: bool = True) -> None:
        super().__init__()
        self.show_character = show_character
        self.section_toggles: list[str] = []

    def build_rail(self) -> ConsoleLeftRail:
        return ConsoleLeftRail(
            rail_state=_all_open_rail_state(),
            workspace_context_state=_workspace_state(),
            settings_summary_state=ConsoleSettingsSummaryState(
                model_row="Model: test",
                context_row="Context: 0",
                sampling_row="T 0.7 · max_tokens 100",
                identity_row="Identity: character",
            ),
            system_line_text="System: none",
            system_line_dim=True,
            fleet_line="2 agents running",
            agent_status_line="Running",
            agent_steps_text="one\ntwo\nthree",
            agent_fleet_section_state=ConsoleInspectorSectionState(rows=(), summary=""),
            agent_drilldown_active=False,
            agent_full_log_available=False,
            show_character_section=self.show_character,
            character_avatar_widget_builder=lambda: Static("avatar"),
            character_avatar_name="Samira",
        )

    def compose(self) -> ComposeResult:
        yield self.build_rail()

    def on_console_left_rail_section_toggled(
        self, event: ConsoleLeftRail.SectionToggled
    ) -> None:
        self.section_toggles.append(event.section_id)
        self.query_one(ConsoleLeftRail).apply_section_open(
            event.section_id,
            event.opened,
        )


class _ProductionConsoleHarness(ConsoleHarness):
    """Real ChatScreen host with the complete production CSS cascade."""

    CSS_PATH = TldwCli.CSS_PATH


async def _settle(pilot, passes: int = 5) -> None:
    for _ in range(passes):
        await pilot.pause()


async def _wait_for_rail_condition(
    pilot,
    rail: ConsoleLeftRail,
    condition,
    *,
    attempts: int = 20,
) -> None:
    """Wait until a mounted rail condition holds across two refresh turns."""

    stable_passes = 0
    for _ in range(attempts):
        await pilot.pause()
        idle = not rail._allocation_reconcile_scheduled and all(
            not section._reconcile_scheduled for section in _sections(rail)
        )
        if idle and condition():
            stable_passes += 1
            if stable_passes == 2:
                return
        else:
            stable_passes = 0
    pytest.fail("Context rail condition did not stabilize within the refresh bound")


def _install_demands(
    monkeypatch: pytest.MonkeyPatch,
    demands: dict[str, int],
) -> None:
    """Drive real body reconciliation from deterministic physical demands."""

    original = ConsoleBoundedSection._measure_content_lines

    def measure(viewport) -> int:
        section = viewport.parent
        if isinstance(section, ConsoleBoundedSection):
            return demands[section.section_id]
        return original(viewport)

    monkeypatch.setattr(
        ConsoleBoundedSection,
        "_measure_content_lines",
        staticmethod(measure),
    )


def _force_geometry(
    rail: ConsoleLeftRail,
    *,
    viewport_height: int,
    header_chrome_height: int,
) -> None:
    rail._snapshot_outer_viewport_height = MethodType(  # type: ignore[attr-defined]
        lambda self: viewport_height,
        rail,
    )
    rail._measure_visible_header_chrome_height = MethodType(  # type: ignore[attr-defined]
        lambda self, descriptors: header_chrome_height,
        rail,
    )


def _sections(rail: ConsoleLeftRail) -> Iterator[ConsoleBoundedSection]:
    return iter(rail.query("#console-left-rail-body ConsoleBoundedSection"))


async def _open_all_production_context_sections(host, pilot) -> ConsoleLeftRail:
    screen = host.screen_stack[-1]
    await _wait_for_selector(screen, pilot, "#console-left-rail")
    rail = screen.query_one("#console-left-rail", ConsoleLeftRail)
    if rail.display:
        screen.query_one("#console-context-rail-collapse", Button).press()
        await _settle(pilot)
    screen.query_one("#console-context-rail-open", Button).press()
    await _settle(pilot)
    for section_id in SECTION_IDS:
        header = rail.query_one(f"#console-rail-section-header-{section_id}")
        if not header.open:
            rail.query_one(f"#console-rail-section-toggle-{section_id}", Button).press()
            await _settle(pilot, passes=2)
    await _settle(pilot, passes=8)
    return rail


async def _open_production_inspector(host, pilot) -> ConsoleInspectorRail:
    """Open the mounted Inspector without replacing the production shell."""

    screen = host.screen_stack[-1]
    rail = screen.query_one("#console-right-rail", ConsoleInspectorRail)
    if not rail.display:
        screen.query_one("#console-inspector-rail-open", Button).press()
        await _settle(pilot, passes=8)
    assert rail.display
    return rail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_size",
    [pytest.param((235, 52), id="235x52"), pytest.param((160, 45), id="160x45")],
)
async def test_bounded_rail_shell_regions_are_compositor_contained(
    terminal_size: tuple[int, int],
) -> None:
    """The real shell paints a 20-row Sources viewport plus its 21st-row cue."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=terminal_size) as pilot:
        context = await _open_all_production_context_sections(host, pilot)
        inspector = await _open_production_inspector(host, pilot)
        screen = host.screen_stack[-1]
        tray = inspector.query_one(
            "#console-staged-context-tray", ConsoleStagedContextTray
        )
        sources = tray.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await sources.viewport.remove_children()
        content = Static(
            "\n".join(f"source content {row}" for row in range(20)),
            id="production-sources-boundary-content",
        )
        await sources.viewport.mount(content)
        sources.request_reconcile()
        inspector.request_outer_reconcile()
        await _wait_for_rail_condition(
            pilot,
            context,
            lambda: (
                sources.desired_content_lines == 20
                and sources.viewport.content_region.height == 20
                and not sources.hint.display
                and not inspector._outer_reconcile_scheduled
            ),
        )
        assert tray.region.contains_region(sources.region)
        assert sources.region.contains_region(sources.viewport.region)

        content.update("\n".join(f"source content {row}" for row in range(21)))
        sources.request_reconcile()
        inspector.request_outer_reconcile()
        await _wait_for_rail_condition(
            pilot,
            context,
            lambda: (
                sources.desired_content_lines == 21
                and sources.viewport.content_region.height == 20
                and sources.hint.display
                and str(sources.hint.renderable) == LOCAL_HINT
                and not inspector._outer_reconcile_scheduled
            ),
        )

        assert context.display and inspector.display
        assert all(
            context.query_one(f"#console-rail-section-header-{section_id}").open
            for section_id in SECTION_IDS
        )
        assert tray.region.contains_region(sources.region)
        assert sources.region.contains_region(sources.viewport.region)
        assert sources.region.contains_region(sources.hint.region)
        assert not sources.viewport.region.overlaps(sources.hint.region)
        outer_hint = inspector.query_one("#console-inspector-outer-scroll-hint", Static)
        assert outer_hint.display
        assert str(outer_hint.renderable) == OUTER_HINT
        assert not sources.hint.region.overlaps(outer_hint.region)
        assert inspector.region.contains_region(outer_hint.region)

        local_point = (
            sources.hint.region.x + 1,
            sources.hint.region.y,
        )
        local_hit = screen.get_widget_at(*local_point)[0]
        assert local_hit is sources.hint or sources.hint in local_hit.ancestors
        outer_point = (outer_hint.region.x + 1, outer_hint.region.y)
        outer_hit = screen.get_widget_at(*outer_point)[0]
        assert outer_hit is outer_hint or outer_hint in outer_hit.ancestors
        rendered = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in screen._compositor.render_strips()
        )
        assert LOCAL_HINT in rendered
        assert OUTER_HINT in rendered

        hint_region = sources.hint.region
        sources.viewport.scroll_end(animate=False, immediate=True)
        await _wait_for_rail_condition(
            pilot,
            context,
            lambda: (
                sources.viewport.scroll_y == sources.viewport.max_scroll_y
                and str(sources.hint.renderable) == ""
            ),
        )
        assert sources.hint.display
        assert sources.hint.region == hint_region


@pytest.mark.asyncio
async def test_production_inspector_counterfactual_ten_eleven_ten_reconciles() -> None:
    """The real Inspector adds and removes its pinned slot without feedback."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        inspector = await _open_production_inspector(host, pilot)
        outer = inspector.query_one("#console-inspector-rail-body")
        hint = inspector.query_one("#console-inspector-outer-scroll-hint", Static)
        await outer.remove_children()
        content = Static("counterfactual content", id="production-outer-demand")
        content.styles.height = 10
        await outer.mount(content)
        inspector.request_outer_reconcile()
        await _settle(pilot, passes=8)
        target_rail_height = inspector.region.height - (
            outer.content_region.height - 10
        )
        for _ in range(2):
            inspector.styles.height = target_rail_height
            inspector.styles.min_height = target_rail_height
            inspector.styles.max_height = target_rail_height
            inspector.refresh(layout=True)
            inspector.request_outer_reconcile()
            await _settle(pilot, passes=8)
            correction = outer.content_region.height - 10
            if correction == 0:
                break
            target_rail_height -= correction

        assert outer.content_region.height == 10
        assert outer.virtual_size.height == 10
        assert hint.display is False

        content.styles.height = 11
        content.refresh(layout=True)
        inspector.request_outer_reconcile()
        await _settle(pilot, passes=8)
        assert outer.virtual_size.height == 11
        assert outer.content_region.height == 9
        assert hint.display
        assert hint.region.height == 1
        assert str(hint.renderable) == OUTER_HINT
        assert not outer.region.overlaps(hint.region)

        outer.scroll_end(animate=False, immediate=True)
        await _settle(pilot, passes=3)
        assert outer.scroll_y == outer.max_scroll_y
        assert str(hint.renderable) == ""

        content.styles.height = 10
        content.refresh(layout=True)
        inspector.request_outer_reconcile()
        await _settle(pilot, passes=8)
        assert outer.virtual_size.height == 10
        assert outer.content_region.height == 10
        assert hint.display is False
        assert outer.scroll_y == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_size", "uses_outer_scroll"),
    [((120, 30), False), ((80, 24), True)],
)
async def test_production_css_uses_uncompressed_header_demand_and_reaches_every_section(
    terminal_size: tuple[int, int],
    uses_outer_scroll: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real CSS keeps two-row headers and selects fallback from fixed demand."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)
    allocation_calls = []
    real_allocate = left_rail_module.allocate_context_sections

    def spy_allocate(**kwargs):
        allocation_calls.append(kwargs)
        return real_allocate(**kwargs)

    monkeypatch.setattr(left_rail_module, "allocate_context_sections", spy_allocate)

    async with host.run_test(size=(120, 30)) as pilot:
        rail = await _open_all_production_context_sections(host, pilot)
        if terminal_size != (120, 30):
            await pilot.resize_terminal(*terminal_size)
            await _settle(pilot, passes=10)
        outer = rail.query_one("#console-left-rail-body")
        cue = rail.query_one("#console-left-rail-outer-hint", Static)
        headers = [
            rail.query_one(f"#console-rail-section-header-{section_id}")
            for section_id in SECTION_IDS
        ]

        def assert_stable_sibling_geometry() -> None:
            direct_children = list(outer.children)
            for current, following in zip(direct_children, direct_children[1:]):
                assert not current.virtual_region.overlaps(following.virtual_region), (
                    current.id,
                    current.virtual_region,
                    following.id,
                    following.virtual_region,
                )
            for section_id in SECTION_IDS:
                bounded = rail.query_one(
                    f"#console-bounded-section-{section_id}", ConsoleBoundedSection
                )
                viewport = bounded.viewport
                hint = bounded.hint
                expected_demand = max(
                    (
                        child.virtual_region_with_margin.bottom
                        for child in viewport.children
                        if child.display
                    ),
                    default=0,
                )
                assert bounded.desired_content_lines == expected_demand
                assert hint not in viewport.children
                assert bounded.region.contains_region(viewport.region), (
                    section_id,
                    bounded.region,
                    viewport.region,
                )
                if hint.display:
                    assert bounded.region.contains_region(hint.region)
                next_header_index = SECTION_IDS.index(section_id) + 1
                if next_header_index < len(SECTION_IDS):
                    next_header = headers[next_header_index]
                    assert not viewport.region.overlaps(next_header.region)
                    if hint.display:
                        assert not hint.region.overlaps(next_header.region)

        header_heights = [header.virtual_region.height for header in headers]
        assert all(height >= 2 for height in header_heights), (
            header_heights,
            outer.content_region.height,
            str(outer.styles.overflow_y),
            cue.display,
        )
        assert str(outer.styles.overflow_y) == (
            "auto" if uses_outer_scroll else "hidden"
        )
        assert cue.display is uses_outer_scroll

        if not uses_outer_scroll:
            assert_stable_sibling_geometry()
            assert outer.scroll_y == 0
            assert all(
                outer.content_region.contains_region(header.region)
                for header in headers
            )
            constrained = next(
                rail.query_one(f"#console-rail-section-toggle-{section_id}", Button)
                for section_id in SECTION_IDS
                if str(
                    rail.query_one(
                        f"#console-rail-section-toggle-{section_id}", Button
                    ).label
                )
                == "[>]"
            )
            constrained_section_id = constrained.id.removeprefix(
                "console-rail-section-toggle-"
            )
            constrained_header = rail.query_one(
                f"#console-rail-section-header-{constrained_section_id}"
            )
            constrained_title = rail.query_one(
                f"#console-rail-section-title-{constrained_section_id}", Static
            )
            assert str(constrained_title.renderable).endswith(" · no room")
            assert (
                cell_len(str(constrained_title.renderable))
                <= constrained_title.content_region.width
            )
            assert str(constrained_title.tooltip) == constrained_header.title
            assert constrained_header.virtual_region.height == 2
            constrained_body = rail.query_one(
                f"#console-bounded-section-{constrained_section_id}",
                ConsoleBoundedSection,
            )
            assert constrained_body.allocation == 0
            allocation_calls.clear()
            constrained.press()
            await _settle(pilot, passes=8)
            assert constrained_body.allocation is not None
            assert constrained_body.allocation > 0
            assert_stable_sibling_geometry()
            stable_call_count = len(allocation_calls)
            await _settle(pilot, passes=8)
            assert len(allocation_calls) == stable_call_count
            assert rail._allocation_reconcile_scheduled is False
            return

        outer.scroll_home(animate=False)
        await pilot.pause()
        assert str(cue.renderable) == OUTER_HINT
        for section_id, header in zip(SECTION_IDS, headers):
            bounded = rail.query_one(
                f"#console-bounded-section-{section_id}", ConsoleBoundedSection
            )
            outer.scroll_to(y=header.virtual_region.y, animate=False, immediate=True)
            await pilot.pause()
            assert header.region.overlaps(outer.content_region)
            assert bounded.viewport.region.overlaps(outer.content_region)


@pytest.mark.asyncio
async def test_mounted_context_activation_never_persists_but_toggle_writes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real ChatScreen mutation, focus, pointer, and [>] paths keep write scope."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(120, 30)) as pilot:
        console = host.screen_stack[-1]
        rail = await _open_all_production_context_sections(host, pilot)
        ordinary = next(
            rail.query_one(f"#console-rail-section-toggle-{section_id}", Button)
            for section_id in SECTION_IDS
            if str(
                rail.query_one(
                    f"#console-rail-section-toggle-{section_id}", Button
                ).label
            )
            != "[>]"
            and rail.query_one(
                f"#console-bounded-section-{section_id}", ConsoleBoundedSection
            ).desired_content_lines
            > 0
        )
        ordinary_section = ordinary.id.removeprefix("console-rail-section-toggle-")
        ordinary_header = rail.query_one(
            f"#console-rail-section-header-{ordinary_section}"
        )
        ordinary_body = rail.query_one(
            f"#console-bounded-section-{ordinary_section}", ConsoleBoundedSection
        )
        ordinary.press()
        await _wait_for_rail_condition(
            pilot,
            rail,
            lambda: not ordinary_header.open and ordinary_body.allocation == 0,
        )

        persisted: list[tuple[str, dict[str, bool]]] = []
        reconcile_runs = 0
        original_save = console._save_console_rail_preferences
        original_reconcile = rail._run_allocation_reconcile

        def save_spy(
            key: str,
            serialized: dict[str, bool],
            *,
            notify_on_failure: bool = False,
        ):
            persisted.append((key, serialized))
            return original_save(
                key,
                serialized,
                notify_on_failure=notify_on_failure,
            )

        def reconcile_spy() -> None:
            nonlocal reconcile_runs
            reconcile_runs += 1
            original_reconcile()

        monkeypatch.setattr(console, "_save_console_rail_preferences", save_spy)
        monkeypatch.setattr(rail, "_run_allocation_reconcile", reconcile_spy)

        if str(ordinary.label) == "[>]":
            assert await pilot.click(ordinary)
            await _wait_for_rail_condition(
                pilot,
                rail,
                lambda: (
                    rail._active_section_id == ordinary_section
                    and str(ordinary.label) != "[>]"
                ),
            )
            assert persisted == []
        assert await pilot.click(ordinary)
        await _wait_for_rail_condition(
            pilot,
            rail,
            lambda: ordinary_header.open,
        )
        assert len(persisted) == 1
        assert ordinary_body.allocation == 0
        assert str(ordinary.label) == "[>]"
        assert ordinary_section in rail._no_room_section_ids
        assert await pilot.click(ordinary)
        await _wait_for_rail_condition(
            pilot,
            rail,
            lambda: (
                rail._active_section_id == ordinary_section
                and (ordinary_body.allocation or 0) > 0
                and str(ordinary.label) != "[>]"
            ),
        )
        assert len(persisted) == 1
        assert reconcile_runs >= 1
        stable_runs = reconcile_runs
        await pilot.pause()
        assert reconcile_runs == stable_runs

        model_toggle = rail.query_one("#console-rail-section-toggle-model", Button)
        model_body = rail.query_one(
            "#console-bounded-section-model", ConsoleBoundedSection
        )
        model_toggle.focus()
        await _wait_for_rail_condition(
            pilot,
            rail,
            lambda: (
                rail._active_section_id == "model" and (model_body.allocation or 0) > 0
            ),
        )
        assert len(persisted) == 1

        provider_value = rail.query_one(
            "#console-model-section-provider .console-model-section-value", Static
        )
        provider_value.scroll_visible(animate=False)
        await pilot.pause()
        assert await pilot.click(provider_value)
        await _wait_for_rail_condition(
            pilot,
            rail,
            lambda: rail._active_section_id == "model",
        )
        assert len(persisted) == 1
        stable_runs = reconcile_runs
        await pilot.pause()
        assert reconcile_runs == stable_runs
        assert rail._allocation_reconcile_scheduled is False


@pytest.mark.asyncio
async def test_detached_rail_rejects_a_queued_active_reveal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 8)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 12)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=10, header_chrome_height=14)
        rail.activate_section("agent")
        await _settle(pilot)
        outer = rail.query_one("#console-left-rail-body")
        calls = []
        original_scroll_to = outer.scroll_to

        def spy_scroll_to(*args, **kwargs):
            calls.append((args, kwargs))
            return original_scroll_to(*args, **kwargs)

        monkeypatch.setattr(outer, "scroll_to", spy_scroll_to)
        queued_callbacks = []

        def capture_after_refresh(callback, *args):
            queued_callbacks.append((callback, args))

        monkeypatch.setattr(rail, "call_after_refresh", capture_after_refresh)
        rail._queue_active_reveal("agent")
        token = rail._active_reveal_token
        await rail.remove()

        assert rail._active_reveal_is_current(token, "agent") is False
        assert len(queued_callbacks) == 1
        callback, args = queued_callbacks.pop()
        callback(*args)
        assert queued_callbacks == []
        assert calls == []


@pytest.mark.asyncio
async def test_request_path_coalesces_and_allocates_one_complete_latest_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 30)
    _install_demands(monkeypatch, demands)
    calls = []
    real_allocate = left_rail_module.allocate_context_sections

    def spy_allocate(**kwargs):
        calls.append(kwargs)
        return real_allocate(**kwargs)

    monkeypatch.setattr(left_rail_module, "allocate_context_sections", spy_allocate)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18, header_chrome_height=14)
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert [section.allocation for section in _sections(rail)] == [
            1,
            1,
            0,
            0,
            0,
            0,
            0,
        ]
        calls.clear()

        demands["session"] = 1
        demands["workspace"] = 1
        rail.query_one(
            "#console-bounded-section-session", ConsoleBoundedSection
        ).request_reconcile()
        rail.query_one(
            "#console-bounded-section-workspace", ConsoleBoundedSection
        ).request_reconcile()
        await _settle(pilot)

        assert len(calls) == 1
        snapshot = calls[0]["sections"]
        assert [section.section_id for section in snapshot] == list(SECTION_IDS)
        assert [section.desired_content_rows for section in snapshot[:2]] == [1, 1]
        result = real_allocate(**calls[0])
        assert [section.allocation for section in _sections(rail)] == [
            item.allocated_content_rows for item in result.allocations
        ]
        assert [section.allocation for section in _sections(rail)] == [
            1,
            1,
            1,
            0,
            0,
            0,
            0,
        ]


@pytest.mark.asyncio
async def test_normal_allocation_is_active_first_and_constrained_press_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 30)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18, header_chrome_height=14)
        rail.request_allocation_reconcile()
        await _settle(pilot)

        assert [section.allocation for section in _sections(rail)] == [
            1,
            1,
            0,
            0,
            0,
            0,
            0,
        ]
        constrained = app.query_one("#console-rail-section-toggle-agent", Button)
        constrained.scroll_visible(animate=False)
        await pilot.pause()
        assert str(constrained.label) == "[>]"
        assert (
            str(app.query_one("#console-rail-section-title-agent", Static).renderable)
            == "Agent · no room"
        )

        await pilot.click(constrained)
        await _settle(pilot)

        assert app.section_toggles == []
        assert rail._rail_state.agent_open is True
        assert (
            rail.query_one(
                "#console-bounded-section-agent", ConsoleBoundedSection
            ).allocation
            == 1
        )
        assert (
            rail.query_one(
                "#console-bounded-section-session", ConsoleBoundedSection
            ).allocation
            == 1
        )
        assert (
            rail.query_one(
                "#console-bounded-section-workspace", ConsoleBoundedSection
            ).allocation
            == 0
        )
        assert str(constrained.label) != "[>]"

        expected_allocations = [1, 0, 0, 0, 1, 0, 0]
        assert [
            section.allocation for section in _sections(rail)
        ] == expected_allocations
        await rail.recompose()
        await _settle(pilot)

        assert [
            section.allocation for section in _sections(rail)
        ] == expected_allocations
        assert (
            str(app.query_one("#console-rail-section-toggle-agent", Button).label)
            != "[>]"
        )
        assert (
            str(app.query_one("#console-rail-section-toggle-workspace", Button).label)
            == "[>]"
        )
        assert len(list(rail.query("#console-bounded-section-agent"))) == 1


@pytest.mark.asyncio
async def test_normal_mode_disables_outer_scroll_and_short_mode_keeps_every_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 8)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        outer = app.query_one("#console-left-rail-body")
        cue = app.query_one("#console-left-rail-outer-hint", Static)

        _force_geometry(rail, viewport_height=30, header_chrome_height=14)
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert str(outer.styles.overflow_y) == "hidden"
        assert outer.scroll_y == 0
        assert cue.display is False

        await pilot.resize_terminal(60, 12)
        await _settle(pilot)
        _force_geometry(rail, viewport_height=10, header_chrome_height=14)
        rail.activate_section("agent")
        await _settle(pilot)
        allocations = [section.allocation for section in _sections(rail)]
        assert allocations == [1, 1, 1, 1, 7, 1, 1]
        assert all(
            allocation is not None and allocation > 0 for allocation in allocations
        )
        assert str(outer.styles.overflow_y) == "auto"
        assert cue.can_focus is False
        assert cue.display is True
        active_header = rail.query_one("#console-rail-section-header-agent")
        active = rail.query_one("#console-bounded-section-agent", ConsoleBoundedSection)
        assert active_header.region.overlaps(outer.content_region), (
            outer.scroll_y,
            outer.max_scroll_y,
            active_header.virtual_region,
            active_header.region,
            outer.content_region,
        )
        assert active.viewport.region.overlaps(outer.content_region)

        expected_allocations = [1, 1, 1, 1, 7, 1, 1]
        await rail.recompose()
        await _settle(pilot)
        outer = app.query_one("#console-left-rail-body")
        cue = app.query_one("#console-left-rail-outer-hint", Static)
        assert [
            section.allocation for section in _sections(rail)
        ] == expected_allocations
        assert str(outer.styles.overflow_y) == "auto"
        assert cue.display is True
        outer.scroll_home(animate=False)
        await pilot.pause()
        assert str(cue.renderable) == OUTER_HINT

        for section_id in SECTION_IDS:
            header = rail.query_one(f"#console-rail-section-header-{section_id}")
            bounded = rail.query_one(
                f"#console-bounded-section-{section_id}", ConsoleBoundedSection
            )
            outer.scroll_to(y=header.virtual_region.y, animate=False)
            await pilot.pause()
            assert header.region.overlaps(outer.content_region)
            assert bounded.viewport.region.overlaps(outer.content_region)


@pytest.mark.asyncio
async def test_fallback_reveal_runs_on_entry_but_unchanged_work_preserves_user_scroll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 8)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        outer = app.query_one("#console-left-rail-body")
        _force_geometry(rail, viewport_height=30, header_chrome_height=14)
        rail.activate_section("agent")
        await _settle(pilot)

        _force_geometry(rail, viewport_height=10, header_chrome_height=14)
        rail.request_allocation_reconcile()
        await _settle(pilot)
        active_header = rail.query_one("#console-rail-section-header-agent")
        assert outer.scroll_y > 0
        assert active_header.region.overlaps(outer.content_region)

        outer.scroll_home(animate=False)
        await pilot.pause()
        assert outer.scroll_y == 0
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert outer.scroll_y == 0

        demands["session"] = 9
        rail.query_one(
            "#console-bounded-section-session", ConsoleBoundedSection
        ).request_reconcile()
        await _settle(pilot)
        assert outer.scroll_y == 0


@pytest.mark.asyncio
async def test_no_room_suffix_uses_the_mounted_headers_canonical_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 30)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18, header_chrome_height=14)
        header = rail.query_one("#console-rail-section-header-agent")
        title = rail.query_one("#console-rail-section-title-agent", Static)
        header.title = "Workers"
        title.update("Workers")

        demands["session"] = 1
        rail.query_one(
            "#console-bounded-section-session", ConsoleBoundedSection
        ).request_reconcile()
        await _settle(pilot)
        assert str(title.renderable) == "Workers · no room"

        rail.activate_section("agent")
        await _settle(pilot)
        assert str(title.renderable) == "Workers"
        assert (
            str(rail.query_one("#console-rail-section-toggle-agent", Button).tooltip)
            == "Collapse Workers"
        )


@pytest.mark.asyncio
async def test_local_and_outer_hints_use_distinct_counterfactual_predicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 2)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        cue = app.query_one("#console-left-rail-outer-hint", Static)
        _force_geometry(rail, viewport_height=10, header_chrome_height=11)
        rail.activate_section("agent")
        await _settle(pilot)
        outer = app.query_one("#console-left-rail-body")
        outer.scroll_home(animate=False)
        await pilot.pause()

        local = rail.query_one("#console-bounded-section-session-hint", Static)
        assert local.display is True
        assert str(local.renderable) == LOCAL_HINT
        assert cue.display is True
        assert str(cue.renderable) == OUTER_HINT
        outer.scroll_end(animate=False)
        await pilot.pause()
        assert cue.display is True
        assert str(cue.renderable) == ""
        outer.scroll_home(animate=False)
        await pilot.pause()
        assert str(cue.renderable) == OUTER_HINT

        demands.update(dict.fromkeys(SECTION_IDS, 0))
        geometry = {"chrome": 10}
        rail._measure_visible_header_chrome_height = MethodType(  # type: ignore[attr-defined]
            lambda self, descriptors: geometry["chrome"],
            rail,
        )
        for section in _sections(rail):
            section.request_reconcile()
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert cue.display is False

        geometry["chrome"] = 11
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert cue.display is True
        assert str(cue.renderable) == OUTER_HINT

        geometry["chrome"] = 10
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert cue.display is False
        assert str(cue.renderable) == ""


@pytest.mark.asyncio
async def test_focus_and_pointer_activation_are_transient_and_open_close_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 3)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 36)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=28, header_chrome_height=14)
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert rail._active_section_id is None

        configure = rail.query_one("#console-model-section-configure", Button)
        configure.focus()
        await _settle(pilot)
        assert rail._active_section_id == "model"
        assert app.section_toggles == []

        reaction = rail.query_one("#console-character-reaction-open", Button)
        reaction.scroll_visible(animate=False)
        await pilot.pause()
        assert await pilot.click(reaction)
        await _settle(pilot)
        assert rail._active_section_id == "character"
        assert app.section_toggles == []

        rail.apply_section_open("workspace", False)
        await _settle(pilot)
        workspace_toggle = rail.query_one(
            "#console-rail-section-toggle-workspace", Button
        )
        await pilot.click(workspace_toggle)
        await _settle(pilot)
        assert rail._active_section_id == "workspace"
        assert app.section_toggles == ["workspace"]

        await pilot.click(workspace_toggle)
        await _settle(pilot)
        assert rail._active_section_id == "session"
        assert app.section_toggles == ["workspace", "workspace"]

        session_toggle = rail.query_one("#console-rail-section-toggle-session", Button)
        await pilot.click(session_toggle)
        await _settle(pilot)
        assert rail._active_section_id == "conversations"


def _reconcile_probe_file() -> ConversationFileEntry:
    return ConversationFileEntry(
        root="/tmp/project",
        path="owner-probe.py",
        label="owner-probe.py",
        status="M",
        adds=2,
        dels=1,
        run_id="run-owner-probe",
        snapshot_id=71,
        note_count=0,
    )


@pytest.mark.parametrize("owner_name", ("sources", "changed-files", "settings", "run"))
@pytest.mark.asyncio
async def test_inspector_descendant_owners_reconcile_local_then_outer(
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-settings-summary")
        screen.query_one("#console-inspector-rail-open", Button).press()
        await _settle(pilot)
        screen._stop_console_transcript_sync_timer()

        rail = screen.query_one("#console-right-rail", ConsoleInspectorRail)
        events: list[str] = []
        target_section = {
            "sources": "sources",
            "changed-files": "changed-files",
            "settings": "session-settings",
            "run": "chat-dictionaries",
        }[owner_name]
        original_local = ConsoleBoundedSection.request_reconcile
        original_outer = rail.request_outer_reconcile

        def observe_local(section: ConsoleBoundedSection) -> None:
            if section.section_id == target_section:
                events.append("local")
            original_local(section)

        def observe_outer() -> None:
            events.append("outer")
            original_outer()

        monkeypatch.setattr(ConsoleBoundedSection, "request_reconcile", observe_local)
        monkeypatch.setattr(rail, "request_outer_reconcile", observe_outer)
        baseline = rail._outer_reconcile_count

        if owner_name == "sources":
            owner = screen.query_one(
                "#console-staged-context-tray", ConsoleStagedContextTray
            )
            owner._on_reconcile = observe_outer
            owner.sync_state(
                replace(
                    owner.state,
                    summary="Sources owner probe",
                    rows=(ConsoleDisplayRow("Source", "ready"),),
                )
            )
        elif owner_name == "changed-files":
            owner = screen.query_one(
                "#console-changed-files-section", ConsoleChangedFilesSection
            )
            owner._on_reconcile = observe_outer
            owner.update_state(
                ConsoleChangedFilesState(entries=(_reconcile_probe_file(),))
            )
        elif owner_name == "settings":
            owner = screen.query_one(
                "#console-settings-summary", ConsoleSettingsSummary
            )
            owner._on_reconcile = observe_outer
            owner.sync_state(replace(owner.state, model_row="Model: owner-path-probe"))
        else:
            owner = screen.query_one(
                "#console-run-inspector-state", ConsoleRunInspector
            )
            owner._on_reconcile = observe_outer
            owner.sync_state(
                replace(
                    owner.state,
                    dictionary_rows=(
                        ConsoleDisplayRow("Dictionary A", "attached"),
                        ConsoleDisplayRow("Dictionary B", "attached"),
                    ),
                    world_book_rows=(
                        ConsoleDisplayRow("World Book A", "attached"),
                        ConsoleDisplayRow("World Book B", "attached"),
                    ),
                )
            )
        await _settle(pilot, passes=10)

        assert "local" in events
        assert "outer" in events
        assert events.index("local") < events.index("outer")
        assert events.count("outer") == 1
        assert rail._outer_reconcile_count == baseline + 1
        assert rail._outer_reconcile_scheduled is False
        assert not any(
            section._reconcile_scheduled
            for section in rail.query(ConsoleBoundedSection)
        )

        section = rail.query_one(
            f"#console-bounded-section-{target_section}", ConsoleBoundedSection
        )
        assert section.desired_content_lines > 0
        assert section.viewport.content_region.height == min(
            section.desired_content_lines, 20
        )
        if owner_name == "sources":
            assert (
                str(
                    rail.query_one("#console-staged-context-summary", Static).renderable
                )
                == "Sources owner probe"
            )
        elif owner_name == "changed-files":
            assert rail.query_one("#console-changed-files-row-0", Button)
        elif owner_name == "settings":
            assert (
                str(rail.query_one("#console-settings-model-row", Static).renderable)
                == "Model: owner-path-probe"
            )
        else:
            assert rail.query_one("#console-inspector-dictionaries-heading", Static)
            assert rail.query_one("#console-inspector-worldbooks-heading", Static)
            assert rail.query_one("#console-bounded-section-world-books")


@pytest.mark.parametrize(
    "mutation_path", ("sources", "changed-files", "settings", "run")
)
@pytest.mark.asyncio
async def test_chat_screen_inspector_mutation_paths_delegate_one_owner_request(
    monkeypatch: pytest.MonkeyPatch,
    mutation_path: str,
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-settings-summary")
        screen.query_one("#console-inspector-rail-open", Button).press()
        await _settle(pilot)
        screen._stop_console_transcript_sync_timer()

        rail = screen.query_one("#console-right-rail", ConsoleInspectorRail)
        target_section = {
            "sources": "sources",
            "changed-files": "changed-files",
            "settings": "session-settings",
            "run": "chat-dictionaries",
        }[mutation_path]
        events: list[str] = []
        original_local = ConsoleBoundedSection.request_reconcile
        original_outer = rail.request_outer_reconcile

        def observe_local(section: ConsoleBoundedSection) -> None:
            if section.section_id == target_section:
                events.append("local")
            original_local(section)

        def observe_outer() -> None:
            events.append("outer")
            original_outer()

        monkeypatch.setattr(ConsoleBoundedSection, "request_reconcile", observe_local)
        monkeypatch.setattr(rail, "request_outer_reconcile", observe_outer)
        baseline = rail._outer_reconcile_count

        if mutation_path == "sources":
            owner = screen.query_one(
                "#console-staged-context-tray", ConsoleStagedContextTray
            )
            owner._on_reconcile = observe_outer
            screen._pending_console_launch_context = ConsoleLiveWorkLaunch.from_values(
                source="owner-path", title="Sources ChatScreen probe"
            )
            screen._sync_console_staged_context_tray()
        elif mutation_path == "changed-files":
            owner = screen.query_one(
                "#console-changed-files-section", ConsoleChangedFilesSection
            )
            owner._on_reconcile = observe_outer
            screen._console_changed_files_summary = (_reconcile_probe_file(),)
            screen._console_changed_files_pruned_rows = 0
            screen._sync_console_changed_files_section()
        elif mutation_path == "settings":
            owner = screen.query_one(
                "#console-settings-summary", ConsoleSettingsSummary
            )
            owner._on_reconcile = observe_outer
            state = replace(owner.state, model_row="Model: screen-path-probe")
            monkeypatch.setattr(
                screen, "_build_console_settings_summary_state", lambda: state
            )
            screen._sync_console_settings_summary()
        else:
            owner = screen.query_one(
                "#console-run-inspector-state", ConsoleRunInspector
            )
            owner._on_reconcile = observe_outer
            state = replace(
                owner.state,
                dictionary_rows=(
                    ConsoleDisplayRow("Screen dictionary A", "attached"),
                    ConsoleDisplayRow("Screen dictionary B", "attached"),
                ),
                world_book_rows=(
                    ConsoleDisplayRow("Screen world book A", "attached"),
                    ConsoleDisplayRow("Screen world book B", "attached"),
                ),
            )
            monkeypatch.setattr(
                screen,
                "_build_console_inspector_state",
                lambda _launch: state,
            )
            screen._sync_console_control_bar(screen._current_console_rail_state())
        await _settle(pilot, passes=10)

        assert "local" in events
        assert events.index("local") < events.index("outer")
        assert events.count("outer") == 1
        assert rail._outer_reconcile_count == baseline + 1
        assert rail._outer_reconcile_scheduled is False
        assert not any(
            section._reconcile_scheduled
            for section in rail.query(ConsoleBoundedSection)
        )

        section = rail.query_one(
            f"#console-bounded-section-{target_section}", ConsoleBoundedSection
        )
        assert section.desired_content_lines > 0
        assert section.viewport.content_region.height == min(
            section.desired_content_lines, 20
        )


@pytest.mark.asyncio
async def test_pointer_activation_preserves_the_pressed_toggle_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activation reflow cannot turn a Details press into a neighboring press."""

    demands = dict.fromkeys(SECTION_IDS, 3)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        rail.apply_section_open("details", False)
        await _settle(pilot)
        details = rail.query_one("#console-rail-section-toggle-details", Button)
        details.scroll_visible(animate=False)
        await pilot.pause()

        assert await pilot.click(details)
        await _settle(pilot)

        assert app.section_toggles == ["details"]
        assert rail._active_section_id == "details"
        assert rail._pointer_activation_pending is None
        assert rail._pointer_activation_waits_for_button is False
        await _settle(pilot)
        assert app.section_toggles == ["details"]
        assert rail._active_section_id == "details"


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_pressed_section", [False, True])
async def test_canceled_header_press_releases_pointer_activation_latch(
    monkeypatch: pytest.MonkeyPatch,
    remove_pressed_section: bool,
) -> None:
    """A drag-away or removed press cannot block later focus reconciliation."""

    demands = dict.fromkeys(SECTION_IDS, 3)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        requests = 0
        original_request = rail.request_allocation_reconcile

        def request_spy() -> None:
            nonlocal requests
            requests += 1
            original_request()

        monkeypatch.setattr(rail, "request_allocation_reconcile", request_spy)
        pressed = rail.query_one("#console-rail-section-toggle-details", Button)
        release = rail.query_one("#console-rail-section-title-model", Static)
        pressed_offset = pressed.region.offset
        pressed.post_message(
            MouseDown(
                pressed,
                0,
                0,
                0,
                0,
                1,
                False,
                False,
                False,
                screen_x=pressed_offset.x,
                screen_y=pressed_offset.y,
            )
        )
        await pilot.pause()
        assert rail._pointer_activation_pending == "details"
        assert rail._pointer_activation_waits_for_button is True

        if remove_pressed_section:
            await rail.query_one("#console-rail-section-header-details").remove()
            await rail.query_one("#console-bounded-section-details").remove()
        release_offset = release.region.offset
        release.post_message(
            MouseUp(
                release,
                0,
                0,
                0,
                0,
                1,
                False,
                False,
                False,
                screen_x=release_offset.x,
                screen_y=release_offset.y,
            )
        )
        await _settle(pilot)

        assert app.section_toggles == []
        assert rail._pointer_activation_pending is None
        assert rail._pointer_activation_waits_for_button is False
        cleanup_requests = requests
        configure = rail.query_one("#console-model-section-configure", Button)
        configure.focus()
        await _settle(pilot)
        assert requests > cleanup_requests
        assert app.section_toggles == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("removed_section", "expected_fallback"),
    (("character", "details"), ("agent", "model")),
)
async def test_absent_active_section_falls_back_in_stable_descriptor_order(
    monkeypatch: pytest.MonkeyPatch,
    removed_section: str,
    expected_fallback: str,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 4)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 36)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        rail.activate_section(removed_section)
        await _settle(pilot)
        await rail.query_one(f"#console-rail-section-header-{removed_section}").remove()
        await rail.query_one(f"#console-bounded-section-{removed_section}").remove()
        rail.request_allocation_reconcile()
        await _settle(pilot)

        assert rail._active_section_id == expected_fallback


@pytest.mark.asyncio
async def test_pointer_boundary_owns_header_title_nested_body_and_viewport_but_not_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 0)
    demands["workspace"] = 4
    demands["model"] = 10
    demands["character"] = 4
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18, header_chrome_height=14)
        rail.request_allocation_reconcile()
        await _settle(pilot)

        model = rail.query_one("#console-bounded-section-model", ConsoleBoundedSection)
        model.query_one("#console-rail-section-body-model").styles.min_height = 10
        rail.request_allocation_reconcile()
        await _settle(pilot)
        viewport = model.viewport
        assert viewport.can_focus is True

        wheel = MouseScrollDown(
            viewport,
            0,
            0,
            0,
            1,
            0,
            False,
            False,
            False,
        )
        viewport.post_message(wheel)
        await pilot.pause()
        assert rail._active_section_id is None

        viewport.post_message(MouseDown(viewport, 0, 0, 0, 0, 1, False, False, False))
        viewport.post_message(MouseUp(viewport, 0, 0, 0, 0, 1, False, False, False))
        await _settle(pilot)
        assert rail._active_section_id == "model"

        title = rail.query_one("#console-rail-section-title-workspace", Static)
        title.scroll_visible(animate=False)
        await pilot.pause()
        title.post_message(MouseDown(title, 0, 0, 0, 0, 1, False, False, False))
        title.post_message(MouseUp(title, 0, 0, 0, 0, 1, False, False, False))
        await _settle(pilot)
        assert rail._active_section_id == "workspace"

        avatar_child = rail.query_one("#console-character-avatar Static", Static)
        assert avatar_child.focusable is False
        avatar_child.post_message(
            MouseDown(avatar_child, 0, 0, 0, 0, 1, False, False, False)
        )
        avatar_child.post_message(
            MouseUp(avatar_child, 0, 0, 0, 0, 1, False, False, False)
        )
        await _settle(pilot)
        assert rail._active_section_id == "character"


@pytest.mark.asyncio
async def test_overflow_focus_order_and_recovery_stay_within_context_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 0)
    demands["model"] = 8
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18, header_chrome_height=14)
        rail.request_allocation_reconcile()
        await _settle(pilot)

        toggle = rail.query_one("#console-rail-section-toggle-model", Button)
        model = rail.query_one("#console-bounded-section-model")
        model.query_one("#console-rail-section-body-model").styles.min_height = 8
        model.request_reconcile()
        rail.request_allocation_reconcile()
        await _settle(pilot)
        viewport = model.viewport
        configure = rail.query_one("#console-model-section-configure", Button)
        next_toggle = rail.query_one("#console-rail-section-toggle-agent", Button)
        assert viewport.can_focus is True

        toggle.focus()
        await pilot.press("tab")
        assert app.focused is viewport
        title = rail.query_one("#console-rail-section-title-model", Static)
        assert "underline" in str(title.styles.text_style)
        await pilot.press("tab")
        assert app.focused is configure
        await pilot.press("tab")
        assert app.focused is next_toggle
        await pilot.press("shift+tab")
        assert app.focused is configure
        await pilot.press("shift+tab")
        assert app.focused is viewport
        await pilot.press("shift+tab")
        assert app.focused is toggle

        outer = rail.query_one("#console-left-rail-body")
        collapse = rail.query_one("#console-context-rail-collapse", Button)
        outer.can_focus = True
        outer.focus()
        await pilot.pause()
        assert "underline" in str(collapse.styles.text_style)
        assert "underline" not in str(title.styles.text_style)

        body = rail.query_one("#console-rail-section-body-model")
        first = Button("First", id="context-recovery-first", compact=True)
        second = Button("Second", id="context-recovery-second", compact=True)
        await body.mount(first, second, before=configure)
        await _settle(pilot)
        first.focus()
        await pilot.pause()
        await first.remove()
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert app.focused is second

        viewport.focus()
        await pilot.pause()
        demands["model"] = 1
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert viewport.can_focus is False
        assert app.focused is second


@pytest.mark.asyncio
async def test_focus_recovery_prefers_next_from_removed_target_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 0)
    demands["model"] = 8
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        body = rail.query_one("#console-rail-section-body-model")
        configure = rail.query_one("#console-model-section-configure", Button)
        first = Button("First", id="context-next-first", compact=True)
        second = Button("Second", id="context-next-second", compact=True)
        third = Button("Third", id="context-next-third", compact=True)
        await body.mount(first, second, third, before=configure)
        await _settle(pilot)

        second.focus()
        await pilot.pause()
        await third.remove()
        await second.remove()
        rail.request_allocation_reconcile()
        await _settle(pilot)

        assert app.focused is configure


@pytest.mark.asyncio
async def test_focus_recovery_uses_previous_only_then_header_then_context_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 0)
    demands["model"] = 8
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18, header_chrome_height=14)
        rail.request_allocation_reconcile()
        await _settle(pilot)
        body = rail.query_one("#console-rail-section-body-model")
        body.styles.min_height = 8
        configure = rail.query_one("#console-model-section-configure", Button)
        first = Button("First", id="context-tier-first", compact=True)
        second = Button("Second", id="context-tier-second", compact=True)
        third = Button("Third", id="context-tier-third", compact=True)
        await body.mount(first, second, third, before=configure)
        await _settle(pilot)

        configure.disabled = True
        second.focus()
        await pilot.pause()
        await third.remove()
        await second.remove()
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert app.focused is first

        first.focus()
        await pilot.pause()
        await first.remove()
        rail.request_allocation_reconcile()
        await _settle(pilot)
        header_toggle = rail.query_one("#console-rail-section-toggle-model", Button)
        assert app.focused is header_toggle

        model = rail.query_one("#console-bounded-section-model", ConsoleBoundedSection)
        assert model.viewport.can_focus is True
        model.viewport.focus()
        await rail.query_one("#console-rail-section-header-model").remove()
        demands["model"] = 1
        model.request_reconcile()
        await _settle(pilot)
        assert app.focused is rail.query_one("#console-context-rail-collapse", Button)


@pytest.mark.asyncio
async def test_focus_recovery_does_not_steal_valid_outside_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 3)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        body = rail.query_one("#console-rail-section-body-model")
        configure = rail.query_one("#console-model-section-configure", Button)
        intentional = Button(
            "Intentional", id="context-intentional-reentry", compact=True
        )
        target = Button("Target", id="context-owned-before-outside", compact=True)
        outside = Button("Outside", id="outside-context-focus", compact=True)
        await body.mount(intentional, target, before=configure)
        await app.screen.mount(outside)
        await _settle(pilot)
        target.focus()
        await pilot.pause()
        outside.focus()
        await target.remove()
        rail.request_allocation_reconcile()
        await _settle(pilot)

        assert app.focused is outside

        intentional.focus()
        await _settle(pilot)
        assert app.focused is intentional


@pytest.mark.asyncio
async def test_new_rail_remount_resets_active_and_demand_shrink_clamps_local_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 0)
    demands["model"] = 12
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18, header_chrome_height=14)
        body = rail.query_one("#console-rail-section-body-model")
        body.styles.min_height = 12
        rail.activate_section("model")
        await _settle(pilot)
        model = rail.query_one("#console-bounded-section-model", ConsoleBoundedSection)
        model.viewport.scroll_end(animate=False)
        await pilot.pause()
        assert model.viewport.scroll_y > 0

        demands["model"] = 2
        body.styles.min_height = 2
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert model.viewport.scroll_y <= model.viewport.max_scroll_y

        await rail.remove()
        replacement = app.build_rail()
        await app.screen.mount(replacement)
        await _settle(pilot)
        assert replacement._active_section_id is None


@pytest.mark.asyncio
async def test_active_section_falls_back_when_content_disappears_and_offsets_survive_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 8)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18, header_chrome_height=14)
        rail.activate_section("agent")
        await _settle(pilot)
        agent = rail.query_one("#console-bounded-section-agent")
        agent.query_one("#console-rail-section-body-agent").styles.min_height = 8
        agent.request_reconcile()
        rail.request_allocation_reconcile()
        await _settle(pilot)
        agent.viewport.scroll_to(y=2, animate=False, immediate=True)
        await pilot.pause()
        retained_offset = agent.viewport.scroll_y
        assert retained_offset > 0

        rail.sync_workspace_context(_workspace_state())
        await _settle(pilot)
        assert agent.viewport.scroll_y == retained_offset
        assert rail._active_section_id == "agent"

        rail.display = False
        await _settle(pilot)
        rail.display = True
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert agent.viewport.scroll_y == retained_offset
        assert rail._active_section_id == "agent"

        demands["agent"] = 0
        rail.request_allocation_reconcile()
        await _settle(pilot)
        assert rail._active_section_id == "model"
