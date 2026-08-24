"""Atomic allocation contracts for the mounted Console Context rail."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from types import MethodType

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.events import Click, MouseDown, MouseScrollDown, MouseScrollUp, MouseUp
from textual.widget import Widget
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
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
from tldw_chatbook.UI.Console_Modules.rail_section_layout import outer_hint_required
from tldw_chatbook.UI.Console_Modules.right_rail import ConsoleInspectorRail
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)
from tldw_chatbook.Widgets.Console.console_workspace_tree import (
    ConsoleWorkspaceTree,
    WorkspaceTreeExpansionChanged,
    WorkspaceTreeFocusRecoveryRequested,
    WorkspaceTreeStarRequested,
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
from tldw_chatbook.Workspaces.workspace_tree_state import (
    WorkspaceTreeConversation,
    WorkspaceTreeWorkspace,
)


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
SCHEDULER_CALLBACK_LIMIT = 4


def _drain_scheduler_callbacks(
    callbacks: list[tuple[Callable[..., None], tuple[object, ...]]],
) -> None:
    """Run one expected scheduler generation without permitting a test hang."""

    for _ in range(SCHEDULER_CALLBACK_LIMIT):
        if not callbacks:
            return
        callback, args = callbacks.pop(0)
        callback(*args)


def _contains_widget_reference(value: object) -> bool:
    """Return whether queued state recursively retains a Textual widget."""

    if isinstance(value, Widget):
        return True
    if isinstance(value, dict):
        return any(
            _contains_widget_reference(item) for pair in value.items() for item in pair
        )
    if isinstance(value, (tuple, list, set)):
        return any(_contains_widget_reference(item) for item in value)
    dataclass_fields = getattr(value, "__dataclass_fields__", None)
    if dataclass_fields is not None:
        return any(
            _contains_widget_reference(getattr(value, name))
            for name in dataclass_fields
        )
    return False


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

    def __init__(
        self,
        *,
        show_character: bool = True,
        workspace_state: ConsoleWorkspaceContextState | None = None,
        workspace_tree_expanded_ids: frozenset[str] | None = None,
        expansion_preferences_changed: Callable[[frozenset[str]], None] | None = None,
    ) -> None:
        super().__init__()
        self.show_character = show_character
        self.workspace_state = workspace_state or _workspace_state()
        self.workspace_tree_expanded_ids = workspace_tree_expanded_ids
        self.expansion_preferences_changed = expansion_preferences_changed
        self.section_toggles: list[str] = []
        self.star_requests: list[WorkspaceTreeStarRequested] = []
        self.expansion_requests: list[WorkspaceTreeExpansionChanged] = []

    def build_rail(self) -> ConsoleLeftRail:
        return ConsoleLeftRail(
            rail_state=_all_open_rail_state(),
            workspace_context_state=self.workspace_state,
            workspace_tree_expanded_ids=self.workspace_tree_expanded_ids,
            workspace_tree_expansion_preferences_changed=(
                self.expansion_preferences_changed
            ),
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
            character_avatar_widget_builder=lambda _box=None: Static("avatar"),
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

    def on_workspace_tree_star_requested(
        self, event: WorkspaceTreeStarRequested
    ) -> None:
        self.star_requests.append(event)

    def on_workspace_tree_expansion_changed(
        self, event: WorkspaceTreeExpansionChanged
    ) -> None:
        self.expansion_requests.append(event)


class _ProductionConsoleHarness(ConsoleHarness):
    """Real ChatScreen host with the complete production CSS cascade."""

    CSS_PATH = TldwCli.CSS_PATH


def _native_workspace_tree_state() -> ConsoleWorkspaceContextState:
    conversation = WorkspaceTreeConversation(
        conversation_id="conversation-1",
        title="Planning",
        starred=False,
        updated_sort="2026-08-22T00:00:00",
        selected=False,
        run_marker="",
    )
    workspace = WorkspaceTreeWorkspace(
        workspace_id="workspace-1",
        label="Workspace One",
        conversations=(conversation,),
        next_cursor=None,
    )
    return replace(
        _workspace_state(),
        workspace_tree=(workspace,),
        workspace_marks_available=True,
    )


@pytest.mark.asyncio
async def test_native_tree_restores_persisted_disclosure_and_exposes_pointer_star() -> (
    None
):
    writes: list[frozenset[str]] = []
    app = _RailHarness(
        workspace_state=_native_workspace_tree_state(),
        workspace_tree_expanded_ids=frozenset(),
        expansion_preferences_changed=writes.append,
    )

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        tree = app.query_one(ConsoleWorkspaceTree)
        workspace = tree.workspace_nodes["workspace-1"]
        action_row = app.query_one("#console-workspace-context-action-row")
        bounded = app.query_one(
            "#console-bounded-section-workspace", ConsoleBoundedSection
        )
        assert action_row.display is False
        assert workspace.is_collapsed

        workspace.expand()
        await _settle(pilot)
        assert writes == [frozenset({"workspace-1"})]
        hidden_demand = bounded.desired_content_lines

        tree.move_cursor(tree.conversation_nodes["conversation-1"])
        await _settle(pilot)
        star = app.query_one("#console-workspace-tree-star", Button)
        assert app.query_one("#console-workspace-context-action-row").display is True
        assert bounded.desired_content_lines == hidden_demand + 1
        assert star.disabled is False
        assert str(star.label) == "Star"

        app.query_one(ConsoleLeftRail).activate_section("workspace")
        star.scroll_visible(animate=False, force=True)
        await _settle(pilot)
        assert await pilot.click(star)
        await pilot.pause()
        assert len(app.star_requests) == 1
        assert app.star_requests[0].conversation_id == "conversation-1"
        assert app.star_requests[0].starred is False

        tree.move_cursor(workspace)
        await _settle(pilot)
        assert app.query_one("#console-workspace-context-action-row").display is False
        expected_demand = sum(
            child.virtual_region_with_margin.height
            for child in bounded._native_content
            if child.is_mounted and child.display
        ) + max(0, tree.virtual_size.height)
        assert bounded.desired_content_lines == expected_demand

        tree.set_search_active(True, forced_workspace_ids={"workspace-1"})
        workspace.collapse()
        await pilot.pause()
        assert writes == [frozenset({"workspace-1"})]

        tree.set_search_active(False)
        assert workspace.is_expanded
        assert writes == [frozenset({"workspace-1"})]


@pytest.mark.asyncio
async def test_native_tree_hides_star_for_unmarkable_conversation() -> None:
    state = _native_workspace_tree_state()
    conversation = replace(
        state.workspace_tree[0].conversations[0],
        conversation_id="native:session-7",
        star_enabled=False,
    )
    state = replace(
        state,
        workspace_tree=(
            replace(state.workspace_tree[0], conversations=(conversation,)),
        ),
    )
    app = _RailHarness(
        workspace_state=state,
        workspace_tree_expanded_ids=frozenset({"workspace-1"}),
    )

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        tree = app.query_one(ConsoleWorkspaceTree)
        tree.move_cursor(tree.conversation_nodes["native:session-7"])
        await _settle(pilot)

        action_row = app.query_one("#console-workspace-context-action-row")
        star = app.query_one("#console-workspace-tree-star", Button)
        assert action_row.display is False
        assert star.disabled is True


@pytest.mark.asyncio
async def test_rapid_workspace_context_show_hide_restores_exact_geometry() -> None:
    app = _RailHarness(
        workspace_state=_native_workspace_tree_state(),
        workspace_tree_expanded_ids=frozenset({"workspace-1"}),
    )

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        tree = app.query_one(ConsoleWorkspaceTree)
        workspace = tree.workspace_nodes["workspace-1"]
        tray = app.query_one("#console-workspaces-context")
        bounded = app.query_one(
            "#console-bounded-section-workspace", ConsoleBoundedSection
        )
        initial_tray_height = int(tray.region.height)
        initial_demand = bounded.desired_content_lines

        tree.move_cursor(tree.conversation_nodes["conversation-1"])
        tree.move_cursor(workspace)
        await _settle(pilot)

        assert app.query_one("#console-workspace-context-action-row").display is False
        assert int(tray.region.height) == initial_tray_height
        assert bounded.desired_content_lines == initial_demand


@pytest.mark.asyncio
async def test_workspace_resize_preserves_visible_contextual_star() -> None:
    app = _RailHarness(
        workspace_state=_native_workspace_tree_state(),
        workspace_tree_expanded_ids=frozenset({"workspace-1"}),
    )

    async with app.run_test(size=(70, 30)) as pilot:
        await _settle(pilot)
        tree = app.query_one(ConsoleWorkspaceTree)
        tree.move_cursor(tree.conversation_nodes["conversation-1"])
        await _settle(pilot)

        tray = app.query_one("#console-workspaces-context")
        initial_content_width = tray._row_content_width
        assert app.query_one("#console-workspace-context-action-row").display is True

        app.query_one(ConsoleLeftRail).styles.width = 42
        await _settle(pilot)

        assert tray._row_content_width == initial_content_width + 2
        action_row = app.query_one("#console-workspace-context-action-row")
        star = app.query_one("#console-workspace-tree-star", Button)
        assert action_row.display is True
        assert star.disabled is False
        assert star.conversation_id == "conversation-1"


@pytest.mark.asyncio
async def test_workspace_context_change_survives_transient_relabel_controls() -> None:
    app = _RailHarness(
        workspace_state=_native_workspace_tree_state(),
        workspace_tree_expanded_ids=frozenset({"workspace-1"}),
    )

    async with app.run_test(size=(70, 30)) as pilot:
        await _settle(pilot)
        tree = app.query_one(ConsoleWorkspaceTree)
        tree.move_cursor(tree.conversation_nodes["conversation-1"])
        await _settle(pilot)

        tray = app.query_one("#console-workspaces-context")
        assert tray._workspace_tree_context_data.conversation_id == "conversation-1"

        # Width relabeling temporarily removes the action controls. Deliver
        # the cursor's new workspace context in that window, then let compose
        # rebuild from the cached truth.
        await app.query_one("#console-workspace-context-action-row").remove()
        workspace = tree.workspace_nodes["workspace-1"]
        assert tray.sync_workspace_tree_context(workspace.data) is False
        tray.refresh(recompose=True)
        await _settle(pilot)

        assert tray._workspace_tree_context_data is workspace.data
        selection_context = tray.query_one(
            "#console-workspace-tree-selection-context", Static
        )
        assert str(selection_context.renderable) == (
            "Selected: Workspace One · Enter open"
        )
        assert app.query_one("#console-workspace-context-action-row").display is False
        star = app.query_one("#console-workspace-tree-star", Button)
        assert star.disabled is True
        assert star.conversation_id is None


@pytest.mark.asyncio
async def test_native_tree_requests_page_zero_for_initial_persisted_expansion() -> None:
    app = _RailHarness(
        workspace_state=_native_workspace_tree_state(),
        workspace_tree_expanded_ids=frozenset({"workspace-1"}),
    )

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)

        assert [
            (event.workspace_id, event.expanded) for event in app.expansion_requests
        ] == [("workspace-1", True)]


@pytest.mark.asyncio
async def test_empty_native_tree_focuses_the_workspace_disclosure_control() -> None:
    app = _RailHarness(
        workspace_state=_native_workspace_tree_state(),
        workspace_tree_expanded_ids=frozenset({"workspace-1"}),
    )

    async with app.run_test(size=(60, 60)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        tree = app.query_one(ConsoleWorkspaceTree)
        tree.move_cursor(tree.conversation_nodes["conversation-1"])
        tree.focus()
        await pilot.pause()
        assert app.focused is tree

        tree.post_message(WorkspaceTreeFocusRecoveryRequested())
        await _settle(pilot)
        disclosure = app.query_one("#console-rail-section-toggle-workspace", Button)
        assert app.focused is disclosure

        tree.focus()
        await pilot.pause()
        assert app.focused is tree

        rail.sync_workspace_context(
            replace(_native_workspace_tree_state(), workspace_tree=())
        )
        await _settle(pilot)

        assert app.focused is disclosure


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


@pytest.mark.parametrize("terminal_height", [45, 18], ids=["tall", "short"])
@pytest.mark.asyncio
async def test_all_open_context_sections_keep_their_own_complete_ceiling_and_outer_reachability(
    terminal_height: int,
) -> None:
    app = _RailHarness()

    async with app.run_test(size=(60, terminal_height)) as pilot:
        await _settle(pilot, passes=8)
        rail = app.query_one(ConsoleLeftRail)
        outer = rail.query_one("#console-left-rail-body")
        cue = rail.query_one("#console-left-rail-outer-hint", Static)
        initial_state = rail._rail_state
        ceilings = dict(zip(SECTION_IDS, (15, 20, 20, 15, 15, 15, 35), strict=True))

        sections = list(_sections(rail))
        assert [section.section_id for section in sections] == list(SECTION_IDS)
        for section in sections:
            ceiling = ceilings[section.section_id]
            assert section.desired_content_lines > 0
            assert section.max_content_lines == ceiling
            assert section.allocation is None
            if section.native_scroll_owner is None:
                assert section.viewport.content_region.height == min(
                    section.desired_content_lines,
                    ceiling,
                )
            else:
                virtual = section.viewport.virtual_size.height
                fixed = section.desired_content_lines - virtual
                assert section.viewport.content_region.height == min(
                    virtual,
                    max(0, ceiling - fixed),
                )

            header = rail.query_one(
                f"#console-rail-section-header-{section.section_id}"
            )
            title = rail.query_one(
                f"#console-rail-section-title-{section.section_id}", Static
            )
            toggle = rail.query_one(
                f"#console-rail-section-toggle-{section.section_id}", Button
            )
            assert header.open is True
            assert str(title.renderable) == header.title
            assert "· no room" not in str(title.renderable)
            assert str(toggle.label) != "[>]"

        assert str(outer.styles.overflow_y) == "auto"
        assert outer.max_scroll_y > 0
        assert outer.can_focus is True
        assert cue.display is True
        outer.scroll_home(animate=False)
        await pilot.pause()
        assert str(cue.renderable) == OUTER_HINT

        for section in sections:
            header = rail.query_one(
                f"#console-rail-section-header-{section.section_id}"
            )
            outer.scroll_to(y=header.virtual_region.y, animate=False)
            await pilot.pause()
            assert header.region.overlaps(outer.content_region)
            assert section.viewport.region.overlaps(outer.content_region)

        assert rail._rail_state == initial_state
        assert app.section_toggles == []


@pytest.mark.asyncio
async def test_production_deliberate_context_activation_reveals_the_complete_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberate activation reveals an ordinary-outer-scroll section physically."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(120, 30)) as pilot:
        rail = await _open_all_production_context_sections(host, pilot)
        console = host.screen_stack[-1]
        outer = rail.query_one("#console-left-rail-body")
        header = rail.query_one("#console-rail-section-header-character")
        bounded = rail.query_one(
            "#console-bounded-section-character", ConsoleBoundedSection
        )
        persisted: list[tuple[str, dict[str, bool]]] = []
        monkeypatch.setattr(
            console,
            "_save_console_rail_preferences",
            lambda key, serialized, **_kwargs: persisted.append((key, serialized)),
        )

        outer.scroll_home(animate=False, immediate=True)
        await pilot.pause()
        assert not header.region.overlaps(outer.content_region)

        rail.activate_section("character")
        await _wait_for_rail_condition(
            pilot,
            rail,
            lambda: (
                rail._active_section_id == "character"
                and header.region.overlaps(outer.content_region)
                and bounded.viewport.region.overlaps(outer.content_region)
            ),
        )

        outer.scroll_home(animate=False, immediate=True)
        await pilot.pause()
        focus_target = rail.query_one("#console-character-reaction-open", Button)
        focus_target.focus()
        await _wait_for_rail_condition(
            pilot,
            rail,
            lambda: (
                pilot.app.focused is focus_target
                and header.region.overlaps(outer.content_region)
                and bounded.viewport.region.overlaps(outer.content_region)
            ),
        )

        assert persisted == []


@pytest.mark.asyncio
async def test_production_context_outer_and_local_offsets_reconcile_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid outer/local offsets persist; demand shrink clamps only invalid outer."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(120, 30)) as pilot:
        rail = await _open_all_production_context_sections(host, pilot)
        console = host.screen_stack[-1]
        outer = rail.query_one("#console-left-rail-body")
        sections = list(_sections(rail))
        persisted: list[tuple[str, dict[str, bool]]] = []
        monkeypatch.setattr(
            console,
            "_save_console_rail_preferences",
            lambda key, serialized, **_kwargs: persisted.append((key, serialized)),
        )

        fillers: dict[str, Static] = {}
        for section in sections:
            body = section.query_one(f"#console-rail-section-body-{section.section_id}")
            filler = Static(
                "\n".join(
                    f"{section.section_id} {row}"
                    for row in range(section.max_content_lines + 5)
                ),
                id=f"context-offset-filler-{section.section_id}",
            )
            fillers[section.section_id] = filler
            await body.mount(filler)
            section.request_reconcile()
        rail.request_allocation_reconcile()
        await _wait_for_rail_condition(
            pilot,
            rail,
            lambda: True,
        )

        local_section = rail.query_one(
            "#console-bounded-section-session", ConsoleBoundedSection
        )
        local_section.viewport.scroll_to(y=2, animate=False, immediate=True)
        outer.scroll_to(
            y=min(12, outer.max_scroll_y - 1), animate=False, immediate=True
        )
        await pilot.pause()
        local_offset = local_section.viewport.scroll_y
        outer_offset = outer.scroll_y
        assert local_offset == 2
        assert 0 < outer_offset < outer.max_scroll_y

        rail.request_allocation_reconcile()
        await _wait_for_rail_condition(pilot, rail, lambda: True)
        assert outer.scroll_y == outer_offset
        assert local_section.viewport.scroll_y == local_offset

        workspace = rail.query_one(
            "#console-bounded-section-workspace", ConsoleBoundedSection
        )
        fillers["workspace"].update(
            "\n".join(
                f"workspace changed {row}"
                for row in range(workspace.max_content_lines + 6)
            )
        )
        workspace.request_reconcile()
        rail.request_allocation_reconcile()
        await _wait_for_rail_condition(pilot, rail, lambda: True)
        assert outer.scroll_y == outer_offset
        assert local_section.viewport.scroll_y == local_offset

        outer.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        invalid_after_shrink = outer.scroll_y
        for section in sections:
            if section is local_section:
                continue
            await fillers[section.section_id].remove()
            section.request_reconcile()
        rail.request_allocation_reconcile()
        await _wait_for_rail_condition(
            pilot,
            rail,
            lambda: outer.max_scroll_y < invalid_after_shrink,
        )

        assert outer.scroll_y == outer.max_scroll_y
        assert local_section.viewport.scroll_y == local_offset
        assert persisted == []


@pytest.mark.asyncio
async def test_delayed_context_reveal_rejects_stale_generation_focus_and_unmount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delayed reveal cannot outlive newer intent, focus, or its rail."""

    app = _RailHarness()
    async with app.run_test(size=(60, 18)) as pilot:
        await _settle(pilot, passes=8)
        rail = app.query_one(ConsoleLeftRail)
        outer = rail.query_one("#console-left-rail-body")
        target = rail.query_one("#console-character-reaction-open", Button)
        outside = Button("Outside", id="outside-delayed-context-reveal")
        await app.screen.mount(outside)
        await pilot.pause()

        callbacks: list[tuple[Callable[..., None], tuple[object, ...]]] = []
        monkeypatch.setattr(
            rail,
            "call_after_refresh",
            lambda callback, *args: callbacks.append((callback, args)),
        )

        outer.scroll_home(animate=False, immediate=True)
        rail._queue_active_reveal("character", None)
        rail.activate_section("model", request_reconcile=False)
        callback, args = callbacks.pop(0)
        callback(*args)
        assert outer.scroll_y == 0
        assert callbacks == []

        target.focus()
        await pilot.pause()
        callbacks.clear()
        rail.activate_section("character", request_reconcile=False)
        outer.scroll_home(animate=False, immediate=True)
        rail._queue_active_reveal("character", target)
        outside.focus()
        await pilot.pause()
        callback, args = callbacks.pop(0)
        callback(*args)
        assert outer.scroll_y == 0
        _drain_scheduler_callbacks(callbacks)
        assert callbacks == []

        rail._queue_active_reveal("character", None)
        await rail.remove()
        callback, args = callbacks.pop(0)
        callback(*args)
        assert outer.scroll_y == 0
        assert callbacks == []


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
) -> None:
    rail._snapshot_outer_viewport_height = MethodType(  # type: ignore[attr-defined]
        lambda self: viewport_height,
        rail,
    )


def _sections(rail: ConsoleLeftRail) -> Iterator[ConsoleBoundedSection]:
    return iter(rail.query("#console-left-rail-body ConsoleBoundedSection"))


def _assert_compositor_hit(screen, widget) -> None:
    """Assert one positive physical region is owned at its center hit point."""

    assert widget.region.width > 0 and widget.region.height > 0, (
        widget.id,
        widget.region,
    )
    point = (
        widget.region.x + widget.region.width // 2,
        widget.region.y + (widget.region.height - 1) // 2,
    )
    hit = screen.get_widget_at(*point)[0]
    assert hit is widget or widget in hit.ancestors, (widget.id, point, hit)


async def _assert_context_direct_sections_are_compositor_reachable(
    screen,
    rail: ConsoleLeftRail,
    pilot,
) -> None:
    """Prove ordinary outer scrolling paints every complete direct section."""

    outer = rail.query_one("#console-left-rail-body")
    assert rail.region.contains_region(outer.region)
    headers = [
        rail.query_one(f"#console-rail-section-header-{section_id}")
        for section_id in SECTION_IDS
    ]
    bounded_sections = [
        rail.query_one(f"#console-bounded-section-{section_id}", ConsoleBoundedSection)
        for section_id in SECTION_IDS
    ]

    for index, (header, bounded) in enumerate(zip(headers, bounded_sections)):
        assert header.parent is outer
        assert bounded.parent is outer
        outer.scroll_to(y=header.virtual_region.y, animate=False, immediate=True)
        await pilot.pause()
        assert header.region.overlaps(outer.content_region)
        assert bounded.viewport.region.overlaps(outer.content_region)
        assert rail.region.contains_region(header.region)
        _assert_compositor_hit(screen, header)

        assert not header.region.overlaps(bounded.region)
        assert bounded.region.contains_region(bounded.viewport.region)

        if index + 1 < len(headers):
            following_header = headers[index + 1]
            assert not bounded.region.overlaps(following_header.region)
            if bounded.hint.display:
                assert not bounded.hint.region.overlaps(following_header.region)


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


@pytest.mark.asyncio
async def test_production_workspace_pointer_keeps_pressed_key_across_outer_reflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete-stylesheet pointer gesture cannot retarget after rail reveal."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)
    workspaces = tuple(
        WorkspaceTreeWorkspace(
            workspace_id=f"workspace-{index}",
            label=f"Workspace {index}",
            conversations=tuple(
                WorkspaceTreeConversation(
                    conversation_id=f"conversation-{index}-{child}",
                    title=f"Conversation {index}-{child}",
                    starred=False,
                    updated_sort=f"{child}",
                    selected=False,
                    run_marker="",
                )
                for child in range(4)
            ),
            next_cursor=None,
        )
        for index in range(8)
    )

    async with host.run_test(size=(120, 30)) as pilot:
        rail = await _open_all_production_context_sections(host, pilot)
        console = host.screen_stack[-1]
        rail.sync_workspace_context(
            replace(
                _workspace_state(),
                workspace_tree=workspaces,
                workspace_marks_available=True,
            )
        )
        await _settle(pilot, passes=8)
        tree = rail.query_one(ConsoleWorkspaceTree)
        bounded = rail.query_one(
            "#console-bounded-section-workspace", ConsoleBoundedSection
        )
        workspace = tree.workspace_nodes["workspace-1"]
        workspace_requests: list[str] = []
        conversation_requests: list[str] = []
        monkeypatch.setattr(
            console._workspace,
            "activate_workspace_id",
            workspace_requests.append,
        )

        async def record_conversation(conversation_id: str, **_kwargs) -> None:
            conversation_requests.append(conversation_id)

        monkeypatch.setattr(
            console._workspace,
            "open_console_workspace_conversation",
            record_conversation,
        )

        assert bounded.max_content_lines == 20
        assert bounded.native_scroll_owner is tree
        assert tree.max_scroll_y > 0
        rail.activate_section("session", deliberate_reveal=False)
        workspace_line = int(workspace._line)
        tree.scroll_to(y=max(0, workspace_line - 1), animate=False, immediate=True)
        workspace_header_y = rail.query_one(
            "#console-rail-section-header-workspace"
        ).virtual_region.y
        outer = rail.query_one("#console-left-rail-body", VerticalScroll)
        outer.scroll_to(
            y=max(0, workspace_header_y - 3),
            animate=False,
            immediate=True,
        )
        await _settle(pilot, passes=4)
        assert rail._active_section_id == "session"
        click_y = workspace_line - int(tree.scroll_y)
        assert 0 <= click_y < tree.content_region.height
        pressed_key = "workspace:workspace-1"
        old_tree_y = tree.content_region.y
        old_outer_scroll_y = outer.scroll_y
        pressed_coordinate = (
            tree.content_region.x + 4,
            tree.content_region.y + click_y,
        )

        assert await pilot.mouse_down(offset=pressed_coordinate)
        await _settle(pilot, passes=8)

        assert tree._pressed_node_key == pressed_key
        assert rail._active_section_id == "workspace"
        assert outer.scroll_y != old_outer_scroll_y
        assert tree.content_region.y != old_tree_y
        await pilot._post_mouse_events(
            [MouseUp, Click],
            offset=pressed_coordinate,
            button=1,
        )
        await _settle(pilot, passes=4)

        assert tree.cursor_node is not None
        assert tree.cursor_node.data.key == pressed_key
        assert workspace_requests == []
        assert conversation_requests == []

        new_click_y = int(workspace._line) - int(tree.scroll_y)
        assert await pilot.click(tree, offset=(4, new_click_y), times=2)
        await _settle(pilot, passes=4)
        assert workspace_requests == ["workspace-1"]
        replacement_coordinate = (
            tree.content_region.x + 4,
            tree.content_region.y + new_click_y,
        )
        assert await pilot.mouse_down(offset=replacement_coordinate)
        await _settle(pilot, passes=2)
        assert tree._pressed_node_key == pressed_key

        replacement = WorkspaceTreeWorkspace(
            workspace_id="replacement",
            label="Replacement",
            conversations=tuple(
                WorkspaceTreeConversation(
                    conversation_id=f"replacement-{index}",
                    title=f"Replacement conversation {index}",
                    starred=False,
                    updated_sort=f"{index}",
                    selected=False,
                    run_marker="",
                )
                for index in range(30)
            ),
            next_cursor=None,
        )
        tree.sync_projection(
            (replacement,),
            expanded_workspace_ids={"replacement"},
        )
        await _settle(pilot, passes=6)
        assert tree._pressed_node_key is None
        assert tree._last_pointer_click_key is None
        assert tree.conversation_nodes, tree.workspace_nodes["replacement"].data
        replacement_node = tree.conversation_nodes["replacement-15"]
        replacement_local_y = replacement_coordinate[1] - tree.content_region.y
        tree.scroll_to(
            y=max(0, int(replacement_node._line) - replacement_local_y),
            animate=False,
            immediate=True,
        )
        await _settle(pilot, passes=2)
        replacement_line = int(tree.scroll_y) + replacement_local_y
        replacement_node = tree.get_node_at_line(replacement_line)
        assert replacement_node is not None
        assert replacement_node.data is not None
        assert replacement_node.data.kind == "conversation"

        await pilot._post_mouse_events(
            [MouseUp, Click],
            offset=replacement_coordinate,
            button=1,
        )
        await _settle(pilot, passes=4)
        assert tree.cursor_node is not replacement_node
        assert conversation_requests == []


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
        await _assert_context_direct_sections_are_compositor_reachable(
            screen,
            context,
            pilot,
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
async def test_production_content_demand_counts_internal_box_only() -> None:
    """Internal padding and margins count in D; external chrome does not."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        context = await _open_all_production_context_sections(host, pilot)
        inspector = await _open_production_inspector(host, pilot)
        tray = inspector.query_one(
            "#console-staged-context-tray", ConsoleStagedContextTray
        )
        sources = tray.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await sources.viewport.remove_children()
        content = Static(
            "\n".join(f"controlled content {row}" for row in range(18)),
            id="production-demand-box",
        )
        content.styles.padding = (1, 0)
        content.styles.margin = (1, 0)
        await sources.viewport.mount(content)
        sources.request_reconcile()
        inspector.request_outer_reconcile()
        await _wait_for_rail_condition(
            pilot,
            context,
            lambda: (
                not sources._reconcile_scheduled
                and not inspector._outer_reconcile_scheduled
            ),
        )

        assert content.virtual_region.height == 20
        assert content.virtual_region.bottom == 21
        assert content.virtual_region_with_margin.bottom == 22
        assert sources.desired_content_lines == 22
        assert sources.viewport.content_region.height == 20
        assert sources.hint.display
        header = tray.query_one(".console-staged-context-header")
        assert header not in sources.viewport.children
        assert sources.hint not in sources.viewport.children

        header.styles.margin = (2, 0)
        sources.hint.styles.margin = (3, 0)
        sources.request_reconcile()
        inspector.request_outer_reconcile()
        await _wait_for_rail_condition(
            pilot,
            context,
            lambda: (
                sources.desired_content_lines == 22
                and sources.hint.display
                and not inspector._outer_reconcile_scheduled
            ),
        )
        assert sources.desired_content_lines == 22


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
async def test_inspector_geometry_only_generation_reconciles_without_owner_pass() -> (
    None
):
    """A virtual-size invalidation transitions the fold without an owner pass."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        inspector = await _open_production_inspector(host, pilot)
        outer = inspector.query_one("#console-inspector-rail-body")
        hint = inspector.query_one("#console-inspector-outer-scroll-hint", Static)
        await outer.remove_children()
        content = Static("geometry-only content", id="geometry-only-outer-demand")
        content.styles.height = 10
        await outer.mount(content)
        inspector.request_outer_reconcile()
        await _settle(pilot, passes=8)
        assert hint.display is False
        assert inspector._outer_reconcile_scheduled is False
        fitting_signature = (
            max(
                child.virtual_region_with_margin.bottom
                for child in outer.children
                if child.display
            ),
            outer.content_region.height,
        )
        assert fitting_signature[0] <= fitting_signature[1]
        logical_owner_count = inspector._outer_owner_reconcile_count

        content.styles.height = fitting_signature[1] + 2
        content.refresh(layout=True)
        await _settle(pilot, passes=8)

        overflow_signature = (
            max(
                child.virtual_region_with_margin.bottom
                for child in outer.children
                if child.display
            ),
            outer.content_region.height + hint.region.height,
        )
        assert overflow_signature != fitting_signature
        assert overflow_signature[0] > overflow_signature[1]
        assert hint.display is True
        assert inspector._outer_reconcile_scheduled is False
        assert inspector._outer_owner_reconcile_count == logical_owner_count
        await _settle(pilot, passes=2)
        assert inspector._outer_reconcile_scheduled is False
        assert inspector._outer_owner_reconcile_count == logical_owner_count


@pytest.mark.asyncio
async def test_inspector_owner_demand_latches_while_geometry_generation_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        inspector = await _open_production_inspector(host, pilot)
        await _settle(pilot, passes=8)
        callbacks: list[tuple[Callable[..., None], tuple[object, ...]]] = []

        def capture_after_refresh(callback, *args) -> None:
            callbacks.append((callback, args))

        monkeypatch.setattr(inspector, "call_after_refresh", capture_after_refresh)
        baseline = inspector._outer_owner_reconcile_count
        inspector._request_outer_geometry_reconcile()
        inspector.request_outer_reconcile()

        assert inspector._outer_reconcile_scheduled is True
        assert inspector._outer_reconcile_dirty is True
        assert inspector._outer_reconcile_owner_demand is True
        assert len(callbacks) == 1
        _drain_scheduler_callbacks(callbacks)

        assert callbacks == [], (
            "owner-demand scheduler did not drain within "
            f"{SCHEDULER_CALLBACK_LIMIT} callbacks; remaining={callbacks!r}"
        )
        assert inspector._outer_reconcile_scheduled is False
        assert inspector._outer_reconcile_dirty is False
        assert inspector._outer_reconcile_owner_demand is False
        assert inspector._outer_owner_reconcile_count == baseline + 1


@pytest.mark.asyncio
async def test_inspector_owner_demand_survives_hint_toggle_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        inspector = await _open_production_inspector(host, pilot)
        await _settle(pilot, passes=8)
        callbacks: list[tuple[Callable[..., None], tuple[object, ...]]] = []
        fold_passes = 0

        def capture_after_refresh(callback, *args) -> None:
            callbacks.append((callback, args))

        def reconcile_with_hint_continuation() -> bool:
            nonlocal fold_passes
            fold_passes += 1
            if fold_passes == 1:
                inspector._request_outer_geometry_reconcile()
                return False
            return True

        monkeypatch.setattr(inspector, "call_after_refresh", capture_after_refresh)
        monkeypatch.setattr(
            inspector,
            "_reconcile_outer_fold",
            reconcile_with_hint_continuation,
        )
        baseline = inspector._outer_owner_reconcile_count
        inspector.request_outer_reconcile()
        _drain_scheduler_callbacks(callbacks)

        assert fold_passes == 2
        assert callbacks == [], (
            "hint-toggle continuation did not drain within "
            f"{SCHEDULER_CALLBACK_LIMIT} callbacks; remaining={callbacks!r}"
        )
        assert inspector._outer_reconcile_scheduled is False
        assert inspector._outer_reconcile_dirty is False
        assert inspector._outer_reconcile_owner_demand is False
        assert inspector._outer_owner_reconcile_count == baseline + 1


@pytest.mark.asyncio
async def test_inspector_unmount_before_callback_clears_pending_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        inspector = await _open_production_inspector(host, pilot)
        await _settle(pilot, passes=8)
        callbacks: list[tuple[Callable[..., None], tuple[object, ...]]] = []

        def capture_after_refresh(callback, *args) -> None:
            callbacks.append((callback, args))

        monkeypatch.setattr(inspector, "call_after_refresh", capture_after_refresh)
        baseline = inspector._outer_owner_reconcile_count
        inspector.request_outer_reconcile()
        inspector._outer_reconcile_dirty = True
        assert len(callbacks) == 1
        await inspector.remove()
        assert inspector._outer_reconcile_scheduled is False
        assert inspector._outer_reconcile_dirty is False
        assert inspector._outer_reconcile_owner_demand is False
        callback, args = callbacks.pop(0)
        callback(*args)

        assert callbacks == []
        assert inspector._outer_reconcile_scheduled is False
        assert inspector._outer_reconcile_dirty is False
        assert inspector._outer_reconcile_owner_demand is False
        assert inspector._outer_owner_reconcile_count == baseline


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_size", [(120, 30), (80, 24)])
async def test_production_css_uses_uncompressed_header_demand_and_reaches_every_section(
    terminal_size: tuple[int, int],
) -> None:
    """Real CSS keeps complete sections reachable through ordinary outer scroll."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)
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
        assert str(outer.styles.overflow_y) == "auto"
        assert cue.display is True
        assert all(section.allocation is None for section in _sections(rail))
        assert all(
            (
                section.viewport.content_region.height
                == min(section.desired_content_lines, section.max_content_lines)
            )
            if section.native_scroll_owner is None
            else (
                section.viewport.content_region.height
                == min(
                    section.viewport.virtual_size.height,
                    max(
                        0,
                        section.max_content_lines
                        - (
                            section.desired_content_lines
                            - section.viewport.virtual_size.height
                        ),
                    ),
                )
            )
            for section in _sections(rail)
        )
        assert all(
            str(
                rail.query_one(
                    f"#console-rail-section-toggle-{section_id}", Button
                ).label
            )
            != "[>]"
            for section_id in SECTION_IDS
        )

        assert_stable_sibling_geometry()
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
    """Real ChatScreen mutation, focus, and pointer paths keep write scope."""

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
            lambda: not ordinary_header.open and ordinary_body.allocation is None,
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

        assert str(ordinary.label) != "[>]"
        assert await pilot.click(ordinary)
        await _wait_for_rail_condition(
            pilot,
            rail,
            lambda: ordinary_header.open,
        )
        assert len(persisted) == 1
        assert ordinary_body.allocation is None
        assert str(ordinary.label) != "[>]"
        assert "· no room" not in str(
            rail.query_one(
                f"#console-rail-section-title-{ordinary_section}", Static
            ).renderable
        )
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
            lambda: rail._active_section_id == "model",
        )
        assert model_body.allocation is None
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
async def test_local_and_outer_hints_use_distinct_counterfactual_predicates() -> None:
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        cue = app.query_one("#console-left-rail-outer-hint", Static)
        session = rail.query_one(
            "#console-bounded-section-session", ConsoleBoundedSection
        )
        session.query_one("#console-rail-section-body-session").styles.min_height = 16
        session.request_reconcile()
        rail.request_allocation_reconcile()
        rail.activate_section("agent")
        await _settle(pilot, passes=8)
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

        await pilot.resize_terminal(60, 200)
        await _settle(pilot, passes=8)
        assert cue.display is False
        assert str(cue.renderable) == ""

        await pilot.resize_terminal(60, 18)
        await _settle(pilot, passes=8)
        outer.scroll_home(animate=False)
        await pilot.pause()
        assert cue.display is True
        assert str(cue.renderable) == OUTER_HINT


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
        _force_geometry(rail, viewport_height=28)
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
        workspace_toggle.scroll_visible(animate=False)
        await pilot.pause()
        assert await pilot.click(workspace_toggle)
        await _settle(pilot)
        assert rail._active_section_id == "workspace"
        assert app.section_toggles == ["workspace"]

        workspace_toggle.scroll_visible(animate=False)
        await pilot.pause()
        assert await pilot.click(workspace_toggle)
        await _settle(pilot)
        assert rail._active_section_id == "session"
        assert app.section_toggles == ["workspace", "workspace"]

        session_toggle = rail.query_one("#console-rail-section-toggle-session", Button)
        session_toggle.scroll_visible(animate=False)
        await pilot.pause()
        assert await pilot.click(session_toggle)
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
        baseline = rail._outer_owner_reconcile_count

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
        assert rail._outer_owner_reconcile_count == baseline + 1
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
        baseline = rail._outer_owner_reconcile_count

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
        assert rail._outer_owner_reconcile_count == baseline + 1
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
    demands["model"] = 20
    demands["character"] = 4
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18)
        rail.request_allocation_reconcile()
        await _settle(pilot)

        model = rail.query_one("#console-bounded-section-model", ConsoleBoundedSection)
        model.query_one("#console-rail-section-body-model").styles.min_height = 20
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
    demands["model"] = 20
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18)
        rail.request_allocation_reconcile()
        await _settle(pilot)

        toggle = rail.query_one("#console-rail-section-toggle-model", Button)
        model = rail.query_one("#console-bounded-section-model")
        model.query_one("#console-rail-section-body-model").styles.min_height = 20
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
        await body.mount(second, before=configure)
        await body.mount(first, before=second)
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
    demands["model"] = 20
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


@pytest.mark.parametrize(
    "signal_order",
    ["rail_then_bounded", "bounded_then_rail"],
)
@pytest.mark.asyncio
async def test_context_focus_recovery_coalesces_adversarial_signal_order(
    monkeypatch: pytest.MonkeyPatch,
    signal_order: str,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 0)
    demands["model"] = 20
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        model = rail.query_one("#console-bounded-section-model", ConsoleBoundedSection)
        body = rail.query_one("#console-rail-section-body-model")
        configure = rail.query_one("#console-model-section-configure", Button)
        first = Button("First", id=f"context-order-first-{signal_order}")
        second = Button("Second", id=f"context-order-second-{signal_order}")
        await body.mount(first, second, before=configure)
        await _settle(pilot)
        first.focus()
        await pilot.pause()
        app.screen.set_focus(model.viewport)

        callbacks: list[tuple[Callable[..., None], tuple[object, ...]]] = []
        monkeypatch.setattr(
            rail,
            "call_after_refresh",
            lambda callback, *args: callbacks.append((callback, args)),
        )
        signals = {
            "rail": lambda: rail._ensure_focus_recovery("model"),
            "bounded": lambda: rail.recover_section_focus("model"),
        }
        for signal in signal_order.split("_then_"):
            signals[signal]()

        incident = rail._pending_focus_recoveries["model"]
        recovery_callbacks = [
            (callback, args)
            for callback, args in callbacks
            if callback.__name__ == "_recover_pending_focus"
        ]
        assert len(recovery_callbacks) == 1
        assert recovery_callbacks[0][1] == ("model", incident)

        await first.remove()
        callback, args = recovery_callbacks[0]
        callback(*args)

        assert app.focused is second
        assert rail._pending_focus_recoveries == {}
        assert model._focused_descendant is second
        assert model._focus_recovery_notified is True

        model._recover_removed_focus_target()
        rail.recover_section_focus("model")
        assert app.focused is second
        assert rail._pending_focus_recoveries == {}


@pytest.mark.asyncio
async def test_context_focus_recovery_prefers_same_id_remount_over_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 0)
    demands["model"] = 20
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        body = rail.query_one("#console-rail-section-body-model")
        configure = rail.query_one("#console-model-section-configure", Button)
        original = Button("Original", id="context-stable-remount")
        ordinal = Button("Ordinal", id="context-stable-ordinal")
        await body.mount(original, ordinal, before=configure)
        await _settle(pilot)
        original.focus()
        await pilot.pause()

        callbacks: list[tuple[Callable[..., None], tuple[object, ...]]] = []
        monkeypatch.setattr(
            rail,
            "call_after_refresh",
            lambda callback, *args: callbacks.append((callback, args)),
        )
        incident = rail._ensure_focus_recovery("model")
        await original.remove()
        replacement = Button("Replacement", id="context-stable-remount")
        await body.mount(replacement)
        await pilot.pause()

        callback, args = next(
            (callback, args)
            for callback, args in callbacks
            if callback.__name__ == "_recover_pending_focus"
        )
        assert args == ("model", incident)
        callback(*args)

        assert app.focused is replacement
        assert app.focused is not ordinal
        assert rail._pending_focus_recoveries == {}


@pytest.mark.asyncio
async def test_stale_context_focus_recovery_callback_cannot_consume_new_incident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        body = rail.query_one("#console-rail-section-body-model")
        configure = rail.query_one("#console-model-section-configure", Button)
        first = Button("First", id="context-stale-first")
        second = Button("Second", id="context-stale-second")
        outside = Button("Outside", id="context-stale-outside")
        await body.mount(first, second, before=configure)
        await app.screen.mount(outside)
        await _settle(pilot)

        callbacks: list[tuple[Callable[..., None], tuple[object, ...]]] = []
        monkeypatch.setattr(
            rail,
            "call_after_refresh",
            lambda callback, *args: callbacks.append((callback, args)),
        )
        first.focus()
        rail._record_section_focus("model", first)
        stale_incident = rail._ensure_focus_recovery("model")
        stale_callback, stale_args = next(
            (callback, args)
            for callback, args in callbacks
            if callback.__name__ == "_recover_pending_focus"
        )

        app.screen.set_focus(outside)
        rail._clear_focus_owner_if_focus_left()
        app.screen.set_focus(second)
        rail._record_section_focus("model", second)
        current_incident = rail._ensure_focus_recovery("model")
        assert current_incident is not stale_incident

        stale_callback(*stale_args)

        assert app.focused is second
        assert rail._pending_focus_recoveries == {"model": current_incident}


@pytest.mark.asyncio
async def test_semantic_focus_incident_releases_widget_history_before_unmount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        target = rail.query_one("#console-model-section-configure", Button)
        target.focus()
        await pilot.pause()

        callbacks: list[tuple[Callable[..., None], tuple[object, ...]]] = []
        monkeypatch.setattr(
            rail,
            "call_after_refresh",
            lambda callback, *args: callbacks.append((callback, args)),
        )
        incident = rail._ensure_focus_recovery("model")

        assert incident is not None
        assert incident.target_id == target.id
        assert "model" not in rail._section_focus_history
        assert rail._pending_focus_recoveries == {"model": incident}
        assert not _contains_widget_reference(rail._pending_focus_recoveries)

        callback, args = next(
            (callback, args)
            for callback, args in callbacks
            if callback.__name__ == "_recover_pending_focus"
        )
        await rail.remove()
        assert not _contains_widget_reference(rail._pending_focus_recoveries)
        callback(*args)

        assert rail._section_focus_history == {}
        assert rail._pending_focus_recoveries == {}


@pytest.mark.asyncio
async def test_active_reveal_queue_retains_only_identity_across_target_and_rail_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _RailHarness()

    async with app.run_test(size=(60, 18)) as pilot:
        await _settle(pilot, passes=8)
        rail = app.query_one(ConsoleLeftRail)
        outer = rail.query_one("#console-left-rail-body")
        target = rail.query_one("#console-character-reaction-open", Button)
        target.focus()
        await pilot.pause()

        callbacks: list[tuple[Callable[..., None], tuple[object, ...]]] = []
        monkeypatch.setattr(
            rail,
            "call_after_refresh",
            lambda callback, *args: callbacks.append((callback, args)),
        )
        rail.activate_section(
            "character",
            request_reconcile=False,
            reveal_target=target,
        )
        assert rail._pending_active_reveal is not None
        assert not _contains_widget_reference(rail._pending_active_reveal)

        rail._queue_pending_active_reveal()
        callback, args = callbacks.pop(0)
        assert not _contains_widget_reference(args)
        await target.remove()
        outer.scroll_home(animate=False, immediate=True)
        callback(*args)
        assert outer.scroll_y == 0
        assert rail._pending_active_reveal is None

        rail.activate_section("model", request_reconcile=False)
        rail._queue_pending_active_reveal()
        callback, args = callbacks.pop(0)
        assert not _contains_widget_reference(args)
        await rail.remove()
        callback(*args)
        assert rail._pending_active_reveal is None


@pytest.mark.asyncio
async def test_focus_recovery_uses_previous_only_then_header_then_context_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demands = dict.fromkeys(SECTION_IDS, 0)
    demands["model"] = 20
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18)
        rail.request_allocation_reconcile()
        await _settle(pilot)
        body = rail.query_one("#console-rail-section-body-model")
        body.styles.min_height = 20
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
    demands["model"] = 20
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18)
        body = rail.query_one("#console-rail-section-body-model")
        body.styles.min_height = 20
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
    demands = dict.fromkeys(SECTION_IDS, 20)
    _install_demands(monkeypatch, demands)
    app = _RailHarness()

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        _force_geometry(rail, viewport_height=18)
        rail.activate_section("agent")
        await _settle(pilot)
        agent = rail.query_one("#console-bounded-section-agent")
        agent.query_one("#console-rail-section-body-agent").styles.min_height = 20
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


# --- TASK-21117: the Inspector's pure-scroll path -----------------------------

WHEEL_NOTCH_BOUND = 24


def _post_wheel(body: VerticalScroll, *, down: bool) -> None:
    """Post one real terminal wheel notch at the Inspector's outer body."""

    event_type = MouseScrollDown if down else MouseScrollUp
    body.post_message(event_type(body, 0, 0, 0, 1, 0, False, False, False))


def _arm_rail_layout_probe(
    monkeypatch: pytest.MonkeyPatch,
    rail: ConsoleInspectorRail,
    body: VerticalScroll,
) -> list[str]:
    """Record every whole-rail ``refresh(layout=True)`` with its scroll offset."""

    observed: list[str] = []
    original_refresh = rail.refresh

    def counting_refresh(*regions, **kwargs):
        if kwargs.get("layout"):
            observed.append(f"layout refresh at scroll_y={body.scroll_y}")
        return original_refresh(*regions, **kwargs)

    monkeypatch.setattr(rail, "refresh", counting_refresh)
    return observed


async def _wheel_to_bottom(pilot, body: VerticalScroll) -> int:
    """Wheel down until the body is parked at the bottom; return the notches."""

    notches = 0
    for _ in range(WHEEL_NOTCH_BOUND):
        if body.scroll_y >= body.max_scroll_y:
            return notches
        _post_wheel(body, down=True)
        await _settle(pilot, passes=3)
        notches += 1
    pytest.fail(
        f"outer body never reached the bottom within {WHEEL_NOTCH_BOUND} notches "
        f"(scroll_y={body.scroll_y}, max_scroll_y={body.max_scroll_y})"
    )


@pytest.mark.asyncio
async def test_pure_inspector_scroll_never_relayouts_the_rail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wheel gesture repaints only the outer hint copy, never the rail layout."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        inspector = await _open_production_inspector(host, pilot)
        outer = inspector.query_one("#console-inspector-rail-body", VerticalScroll)
        hint = inspector.query_one("#console-inspector-outer-scroll-hint", Static)
        await outer.remove_children()
        content = Static("pure scroll content", id="pure-scroll-outer-demand")
        content.styles.height = outer.content_region.height + 8
        await outer.mount(content)
        inspector.request_outer_reconcile()
        await _settle(pilot, passes=8)

        assert hint.display is True
        assert str(hint.renderable) == OUTER_HINT
        assert outer.scroll_y == 0
        assert outer.max_scroll_y > 0
        assert inspector._outer_reconcile_scheduled is False

        observed = _arm_rail_layout_probe(monkeypatch, inspector, outer)
        owner_passes = inspector._outer_owner_reconcile_count
        # The copy is repainted without a layout pass, which is only sound
        # while the slot's height is pinned: hold that geometry to account.
        hint_region = hint.region
        assert hint_region.height == 1
        # Record what the scroll path writes to the slot. Both halves of the
        # repaint -- layout=False, and the skip when the copy is unchanged --
        # are measured optimizations, so assert them instead of trusting the
        # source to keep saying so.
        hint_writes: list[tuple[str, object]] = []
        original_hint_update = hint.update

        def recording_update(content: object = "", **kwargs: object):
            hint_writes.append((str(content), kwargs.get("layout", True)))
            return original_hint_update(content, **kwargs)

        monkeypatch.setattr(hint, "update", recording_update)

        notches = await _wheel_to_bottom(pilot, outer)
        assert notches >= 2, "the probe must cover several wheel frames"
        assert outer.scroll_y == outer.max_scroll_y
        # Reaching the bottom is the one thing a pure scroll DOES change.
        assert str(hint.renderable) == ""

        for _ in range(notches):
            _post_wheel(outer, down=False)
            await _settle(pilot, passes=3)
        assert outer.scroll_y == 0
        assert str(hint.renderable) == OUTER_HINT
        assert hint.display is True
        assert hint.region == hint_region
        rendered = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in host.screen_stack[-1]._compositor.render_strips()
        )
        assert OUTER_HINT in rendered, "the repainted copy must reach the compositor"

        assert observed == [], (
            f"{2 * notches} pure wheel frames forced {len(observed)} whole-rail "
            f"layout refreshes: {observed}"
        )

        # The same gesture delivered as one coalesced burst -- the shape the
        # audit measured (~2-3 layout passes survive the existing coalescing).
        burst = list(observed)
        for _ in range(notches):
            _post_wheel(outer, down=True)
        await _settle(pilot, passes=8)
        assert outer.scroll_y == outer.max_scroll_y
        assert str(hint.renderable) == ""
        assert observed == burst, (
            f"a coalesced {notches}-notch wheel burst forced "
            f"{len(observed) - len(burst)} whole-rail layout refreshes: "
            f"{observed[len(burst) :]}"
        )

        assert inspector._outer_owner_reconcile_count == owner_passes
        assert inspector._outer_reconcile_scheduled is False

        # The fold copy is the only thing the scroll path writes.
        frames = 3 * notches
        assert hint_writes, "the gesture must repaint the fold copy at least once"
        assert all(layout is False for _copy, layout in hint_writes), (
            "the scroll path must repaint the fold copy without a layout pass "
            f"(the slot's height is pinned): {hint_writes}"
        )
        assert len(hint_writes) <= 4, (
            f"{frames} scroll frames wrote the fold copy {len(hint_writes)} "
            f"times -- the unchanged-copy skip is gone: {hint_writes}"
        )


@pytest.mark.asyncio
async def test_inspector_section_collapse_still_runs_the_full_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real section demand keeps the layout + refold path the scroll path skips."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        inspector = await _open_production_inspector(host, pilot)
        outer = inspector.query_one("#console-inspector-rail-body", VerticalScroll)
        hint = inspector.query_one("#console-inspector-outer-scroll-hint", Static)
        tray = inspector.query_one(
            "#console-staged-context-tray", ConsoleStagedContextTray
        )
        sources = tray.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await sources.viewport.remove_children()
        await sources.viewport.mount(
            Static(
                "\n".join(f"source content {row}" for row in range(21)),
                id="collapse-probe-content",
            )
        )
        sources.request_reconcile()
        inspector.request_outer_reconcile()
        await _settle(pilot, passes=10)
        assert sources.display is True
        assert hint.display is True
        assert inspector._outer_reconcile_scheduled is False

        observed = _arm_rail_layout_probe(monkeypatch, inspector, outer)

        # Collapse the section: a real demand change, not a scroll. No explicit
        # reconcile request -- the collapse must reach the outer fold through
        # the body's own committed-geometry trigger, exactly as production does.
        sources.display = False
        await _settle(pilot, passes=10)
        assert observed, "a section collapse must still take the full reconcile path"
        _assert_outer_fold_contract(inspector, outer, hint)

        collapsed_refreshes = len(observed)
        sources.display = True
        await _settle(pilot, passes=10)
        assert len(observed) > collapsed_refreshes, (
            "re-expanding the section must reconcile the outer fold too"
        )
        assert hint.display is True
        assert str(hint.renderable) == OUTER_HINT
        _assert_outer_fold_contract(inspector, outer, hint)


def _assert_outer_fold_contract(
    inspector: ConsoleInspectorRail,
    outer: VerticalScroll,
    hint: Static,
) -> None:
    """Assert the live geometry still satisfies the outer-hint predicate."""

    assert inspector._outer_reconcile_scheduled is False
    desired_rows = max(
        (
            child.virtual_region_with_margin.bottom
            for child in outer.children
            if child.display
        ),
        default=0,
    )
    hint_rows = hint.region.height if hint.display else 0
    viewport_without_hint = outer.content_region.height + hint_rows
    assert viewport_without_hint > 0
    assert hint.display is outer_hint_required(desired_rows, viewport_without_hint)
    assert outer.scroll_y <= max(0, outer.max_scroll_y)
    expected_copy = (
        OUTER_HINT
        if hint.display
        and outer.max_scroll_y > 0
        and outer.scroll_y < outer.max_scroll_y
        else ""
    )
    assert str(hint.renderable) == expected_copy


@pytest.mark.asyncio
async def test_scroll_cost_probe_still_detects_pre_split_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation arm: restore the old routing and the probe must go red.

    Without this arm the zero-refresh assertion above could silently stop
    measuring its subject (a probe that can no longer see the defect passes
    forever). Restoring the pre-split ``watch_scroll_y`` -> geometry reconcile
    routing must make the same gesture cost whole-rail layout passes again.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _ProductionConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        inspector = await _open_production_inspector(host, pilot)
        outer = inspector.query_one("#console-inspector-rail-body", VerticalScroll)
        await outer.remove_children()
        content = Static("pre-split routing", id="pre-split-outer-demand")
        content.styles.height = outer.content_region.height + 8
        await outer.mount(content)
        inspector.request_outer_reconcile()
        await _settle(pilot, passes=8)
        assert outer.max_scroll_y > 0
        assert inspector._outer_reconcile_scheduled is False

        def legacy_watch_scroll_y(self, old_value: float, new_value: float) -> None:
            VerticalScroll.watch_scroll_y(self, old_value, new_value)
            if old_value != new_value:
                self._on_geometry_changed()

        monkeypatch.setattr(
            type(outer), "watch_scroll_y", legacy_watch_scroll_y, raising=True
        )
        observed = _arm_rail_layout_probe(monkeypatch, inspector, outer)

        notches = await _wheel_to_bottom(pilot, outer)
        sequential = len(observed)
        outer.scroll_to(y=0, animate=False, immediate=True)
        await _settle(pilot, passes=8)
        burst_baseline = len(observed)
        for _ in range(notches):
            _post_wheel(outer, down=True)
        await _settle(pilot, passes=8)
        burst = len(observed) - burst_baseline

        print(
            f"\n[TASK-21117 probe] pre-split routing over {notches} notches: "
            f"sequential={sequential} coalesced_burst={burst} "
            f"whole-rail refresh(layout=True) calls"
        )
        assert sequential > 0, (
            "the probe no longer observes the pre-split per-frame layout cost"
        )
        assert burst > 0, (
            "the probe no longer observes the pre-split coalesced layout cost"
        )
