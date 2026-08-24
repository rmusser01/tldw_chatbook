"""Mounted Console workspace context rail regressions."""

from __future__ import annotations

import inspect
import time
from dataclasses import replace
from pathlib import Path

import pytest
from rich.text import Text
from textual.content import Content
from textual.widgets import Button, Input, Static
from textual.widgets._tooltip import Tooltip

from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Widgets.Console import (
    ConsoleBoundedSection,
    ConsoleWorkspaceContextTray,
    ConsoleWorkspaceSwitcherModal,
)
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceStatusPair,
)
from tldw_chatbook.Widgets.Console.console_workspace_tree import (
    ConsoleWorkspaceTree,
    WorkspaceTreeNodeData,
)
from tldw_chatbook.Widgets.Console.console_workspace_details import (
    ConsoleWorkspaceDetailsTray,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Workspaces import (
    CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT,
    ConsoleWorkspaceACPHandoffState,
    ConsoleConversationBrowserInputRow,
    DEFAULT_WORKSPACE_ID,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceAuthority,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
    WorkspaceTransferPolicy,
    build_console_conversation_browser_state,
)
from tldw_chatbook.Workspaces.display_state import (
    CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
    ConsoleWorkspaceContextState,
    ConsoleWorkspaceConversationRow,
    ConsoleWorkspaceConversationSectionState,
    ConsoleWorkspaceServerAdapterState,
    console_workspace_conversation_result_copy,
    console_workspace_conversation_visible_rows,
)
from tldw_chatbook.Workspaces.workspace_tree_state import WorkspaceTreeWorkspace

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp


def _visible_text(screen) -> str:
    visible_chunks: list[str] = []
    for widget in screen.query(Static):
        if widget.display and hasattr(widget, "renderable"):
            visible_chunks.append(
                getattr(widget.renderable, "plain", str(widget.renderable))
            )
    for button in screen.query(Button):
        if button.display:
            visible_chunks.append(str(button.label))
    return " ".join(visible_chunks)


def _browser_group_toggle(screen, group_id: str) -> Button:
    for button in screen.query(".console-workspace-conversations-toggle"):
        if getattr(button, "group_id", None) == group_id:
            return button
    toggles = [
        (getattr(button, "id", None), getattr(button, "group_id", None))
        for button in screen.query(".console-workspace-conversations-toggle")
    ]
    raise AssertionError(f"Browser group toggle {group_id!r} not found: {toggles!r}")


def _conversation_row_texts(screen) -> list[str]:
    return [
        str(getattr(row, "label", ""))
        for row in screen.query(".console-workspace-conversation-row")
        if row.display
    ]


def _static_plain(screen, selector: str) -> str:
    widget = screen.query_one(selector, Static)
    return getattr(widget.render(), "plain", str(widget.render()))


async def _wait_for_condition(pilot, predicate, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate()


async def _wait_for_production_chat_screen(
    app, pilot, *, timeout: float = 6.0
) -> ChatScreen:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        screen = app.screen
        if isinstance(screen, ChatScreen) and screen.region.width > 0:
            await pilot.pause()
            return screen
        await pilot.pause(0.01)
    raise AssertionError(
        f"Timed out waiting for production ChatScreen; active={type(app.screen).__name__}"
    )


def _assert_status_row(
    screen,
    *,
    label_selector: str,
    value_selector: str,
    label: str,
    value_contains: str,
) -> None:
    assert _static_plain(screen, label_selector) == label
    assert value_contains in _static_plain(screen, value_selector)


def _section_state(
    *,
    collapsed: bool = False,
    rows: int = 6,
    query: str = "",
    search_enabled: bool = True,
) -> ConsoleWorkspaceConversationSectionState:
    conversation_rows = tuple(
        ConsoleWorkspaceConversationRow(
            conversation_id=f"conv-{index}",
            title=f"Conversation {index}",
            status="workspace-thread",
            selected=index == 2,
        )
        for index in range(rows)
    )
    return ConsoleWorkspaceConversationSectionState(
        workspace_id="ws-a",
        collapsed=collapsed,
        query=query,
        selected_summary="Conversation 2 - saved chat",
        rows=conversation_rows,
        workspace_total_count=rows,
        result_total_count=None,
        status_copy="",
        empty_copy="No active workspace conversations.",
        search_enabled=search_enabled,
    )


def _base_workspace_state(
    section: ConsoleWorkspaceConversationSectionState,
) -> ConsoleWorkspaceContextState:
    return ConsoleWorkspaceContextState(
        heading="Convos & Workspaces",
        workspace_label="Workspace: Test",
        authority_label="Authority: local registry ready",
        sync_label="Sync: not configured",
        runtime_label="Runtime: none",
        conversation_rows=section.rows,
        conversation_section=section,
        conversation_empty_copy="No active workspace conversations.",
        change_workspace_enabled=True,
        change_workspace_recovery="",
        new_conversation_enabled=True,
        new_conversation_recovery="",
        recovery_copy="",
    )


def _browser_row(
    row_key: str,
    title: str,
    *,
    conversation_id: str | None = None,
    native_session_id: str | None = None,
    scope_type: str = "workspace",
    workspace_id: str | None = "ws-a",
    workspace_label: str = "Workspace A",
    status: str = "workspace-thread",
    updated_label: str = "1d",
    selected: bool = False,
    starred: bool = False,
    star_enabled: bool = True,
    source_kind: str = "persisted",
    starred_sort: str = "",
    updated_sort: str = "",
    run_marker: str = "",
    queued_count: int = 0,
) -> ConsoleConversationBrowserInputRow:
    return ConsoleConversationBrowserInputRow(
        row_key=row_key,
        conversation_id=conversation_id if conversation_id is not None else row_key,
        native_session_id=native_session_id,
        title=title,
        scope_type=scope_type,
        workspace_id=workspace_id,
        workspace_label=workspace_label,
        status=status,
        updated_label=updated_label,
        selected=selected,
        starred=starred,
        star_enabled=star_enabled,
        source_kind=source_kind,
        starred_sort=starred_sort,
        updated_sort=updated_sort,
        run_marker=run_marker,
        queued_count=queued_count,
    )


def _grouped_browser_state(
    *,
    marks_available: bool = True,
    query: str = "",
    rows: tuple[ConsoleConversationBrowserInputRow, ...] | None = None,
    group_collapse_preferences: dict[str, bool] | None = None,
):
    preferences = {"section:chats": False}
    preferences.update(group_collapse_preferences or {})
    return build_console_conversation_browser_state(
        rows=rows
        or (
            _browser_row(
                "conv-starred",
                "Starred planning",
                starred=True,
                selected=True,
                starred_sort="2026-06-27T10:00:00",
                updated_sort="2026-06-27T09:00:00",
            ),
            _browser_row(
                "conv-workspace",
                "Workspace review",
                workspace_id="ws-a",
                workspace_label="Workspace A",
                updated_sort="2026-06-26T09:00:00",
            ),
            _browser_row(
                "conv-chat",
                "Loose chat",
                scope_type="global",
                workspace_id=None,
                workspace_label="Chats",
                updated_sort="2026-06-25T09:00:00",
            ),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences=preferences,
        query=query,
        marks_available=marks_available,
    )


def _base_grouped_workspace_state(
    *,
    marks_available: bool = True,
    query: str = "",
    rows: tuple[ConsoleConversationBrowserInputRow, ...] | None = None,
    configured_server: bool = False,
    group_collapse_preferences: dict[str, bool] | None = None,
) -> ConsoleWorkspaceContextState:
    """Build grouped-browser state; ``configured_server`` renders the full
    Sync/Server/ACP rows instead of the TASK-715 collapsed line."""
    section = _section_state(rows=0)
    state = _base_workspace_state(section)
    return ConsoleWorkspaceContextState(
        heading=state.heading,
        workspace_label=state.workspace_label,
        authority_label=state.authority_label,
        sync_label="Sync: syncing" if configured_server else state.sync_label,
        runtime_label=state.runtime_label,
        conversation_rows=(),
        conversation_empty_copy=state.conversation_empty_copy,
        conversation_section=state.conversation_section,
        conversation_browser=_grouped_browser_state(
            marks_available=marks_available,
            query=query,
            rows=rows,
            group_collapse_preferences=group_collapse_preferences,
        ),
        change_workspace_enabled=state.change_workspace_enabled,
        change_workspace_recovery=state.change_workspace_recovery,
        new_conversation_enabled=state.new_conversation_enabled,
        new_conversation_recovery=state.new_conversation_recovery,
        recovery_copy=state.recovery_copy,
        server_readiness_label=(
            "Server: adapter ready"
            if configured_server
            else state.server_readiness_label
        ),
        server_readiness_detail=state.server_readiness_detail,
        handoff_rows=state.handoff_rows,
        acp_handoff_label=(
            "ACP task/run: ready" if configured_server else state.acp_handoff_label
        ),
        acp_handoff_detail=state.acp_handoff_detail,
        acp_handoff_audit=state.acp_handoff_audit,
    )


def test_console_conversation_status_labels_use_saved_chat_vocabulary() -> None:
    """Persisted-but-not-archived chats read "saved chat" everywhere.

    Membership roles ("workspace-thread"/"workspace") and the default
    persisted conversation state ("in-progress") all describe the same
    user-visible thing: a chat saved locally that is not open in a tab.
    Library Browse ▸ Conversations lists the same records, so these labels
    must not contradict its copy (task-179 vocabulary alignment).
    """
    detail = ConsoleWorkspaceContextTray._conversation_detail_status
    status = ConsoleWorkspaceContextTray._conversation_status

    assert detail("workspace-thread") == "saved chat"
    assert detail("workspace") == "saved chat"
    assert detail("in-progress") == "saved chat"
    assert detail("active") == "active session"
    assert detail("open") == "open session"

    assert status("workspace-thread") == "saved"
    assert status("workspace") == "saved"
    assert status("in-progress") == "saved"
    assert status("active") == "active"
    assert status("open") == "open"


def test_conversation_row_secondary_drops_boilerplate_keeps_differentiator() -> None:
    """TASK-374 AC#2: the row subtitle compresses to just the differentiator.

    Every row previously repeated ``<workspace> - saved chat - <age>``, so only
    the age digits differed and half the section's vertical space carried no
    information. The subtitle now keeps the age always and the state ONLY when it
    is not the default ``saved chat``; the section-level workspace label is dropped.
    """
    secondary = ConsoleWorkspaceContextTray._conversation_row_secondary

    # Default saved state -> only the differentiating age remains.
    assert secondary("saved chat", "2d") == "2d"
    # A non-default state IS the differentiator and is kept alongside the age.
    assert secondary("active session", "5m") == "active session - 5m"
    assert secondary("open session", "1h") == "open session - 1h"
    # Degenerate inputs stay sane.
    assert secondary("active session", "") == "active session"
    assert secondary("saved chat", "") == ""
    # The repeated workspace label is never part of the compressed grouped subtitle.
    assert "Workspace" not in secondary("saved chat", "3d")

    # Qodo #812: header-less sections (the cross-workspace Starred section) pass
    # the workspace explicitly, and it leads the subtitle as the differentiator
    # so same-titled conversations from different workspaces stay distinguishable.
    assert (
        secondary("saved chat", "3d", workspace_label="Workspace A")
        == "Workspace A - 3d"
    )
    assert (
        secondary("active session", "5m", workspace_label="Workspace B")
        == "Workspace B - active session - 5m"
    )


@pytest.mark.asyncio
async def test_flat_conversations_has_no_cross_owner_starred_aggregate() -> None:
    """Default/unassigned rows are flat; stars only affect local ordering."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 50)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_base_grouped_workspace_state())
        await pilot.pause()

        # The old cross-owner Starred aggregate is retired. The flat row does
        # not carry named-workspace disambiguation because named owners live
        # in the native Workspaces Tree.
        starred_label = str(
            console.query_one("#console-workspace-conversation-0", Button).label
        )
        assert "Workspace A" not in starred_label

        # The same conversation under its Workspaces group header drops it.
        grouped_labels = [
            str(button.label)
            for button in console.query(".console-workspace-conversation-row")
            if button.display and "\nWorkspace A" not in str(button.label)
        ]
        assert grouped_labels, (
            "expected at least one grouped row without a workspace prefix"
        )


def test_console_workspace_conversation_section_state_defaults() -> None:
    section = ConsoleWorkspaceConversationSectionState(
        workspace_id="ws-a",
        collapsed=False,
        query="",
        selected_summary="No active conversation.",
        rows=(),
    )

    assert section.workspace_id == "ws-a"
    assert section.workspace_total_count is None
    assert section.result_total_count is None
    assert section.result_limit == CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT
    assert section.search_enabled is True
    assert section.new_conversation_enabled is True
    assert section.error_copy == ""


@pytest.mark.asyncio
async def test_console_workspace_context_mounts_native_tree_without_legacy_groups() -> (
    None
):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_base_grouped_workspace_state())
        await pilot.pause()

        assert len(console.query("#console-workspace-tree")) == 1
        assert len(console.query("#console-workspace-search")) == 1
        assert len(console.query("#console-conversation-browser-starred-title")) == 0


@pytest.mark.asyncio
async def test_workspace_tree_selection_context_is_one_reserved_updating_row() -> None:
    state = replace(_base_grouped_workspace_state(), workspace_name="Research Lab")

    class TestApp(ConsolidatedCSSApp):
        CSS_PATH = str(BUNDLED_STYLESHEET)

        def compose(self):
            yield ConsoleWorkspaceContextTray(
                state,
                show_heading=False,
                content="workspace",
                id="console-workspaces-context",
            )

    app = TestApp()
    async with app.run_test(size=(40, 20)) as pilot:
        tray = app.query_one(ConsoleWorkspaceContextTray)
        context = app.query_one("#console-workspace-tree-selection-context", Static)
        action_row = app.query_one("#console-workspace-context-action-row")

        assert context.region.height == 1
        assert context.styles.height.value == 1
        assert context.styles.text_wrap == "nowrap"
        assert context.styles.text_overflow == "ellipsis"
        assert str(context.renderable) == "Selected: Research Lab · Enter open"
        assert action_row.display is False

        conversation = WorkspaceTreeNodeData.conversation(
            "workspace-1",
            "conversation-1",
            "Planning notes",
            starred=False,
            selected=False,
            star_enabled=True,
        )
        assert tray.sync_workspace_tree_context(conversation) is False
        await pilot.pause()
        context = app.query_one("#console-workspace-tree-selection-context", Static)
        action_row = app.query_one("#console-workspace-context-action-row")
        assert str(context.renderable) == "Selected: Planning notes · Enter open"
        assert action_row.display is False

        auxiliary = WorkspaceTreeNodeData.auxiliary(
            "load-more",
            "workspace-1",
            "action:workspace-1:load-more",
            "Load more…",
        )
        assert tray.sync_workspace_tree_context(auxiliary) is False
        await pilot.pause()
        context = app.query_one("#console-workspace-tree-selection-context", Static)
        action_row = app.query_one("#console-workspace-context-action-row")
        assert str(context.renderable) == "Selected: Load more… · Enter open"
        assert action_row.display is False

        assert tray.sync_workspace_tree_context(None) is False
        await pilot.pause()
        context = app.query_one("#console-workspace-tree-selection-context", Static)
        assert str(context.renderable) == "Selected: Research Lab · Enter open"
        assert context.region.height == 1


@pytest.mark.asyncio
async def test_workspace_tree_selection_context_tooltip_only_when_clipped() -> None:
    state = replace(_base_grouped_workspace_state(), workspace_name="Research Lab")

    class TestApp(ConsolidatedCSSApp):
        CSS_PATH = str(BUNDLED_STYLESHEET)

        def compose(self):
            yield ConsoleWorkspaceContextTray(
                state,
                show_heading=False,
                content="workspace",
                id="console-workspaces-context",
            )

    app = TestApp()
    async with app.run_test(size=(90, 20)) as pilot:
        tray = app.query_one(ConsoleWorkspaceContextTray)
        context = app.query_one("#console-workspace-tree-selection-context", Static)
        assert context.tooltip is None

        long_label = "研究🙂" * 8
        tray.sync_workspace_tree_context(
            WorkspaceTreeNodeData.workspace("workspace-1", long_label)
        )
        await pilot.resize_terminal(28, 20)
        await pilot.pause()
        context = app.query_one("#console-workspace-tree-selection-context", Static)
        full_copy = f"Selected: {long_label} · Enter open"
        assert context.region.height == 1
        assert isinstance(context.tooltip, Text)
        assert context.tooltip.plain == full_copy

        await pilot.resize_terminal(90, 20)
        await pilot.pause()
        context = app.query_one("#console-workspace-tree-selection-context", Static)
        assert context.tooltip is None


@pytest.mark.asyncio
async def test_workspace_tree_selection_context_renders_markup_label_literally() -> (
    None
):
    state = replace(_base_grouped_workspace_state(), workspace_name="Research Lab")
    raw = "[bold]abc"

    class TestApp(ConsolidatedCSSApp):
        CSS_PATH = str(BUNDLED_STYLESHEET)

        def compose(self):
            yield ConsoleWorkspaceContextTray(
                state,
                show_heading=False,
                content="workspace",
                id="console-workspaces-context",
            )

    app = TestApp()
    app.TOOLTIP_DELAY = 0.01
    async with app.run_test(size=(28, 20), tooltips=True) as pilot:
        tray = app.query_one(ConsoleWorkspaceContextTray)
        tray.sync_workspace_tree_context(
            WorkspaceTreeNodeData.workspace("workspace-1", raw)
        )
        await pilot.pause()
        context = app.query_one("#console-workspace-tree-selection-context", Static)
        assert raw in str(context.render())

        assert await pilot.hover(context)
        await pilot.pause(0.05)
        tooltip = app.screen.get_child_by_type(Tooltip)
        assert tooltip.display is True
        assert raw in str(tooltip.render())


@pytest.mark.asyncio
async def test_console_f1_exposes_full_selected_tree_label_and_complete_grammar() -> (
    None
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    long_label = "Research Lab " + "研究🙂" * 12

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-tree")
        rail = console.query_one("#console-left-rail")
        state = replace(
            _base_grouped_workspace_state(),
            workspace_tree=(
                WorkspaceTreeWorkspace(
                    workspace_id="workspace-1",
                    label=long_label,
                    conversations=(),
                    next_cursor=None,
                ),
            ),
        )
        rail.sync_workspace_context(state)
        await pilot.pause()
        tree = console.query_one(ConsoleWorkspaceTree)
        tree.move_cursor(tree.workspace_nodes["workspace-1"])
        tree.focus()
        await pilot.pause()

        await console.action_show_workbench_help()
        await pilot.pause()
        panel = host.screen_stack[-1]
        assert isinstance(panel, WorkbenchHelpPanel)
        rendered = panel.state.render_text()

        assert long_label in rendered
        for gesture in (
            "Single click",
            "Double-click",
            "Enter",
            "Space",
            "Left",
            "Right",
        ):
            assert gesture in rendered


@pytest.mark.asyncio
async def test_console_f1_renders_markup_looking_selected_label_literally() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    raw = "[bold]abc"

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-tree")
        rail = console.query_one("#console-left-rail")
        rail.sync_workspace_context(
            replace(
                _base_grouped_workspace_state(),
                workspace_tree=(
                    WorkspaceTreeWorkspace(
                        workspace_id="workspace-1",
                        label=raw,
                        conversations=(),
                        next_cursor=None,
                    ),
                ),
            )
        )
        await pilot.pause()
        tree = console.query_one(ConsoleWorkspaceTree)
        tree.move_cursor(tree.workspace_nodes["workspace-1"])
        tree.focus()
        await pilot.pause()

        await console.action_show_workbench_help()
        await pilot.pause()
        panel = host.screen_stack[-1]
        assert isinstance(panel, WorkbenchHelpPanel)
        body = panel.query_one("#workbench-help-body", Static)
        assert raw in str(body.render())


@pytest.mark.asyncio
async def test_zero_result_workspace_search_stays_editable_and_initializes_without_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-search")
        transitions: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            console._workspace,
            "transition_workspace_tree_search",
            lambda query, disabled: transitions.append((query, disabled)),
        )
        state = replace(
            _base_grouped_workspace_state(),
            workspace_tree=(),
            workspace_query="older",
        )

        console.query_one("#console-left-rail").sync_workspace_context(state)
        await pilot.pause()

        search = console.query_one("#console-workspace-search", Input)
        assert search.disabled is False
        assert search.value == "older"
        assert transitions == []

        search.value = "newer"
        await pilot.pause()
        assert transitions == [("newer", False)]


def _workspace_rows_with_marker_at(index: int, marker: str, *, count: int):
    """``count`` rows in the active workspace ``ws-a``, newest-first by
    construction (descending ``updated_sort``) so display order matches
    ``range(count)`` exactly -- the row at ``index`` carries ``marker``."""
    return tuple(
        _browser_row(
            f"ws-a-{i}",
            f"Chat {i}",
            workspace_id="ws-a",
            workspace_label="Workspace A",
            updated_sort=f"2026-07-{31 - i:02d}T00:00:00",
            run_marker=marker if i == index else "",
        )
        for i in range(count)
    )


def _rendered_tooltip(widget) -> str:
    """Return a widget's tooltip as Textual will actually DISPLAY it.

    TASK-1233 review round 1: `Widget.tooltip` is a markup *source*
    string, not the rendered text -- the `Tooltip` widget parses it the
    same way `Content.from_markup` does, at display time. Asserting the
    raw attribute (as round 0 of this task did) passed against a tooltip
    that was silently broken once rendered: `status_suffix`'s literal
    ``"[saved]"`` was read as an unrecognized style tag and DROPPED from
    the rendered text entirely (confirmed via this exact helper against
    the pre-fix code: rendered to ``"Switch to Alpha "``, the word gone).
    Render through the same path the app uses before asserting.
    """
    return Content.from_markup(widget.tooltip or "").plain


# TASK-912 review fix round 1 (CRITICAL): the identical cap bug survived in
# the flat Starred/Chats sections. These mirror the group-cap tests above
# exactly, against an expanded Chats section instead.


def _chat_rows_with_marker_at(index: int, marker: str, *, count: int):
    """``count`` global Chats-section rows, newest-first by construction
    (descending ``updated_sort``) so display order matches ``range(count)``
    exactly -- the row at ``index`` carries ``marker``."""
    return tuple(
        _browser_row(
            f"chat-{i}",
            f"Chat {i}",
            scope_type="global",
            workspace_id=None,
            workspace_label="Chats",
            updated_sort=f"2026-07-{31 - i:02d}T00:00:00",
            run_marker=marker if i == index else "",
        )
        for i in range(count)
    )


@pytest.mark.asyncio
async def test_expanded_chats_section_capped_row_marker_surfaces_on_header() -> None:
    """TASK-912 review fix round 1: an expanded Chats section (the default
    once it has rows) with more rows than the cap surfaces the marker on a
    row pushed past the cap -- otherwise it renders nowhere at all."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    marked_index = CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 1

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(
            _base_grouped_workspace_state(
                rows=_chat_rows_with_marker_at(
                    marked_index,
                    "●",
                    count=CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 3,
                ),
            )
        )
        await pilot.pause()

        assert (
            _static_plain(console, "#console-conversation-browser-chats-title")
            == "Chats ●"
        )
        # The marked row itself stays unmounted -- it really is past the cap.
        assert (
            len(console.query(".console-workspace-conversation-row"))
            == CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT
        )


@pytest.mark.asyncio
async def test_expanded_chats_section_visible_row_marker_has_no_header_echo() -> None:
    """A marked row still within the visible cap already shows its own
    glyph -- the header must not also echo it (no double marker)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(
            _base_grouped_workspace_state(
                rows=_chat_rows_with_marker_at(
                    2,  # well within the cap
                    "●",
                    count=CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 3,
                ),
            )
        )
        await pilot.pause()

        assert (
            _static_plain(console, "#console-conversation-browser-chats-title")
            == "Chats"
        )


@pytest.mark.asyncio
async def test_collapsed_chats_section_header_shows_aggregate_unchanged() -> None:
    """Collapsed-section rendering (`section.run_marker`, the full
    aggregate) is unchanged by this fix -- it still wins over
    `capped_run_marker` while collapsed."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    marked_index = CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 1

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(
            _base_grouped_workspace_state(
                rows=_chat_rows_with_marker_at(
                    marked_index,
                    "●",
                    count=CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 3,
                ),
                group_collapse_preferences={"section:chats": True},
            )
        )
        await pilot.pause()

        assert (
            _static_plain(console, "#console-conversation-browser-chats-title")
            == "Chats ●"
        )
        assert len(console.query(".console-workspace-conversation-row")) == 0


@pytest.mark.asyncio
async def test_rail_title_budget_scales_with_terminal_width() -> None:
    """TASK-374 AC#1: titles get the available width instead of a fixed cap.

    The review saw 17-char titles on a wide terminal (pre-width-aware code). The
    grouped-browser title budget is now measured from the real rail width, so it
    grows as the terminal widens -- a wide terminal yields 25+ char titles. This
    locks that responsiveness so it cannot regress to a fixed cap.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_base_grouped_workspace_state())
        await pilot.pause()
        narrow_budget = tray._browser_title_budget()

        await pilot.resize_terminal(260, 60)
        await pilot.pause()
        tray.sync_state(_base_grouped_workspace_state())
        await pilot.pause()
        wide_budget = tray._browser_title_budget()

    assert wide_budget > narrow_budget, (
        f"title budget must grow with rail width, not stay fixed "
        f"(narrow={narrow_budget}, wide={wide_budget})"
    )
    assert wide_budget >= 25, (
        f"a wide terminal must give titles the available width (25+ cells), "
        f"got {wide_budget}"
    )


@pytest.mark.asyncio
async def test_on_resize_alone_regrows_wrap_budget_within_one_pause() -> None:
    """TASK-1191 fast-follow: `on_resize` must regrow the row-wrap budget on
    its OWN, isolated from any caller also driving an explicit
    `sync_state()`.

    `test_rail_title_budget_scales_with_terminal_width` above proves the
    budget grows with terminal width, but it re-calls `tray.sync_state()`
    after every resize -- so it cannot tell whether `on_resize`'s own
    `call_after_refresh(self._fit_height_to_content)` pass (see
    `ConsoleWorkspaceContextTray.on_resize`) is doing the regrow work by
    itself, or whether the explicit sync is silently carrying it. This test
    resizes the mounted tray and reads its measured width/budget back with
    no `sync_state()` call anywhere in between, single `pilot.pause()`
    only -- the same one-deferred-pass path TASK-1191 collapsed
    `_schedule_recomposed_content_fit` down to (`_fit_height_to_content`'s
    docstring: `on_resize` already used `call_after_refresh` for this job
    before TASK-1191 and is unchanged by it, but this is still the first
    test to isolate that on_resize path from a follow-up sync).
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_base_grouped_workspace_state())
        await pilot.pause()

        narrow_row_width = tray._row_content_width
        narrow_budget = tray._browser_title_budget()

        # Resize only -- deliberately no `tray.sync_state()` call anywhere
        # below, so any regrow observed here is `on_resize`'s own doing.
        await pilot.resize_terminal(260, 60)
        await pilot.pause()  # one pause only: the single fit pass must land here

        wide_row_width = tray._row_content_width
        wide_budget = tray._browser_title_budget()
        settled_height = int(tray.region.height)

    assert wide_row_width > narrow_row_width, (
        "on_resize alone (no sync_state in between) must regrow the "
        f"measured row content width (narrow={narrow_row_width}, "
        f"wide={wide_row_width})"
    )
    assert wide_budget > narrow_budget, (
        "on_resize alone (no sync_state in between) must regrow the row "
        f"wrap budget (narrow={narrow_budget}, wide={wide_budget})"
    )
    assert settled_height > 0, (
        "the tray height must converge within on_resize's single fit pass, "
        f"got {settled_height}"
    )


@pytest.mark.asyncio
async def test_sync_state_schedules_exactly_one_deferred_fit_pass(monkeypatch) -> None:
    """TASK-1191 regression guard for `_schedule_recomposed_content_fit`'s
    scheduling shape, not just its outcome.

    The other TASK-1191 tests in this file (`test_rail_title_budget_scales_
    with_terminal_width`, `test_on_resize_alone_regrows_wrap_budget_within_
    one_pause`) prove the tray still converges within a single
    `pilot.pause()` -- an outcome that a reintroduced two-`call_later`-hop
    fan-out could still satisfy by coincidence in a fast test run. This test
    instead pins the SCHEDULING CALLS `sync_state()` itself makes: exactly
    one `call_after_refresh` registration for the fit-and-restore-scroll
    closure, and zero `call_later`/`set_timer` calls from this seam --
    the old shape `_schedule_recomposed_content_fit`'s docstring describes
    (two `call_later` hops plus a 0.01s `set_timer` scroll-restore, commit
    1115fa624, collapsed by TASK-1191's f10c6bcdd).

    `call_after_refresh`/`call_later`/`set_timer` are spied at the INSTANCE
    level with record-and-forward wrappers, so the tray still settles
    normally (behavior preserved) while every scheduling call is captured.
    The fit seam's own callback is a nested closure literally named
    `fit_and_restore_scroll` inside `_schedule_recomposed_content_fit`, so
    it is identified by `callback.__name__` -- discriminating it from any
    other legitimate `call_after_refresh` use on this widget (`on_mount`,
    `on_resize`) rather than assuming every recorded call belongs to this
    seam.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        # Settle the tray's own on_mount fit pass first so it cannot leak
        # into the calls recorded below.
        await pilot.pause()

        recorded: dict[str, list] = {
            "call_after_refresh": [],
            "call_later": [],
            "set_timer": [],
        }
        original_call_after_refresh = tray.call_after_refresh
        original_call_later = tray.call_later
        original_set_timer = tray.set_timer

        def _call_after_refresh_spy(callback, *args, **kwargs):
            recorded["call_after_refresh"].append(callback)
            return original_call_after_refresh(callback, *args, **kwargs)

        def _call_later_spy(callback, *args, **kwargs):
            recorded["call_later"].append(callback)
            return original_call_later(callback, *args, **kwargs)

        def _set_timer_spy(delay, callback=None, *, name=None, pause=False):
            recorded["set_timer"].append(callback)
            return original_set_timer(delay, callback, name=name, pause=pause)

        monkeypatch.setattr(tray, "call_after_refresh", _call_after_refresh_spy)
        monkeypatch.setattr(tray, "call_later", _call_later_spy)
        monkeypatch.setattr(tray, "set_timer", _set_timer_spy)

        tray.sync_state(_base_grouped_workspace_state())

        # No call_later fan-out and no set_timer scroll-restore hop -- the
        # old two-hop-plus-timer shape this pins against regressing to.
        assert recorded["call_later"] == []
        assert recorded["set_timer"] == []

        # Exactly one deferred fit pass was scheduled via call_after_refresh,
        # and it is THE fit seam's own callback (by name), not some other
        # call_after_refresh use miscounted as this seam.
        fit_callbacks = [
            callback
            for callback in recorded["call_after_refresh"]
            if getattr(callback, "__name__", None) == "fit_and_restore_scroll"
        ]
        assert len(fit_callbacks) == 1
        assert recorded["call_after_refresh"] == fit_callbacks

        # Let the scheduled pass actually run so the tray settles normally
        # before the harness tears down (spies forward to the real
        # primitives, so this is unaffected by the patch above).
        await pilot.pause()


@pytest.mark.asyncio
async def test_console_conversation_star_uses_recognizable_star_glyphs():
    """TASK-357: the star toggle must use a recognizable ★/☆ pair, not the
    near-invisible one-cell '*'/'.' distinction."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_base_grouped_workspace_state())
        await pilot.pause()

        for star in console.query(".console-conversation-star"):
            label = str(star.label)
            assert "*" not in label and label.strip() != "."
            if getattr(star, "starred", False):
                assert "★" in label
            else:
                assert "☆" in label


@pytest.mark.asyncio
async def test_console_conversation_star_press_confirms_the_toggle():
    """TASK-357: starring must confirm the change ('Starred "<title>"') rather
    than toggle state silently (the review saw an accidental star go unnoticed)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_base_grouped_workspace_state())
        await pilot.pause()

        star = next(
            s
            for s in console.query(".console-conversation-star")
            if not getattr(s, "starred", False)
        )
        # A title with Rich markup must be escaped in the toast, not interpreted.
        star.conversation_title = "[b]Plan[/b]"

        class _Marks:
            def is_starred(self, conversation_id):
                return False

            def star_conversation(self, conversation_id):
                return None

            def unstar_conversation(self, conversation_id):
                return None

        console.app_instance.conversation_local_marks_service = _Marks()
        notes: list[str] = []
        console.app_instance.notify = lambda message, **kwargs: notes.append(message)

        await console.on_button_pressed(Button.Pressed(star))
        # task-15471: the durable write + confirmation run on a worker now.
        await console.workers.wait_for_complete()
        await pilot.pause()

        assert any("Starred" in note for note in notes)
        # The markup is escaped (literal backslash-brackets), never interpreted.
        assert any(r"\[b]Plan\[/b]" in note for note in notes)


@pytest.mark.asyncio
async def test_console_conversation_star_confirms_an_untitled_conversation():
    """task-3024: an empty title must still confirm, not crash after the write.

    `"".splitlines()` is `[]`, so the first-line read raised `IndexError` on an
    untitled conversation -- and it did so AFTER the durable star write, so the
    star landed while the user saw no confirmation and the context rail never
    re-synced. The toast is supposed to simply omit the quoted name here, which
    the suffix logic already handled; only the first-line read was unguarded.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_base_grouped_workspace_state())
        await pilot.pause()

        star = next(
            s
            for s in console.query(".console-conversation-star")
            if not getattr(s, "starred", False)
        )
        star.conversation_title = ""

        starred: list[str] = []

        class _Marks:
            def is_starred(self, conversation_id):
                return False

            def star_conversation(self, conversation_id):
                starred.append(conversation_id)

            def unstar_conversation(self, conversation_id):
                return None

        console.app_instance.conversation_local_marks_service = _Marks()
        notes: list[str] = []
        console.app_instance.notify = lambda message, **kwargs: notes.append(message)

        await console.on_button_pressed(Button.Pressed(star))
        # task-15471: the durable write + confirmation run on a worker now.
        await console.workers.wait_for_complete()
        await pilot.pause()

        # The durable write always landed; the confirmation is what was lost.
        assert starred, "the star write did not happen"
        assert notes == ["Starred."], notes


@pytest.mark.asyncio
async def test_console_workspace_context_disables_star_controls_when_marks_unavailable() -> (
    None
):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_base_grouped_workspace_state(marks_available=False))
        await pilot.pause()

        stars = list(console.query(".console-conversation-star"))

        assert len(console.query(".console-workspace-conversation-row")) >= 1
        assert stars
        assert all(star.disabled for star in stars)
        assert "Local stars unavailable" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_workspace_context_search_controls_keep_stable_ids() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_base_grouped_workspace_state(query="planning"))
        await pilot.pause()

        search = console.query_one("#console-workspace-conversation-search", Input)
        clear = console.query_one(
            "#console-workspace-conversation-search-clear", Button
        )
        list_container = console.query_one("#console-workspace-conversations")

        assert search.value == "planning"
        assert clear.disabled is False
        assert list_container is not None
        assert len(console.query("#console-workspace-conversation-search")) == 1
        assert len(console.query("#console-workspace-conversation-search-clear")) == 1
        assert len(console.query("#console-workspace-conversations")) == 1


def test_console_workspace_context_grouped_browser_styles_are_declared() -> None:
    for css_path in (
        Path("tldw_chatbook/css/components/_agentic_terminal.tcss"),
        Path("tldw_chatbook/css/tldw_cli_modular.tcss"),
    ):
        css = css_path.read_text(encoding="utf-8")

        assert ".console-conversation-browser-section-header {" in css
        assert ".console-conversation-browser-section-title {" in css
        assert ".console-conversation-browser-group-header {" in css
        assert ".console-conversation-browser-group-title {" in css
        assert ".console-conversation-browser-row-line {" in css
        assert ".console-conversation-star {" in css
        list_selector = "#console-workspace-conversations {"
        assert list_selector in css
        list_block = css.split(list_selector, 1)[1].split("}", 1)[0]
        assert "overflow-y: auto" not in list_block
        assert "scrollbar-size:" not in list_block
        assert "#console-workspace-conversations:focus {" in css

        # Row lines must size to their explicitly-heighted buttons; Textual's
        # Horizontal defaults to `height: 1fr`, which divides the list height
        # equally and breaks mixed wrapped/badge row heights.
        row_line_block = css.split(".console-conversation-browser-row-line {", 1)[
            1
        ].split("}", 1)[0]
        assert "height: auto" in row_line_block
        # Reserve the scrollbar cell permanently so row-wrap width does not
        # depend on scroll state (scrollbar toggle <-> rewrap feedback loop).
        rail_body_block = css.split("#console-left-rail-body {", 1)[1].split("}", 1)[0]
        assert "scrollbar-gutter: stable" in rail_body_block


def test_console_workspace_conversation_visible_rows_are_clamped() -> None:
    assert console_workspace_conversation_visible_rows(None) == 4
    assert console_workspace_conversation_visible_rows(10) == 4
    assert console_workspace_conversation_visible_rows(48) == 7
    assert console_workspace_conversation_visible_rows(120) == 12


def test_console_workspace_conversation_result_copy_is_explicit() -> None:
    assert (
        console_workspace_conversation_result_copy(
            query="research",
            result_total_count=143,
            result_limit=50,
        )
        == "Showing 50 of 143 matches"
    )
    assert (
        console_workspace_conversation_result_copy(
            query="research",
            result_total_count=3,
            result_limit=50,
        )
        == "3 matches"
    )
    assert (
        console_workspace_conversation_result_copy(
            query="",
            result_total_count=None,
            result_limit=50,
        )
        == ""
    )


def _configure_native_ready_console(app, model: str = "local-model") -> None:
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": model},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": model,
            },
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = model


# TASK-1190: five tests that pinned `ConsoleWorkspaceContextTray`'s
# transitional legacy conversation-list compose path (rendered only when
# `state.conversation_browser is None` -- reached here only by directly
# calling `sync_state()` with a hand-built legacy-shaped state, never by any
# production code path, see the reachability note on `compose()` in
# `console_workspace_context.py`) were removed along with that dead path:
#   - test_console_workspace_conversations_render_bounded_expanded_section
#   - test_console_workspace_conversations_collapsed_shows_selected_summary_only
#   - test_console_workspace_legacy_conversation_toggle_collapses_and_expands
#   - test_console_workspace_conversations_fallback_disables_unowned_controls
#   - test_console_workspace_conversations_clear_requires_enabled_search
# `test_console_workspace_context_renders_grouped_conversation_browser`
# above already covers the one real path (`conversation_browser` present).


def _render_screen_lines(console) -> list[str]:
    """Render the full screen to plain text lines -- the same view a human
    (or a tmux capture, matching the UAT's own flow) would see, not the
    widget tree."""
    compositor = console.screen._compositor
    return [
        "".join(seg.text for seg in strip._segments)
        for strip in compositor.render_strips()
    ]


@pytest.mark.asyncio
async def test_narrow_details_rail_paints_full_private_scratch_value() -> None:
    """The human-visible rail must not reduce the authority value to ``Priva…``."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 55)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-runtime-value")
        if not console._current_console_rail_state().details_open:
            console._toggle_console_rail_section("details")
        await pilot.pause()

        details_section = console.query_one(
            "#console-bounded-section-details", ConsoleBoundedSection
        )
        left_rail = console.query_one("#console-left-rail")
        runtime_value = console.query_one("#console-workspace-runtime-value")
        left_rail.activate_section("details")
        await _wait_for_condition(
            pilot,
            lambda: (
                details_section.desired_content_lines > 0
                and not details_section._reconcile_scheduled
                and not left_rail._allocation_reconcile_scheduled
            ),
        )
        details_section.viewport.scroll_to_widget(
            runtime_value, animate=False, immediate=True
        )
        await pilot.pause(0.1)

        rendered_rows = _render_screen_lines(console)
        assert any(
            "Local files" in row and "Private scratch" in row for row in rendered_rows
        ), [row for row in rendered_rows if "Local" in row or "Priva" in row]


async def _wait_for_workspace_switcher_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if host.screen_stack and host.screen_stack[-1].query(
            "#console-workspace-switcher-modal"
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console workspace switcher modal did not open")


async def _wait_for_console_screen(host: ConsoleHarness, console, pilot) -> None:
    for _ in range(40):
        if host.screen_stack and host.screen_stack[-1] is console:
            await pilot.pause()
            return
        await pilot.pause(0.05)
    raise AssertionError("Console workspace switcher did not dismiss")


def test_console_workspace_switcher_modal_documents_constructor_contract() -> None:
    docstring = inspect.getdoc(ConsoleWorkspaceSwitcherModal)

    assert docstring is not None
    assert "Args:" in docstring


def test_console_workspace_runtime_label_is_case_insensitive() -> None:
    assert (
        ConsoleWorkspaceDetailsTray._friendly_status_label(
            "Runtime: 2 bindings, 1 Ready, 1 Missing"
        )
        == "Local file tools: 1 ready, 1 missing"
    )


def test_console_workspace_authority_label_preserves_non_local_state() -> None:
    assert (
        ConsoleWorkspaceDetailsTray._friendly_status_label("Authority: runtime-missing")
        == "Storage: runtime missing"
    )
    assert (
        ConsoleWorkspaceDetailsTray._friendly_status_label("Authority: server-backed")
        == "Storage: server backed"
    )


def test_console_workspace_readiness_detail_preserves_error_copy() -> None:
    assert (
        ConsoleWorkspaceDetailsTray._friendly_detail_copy(
            "Workspace registry service is not ready. No background sync is running."
        )
        == "Workspace registry service is not ready. No background sync is running."
    )
    assert (
        ConsoleWorkspaceDetailsTray._friendly_detail_copy(
            "Workspace registry could not be read. No background sync is running."
        )
        == "Workspace registry could not be read. No background sync is running."
    )
    assert (
        ConsoleWorkspaceDetailsTray._friendly_detail_copy(
            "Local registry fallback is active. No background sync is running."
        )
        == "Chats stay local. Connect a server later for explicit handoff."
    )


@pytest.mark.asyncio
async def test_console_left_rail_splits_staged_context_from_workspace_context() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        left_rail = console.query_one("#console-left-rail")
        staged_context = console.query_one("#console-staged-context-tray")
        session_context = console.query_one("#console-session-context")
        workspaces_context = console.query_one("#console-workspaces-context")
        conversations_context = console.query_one("#console-workspace-context")
        # Task-400 keeps staged sources in the Inspector. TASK-14810 splits
        # the former mixed Session tray into three peer context sections.
        assert session_context.parent.id == "console-rail-section-body-session"
        assert workspaces_context.parent.id == "console-rail-section-body-workspace"
        assert (
            conversations_context.parent.id == "console-rail-section-body-conversations"
        )
        assert staged_context.parent.id == "console-inspector-rail-body"
        assert not list(left_rail.query("#console-staged-context-tray"))
        assert not list(console.query("#console-workspace-conversations-title"))
        assert len(console.query("#console-workspace-recovery")) == 0
        switch_button = console.query_one("#console-change-workspace", Button)
        assert switch_button.disabled is True
        new_conversation = console.query_one(
            "#console-new-workspace-conversation", Button
        )
        assert new_conversation.disabled is False
        text = _visible_text(console)
        assert "Sources" in text
        # The workspace context tray no longer renders its own heading; the
        # "Session" rail-section header labels this section instead.
        assert "Session" in text
        assert "Default" in text
        assert "Workspace switching: locked" not in text
        assert DEFAULT_WORKSPACE_ID in {
            app.workspace_registry_service.get_active_workspace().workspace_id
        }
        assert "until workspace selection is wired" not in text
        assert "read-only" not in text
        assert "Change workspace" not in text
        assert "New conversation" in text


@pytest.mark.asyncio
async def test_console_workspace_context_exposes_new_conversation_for_default_workspace() -> (
    None
):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        await _wait_for_selector(console, pilot, "#console-new-workspace-conversation")

        new_conversation = console.query_one(
            "#console-new-workspace-conversation", Button
        )
        assert new_conversation.disabled is False

        text = _visible_text(console)
        assert "New conversation" in text
        _assert_status_row(
            console,
            label_selector="#console-workspace-runtime-label",
            value_selector="#console-workspace-runtime-value",
            label="Local files",
            value_contains="Private scratch",
        )
        # TASK-715: unconfigured server features collapse into one line
        # instead of a Server status row.
        assert console.query("#console-workspace-server-features-collapsed")
        assert not console.query("#console-workspace-server-readiness-label")
        assert "local registry" not in text.lower()
        assert "authoritative" not in text.lower()
        assert "Workspace conversation creation lands in a later slice" not in text


@pytest.mark.asyncio
async def test_console_workspace_selector_is_compact_plain_status_row() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-active-workspace")

        active_workspace = console.query_one("#console-active-workspace")
        value = active_workspace.query_one("#console-active-workspace-value", Static)
        rendered_label = str(value.renderable)
        border = active_workspace.styles.border

        assert rendered_label == "Default"
        assert active_workspace.region.height == 1
        assert border.top[0] in {"", "none"}
        assert border.right[0] in {"", "none"}
        assert border.bottom[0] in {"", "none"}
        assert border.left[0] in {"", "none"}


@pytest.mark.asyncio
async def test_session_tray_shows_workspace_scope_and_new_button() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Research Sprint")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="Planning thread",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-active-workspace")

        workspace_label = console.query_one(
            "#console-active-workspace .console-workspace-status-label", Static
        )
        scope_label = console.query_one(
            "#console-active-scope .console-workspace-status-label", Static
        )
        assert "Workspace" in str(workspace_label.renderable)
        # RAG-45: this pair shows the active CONVERSATION's identity, not a
        # RAG retrieval scope, so it is labeled "Conversation" -- distinct
        # from the "RAG Scope" button and the Inspector's item-scope row
        # ("Scope: everything" / "Scope: N items").
        assert "Conversation" in str(scope_label.renderable)

        new_button = console.query_one("#console-new-workspace", Button)
        assert new_button.disabled is False


@pytest.mark.asyncio
async def test_conversation_row_shows_placeholder_when_no_active_conversation() -> None:
    """RAG-45: a fresh session with no active conversation must not render a
    bare "Conversation" label with an empty value body. The Sessions section
    names the empty state explicitly."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        # `_base_grouped_workspace_state` leaves `scope_label`/`scope_detail`
        # at their dataclass defaults (""), matching a fresh session with no
        # active conversation.
        state = _base_grouped_workspace_state()
        assert state.scope_label == ""
        tray.sync_state(state)
        await pilot.pause()

        _assert_status_row(
            console,
            label_selector="#console-active-scope-label",
            value_selector="#console-active-scope-value",
            label="Conversation",
            value_contains="None",
        )


@pytest.mark.asyncio
async def test_status_label_width_preserves_one_cell_gutter() -> None:
    """Status labels grow only enough to retain one readable gutter cell."""

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield ConsoleWorkspaceStatusPair(
                "Workspace", "demo", label_id="l", value_id="v"
            )
            yield ConsoleWorkspaceStatusPair(
                "Local file tools", "Private scratch", label_id="fl", value_id="fv"
            )

    app = TestApp()
    async with app.run_test():
        assert app.query_one("#l").styles.width.value == 13
        assert app.query_one("#fl").styles.width.value == 17
        assert app.query_one("#v").styles.min_width.value == 10
        assert app.query_one("#fv").styles.min_width.value == 6


def _composited_rows(container) -> list[str]:
    """`container`'s own row-span exactly as the compositor painted it.

    Source-widget text (label/value `Static.renderable`) is NOT ground
    truth for what a terminal shows -- Textual packs the two `Static`s of a
    `ConsoleWorkspaceStatusPair` into one `Horizontal` with no gutter, so a
    label whose fixed column width exactly matches its text length reads as
    fused with the adjacent value on the actual painted row even though the
    two widgets' own `renderable`s are cleanly separate strings (RAG-47
    lesson: measure the compositor's output, not source labels).
    """
    strips = container.screen._compositor.render_strips()
    region = container.region
    rows = []
    for y in range(region.y, region.y + region.height):
        if 0 <= y < len(strips):
            row_text = "".join(segment.text for segment in strips[y])
            rows.append(row_text[region.x : region.x + region.width])
    return rows


async def _settled_composited_row(pilot, container, label_widget) -> str:
    """The composited row `label_widget` is painted on, once layout settles.

    task-3025: the original assertion read `_composited_rows(container)[0]`,
    which is an INDEX into whatever the compositor had painted at that
    instant. Sampled mid-layout it returned a neighbouring pair's row -- the
    observed failure got `'Model'` where `'Conversation'` was expected -- so
    the test failed nondeterministically (measured 1 pass / 2 fail and 2 pass
    / 1 fail across two arms of three isolated runs each).

    The rail's row ORDER is not the problem and asserting on it is not the
    point of the caller: what matters is that a specific label and its value
    are visually separated. So resolve the row by the label widget's own
    identity, and wait for the paint to agree with the widget tree before
    reading it.

    Args:
        pilot: The active Textual pilot, used to yield between polls.
        container: The status-pair container whose x-span bounds the row.
        label_widget: The `Static` whose painted row is wanted.

    Returns:
        str: The container-width slice of the row the label is painted on.

    Raises:
        AssertionError: If the label's own text never appears on its own
            painted row -- that is a real paint failure, not a race, and it
            must not be silently retried away.
    """
    label_text = str(label_widget.renderable).strip()
    row_text = ""
    for _ in range(100):
        strips = container.screen._compositor.render_strips()
        y = label_widget.region.y
        region = container.region
        if 0 <= y < len(strips):
            painted = "".join(segment.text for segment in strips[y])
            row_text = painted[region.x : region.x + region.width]
            if label_text and label_text in row_text:
                return row_text
        await pilot.pause()
    raise AssertionError(
        f"label {label_text!r} never appeared on its own painted row; "
        f"last read {row_text!r}"
    )


@pytest.mark.asyncio
async def test_conversation_status_row_label_and_value_are_separate_visual_runs() -> (
    None
):
    """I1 (final review): live captures showed `Conversation—` (placeholder
    value) and `ConversationThis conversation` (real title) rendering as one
    run-on token on the main rail -- the 12-char "Conversation" label filled
    its whole fixed-width cell with no separator before the value column.
    Assert the COMPOSITED row (the RAG-47 lesson) shows the label followed
    by a literal space before the value starts, for both the placeholder and
    a real conversation title.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )

        # Placeholder value ("Conversation—" in the pre-fix report).
        tray.sync_state(_base_grouped_workspace_state())
        await pilot.pause()
        scope_pair = console.query_one("#console-active-scope")
        scope_label = console.query_one("#console-active-scope-label", Static)
        row_text = await _settled_composited_row(pilot, scope_pair, scope_label)
        assert "Conversation " in row_text, (
            "label fused with the placeholder value on the composited row: "
            f"{row_text!r}"
        )

        # Real conversation title ("ConversationThis conversation" in the
        # pre-fix report).
        state = replace(
            _base_grouped_workspace_state(),
            scope_label="This conversation",
            scope_detail="conv-1",
        )
        tray.sync_state(state)
        await pilot.pause()
        scope_pair = console.query_one("#console-active-scope")
        scope_label = console.query_one("#console-active-scope-label", Static)
        row_text = await _settled_composited_row(pilot, scope_pair, scope_label)
        assert "Conversation " in row_text, (
            "label fused with the conversation title on the composited row: "
            f"{row_text!r}"
        )
        assert "ConversationThis" not in row_text


@pytest.mark.asyncio
async def test_status_pair_value_truncates_instead_of_letter_stacking() -> None:
    """TASK-384: at a narrow rail the value column shrinks to a few cells; the
    value must nowrap+ellipsize (so "Default" reads "De…") rather than word-wrap
    into a "Def / aul / t" letter stack, with the full value on hover."""
    from textual.widgets import Static

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield ConsoleWorkspaceStatusPair(
                "Workspace", "Default", label_id="l", value_id="v"
            )

    app = TestApp()
    async with app.run_test():
        value = app.query_one("#v", Static)
        assert value.styles.text_wrap == "nowrap"
        assert value.styles.text_overflow == "ellipsis"
        assert value.tooltip == "Default"


@pytest.mark.asyncio
async def test_status_pair_value_tooltip_escapes_markup() -> None:
    """Qodo #821: the value tooltip renders Rich markup, so a value with bracket
    tokens must be escaped (shown literally, not interpreted)."""
    from textual.widgets import Static

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield ConsoleWorkspaceStatusPair(
                "Workspace", "[b]danger[/b]", label_id="l", value_id="v"
            )

    app = TestApp()
    async with app.run_test():
        value = app.query_one("#v", Static)
        assert value.tooltip == r"\[b]danger\[/b]"


@pytest.mark.asyncio
async def test_console_workspace_context_renders_active_workspace() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(
        workspace_id="ws-a",
        name="Research Sprint",
        sync_status=WorkspaceSyncStatus.READY,
    )
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="Planning thread",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-active-workspace")

        text = _visible_text(console)
        assert "Research Sprint" in text
        _assert_status_row(
            console,
            label_selector="#console-workspace-sync-label",
            value_selector="#console-workspace-sync-value",
            label="Sync",
            value_contains="dry-run only",
        )
        assert "Planning thread" in text
        assert len(console.query("#console-new-workspace-conversation")) == 1
        assert "Workspace conversation creation lands in a later slice." not in text


@pytest.mark.asyncio
async def test_console_workspace_context_renders_server_readiness_handoff_and_acp_contracts() -> (
    None
):
    app = _build_test_app()
    app.workspace_server_adapter_state = ConsoleWorkspaceServerAdapterState(
        available=False,
        detail="No tldw_server workspace API configured.",
    )
    app.workspace_acp_handoff_state = ConsoleWorkspaceACPHandoffState(
        status="unavailable",
        detail="ACP task/run package handoff is not wired.",
        audit_detail="Audit: visible only; no package was sent.",
    )
    service = app.workspace_registry_service
    service.create_workspace(
        workspace_id="ws-a",
        name="Server Readiness",
        authority=WorkspaceAuthority.RUNTIME_MISSING,
        sync_status=WorkspaceSyncStatus.BLOCKED,
    )
    service.set_active_workspace("ws-a")
    service.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="ws-a",
            binding_id="acp-run-1",
            binding_kind=RuntimeBindingKind.ACP_SESSION,
            label="ACP run package",
            locator="acp://runs/1",
            status=RuntimeBindingStatus.MISSING,
        )
    )
    service.link_membership(
        "ws-a",
        item_type="note",
        item_id="note-1",
        role="source",
        title="Source note",
        transfer_policy=WorkspaceTransferPolicy.COPY,
    )
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="Conversation package",
        transfer_policy=WorkspaceTransferPolicy.METADATA_ONLY,
    )

    host = ConsoleHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        text = _visible_text(console)
        _assert_status_row(
            console,
            label_selector="#console-workspace-server-readiness-label",
            value_selector="#console-workspace-server-readiness-value",
            label="Server",
            value_contains="Unavailable",
        )
        assert "No tldw_server workspace API configured." in text
        _assert_status_row(
            console,
            label_selector="#console-workspace-runtime-label",
            value_selector="#console-workspace-runtime-value",
            label="Local files",
            value_contains="Private scratch",
        )
        assert "Handoff" in text
        assert "Source note - copy" in text
        assert "Conversation package - metadata-only" in text
        # TASK-715: the ACP status row no longer shares the Handoff section's
        # label - "ACP" fits the 12-cell column and is unambiguous.
        _assert_status_row(
            console,
            label_selector="#console-workspace-handoff-label",
            value_selector="#console-workspace-handoff-value",
            label="ACP",
            value_contains="Not configured",
        )
        assert "Audit: visible only; no package was sent." in text


@pytest.mark.asyncio
async def test_console_workspace_context_renders_markup_titles_literally() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="[bold red]Research[/]")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="[blink]Planning[/]",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        text = _visible_text(console)
        assert "[bold red]Research[/]" in text
        assert "[blink]Planning[/]" in text


@pytest.mark.asyncio
async def test_console_change_workspace_switches_active_context_and_conversation_rows() -> (
    None
):
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-a",
        role="workspace-thread",
        title="Planning A",
    )
    service.link_membership(
        "ws-b",
        item_type="conversation",
        item_id="conv-b",
        role="workspace-thread",
        title="Planning B",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-change-workspace")

        assert "Workspace A" in _visible_text(console)
        assert "Planning A" in _visible_text(console)
        assert "Planning B" not in _visible_text(console)

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        current_workspace = modal_screen.query_one(
            ".console-workspace-switcher-current",
            Static,
        )
        assert str(current_workspace.renderable) == "Workspace A (current)"
        assert all(
            str(button.label) != "Workspace A (current)"
            for button in modal_screen.query(Button)
        )
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )

        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == "ws-b"
        assert "Workspace B" in _visible_text(console)
        assert "Planning B" in _visible_text(console)
        assert "Planning A" not in _visible_text(console)


def test_console_workspace_conversation_subsection_styles_are_declared() -> None:
    css = Path("tldw_chatbook/css/components/_agentic_terminal.tcss").read_text(
        encoding="utf-8"
    )

    assert "#console-workspace-conversations-header {" in css
    assert ".console-workspace-action.console-workspace-conversations-toggle {" in css
    assert "#console-workspace-selected-conversation {" in css
    assert "#console-workspace-conversation-search-row {" in css
    context_selector = "#console-workspace-context"
    assert context_selector in css
    context_blocks = [
        block.split("}", 1)[0] for block in css.split(context_selector)[1:]
    ]
    assert all("overflow-y: auto" not in block for block in context_blocks)
    assert "#console-left-rail-body {" in css
    left_rail_body_block = css.split("#console-left-rail-body {", 1)[1].split("}", 1)[0]
    assert "overflow-y: auto" in left_rail_body_block
    list_selector = "#console-workspace-conversations {"
    assert list_selector in css

    list_block = css.split(list_selector, 1)[1].split("}", 1)[0]
    assert "overflow-y: auto" not in list_block
    assert "scrollbar-size:" not in list_block
    assert "#console-left-rail-body:focus {" in css


def test_console_workspace_aggregate_height_pins_badge_row_cost() -> None:
    """Aggregate row height includes badge row up-charge.

    Regression: row height calculation must sum plain (3px) and badge (4px)
    rows correctly. Under-counting silently overlaps rows on non-scrolling
    Vertical containers.
    """
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        ConsoleWorkspaceContextTray,
    )
    from tldw_chatbook.Workspaces.conversation_browser_state import (
        ConsoleConversationBrowserRow,
    )

    # At budget 20 every title here is a single name line, so:
    # Plain row (no subagent_count): costs 3px (1 name + 1 metadata + 1 margin).
    # Badge row (subagent_count > 0): costs 4px (plus a dedicated badge line).
    plain_rows = tuple(
        ConsoleConversationBrowserRow(
            row_key=f"plain-{i}",
            conversation_id=f"conv-plain-{i}",
            native_session_id=None,
            title=f"Plain row {i}",
            status="workspace-thread",
            selected=False,
            subagent_count=0,
            scope_type="workspace",
            workspace_id="ws-a",
            workspace_label="Workspace A",
            updated_label="1d",
            star_enabled=True,
            starred=False,
        )
        for i in range(3)
    )
    badge_rows = tuple(
        ConsoleConversationBrowserRow(
            row_key=f"badge-{i}",
            conversation_id=f"conv-badge-{i}",
            native_session_id=None,
            title=f"Badge row {i}",
            status="workspace-thread",
            selected=False,
            subagent_count=1,  # Has badge.
            scope_type="workspace",
            workspace_id="ws-a",
            workspace_label="Workspace A",
            updated_label="1d",
            star_enabled=True,
            starred=False,
        )
        for i in range(2)
    )
    mixed_rows = plain_rows + badge_rows

    # Expected sum: 3 plain rows * 3px/row + 2 badge rows * 4px/row = 17px.
    expected_height = 3 * 3 + 2 * 4
    actual_height = ConsoleWorkspaceContextTray._conversation_browser_rows_height(
        mixed_rows, 20
    )

    assert actual_height == expected_height == 17
    assert (
        actual_height
        == ConsoleWorkspaceContextTray._conversation_browser_rows_height(plain_rows, 20)
        + ConsoleWorkspaceContextTray._conversation_browser_rows_height(badge_rows, 20)
    )


_LONG_ROW_TITLE = "A very long conversation title that overflows the rail width easily"


def _long_title_grouped_state():
    return _base_grouped_workspace_state(
        rows=(
            _browser_row(
                "conv-long",
                _LONG_ROW_TITLE,
                selected=True,
                updated_sort="2026-06-27T09:00:00",
            ),
        )
    )


def _first_row_name_lines(console) -> list[str]:
    row_button = console.query_one("#console-workspace-conversation-0", Button)
    lines = str(row_button.label).splitlines()
    # Last line is the metadata line; badge rows are not used in this fixture.
    return lines[:-1]


def test_conversation_search_input_is_tall_enough_to_show_its_value() -> None:
    """The rail search box must have room for its text, not just its border.

    Roleplay UAT regression: the app-tier rule styled this Input `height: 1`
    while it kept Textual's default `tall` border, which needs one row of
    chrome above and below. The content row was squeezed out, so the compiled
    bundle rendered only the border's top edge -- typing "Seraphina" filtered
    the list to "1 match" while the query itself stayed invisible, with no way
    to see, verify or correct the active filter.

    Asserted against the authored component stylesheet because the compiled
    bundle is what the running app loads and it outranks widget DEFAULT_CSS;
    a mounted harness resolves the widget default (height 3) and stays green
    even when the shipped rule is broken.
    """
    css = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "components"
        / "_agentic_terminal.tcss"
    ).read_text()

    block_start = css.index("#console-workspace-conversation-search {")
    block = css[block_start : css.index("}", block_start)]
    heights = [
        line.split(":", 1)[1].strip().rstrip(";")
        for line in block.splitlines()
        if line.strip().startswith("height:")
    ]

    assert heights, "search input declares no height"
    # A bordered Input needs 3 rows: border, content, border.
    assert heights[0] == "3", (
        f"search input height {heights[0]!r} leaves no content row for its value"
    )
