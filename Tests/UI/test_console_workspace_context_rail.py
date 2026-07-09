"""Mounted Console workspace context rail regressions."""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.Widgets.Console import (
    ConsoleWorkspaceContextTray,
    ConsoleWorkspaceSwitcherModal,
)
from tldw_chatbook.Widgets.Console.console_workspace_details import (
    ConsoleWorkspaceDetailsTray,
)
from tldw_chatbook.Workspaces import (
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


def _visible_text(screen) -> str:
    visible_chunks: list[str] = []
    for widget in screen.query(Static):
        if widget.display and hasattr(widget, "renderable"):
            visible_chunks.append(getattr(widget.renderable, "plain", str(widget.renderable)))
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
        selected_summary="Conversation 2 - saved workspace",
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
    )


def _grouped_browser_state(
    *,
    marks_available: bool = True,
    query: str = "",
    rows: tuple[ConsoleConversationBrowserInputRow, ...] | None = None,
):
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
        group_collapse_preferences={"section:chats": False},
        query=query,
        marks_available=marks_available,
    )


def _base_grouped_workspace_state(
    *,
    marks_available: bool = True,
    query: str = "",
    rows: tuple[ConsoleConversationBrowserInputRow, ...] | None = None,
) -> ConsoleWorkspaceContextState:
    section = _section_state(rows=0)
    state = _base_workspace_state(section)
    return ConsoleWorkspaceContextState(
        heading=state.heading,
        workspace_label=state.workspace_label,
        authority_label=state.authority_label,
        sync_label=state.sync_label,
        runtime_label=state.runtime_label,
        conversation_rows=(),
        conversation_empty_copy=state.conversation_empty_copy,
        conversation_section=state.conversation_section,
        conversation_browser=_grouped_browser_state(
            marks_available=marks_available,
            query=query,
            rows=rows,
        ),
        change_workspace_enabled=state.change_workspace_enabled,
        change_workspace_recovery=state.change_workspace_recovery,
        new_conversation_enabled=state.new_conversation_enabled,
        new_conversation_recovery=state.new_conversation_recovery,
        recovery_copy=state.recovery_copy,
        server_readiness_label=state.server_readiness_label,
        server_readiness_detail=state.server_readiness_detail,
        handoff_rows=state.handoff_rows,
        acp_handoff_label=state.acp_handoff_label,
        acp_handoff_detail=state.acp_handoff_detail,
        acp_handoff_audit=state.acp_handoff_audit,
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
async def test_console_workspace_context_renders_grouped_conversation_browser() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_grouped_workspace_state())
        await pilot.pause()

        assert _static_plain(console, "#console-conversation-browser-starred-title") == "Starred"
        assert "Workspaces" in _visible_text(console)
        assert "Chats" in _visible_text(console)
        assert len(console.query("#console-workspace-conversation-search")) == 1
        assert len(console.query("#console-workspace-conversations")) == 1
        assert len(console.query(".console-conversation-star")) >= 1
        assert len(console.query(".console-workspace-conversation-row")) >= 1

        section_toggle = console.query_one(
            "#console-conversation-browser-section-toggle-starred",
            Button,
        )
        workspace_toggle = console.query_one(
            "#console-conversation-browser-group-toggle-0",
            Button,
        )
        row_button = console.query_one("#console-workspace-conversation-0", Button)
        star_button = console.query_one("#console-conversation-star-0", Button)

        assert section_toggle.group_id == "section:starred"
        assert workspace_toggle.group_id == "workspace:ws-a"
        assert row_button.row_key == "conv-starred"
        assert row_button.conversation_id == "conv-starred"
        assert row_button.native_session_id is None
        assert row_button.scope_type == "workspace"
        assert row_button.workspace_id == "ws-a"
        assert star_button.row_key == "conv-starred"
        assert star_button.conversation_id == "conv-starred"
        assert star_button.starred is True


@pytest.mark.asyncio
async def test_console_workspace_context_preserves_duplicate_starred_workspace_row_keys() -> None:
    rows = (
        _browser_row(
            "conv-duplicate",
            "Appears twice",
            starred=True,
            starred_sort="2026-06-27T10:00:00",
            updated_sort="2026-06-27T09:00:00",
        ),
    )
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_grouped_workspace_state(rows=rows))
        await pilot.pause()

        row_buttons = list(console.query(".console-workspace-conversation-row"))
        star_buttons = list(console.query(".console-conversation-star"))
        duplicate_rows = [
            button
            for button in row_buttons
            if getattr(button, "row_key", None) == "conv-duplicate"
        ]
        duplicate_stars = [
            button
            for button in star_buttons
            if getattr(button, "row_key", None) == "conv-duplicate"
        ]

        assert len({button.id for button in row_buttons}) == len(row_buttons)
        assert len({button.id for button in star_buttons}) == len(star_buttons)
        assert len(duplicate_rows) == 2
        assert len(duplicate_stars) == 2
        assert all(button.conversation_id == "conv-duplicate" for button in duplicate_rows)
        assert all(button.conversation_id == "conv-duplicate" for button in duplicate_stars)


@pytest.mark.asyncio
async def test_console_workspace_context_keeps_status_rows_below_grouped_browser() -> None:
    rows = tuple(
        _browser_row(
            f"conv-{index}",
            f"Planning {index}",
            updated_sort=f"2026-06-{27 - min(index, 20):02d}T09:00:00",
        )
        for index in range(16)
    )
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 34)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_grouped_workspace_state(rows=rows))
        # Status rows now live in the collapsible Details section below the
        # grouped conversation browser; expand it to lay out those rows.
        if not console._current_console_rail_state().details_open:
            console._toggle_console_rail_section("details")
        await pilot.pause()

        details_tray = console.query_one("#console-workspace-details")
        left_rail_body = console.query_one("#console-left-rail-body")
        conversation_list = console.query_one("#console-workspace-conversations")
        sync_label = console.query_one("#console-workspace-sync-label")
        server_readiness = console.query_one("#console-workspace-server-readiness-label")
        handoff_label = console.query_one("#console-workspace-handoff-label")
        composer = console.query_one("#console-native-composer")
        details_bottom = details_tray.region.y + details_tray.region.height

        assert _static_plain(console, "#console-workspace-sync-label") == "Sync"
        assert sync_label.region.y > conversation_list.region.y
        assert server_readiness.region.y > conversation_list.region.y

        assert getattr(conversation_list, "max_scroll_y", 0) == 0
        assert left_rail_body.max_scroll_y > 0
        left_rail_body.scroll_end(animate=False)
        await pilot.pause(0.1)

        assert left_rail_body.scroll_y > 0
        assert handoff_label.region.y >= details_tray.region.y
        assert handoff_label.region.y + handoff_label.region.height <= details_bottom
        assert handoff_label.region.y + handoff_label.region.height <= composer.region.y


@pytest.mark.asyncio
async def test_console_workspace_context_renders_disabled_star_for_unpersisted_native_session() -> None:
    rows = (
        _browser_row(
            "native:native-session-1",
            "Unsaved native session",
            conversation_id="",
            native_session_id="native-session-1",
            star_enabled=False,
            source_kind="native",
        ),
    )
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_grouped_workspace_state(rows=rows))
        await pilot.pause()

        star = console.query_one("#console-conversation-star-0", Button)

        assert star.disabled is True
        assert star.tooltip == "Send or save this conversation before starring."
        assert len(console.query(".console-workspace-conversation-row")) >= 1


@pytest.mark.asyncio
async def test_console_workspace_context_disables_star_controls_when_marks_unavailable() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
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
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_grouped_workspace_state(query="planning"))
        await pilot.pause()

        search = console.query_one("#console-workspace-conversation-search", Input)
        clear = console.query_one("#console-workspace-conversation-search-clear", Button)
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
        css = css_path.read_text()

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


@pytest.mark.asyncio
async def test_console_workspace_conversations_render_bounded_expanded_section() -> None:
    app = _build_test_app()
    section = _section_state(collapsed=False, rows=8)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_workspace_state(section))
        await pilot.pause()

        assert _static_plain(console, "#console-workspace-conversations-title") == "Conversations (8)"
        assert _static_plain(console, "#console-workspace-selected-conversation") == "Conversation 2 - saved workspace"
        assert len(console.query("#console-workspace-conversation-search")) == 1
        assert len(console.query("#console-workspace-conversation-search-clear")) == 1
        assert len(console.query("#console-new-workspace-conversation")) == 1
        conversation_list = console.query_one("#console-workspace-conversations")
        rows = list(console.query(".console-workspace-conversation-row"))
        assert len(rows) == 8
        assert conversation_list.region.height >= len(rows) * 2
        assert getattr(conversation_list, "max_scroll_y", 0) == 0


@pytest.mark.asyncio
async def test_console_workspace_conversations_collapsed_shows_selected_summary_only() -> None:
    app = _build_test_app()
    section = _section_state(collapsed=True, rows=8)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_workspace_state(section))
        await pilot.pause()

        assert _static_plain(console, "#console-workspace-conversations-title") == "Conversations (8)"
        assert _static_plain(console, "#console-workspace-selected-conversation") == "Conversation 2 - saved workspace"
        assert len(console.query("#console-workspace-conversation-search")) == 0
        assert len(console.query("#console-workspace-conversations")) == 0
        assert len(console.query("#console-new-workspace-conversation")) == 0


@pytest.mark.asyncio
async def test_console_workspace_legacy_conversation_toggle_collapses_and_expands() -> None:
    app = _build_test_app()
    section = _section_state(collapsed=False, rows=3)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_workspace_state(section))
        await pilot.pause()

        toggle = console.query_one("#console-workspace-conversations-toggle", Button)
        assert toggle.disabled is False
        assert len(console.query("#console-workspace-conversations")) == 1
        assert any("Conversation 0" in text for text in _conversation_row_texts(console))

        toggle.press()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversations-toggle")) == 1
        assert len(console.query("#console-workspace-conversations")) == 0
        assert _static_plain(
            console,
            "#console-workspace-selected-conversation",
        ) == "Conversation 2 - saved workspace"
        assert app.app_config["console"]["conversation_section"]["ws-a"][
            "collapsed"
        ] is True

        console.query_one("#console-workspace-conversations-toggle", Button).press()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversations")) == 1
        assert any("Conversation 0" in text for text in _conversation_row_texts(console))
        assert app.app_config["console"]["conversation_section"]["ws-a"][
            "collapsed"
        ] is False


@pytest.mark.asyncio
async def test_console_workspace_conversations_fallback_disables_unowned_controls() -> None:
    app = _build_test_app()
    section = _section_state(collapsed=False, rows=3)
    state = _base_workspace_state(section)
    legacy_state = ConsoleWorkspaceContextState(
        heading=state.heading,
        workspace_label=state.workspace_label,
        authority_label=state.authority_label,
        sync_label=state.sync_label,
        runtime_label=state.runtime_label,
        conversation_rows=state.conversation_rows,
        conversation_section=None,
        conversation_empty_copy=state.conversation_empty_copy,
        change_workspace_enabled=state.change_workspace_enabled,
        change_workspace_recovery=state.change_workspace_recovery,
        new_conversation_enabled=state.new_conversation_enabled,
        new_conversation_recovery=state.new_conversation_recovery,
        recovery_copy=state.recovery_copy,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(legacy_state)
        await pilot.pause()

        search_input = console.query_one("#console-workspace-conversation-search", Input)
        clear_button = console.query_one(
            "#console-workspace-conversation-search-clear",
            Button,
        )
        toggle_button = console.query_one(
            "#console-workspace-conversations-toggle",
            Button,
        )

        assert search_input.disabled is True
        assert clear_button.disabled is True
        assert toggle_button.disabled is True


@pytest.mark.asyncio
async def test_console_workspace_conversations_clear_requires_enabled_search() -> None:
    app = _build_test_app()
    section = _section_state(
        collapsed=False,
        rows=3,
        query="research",
        search_enabled=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_workspace_state(section))
        await pilot.pause()

        search_input = console.query_one("#console-workspace-conversation-search", Input)
        clear_button = console.query_one(
            "#console-workspace-conversation-search-clear",
            Button,
        )

        assert search_input.disabled is True
        assert clear_button.disabled is True


@pytest.mark.asyncio
async def test_console_workspace_many_conversations_keep_lower_status_reachable() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    for index in range(40):
        service.link_membership(
            active_workspace.workspace_id,
            item_type="conversation",
            item_id=f"overflow-chat-{index}",
            role="workspace-thread",
            title=f"Overflow Chat {index:02d}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 34)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversations")
        # Storage/Server-handoff status rows now live in the collapsible Details
        # section beneath the Session (workspace) section; expand it.
        if not console._current_console_rail_state().details_open:
            console._toggle_console_rail_section("details")
        await pilot.pause()

        workspace_context = console.query_one("#console-workspace-context")
        details_tray = console.query_one("#console-workspace-details")
        left_rail_body = console.query_one("#console-left-rail-body")
        conversation_list = console.query_one("#console-workspace-conversations")
        new_conversation = console.query_one("#console-new-workspace-conversation", Button)
        server_readiness = console.query_one("#console-workspace-server-readiness-label")
        handoff_label = console.query_one("#console-workspace-handoff-label")
        composer = console.query_one("#console-native-composer")
        hit_x = new_conversation.region.x + max(0, new_conversation.region.width // 2)
        hit_y = new_conversation.region.y + max(0, new_conversation.region.height // 2)
        hit_widget, _region = console.get_widget_at(hit_x, hit_y)

        workspace_bottom = workspace_context.region.y + workspace_context.region.height
        details_bottom = details_tray.region.y + details_tray.region.height

        assert conversation_list.region.height > left_rail_body.region.height
        assert new_conversation.region.y + new_conversation.region.height <= composer.region.y
        assert new_conversation.region.y + new_conversation.region.height <= workspace_bottom
        assert hit_widget is new_conversation
        assert server_readiness.region.y > conversation_list.region.y

        assert getattr(conversation_list, "max_scroll_y", 0) == 0
        assert left_rail_body.max_scroll_y > 0
        left_rail_body.scroll_end(animate=False)
        await pilot.pause(0.1)

        assert left_rail_body.scroll_y > 0
        assert handoff_label.region.y >= details_tray.region.y
        assert handoff_label.region.y + handoff_label.region.height <= details_bottom
        assert handoff_label.region.y + handoff_label.region.height <= composer.region.y


@pytest.mark.asyncio
async def test_console_workspace_sync_while_scrolled_keeps_scroll_range_stable() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    for index in range(40):
        service.link_membership(
            active_workspace.workspace_id,
            item_type="conversation",
            item_id=f"stable-scroll-chat-{index}",
            role="workspace-thread",
            title=f"Stable Scroll Chat {index:02d}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 34)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversations")

        workspace_context = console.query_one(
            "#console-workspace-context",
            ConsoleWorkspaceContextTray,
        )
        left_rail_body = console.query_one("#console-left-rail-body")
        assert workspace_context._nearest_scroll_parent() is left_rail_body
        assert left_rail_body.max_scroll_y > 0

        scroll_y = max(1, min(8, left_rail_body.max_scroll_y - 1))
        initial_max_scroll_y = left_rail_body.max_scroll_y
        initial_height = str(workspace_context.styles.height)
        left_rail_body.scroll_to(y=scroll_y, animate=False)
        await pilot.pause(0.1)
        assert left_rail_body.scroll_y == scroll_y, (
            left_rail_body.scroll_y,
            left_rail_body.max_scroll_y,
            str(workspace_context.styles.height),
            workspace_context.region.height,
            workspace_context.virtual_region.height,
        )

        workspace_context.sync_state(workspace_context.state)
        await _wait_for_condition(
            pilot,
            lambda: (
                left_rail_body.scroll_y == scroll_y
                and str(workspace_context.styles.height) == initial_height
            ),
        )

        assert left_rail_body.scroll_y == scroll_y, (
            left_rail_body.scroll_y,
            left_rail_body.max_scroll_y,
            str(workspace_context.styles.height),
            workspace_context.region.height,
            workspace_context.virtual_region.height,
        )
        assert left_rail_body.max_scroll_y == initial_max_scroll_y
        assert str(workspace_context.styles.height) == initial_height


@pytest.mark.asyncio
async def test_console_workspace_browser_group_collapse_persists_locally() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    default_workspace = service.get_active_workspace()
    service.create_workspace(workspace_id="ws-collapse-b", name="Collapse B")
    service.link_membership(
        default_workspace.workspace_id,
        item_type="conversation",
        item_id="collapse-chat-a",
        role="workspace-thread",
        title="Collapse Chat A",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        assert (
            console.query_one(
                "#console-workspace-conversation-search",
                Input,
            ).disabled
            is False
        )
        assert len(console.query("#console-workspace-conversations-toggle")) == 0
        assert any("Collapse Chat A" in text for text in _conversation_row_texts(console))

        _browser_group_toggle(console, "section:chats").press()
        await pilot.pause(0.1)
        assert all("Collapse Chat A" not in text for text in _conversation_row_texts(console))
        collapsed_groups = app.app_config["console"]["conversation_browser"][
            "collapsed_groups"
        ]
        assert collapsed_groups["section:chats"] is True

        service.set_active_workspace("ws-collapse-b")
        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversations")) == 1
        assert all("Collapse Chat A" not in text for text in _conversation_row_texts(console))

        service.set_active_workspace(default_workspace.workspace_id)
        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversations")) == 1
        assert all("Collapse Chat A" not in text for text in _conversation_row_texts(console))


async def _wait_for_workspace_switcher_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if (
            host.screen_stack
            and host.screen_stack[-1].query("#console-workspace-switcher-modal")
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
        == "File tools: 1 ready, 1 missing"
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
        workspace_context = console.query_one("#console-workspace-context")
        # Session (workspace context) now precedes Context (staged sources) in
        # the four-section left rail, so workspace context renders above staged.
        assert workspace_context.region.y < staged_context.region.y
        assert staged_context.region.x == workspace_context.region.x
        assert staged_context.region.x >= left_rail.region.x
        assert (
            staged_context.region.x + staged_context.region.width
            <= left_rail.region.x + left_rail.region.width
        )
        assert staged_context.region.width == workspace_context.region.width
        assert workspace_context.region.height > staged_context.region.height
        conversations_title = console.query_one("#console-workspace-conversations-title")
        assert conversations_title.region.y > workspace_context.region.y
        assert len(console.query("#console-workspace-recovery")) == 0
        assert len(console.query("#console-change-workspace")) == 0
        new_conversation = console.query_one("#console-new-workspace-conversation", Button)
        assert new_conversation.disabled is False
        text = _visible_text(console)
        assert "Staged Context" in text
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
async def test_console_workspace_context_exposes_new_conversation_for_default_workspace() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        new_conversation = console.query_one("#console-new-workspace-conversation", Button)
        assert new_conversation.disabled is False

        text = _visible_text(console)
        assert "New conversation" in text
        _assert_status_row(
            console,
            label_selector="#console-workspace-runtime-label",
            value_selector="#console-workspace-runtime-value",
            label="File tools",
            value_contains="Off in Default workspace",
        )
        _assert_status_row(
            console,
            label_selector="#console-workspace-server-readiness-label",
            value_selector="#console-workspace-server-readiness-value",
            label="Server handoff",
            value_contains="Not configured",
        )
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
        rendered_label = str(active_workspace.renderable)
        border = active_workspace.styles.border

        assert rendered_label == "Default"
        assert active_workspace.region.height == 1
        assert border.top[0] in {"", "none"}
        assert border.right[0] in {"", "none"}
        assert border.bottom[0] in {"", "none"}
        assert border.left[0] in {"", "none"}


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
async def test_console_workspace_conversation_list_expands_for_multiple_rows() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Research Sprint")
    service.set_active_workspace("ws-a")
    for index in range(3):
        service.link_membership(
            "ws-a",
            item_type="conversation",
            item_id=f"conv-{index}",
            role="workspace-thread",
            title=f"Planning thread {index + 1}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        conversation_list = console.query_one("#console-workspace-conversations")
        rows = list(console.query(".console-workspace-conversation-row"))

        assert len(rows) >= 3
        assert conversation_list.region.height >= len(rows)


@pytest.mark.asyncio
async def test_console_workspace_context_renders_server_readiness_handoff_and_acp_contracts() -> None:
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
            label="Server handoff",
            value_contains="Unavailable",
        )
        assert "No tldw_server workspace API configured." in text
        _assert_status_row(
            console,
            label_selector="#console-workspace-runtime-label",
            value_selector="#console-workspace-runtime-value",
            label="File tools",
            value_contains="0 ready, 1 missing",
        )
        assert "Handoff" in text
        assert "Source note - copy" in text
        assert "Conversation package - metadata-only" in text
        _assert_status_row(
            console,
            label_selector="#console-workspace-handoff-label",
            value_selector="#console-workspace-handoff-value",
            label="Handoff",
            value_contains="ACP handoff: Not configured",
        )
        assert "Audit: visible only; no package was sent." in text


@pytest.mark.asyncio
async def test_console_workspace_context_syncs_active_conversation_marker() -> None:
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
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        console.sync_shell_bar_from_session_data(
            ChatSessionData(tab_id="tab-1", conversation_id="conv-1")
        )
        await pilot.pause()

        assert "> Planning thread" in _visible_text(console)


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
async def test_console_change_workspace_switches_active_context_and_conversation_rows() -> None:
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
    css = Path("tldw_chatbook/css/components/_agentic_terminal.tcss").read_text()

    assert "#console-workspace-conversations-header {" in css
    assert ".console-workspace-action.console-workspace-conversations-toggle {" in css
    assert "#console-workspace-selected-conversation {" in css
    assert "#console-workspace-conversation-search-row {" in css
    context_selector = "#console-workspace-context {"
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
