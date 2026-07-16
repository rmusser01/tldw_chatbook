# Tests/UI/test_mcp_permissions_mode.py
from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox, DataTable, Static

import tldw_chatbook
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode, PermRow

_CSS_ROOT = Path(tldw_chatbook.__file__).parent / "css"
_BUNDLED_STYLESHEET = _CSS_ROOT / "tldw_cli_modular.tcss"


def _row(
    *,
    kind: str,
    server_key: str = "",
    server_label: str = "",
    tool_name: str | None = None,
    state_label: str = "Ask",
    tags_label: str = "—",
    cycle_current: str | None = None,
) -> PermRow:
    return PermRow(
        kind=kind,
        server_key=server_key,
        server_label=server_label,
        tool_name=tool_name,
        state_label=state_label,
        tags_label=tags_label,
        cycle_current=cycle_current,
    )


def _global_row(*, state_label: str = "Ask", cycle_current: str | None = "ask") -> PermRow:
    return _row(kind="global", state_label=state_label, cycle_current=cycle_current)


def _server_row(
    *,
    server_key: str,
    server_label: str,
    state_label: str = "Ask",
    cycle_current: str | None = None,
) -> PermRow:
    return _row(
        kind="server",
        server_key=server_key,
        server_label=server_label,
        state_label=state_label,
        cycle_current=cycle_current,
    )


def _tool_row(
    *,
    server_key: str,
    server_label: str,
    tool_name: str,
    state_label: str = "Ask",
    tags_label: str = "—",
    cycle_current: str | None = None,
) -> PermRow:
    return _row(
        kind="tool",
        server_key=server_key,
        server_label=server_label,
        tool_name=tool_name,
        state_label=state_label,
        tags_label=tags_label,
        cycle_current=cycle_current,
    )


class PermissionsModeApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPPermissionsMode(id="mcp-mode-canvas-permissions")

    def on_mcp_permissions_mode_state_cycle_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_permissions_mode_kill_switch_toggled(self, event) -> None:
        self.events.append(event)

    def on_mcp_permissions_mode_row_selected(self, event) -> None:
        self.events.append(event)


def _row_texts(table: DataTable, row_index: int) -> list[str]:
    row = table.get_row_at(row_index)
    return [cell.plain if hasattr(cell, "plain") else str(cell) for cell in row]


# -- rendering ----------------------------------------------------------


@pytest.mark.asyncio
async def test_rows_render_in_given_order_with_pinned_row_keys():
    """The widget is render-only: it renders `PermRow`s in the exact order
    given (grouping/sorting is the workbench's job) but the ROW KEYS it
    assigns must follow the spec-verbatim formats: `__global__`,
    `__server__::<server_key>`, and a tool's `tool_id`
    (`<server_key>::<name>`).
    """
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(state_label="Ask", cycle_current="ask"),
            _server_row(server_key="local:docs", server_label="docs", state_label="Ask"),
            _tool_row(
                server_key="local:docs", server_label="docs", tool_name="fetch",
                state_label="Ask", tags_label="—",
            ),
            _tool_row(
                server_key="local:docs", server_label="docs", tool_name="search",
                state_label="Allow •", tags_label="network",
            ),
            _server_row(server_key="local:notes", server_label="notes", state_label="Off •"),
            _tool_row(
                server_key="local:notes", server_label="notes", tool_name="list_notes",
                state_label="Off •",
            ),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="Global default: Ask.")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count == 6

        assert _row_texts(table, 0) == ["Global default", "Ask", "—"]
        assert _row_texts(table, 1) == ["Server default — docs", "Ask", "—"]
        assert _row_texts(table, 2) == ["fetch", "Ask", "—"]
        assert _row_texts(table, 3) == ["search", "Allow •", "network"]
        assert _row_texts(table, 4) == ["Server default — notes", "Off •", "—"]
        assert _row_texts(table, 5) == ["list_notes", "Off •", "—"]

        expected_keys = [
            "__global__",
            "__server__::local:docs",
            "local:docs::fetch",
            "local:docs::search",
            "__server__::local:notes",
            "local:notes::list_notes",
        ]
        for index, expected_key in enumerate(expected_keys):
            row_key, _ = table.coordinate_to_cell_key((index, 0))
            assert row_key.value == expected_key

        assert str(app.query_one("#mcp-perm-preview", Static).renderable) == "Global default: Ask."


@pytest.mark.asyncio
async def test_state_label_renders_verbatim_and_markup_safe():
    """`state_label` may embed a bullet/warning/flag marker already baked in
    by the workbench -- the widget must render it literally as plain `Text`,
    not parse it as Rich markup (a server label a user typed could otherwise
    inject styling)."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(
                server_key="local:[bold red]x", server_label="[bold red]x", state_label="Ask ⚠",
            ),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        assert _row_texts(table, 1) == ["Server default — [bold red]x", "Ask ⚠", "—"]


# -- kill switch ----------------------------------------------------------


@pytest.mark.asyncio
async def test_mount_alone_posts_no_kill_switch_event():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.events == []


@pytest.mark.asyncio
async def test_update_matrix_sets_kill_switch_without_posting_mount_echo():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix([_global_row()], kill_switch=True, preview="")
        await pilot.pause()
        checkbox = app.query_one("#mcp-perm-kill-switch", Checkbox)
        assert checkbox.value is True
        assert app.events == []


@pytest.mark.asyncio
async def test_user_toggle_posts_kill_switch_toggled_exactly_once():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix([_global_row()], kill_switch=False, preview="")
        await pilot.pause()
        await pilot.click("#mcp-perm-kill-switch")
        await pilot.pause()
        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPPermissionsMode.KillSwitchToggled)
        assert event.value is True


# -- Space cycling ----------------------------------------------------------


@pytest.mark.asyncio
async def test_space_on_tool_row_posts_next_state_per_cycle_helper():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(server_key="local:docs", server_label="docs"),
            _tool_row(
                server_key="local:docs", server_label="docs", tool_name="search",
                cycle_current=None,
            ),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=2)
        await pilot.press("space")
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPPermissionsMode.StateCycleRequested)
        assert event.row_kind == "tool"
        assert event.server_key == "local:docs"
        assert event.tool_name == "search"
        # cycle_ui_state(None) == "allow"
        assert event.new_state == "allow"


@pytest.mark.asyncio
async def test_space_on_server_row_allows_cycling_back_to_inherit():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(server_key="local:docs", server_label="docs", cycle_current="deny"),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=1)
        await pilot.press("space")
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert event.row_kind == "server"
        assert event.server_key == "local:docs"
        assert event.tool_name is None
        # cycle_ui_state("deny") == None (Inherit)
        assert event.new_state is None


@pytest.mark.asyncio
async def test_space_on_global_row_never_posts_none():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(
            [_global_row(state_label="Off", cycle_current="deny")],
            kill_switch=False, preview="",
        )
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.press("space")
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert event.row_kind == "global"
        assert event.new_state is not None
        # cycle_global("deny") == "allow"
        assert event.new_state == "allow"


@pytest.mark.asyncio
async def test_space_with_no_rows_is_a_noop():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        await pilot.press("space")
        await pilot.pause()
        assert app.events == []


# -- Task 7: row selection ----------------------------------------------


@pytest.mark.asyncio
async def test_enter_on_tool_row_posts_row_selected_with_tool_fields():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(server_key="local:docs", server_label="docs"),
            _tool_row(server_key="local:docs", server_label="docs", tool_name="search"),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=2)
        await pilot.press("enter")
        await pilot.pause()

        events = [e for e in app.events if isinstance(e, MCPPermissionsMode.RowSelected)]
        assert len(events) == 1
        assert events[0].row_kind == "tool"
        assert events[0].server_key == "local:docs"
        assert events[0].tool_name == "search"


@pytest.mark.asyncio
async def test_enter_on_global_row_posts_row_selected_with_no_tool_name():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix([_global_row()], kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.press("enter")
        await pilot.pause()

        events = [e for e in app.events if isinstance(e, MCPPermissionsMode.RowSelected)]
        assert len(events) == 1
        assert events[0].row_kind == "global"
        assert events[0].tool_name is None


@pytest.mark.asyncio
async def test_enter_on_server_row_posts_row_selected_with_no_tool_name():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [_global_row(), _server_row(server_key="local:docs", server_label="docs")]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=1)
        await pilot.press("enter")
        await pilot.pause()

        events = [e for e in app.events if isinstance(e, MCPPermissionsMode.RowSelected)]
        assert len(events) == 1
        assert events[0].row_kind == "server"
        assert events[0].server_key == "local:docs"
        assert events[0].tool_name is None


# -- preview ----------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_text_renders_verbatim():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        preview = "docs-server: 2 allowed, 1 asks, 1 off. Global default: Ask."
        await canvas.update_matrix([_global_row()], kill_switch=False, preview=preview)
        await pilot.pause()
        assert str(app.query_one("#mcp-perm-preview", Static).renderable) == preview


# -- real bundled CSS (Phase 3 lesson: DEFAULT_CSS alone can still collapse
# to 0x0 under the real app stylesheet's global widget rules) ---------------


class PermissionsModeAppWithBundledCSS(App):
    """Mirrors `ToolsModeAppWithBundledCSS` (test_mcp_tools_mode.py) --
    loads the real generated bundle as CSS_PATH so the matrix table and
    kill-switch row contest their actual CSS priority battle exactly as
    they do in the live app."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        yield MCPPermissionsMode(id="mcp-mode-canvas-permissions")


@pytest.mark.asyncio
async def test_matrix_and_kill_switch_have_nonzero_geometry_with_bundled_css():
    app = PermissionsModeAppWithBundledCSS()
    async with app.run_test(size=(120, 40)) as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(server_key="local:docs", server_label="docs"),
            _tool_row(server_key="local:docs", server_label="docs", tool_name="search"),
        ]
        await canvas.update_matrix(rows, kill_switch=True, preview="Global default: Ask.")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.size.width > 0, "matrix table collapsed to zero width under bundled CSS"
        assert table.size.height > 0, "matrix table collapsed to zero height under bundled CSS"

        checkbox = app.query_one("#mcp-perm-kill-switch", Checkbox)
        assert checkbox.size.width > 0, "kill-switch row collapsed to zero width under bundled CSS"
        assert checkbox.size.height > 0, "kill-switch row collapsed to zero height under bundled CSS"
