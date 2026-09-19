"""Tool identity and permission state stay visible without losing selection."""

from unittest.mock import AsyncMock

import pytest
from textual.widgets import DataTable, Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_compact_readability import (
    _cell_paint,
    _compact,
    _open,
    _settle,
)
from Tests.UI.test_mcp_tools_mode import ToolsModeBundledCSSApp, _tool
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import format_tool_state_label
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode


def _key(table):
    return table.coordinate_to_cell_key((table.cursor_row, 0))[0].value


def _catalog(tags):
    return [
        _tool(
            server_key="local:docs",
            server_label="Documentation server",
            name=f"工具_read_a_very_long_selected_resource_name_{index:02}",
            tags=("read-only",) if tags else (),
        )
        for index in range(60)
    ]


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_production_tools_keep_tool_and_state_visible_without_reload(
    request, tmp_path, theme, monkeypatch
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        workbench.set_mode("tools")
        await _settle(pilot)
        canvas = workbench.query_one(MCPToolsMode)
        tools = _catalog(True)
        state = EffectiveToolState(
            state="ask", origin="tool_override", config_changed=True
        )
        await canvas.update_tools(
            tools, states={(tool.server_key, tool.name): state for tool in tools}
        )
        table = canvas.query_one(DataTable)
        table.focus()
        table.move_cursor(row=59)
        await _settle(pilot)
        target = tools[-1]
        before = app.unified_mcp_service.permission_store.load()
        reload_observer = AsyncMock(wraps=workbench.reload)
        monkeypatch.setattr(workbench, "reload", reload_observer)
        for size in [(170, 48), (80, 24), (100, 30), (120, 40), (170, 48), (80, 24)]:
            await pilot.resize_terminal(*size)
            await _settle(pilot)
            assert app.focused is table
            assert _key(table) == target.tool_id
            # No navigation/reveal here: resize itself must retain a visible row.
            assert table.scroll_x == 0
            assert _compact(target.name) in _compact(
                _cell_paint(table, table.cursor_row, 0)
            )
            assert format_tool_state_label(state) in _cell_paint(
                table, table.cursor_row, 1
            )
        assert app.unified_mcp_service.permission_store.load() == before
        reload_observer.assert_not_called()


@pytest.mark.parametrize("tags", [False, True])
@private_profile_test
async def test_unicode_filter_refresh_and_reflow_preserve_exact_tool(request, tags):
    app = ToolsModeBundledCSSApp()
    tools = _catalog(tags)
    target = tools[-1]
    state = EffectiveToolState(state="ask", origin="server_default", risk_floored=True)
    states = {(tool.server_key, tool.name): state for tool in tools}
    async with app.run_test(size=(42, 24)) as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(tools, states=states)
        table = canvas.query_one(DataTable)
        table.focus()
        table.move_cursor(row=59)
        await _settle(pilot)
        for width in [42, 90, 34, 42]:
            await pilot.resize_terminal(width, 24)
            await _settle(pilot)
            assert app.focused is table
            assert _key(table) == target.tool_id
            assert _compact(target.name) in _compact(
                _cell_paint(table, table.cursor_row, 0)
            )
            assert format_tool_state_label(state) in _cell_paint(
                table, table.cursor_row, 1
            )
        await pilot.press("end")
        await _settle(pilot)
        assert "raw" in _cell_paint(table, table.cursor_row, 4 if tags else 3)
        await pilot.press("home")
        field = canvas.query_one("#mcp-tools-filter-text", Input)
        field.focus()
        field.value = "resource_name_5"
        await _settle(pilot)
        assert _key(table) == target.tool_id
        assert table.row_count == 10
        server_select = canvas.query_one("#mcp-tools-filter-server", Select)
        server_select.value = target.server_key
        await _settle(pilot)
        canvas.update_states({})
        await _settle(pilot)
        assert _key(table) == target.tool_id
        assert table.get_row_at(table.cursor_row)[1].plain == "—"
        await pilot.resize_terminal(34, 24)
        await _settle(pilot)
        assert app.focused is field
        assert field.value == "resource_name_5"
        assert canvas.query_one("#mcp-tools-filter-server", Select) is server_select
        assert server_select.value == target.server_key
        # Reordering, an inserted earlier tool and duplicate IDs cannot retarget
        # the cursor by a stale list index.
        await canvas.update_tools([tools[0], tools[0], *reversed(tools)], states=states)
        await _settle(pilot)
        assert _key(table) == target.tool_id
        assert app.focused is field
        table.focus()
        await _settle(pilot)
        app.events.clear()
        await pilot.press("enter")
        await _settle(pilot)
        assert [event.tool_id for event in app.events] == [target.tool_id]
        await canvas.update_tools(tools[:-1], states=states)
        await _settle(pilot)
        assert _key(table) == tools[50].tool_id
        field.value = "no matching tool"
        await _settle(pilot)
        assert table.row_count == 0
        field.value = ""
        await _settle(pilot)
        assert _key(table) == tools[0].tool_id


@private_profile_test
async def test_catalog_size_and_scrollbar_changes_keep_complete_allow_state(request):
    app = ToolsModeBundledCSSApp()
    tools = _catalog(False)
    state = EffectiveToolState(state="allow", origin="tool_override")
    states = {(tool.server_key, tool.name): state for tool in tools}
    async with app.run_test(size=(90, 24)) as pilot:
        canvas = app.query_one(MCPToolsMode)
        table = canvas.query_one(DataTable)
        for count in [1, 5, 60, 5, 1]:
            await canvas.update_tools(tools[:count], states=states)
            table.focus()
            table.move_cursor(row=count - 1)
            await _settle(pilot)
            for width in [34, 39, 42, 90]:
                await pilot.resize_terminal(width, 24)
                await _settle(pilot)
                assert _key(table) == tools[count - 1].tool_id
                assert _compact(tools[count - 1].name) in _compact(
                    _cell_paint(table, table.cursor_row, 0)
                )
                assert "Allow •" in _cell_paint(table, table.cursor_row, 1)
