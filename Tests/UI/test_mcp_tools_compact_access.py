"""MCP Tools controls and catalog rows remain reachable in real pane sizes."""

from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Button, DataTable, Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_compact_readability import _compact, _open, _paint, _settle
from Tests.UI.test_mcp_permissions_handoff import _cursor_visible, _visible
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_complete_local_switch_label_survives_compact_pane(
    request, tmp_path, theme
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        workbench.set_mode("tools")
        canvas = workbench.query_one(MCPToolsMode)
        for size in [(80, 24), (100, 30), (120, 40), (170, 48), (80, 24)]:
            await pilot.resize_terminal(*size)
            for enabled in (False, True):
                canvas.update_local_config(
                    enabled=enabled, workspace_root="", visible=True
                )
                button = canvas.query_one("#mcp-tools-local-enabled", Button)
                button.focus()
                await _settle(pilot)
                assert app.focused is button
                _visible(app, button)
                assert _compact(str(button.label)) in _compact(
                    _paint(app.screen, button.content_region)
                )


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_tools_filters_and_first_last_rows_visible_through_resize(
    request, tmp_path, monkeypatch, theme
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        workbench.set_mode("tools")
        await _settle(pilot)
        canvas = workbench.query_one(MCPToolsMode)
        table = canvas.query_one("#mcp-tools-table", DataTable)
        before = app.unified_mcp_service.permission_store.load()
        save = Mock(side_effect=AssertionError("reading must not save config"))
        monkeypatch.setattr(
            "tldw_chatbook.UI.MCP_Modules.mcp_workbench.save_setting_to_cli_config",
            save,
        )
        reload = AsyncMock(wraps=workbench.reload)
        monkeypatch.setattr(workbench, "reload", reload)
        for size in [(80, 24), (100, 30), (120, 40), (170, 48), (80, 24)]:
            await pilot.resize_terminal(*size)
            table.focus()
            await _settle(pilot)
            assert app.focused is table
            _visible(app, table)
            await pilot.press("ctrl+home")
            await _settle(pilot)
            assert table.cursor_row == 0
            _cursor_visible(app, table)
            await pilot.press("ctrl+end")
            await _settle(pilot)
            assert table.cursor_row == table.row_count - 1
            _cursor_visible(app, table)

            text = canvas.query_one("#mcp-tools-filter-text", Input)
            text.focus()
            await _settle(pilot)
            _visible(app, text)
            assert text.content_region.width >= len("Filter tools…")
            text.value = "list_notes"
            await _settle(pilot)
            assert app.focused is text
            assert table.row_count == 1
            table.focus()
            await _settle(pilot)
            _visible(app, table)
            _cursor_visible(app, table)
            assert "list_notes" in _paint(app.screen, table.content_region)
            text.focus()
            text.value = ""
            await _settle(pilot)

            server = canvas.query_one("#mcp-tools-filter-server", Select)
            server.focus()
            await _settle(pilot)
            _visible(app, server)
            assert "All servers" in _paint(app.screen, server.content_region)
            await pilot.press("enter", "end", "enter")
            await _settle(pilot)
            assert canvas._filter_server_key is not None
            assert table.row_count > 0
            await pilot.press("enter", "home", "enter")
            await _settle(pilot)
            assert canvas._filter_server_key is None

        # A queued reveal must follow the latest focus, including other modes.
        canvas.call_after_refresh(canvas.reveal_focused_control)
        rail = workbench.query_one("#mcp-rail-source", Select)
        rail.focus()
        await _settle(pilot)
        assert app.focused is rail
        scroll = canvas.scroll_offset
        canvas.call_after_refresh(canvas.reveal_focused_control)
        workbench.set_mode("servers")
        await _settle(pilot)
        assert app.focused is rail
        assert canvas.scroll_offset == scroll
        assert app.unified_mcp_service.permission_store.load() == before
        save.assert_not_called()
        reload.assert_not_called()


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_overflowing_catalog_retains_visible_cursor_on_resize(
    request, tmp_path, theme
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        workbench.set_mode("tools")
        canvas = workbench.query_one(MCPToolsMode)
        tools = [replace(canvas._tools[0], name=f"tool_{i:02}") for i in range(60)]
        await canvas.update_tools(tools)
        table = canvas.query_one("#mcp-tools-table", DataTable)
        table.focus()
        await _settle(pilot)
        await pilot.press("ctrl+end")
        await _settle(pilot)
        assert table.cursor_row == 59
        key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        for size in [(120, 40), (100, 30), (80, 24), (170, 48)]:
            await pilot.resize_terminal(*size)
            await _settle(pilot)
            assert app.focused is table
            assert table.coordinate_to_cell_key((table.cursor_row, 0))[0] == key
            _visible(app, table)
            _cursor_visible(app, table)
