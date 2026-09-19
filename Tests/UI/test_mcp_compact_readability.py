"""Compact MCP users can read source, tool identity and permission state."""

from unittest.mock import AsyncMock

import pytest
from textual.coordinate import Coordinate
from textual.geometry import Offset
from textual.widgets import DataTable, Select

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_workbench import PermissionsHubService
from Tests.UI.test_tool_profile_review_lifetime import _wait
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import (
    MCPPermissionsMode,
    _row_key,
    _tool_column_text,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


async def _open(pilot, store_path):
    app = pilot.app
    await _wait(pilot, lambda: getattr(app, "_initial_screen_pushed", False))
    app.unified_mcp_service = PermissionsHubService(store_path)
    await app.handle_screen_navigation(NavigateToScreen("mcp"))
    await _wait(pilot, lambda: getattr(app.screen, "screen_name", None) == "mcp")
    workbench = app.screen.workbench
    await _wait(
        pilot,
        lambda: (
            not workbench.is_loading
            and not workbench._reloading
            and bool(workbench.query("#mcp-perm-table"))
        ),
    )
    workbench.set_mode("permissions")
    await _settle(pilot)
    return workbench


async def _settle(pilot):
    await pilot.pause()
    await pilot.wait_for_scheduled_animations()
    await pilot.pause()


def _paint(screen, region):
    return "\n".join(
        strip.crop(region.x, region.right).text
        for strip in screen._compositor.render_strips()[region.y : region.bottom]
    )


def _compact(text):
    return "".join(text.split())


def _cell_paint(table, row_index, column_index):
    cell = table._get_cell_region(Coordinate(row_index, column_index))
    region = cell.translate(
        Offset(
            table.content_region.x - int(table.scroll_x),
            table.content_region.y - int(table.scroll_y),
        )
    )
    clip = table.screen._compositor.visible_widgets[table][1]
    assert region.intersection(clip) == region
    assert region.intersection(table.scrollable_content_region) == region
    return _paint(table.screen, region)


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_purpose_and_both_source_labels_are_fully_painted(
    request, tmp_path, theme
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        for size in [(80, 24), (100, 30), (120, 40), (170, 48), (80, 24)]:
            await pilot.resize_terminal(*size)
            await _settle(pilot)
            purpose = app.screen.query_one("#mcp-purpose")
            assert _compact(
                "most people never need to change anything here."
            ) in _compact(_paint(app.screen, purpose.content_region))
            for value, label in [("server", "Server"), ("local", "Local")]:
                source = app.screen.query_one("#mcp-rail-source", Select)
                source.focus()
                await _settle(pilot)
                await pilot.press(
                    "enter", "end" if value == "server" else "home", "enter"
                )
                await _wait(
                    pilot,
                    lambda value=value: (
                        workbench._source == value and not workbench._reloading
                    ),
                )
                await _settle(pilot)
                source = app.screen.query_one("#mcp-rail-source", Select)
                current = source.query_one("SelectCurrent")
                text = current.query_one("#label")
                assert text.content_region.height == 1
                assert label in _paint(app.screen, text.content_region)


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_tool_and_state_stay_readable_through_resize_without_mutation(
    request, tmp_path, theme, monkeypatch
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        canvas = workbench.query_one(MCPPermissionsMode)
        table = canvas.query_one("#mcp-perm-table", DataTable)
        before = app.unified_mcp_service.permission_store.load()
        reload_observer = AsyncMock(wraps=workbench.reload)
        monkeypatch.setattr(workbench, "reload", reload_observer)
        target = max(canvas._all_rows, key=lambda row: len(_tool_column_text(row)))
        key = _row_key(target)
        table.focus()
        table.move_cursor(
            row=next(
                i for i, row in enumerate(canvas._visible_rows) if _row_key(row) == key
            )
        )
        await _settle(pilot)
        for size in [(80, 24), (100, 30), (120, 40), (170, 48), (80, 24)]:
            await pilot.resize_terminal(*size)
            await _settle(pilot)
            assert app.focused is table
            row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
            assert row_key.value == key
            table.scroll_to(x=0, animate=False)
            table._scroll_cursor_into_view(animate=False)
            await _settle(pilot)
            assert _compact(_tool_column_text(target)) in _compact(
                _cell_paint(table, table.cursor_row, 0)
            )
            assert target.state_label in _cell_paint(table, table.cursor_row, 1)
        assert app.unified_mcp_service.permission_store.load() == before
        reload_observer.assert_not_called()


@pytest.mark.parametrize("tags", [False, True])
@private_profile_test
async def test_wrapped_unicode_rows_keep_filter_and_exact_action_context(request, tags):
    from Tests.UI.test_mcp_permissions_mode import (
        PermissionsModeApp,
        _global_row,
        _server_row,
        _tool_row,
    )
    from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import (
        PermissionProfileContext,
    )

    app = PermissionsModeApp()
    name = "工具_read_a_very_long_selected_resource_name"
    rows = [
        _global_row(),
        _server_row(server_key="local:docs", server_label="Documentation server"),
        *[
            _tool_row(
                server_key="local:docs",
                server_label="Documentation server",
                tool_name=name + str(index),
                state_label="Ask ⚑",
                tags_label="filesystem, read-only" if tags else "—",
            )
            for index in range(12)
        ],
    ]
    context = PermissionProfileContext("reviewed-profile", 7, "a" * 64, 2)
    async with app.run_test(size=(42, 24)) as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(
            rows, kill_switch=False, preview="", profile_context=context
        )
        table = canvas.query_one("#mcp-perm-table", DataTable)
        table.focus()
        await _settle(pilot)
        table.move_cursor(row=8)
        await _settle(pilot)
        target = rows[8]
        for width in [42, 90, 34, 42]:
            await pilot.resize_terminal(width, 24)
            await _settle(pilot)
            assert app.focused is table
            assert table.coordinate_to_cell_key((table.cursor_row, 0))[
                0
            ].value == _row_key(target)
            table.scroll_to(x=0, animate=False)
            table._scroll_cursor_into_view(animate=False)
            await _settle(pilot)
            assert _compact(_tool_column_text(target)) in _compact(
                _cell_paint(table, table.cursor_row, 0)
            )
            assert target.state_label in _cell_paint(table, table.cursor_row, 1)
        if tags:
            await pilot.press("end")
            await _settle(pilot)
            assert target.tags_label in _cell_paint(table, table.cursor_row, 2)
            await pilot.press("home")
            await _settle(pilot)
        field = canvas.query_one("#mcp-perm-filter-text")
        field.focus()
        field.value = target.tool_name
        await _settle(pilot)
        await pilot.resize_terminal(34, 24)
        await _settle(pilot)
        assert app.focused is field
        assert field.value == target.tool_name
        assert table.coordinate_to_cell_key((table.cursor_row, 0))[0].value == _row_key(
            target
        )
        assert all(
            isinstance(event, MCPPermissionsMode.RowSelected) for event in app.events
        )
        table.focus()
        await _settle(pilot)
        await pilot.press("space")
        await _settle(pilot)
        events = [
            e
            for e in app.events
            if isinstance(e, MCPPermissionsMode.StateCycleRequested)
        ]
        assert len(events) == 1
        assert events[0].server_key == target.server_key
        assert events[0].tool_name == target.tool_name
        assert events[0].profile_context == context


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_tools_empty_recovery_action_is_visible_when_focused(
    request, tmp_path, theme
):
    from textual.widgets import Button

    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        workbench.set_mode("tools")
        await _settle(pilot)
        canvas = workbench.query_one("#mcp-mode-canvas-tools")
        message = (
            "No servers configured — add one to see its tools. 9 tool gate(s) "
            "are off. Configure the workspace, web, and Watchlists master switch "
            "in Tools mode; other registration gates are in Servers mode."
        )
        await canvas.update_tools([], empty_diagnosis=(message, "add_server"))
        await _settle(pilot)
        button = canvas.query_one("#mcp-tools-empty-action", Button)
        button.focus()
        await _settle(pilot)
        assert app.focused is button
        painted = app.screen._compositor.visible_widgets.get(button)
        assert painted is not None
        region, clip = painted
        assert region.intersection(clip) == region
        assert "Add server" in _paint(app.screen, region)
