"""Final matrix viewport changes preserve painted state and action identity."""

from unittest.mock import Mock

import pytest
from textual.widgets import DataTable, Input

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_compact_readability import _cell_paint, _compact, _open, _settle
from Tests.UI.test_mcp_permissions_mode import PermissionsModeApp, _tool_row
from Tests.UI.test_tool_profile_review_lifetime import _wait
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import (
    MCPPermissionsMode,
    PermissionProfileContext,
    _row_key,
    _tool_column_text,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


def _rows(count, tags=False):
    return [
        _tool_row(
            server_key="local:docs",
            server_label="Documentation server",
            tool_name=f"工具_read_a_very_long_selected_resource_name_{index:02}",
            state_label="Allow •",
            tags_label="filesystem, read-only" if tags else "—",
        )
        for index in range(count)
    ]


def _readable(table, row):
    assert table.scroll_x == 0
    assert table.coordinate_to_cell_key((table.cursor_row, 0))[0].value == _row_key(row)
    assert _compact(_tool_column_text(row)) in _compact(
        _cell_paint(table, table.cursor_row, 0)
    )
    assert row.state_label in _cell_paint(table, table.cursor_row, 1)


@private_profile_test
async def test_real_catalog_selected_row_stays_visible_after_resize(request):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    async with app.run_test(size=(80, 24)) as pilot:
        await _wait(pilot, lambda: getattr(app, "_initial_screen_pushed", False))
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
        canvas = workbench.query_one(MCPPermissionsMode)
        table = canvas.query_one(DataTable)
        await _wait(pilot, lambda: table.row_count > 0)
        target = max(
            (row for row in canvas._all_rows if row.kind == "tool"),
            key=lambda row: len(_tool_column_text(row)),
        )
        table.focus()
        await pilot.press("ctrl+home")
        await pilot.press(*(["down"] * table.get_row_index(_row_key(target))))
        await _settle(pilot)
        _readable(table, target)
        before = app.unified_mcp_service.permission_store.load()
        for size in [(170, 48), (80, 24), (170, 48), (80, 24)]:
            await pilot.resize_terminal(*size)
            await _settle(pilot)
            assert app.focused is table
            _readable(table, target)
        assert app.unified_mcp_service.permission_store.load() == before


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_production_permission_reflow_converges_through_repeated_resizes(
    request, tmp_path, theme
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        canvas = workbench.query_one(MCPPermissionsMode)
        table = canvas.query_one(DataTable)
        target = max(canvas._all_rows, key=lambda row: len(_tool_column_text(row)))
        table.focus()
        table.move_cursor(
            row=next(
                i
                for i, row in enumerate(canvas._visible_rows)
                if _row_key(row) == _row_key(target)
            )
        )
        await _settle(pilot)
        before = app.unified_mcp_service.permission_store.load()
        for _ in range(3):
            for size in [(80, 24), (100, 30), (120, 40), (170, 48), (80, 24)]:
                await pilot.resize_terminal(*size)
                await _settle(pilot)
                table.scroll_to(x=0, animate=False)
                table._scroll_cursor_into_view(animate=False)
                await _settle(pilot)
                assert app.focused is table
                _readable(table, target)
        assert app.unified_mcp_service.permission_store.load() == before


@pytest.mark.parametrize("count", [1, 60])
@private_profile_test
async def test_child_viewport_change_reflows_without_outer_resize(
    request, monkeypatch, count
):
    app = PermissionsModeApp()
    rows = _rows(count)
    async with app.run_test(size=(42, 60)) as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        table = canvas.query_one(DataTable)
        table.focus()
        await _settle(pilot)
        _readable(table, rows[0])
        outer_size = canvas.size
        outer_virtual_size = canvas.virtual_size
        first_width = table.content_region.width
        render = Mock(wraps=canvas._render_rows)
        monkeypatch.setattr(canvas, "_render_rows", render)
        app.events.clear()
        # Isolate a child-only width change; the outer canvas keeps its size.
        table.styles.width = table.size.width - 3
        await _settle(pilot)
        assert canvas.size == outer_size
        assert canvas.virtual_size == outer_virtual_size
        assert table.content_region.width < first_width
        assert app.focused is table
        _readable(table, rows[0])
        assert not app.events
        settled_rebuilds = render.call_count
        assert settled_rebuilds == 1
        await _settle(pilot)
        assert render.call_count == settled_rebuilds


@pytest.mark.parametrize("tags", [False, True])
@private_profile_test
async def test_matrix_count_resize_and_filter_keep_exact_visible_action(request, tags):
    app = PermissionsModeApp()
    rows = _rows(60, tags)
    context = PermissionProfileContext("reviewed", 7, "a" * 64, 2)
    async with app.run_test(size=(90, 24)) as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        table = canvas.query_one(DataTable)
        for count in [1, 5, 60, 5, 1]:
            await canvas.update_matrix(
                rows[:count], kill_switch=False, preview="", profile_context=context
            )
            table.focus()
            table.move_cursor(row=count - 1)
            await _settle(pilot)
            app.events.clear()
            for width in [34, 39, 42, 90]:
                await pilot.resize_terminal(width, 24)
                await _settle(pilot)
                assert app.focused is table
                _readable(table, rows[count - 1])
            assert not app.events
        await canvas.update_matrix(
            rows, kill_switch=False, preview="", profile_context=context
        )
        field = canvas.query_one(Input)
        field.focus()
        field.value = "resource_name_5"
        await _settle(pilot)
        table.move_cursor(row=9)
        await pilot.resize_terminal(34, 24)
        await _settle(pilot)
        assert app.focused is field and field.value == "resource_name_5"
        assert table.row_count == 10
        table.focus()
        await _settle(pilot)
        _readable(table, rows[59])
        app.events.clear()
        await pilot.press("enter", "space")
        await _settle(pilot)
        assert len(app.events) == 2
        selected, cycle = app.events
        assert isinstance(selected, MCPPermissionsMode.RowSelected)
        assert isinstance(cycle, MCPPermissionsMode.StateCycleRequested)
        assert selected.tool_name == cycle.tool_name == rows[59].tool_name
        assert selected.profile_context == cycle.profile_context == context


@private_profile_test
async def test_child_height_change_reveals_selected_row_without_rebuilding(
    request, monkeypatch
):
    app = PermissionsModeApp()
    rows = [
        _tool_row(
            server_key="local:docs",
            server_label="Docs",
            tool_name=f"read_resource_{i:02}",
        )
        for i in range(60)
    ]
    async with app.run_test(size=(90, 60)) as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        table = canvas.query_one(DataTable)
        table.styles.height = 20
        table.focus()
        table.move_cursor(row=59)
        await _settle(pilot)
        _readable(table, rows[-1])
        geometry = (canvas.size, canvas.virtual_size)
        render = Mock(wraps=canvas._render_rows)
        monkeypatch.setattr(canvas, "_render_rows", render)
        app.events.clear()
        table.styles.height = 10
        await _settle(pilot)
        assert (canvas.size, canvas.virtual_size) == geometry
        assert app.focused is table
        _readable(table, rows[-1])
        render.assert_not_called()
        assert not app.events


@private_profile_test
async def test_reflow_does_not_swallow_new_enter_on_retained_permission_row(request):
    app = PermissionsModeApp()
    rows = _rows(5)
    context = PermissionProfileContext("reviewed", 7, "a" * 64, 2)
    fresh = PermissionProfileContext("reviewed", 8, "b" * 64, 2)
    async with app.run_test(size=(90, 24)) as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(
            rows, kill_switch=False, preview="", profile_context=context
        )
        table = canvas.query_one(DataTable)
        table.focus()
        await _settle(pilot)
        await pilot.press("ctrl+end")
        await _settle(pilot)
        assert app.events[-1].tool_name == rows[-1].tool_name
        await canvas.update_matrix(
            rows, kill_switch=False, preview="", profile_context=fresh
        )
        await pilot.resize_terminal(34, 24)
        await _settle(pilot)
        _readable(table, rows[-1])
        app.events.clear()
        await pilot.press("enter")
        await _settle(pilot)
        assert len(app.events) == 1
        assert app.events[0].tool_name == rows[-1].tool_name
        assert app.events[0].profile_context == fresh
