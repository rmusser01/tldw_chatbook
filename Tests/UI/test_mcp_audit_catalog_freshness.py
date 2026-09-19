"""Audit drilldowns resolve definitions after pending catalog publication."""

import asyncio
from copy import deepcopy

import pytest
from textual.widgets import Button, DataTable, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_compact_readability import _settle
from Tests.UI.test_mcp_workbench import (
    AuditHubService,
    ToolTestApp,
    _audit_record,
    _capture_notifications,
)
from Tests.UI.test_tool_profile_review_lifetime import _wait
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


def _change_catalog(workbench, change):
    record = deepcopy(workbench._catalog_records["docs"])
    record["discovery_snapshot"]["tools"] = (
        []
        if change == "removed"
        else [
            {
                "name": "search",
                "description": "Search the refreshed catalog.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"refreshed_query": {"type": "string"}},
                },
            }
        ]
    )
    record["is_connected"] = change != "disconnected"
    workbench._catalog_records["docs"] = record


async def _navigate(workbench, destination, tool, context):
    if destination == "tools":
        await workbench._open_audit_tool(tool, context)
    else:
        await workbench._open_audit_permission(tool, context)


def _assert_current(inspector, destination, change):
    selected = (
        inspector.current_tool
        if destination == "tools"
        else inspector.current_permission_tool
    )
    assert selected.description == "Search the refreshed catalog."
    assert selected.input_schema["properties"] == {
        "refreshed_query": {"type": "string"}
    }
    assert selected.stale == (change == "disconnected")
    if destination == "tools":
        description = inspector.query_one("#mcp-inspector-tool-description", Static)
        assert str(description.renderable) == "Search the refreshed catalog."
        assert bool(inspector.query("#mcp-inspector-tool-stale")) == (
            change == "disconnected"
        )


@pytest.mark.parametrize("destination", ["tools", "permissions"])
@pytest.mark.parametrize("change", ["replaced", "disconnected", "removed"])
@private_profile_test
async def test_audit_resolves_catalog_published_while_clearing(
    request, monkeypatch, destination, change
):
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        inspector = app.query_one(MCPInspector)
        tool = workbench._tool_for("local:docs", "search")
        context = workbench._tool_policy_profile_context
        notices = _capture_notifications(app)
        original = inspector.show_audit_entry

        async def refresh_during_clear(entry):
            await original(entry)
            _change_catalog(workbench, change)
            await workbench._sync_children()

        monkeypatch.setattr(inspector, "show_audit_entry", refresh_during_clear)
        await _navigate(workbench, destination, tool, context)
        await pilot.pause()
        if change == "removed":
            assert inspector.current_tool is None
            assert inspector.current_permission_tool is None
            assert notices[-1] == (
                "local:docs::search: tool no longer available.",
                "warning",
            )
        else:
            _assert_current(inspector, destination, change)


@pytest.mark.parametrize("destination", ["tools", "permissions"])
@private_profile_test
async def test_audit_waits_for_in_progress_catalog_publication(
    request, monkeypatch, destination
):
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        inspector = app.query_one(MCPInspector)
        old_tool = workbench._tool_for("local:docs", "search")
        context = workbench._tool_policy_profile_context
        published = asyncio.Event()
        release = asyncio.Event()
        original = workbench._sync_tools_mode

        async def pause_publication(tools, states):
            await original(tools, states)
            published.set()
            await release.wait()

        monkeypatch.setattr(workbench, "_sync_tools_mode", pause_publication)
        _change_catalog(workbench, "replaced")
        refresh = asyncio.create_task(workbench._sync_children())
        await asyncio.wait_for(published.wait(), 5)
        navigation = asyncio.create_task(
            _navigate(workbench, destination, old_tool, context)
        )
        try:
            await pilot.pause()
            assert not navigation.done(), (
                "Navigation published before catalog sync finished"
            )
            assert inspector.current_tool is None
            assert inspector.current_permission_tool is None
        finally:
            release.set()
            await asyncio.gather(refresh, navigation)
        _assert_current(inspector, destination, "replaced")


@pytest.mark.parametrize("destination", ["tools", "permissions"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_real_app_audit_action_renders_refreshed_definition(
    request, monkeypatch, destination, theme
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(170, 48)) as pilot:
        await _wait(pilot, lambda: getattr(app, "_initial_screen_pushed", False))
        app.unified_mcp_service = AuditHubService([_audit_record()])
        await app.handle_screen_navigation(NavigateToScreen("mcp"))
        workbench = app.screen.workbench
        await _wait(
            pilot, lambda: not workbench.is_loading and not workbench._reloading
        )
        workbench.set_mode("audit")
        await _settle(pilot)
        workbench.query_one("#mcp-audit-table", DataTable).focus()
        await pilot.press("enter")
        await _settle(pilot)
        inspector = workbench.query_one(MCPInspector)
        original = inspector.show_audit_entry
        replaced = False

        async def refresh_during_clear(entry):
            nonlocal replaced
            await original(entry)
            if entry is None and not replaced:
                replaced = True
                _change_catalog(workbench, "replaced")
                await workbench._sync_children()

        monkeypatch.setattr(inspector, "show_audit_entry", refresh_during_clear)
        button_id = (
            "mcp-audit-open-tool"
            if destination == "tools"
            else "mcp-audit-adjust-permission"
        )
        inspector.query_one("#" + button_id, Button).focus()
        await pilot.press("enter")
        await _wait(
            pilot,
            lambda: (
                (
                    inspector.current_tool
                    if destination == "tools"
                    else inspector.current_permission_tool
                )
                is not None
            ),
        )
        assert workbench.active_mode == destination
        _assert_current(inspector, destination, "replaced")
        table_id = "mcp-tools-table" if destination == "tools" else "mcp-perm-table"
        table = workbench.query_one("#" + table_id, DataTable)
        key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        assert key.value == "local:docs::search"


@pytest.mark.parametrize("destination", ["tools", "permissions"])
@private_profile_test
async def test_catalog_navigation_rejects_profile_change_during_row_selection(
    request, monkeypatch, destination
):
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        inspector = app.query_one(MCPInspector)
        tool = workbench._tool_for("local:docs", "search")
        context = workbench._tool_policy_profile_context

        def change_profile():
            workbench._tool_policy_profile_id = "other"
            workbench._tool_policy_selector_generation += 1
            workbench._tool_policy_profile_context = None

        if destination == "tools":
            canvas = workbench.query_one(MCPToolsMode)
            select = canvas.select_tool_row

            async def select_then_switch(tool_id):
                selected = await select(tool_id)
                change_profile()
                return selected
        else:
            canvas = workbench.query_one(MCPPermissionsMode)
            select = canvas.select_tool_row

            def select_then_switch(server_key, tool_name):
                selected = select(server_key, tool_name)
                change_profile()
                return selected

        monkeypatch.setattr(canvas, "select_tool_row", select_then_switch)
        await _navigate(workbench, destination, tool, context)
        await pilot.pause()
        assert inspector.current_tool is None
        assert inspector.current_permission_tool is None
