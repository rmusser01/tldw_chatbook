"""Audit navigation keeps the displayed identity and checks its destination."""

import pytest
from textual.widgets import Button, DataTable, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_compact_readability import _settle
from Tests.UI.test_mcp_inspector import InspectorApp
from Tests.UI.test_mcp_workbench import (
    AuditHubService,
    ToolTestApp,
    _audit_record,
    _capture_notifications,
)
from Tests.UI.test_tool_profile_review_lifetime import _wait
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import (
    MCPPermissionsMode,
    PermissionProfileContext,
)
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

ACTIONS = [
    ("mcp-audit-open-tool", MCPInspector.AuditOpenToolRequested),
    ("mcp-audit-adjust-permission", MCPInspector.AuditAdjustPermissionRequested),
]


@pytest.mark.parametrize("button_id,event_type", ACTIONS)
@pytest.mark.parametrize("replacement", ["other", "same", "clear"])
@private_profile_test
async def test_retired_audit_press_cannot_target_replacement(
    request, monkeypatch, button_id, event_type, replacement
):
    app = InspectorApp()
    old_context = PermissionProfileContext("old", 1, "a" * 64, 1)
    new_context = PermissionProfileContext("new", 2, "b" * 64, 2)
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_audit_entry(_audit_record(), profile_context=old_context)
        await pilot.pause()
        old_button = inspector.query_one("#" + button_id, Button)
        held = []
        original = Button.post_message

        def hold(self, message):
            if self is old_button and isinstance(message, Button.Pressed):
                held.append(message)
                return True
            return original(self, message)

        monkeypatch.setattr(Button, "post_message", hold)
        old_button.press()
        assert len(held) == 1
        target = "fetch" if replacement == "other" else "search"
        await inspector.show_audit_entry(
            None if replacement == "clear" else _audit_record(tool_name=target),
            profile_context=new_context,
        )
        inspector.on_button_pressed(held[0])
        await pilot.pause()
        assert not [event for event in app.events if isinstance(event, event_type)]

        # A current control still carries exactly its rendered profile/identity.
        await inspector.show_audit_entry(
            _audit_record(tool_name=target), profile_context=new_context
        )
        await pilot.pause()
        inspector.query_one("#" + button_id, Button).press()
        await pilot.pause()
        events = [event for event in app.events if isinstance(event, event_type)]
        assert len(events) == 1
        assert (events[0].server_key, events[0].tool_name) == ("local:docs", target)
        assert events[0].profile_context == new_context


@pytest.mark.parametrize("button_id,event_type", ACTIONS)
@private_profile_test
async def test_audit_action_uses_rendered_identity_not_mutated_entry(
    request, button_id, event_type
):
    app = InspectorApp()
    entry = _audit_record()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_audit_entry(entry)
        await pilot.pause()
        entry.update(server_key="local:other", tool_name="replacement")
        assert "search" in str(
            inspector.query_one("#mcp-inspector-audit-name", Static).renderable
        )
        inspector.query_one("#" + button_id, Button).press()
        await pilot.pause()
        events = [event for event in app.events if isinstance(event, event_type)]
        assert len(events) == 1
        assert (events[0].server_key, events[0].tool_name) == ("local:docs", "search")


@pytest.mark.parametrize("destination", ["tool", "permission"])
@private_profile_test
async def test_audit_navigation_handles_destination_row_removed_during_clear(
    request, monkeypatch, destination
):
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        inspector = app.query_one(MCPInspector)
        tool = workbench._last_hub_tools[0]
        context = workbench._tool_policy_profile_context
        assert context is not None
        notices = _capture_notifications(app)
        original = inspector.show_audit_entry

        async def clear_and_remove(entry):
            await original(entry)
            if destination == "tool":
                await workbench.query_one(MCPToolsMode).update_tools([])
            else:
                await workbench.query_one(MCPPermissionsMode).update_matrix(
                    [], kill_switch=False, preview="", profile_context=context
                )

        monkeypatch.setattr(inspector, "show_audit_entry", clear_and_remove)
        if destination == "tool":
            await workbench._open_audit_tool(tool, context)
        else:
            await workbench._open_audit_permission(tool, context)
        await pilot.pause()
        assert inspector.current_tool is None
        assert inspector.current_permission_tool is None
        assert notices[-1] == (
            f"{tool.server_key}::{tool.name}: tool no longer available.",
            "warning",
        )


@pytest.mark.parametrize("destination", ["tools", "permissions"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_audit_drill_reveals_target_hidden_by_destination_filter(
    request, destination, theme
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
        field_id = (
            "mcp-tools-filter-text"
            if destination == "tools"
            else "mcp-perm-filter-text"
        )
        field = workbench.query_one("#" + field_id, Input)
        field.value = "nonmatching_filter"
        workbench.set_mode("audit")
        await _settle(pilot)
        table = workbench.query_one("#mcp-audit-table", DataTable)
        table.focus()
        await pilot.press("enter")
        await _settle(pilot)
        button_id = (
            "mcp-audit-open-tool"
            if destination == "tools"
            else "mcp-audit-adjust-permission"
        )
        button = workbench.query_one("#" + button_id, Button)
        button.focus()
        await pilot.press("enter")
        inspector = workbench.query_one(MCPInspector)

        def destination_ready():
            selected = (
                inspector.current_tool
                if destination == "tools"
                else inspector.current_permission_tool
            )
            return (
                workbench.active_mode == destination
                and selected is not None
                and selected.name == "search"
            )

        await _wait(pilot, destination_ready)
        assert field.value == ""
        table = workbench.query_one(
            "#mcp-tools-table" if destination == "tools" else "#mcp-perm-table",
            DataTable,
        )
        key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        assert key.value == "local:docs::search"
        assert not workbench.query("#mcp-audit-open-tool")
