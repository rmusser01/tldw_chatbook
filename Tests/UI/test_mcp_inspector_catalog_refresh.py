"""Selected tool details follow refresh without discarding unchanged drafts."""

from dataclasses import replace

import pytest
from textual.widgets import Button, DataTable, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_compact_readability import _settle
from Tests.UI.test_mcp_root_settings import _open
from Tests.UI.test_mcp_workbench import (
    ToolTestHubService,
    WorkbenchAppWithBundledCSS,
)
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unchanged", "description", "schema", "removed"])
@private_profile_test
async def test_catalog_refresh_reconciles_selected_tool_without_reselecting(
    request, monkeypatch, change
):
    app = WorkbenchAppWithBundledCSS()
    app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        await canvas.select_tool_row(original.tool_id)
        canvas.query_one(DataTable).focus()
        await pilot.press("enter")
        await _settle(pilot)
        inspector = app.query_one(MCPInspector)
        assert inspector.current_tool == original
        inspector.query_one("#mcp-inspector-test-tool", Button).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await _settle(pilot)
        field = inspector.query_one("#mcp-schema-field-0", Input)
        field.focus()
        await pilot.press(*"keep me")
        await _settle(pilot)
        assert field.value == "keep me"
        nonce = inspector._test_preview.nonce

        fresh = replace(original)
        if change == "description":
            fresh = replace(original, description="Search the current catalog.")
        elif change == "schema":
            fresh = replace(original, input_schema={"oneOf": [{"type": "object"}]})
        tools = [
            fresh if tool.tool_id == original.tool_id else tool
            for tool in workbench._last_hub_tools
            if change != "removed" or tool.tool_id != original.tool_id
        ]
        monkeypatch.setattr(workbench, "_collect_hub_tools", lambda: tools)
        await workbench._sync_children()
        await app.workers.wait_for_complete()
        await _settle(pilot)

        if change == "unchanged":
            assert inspector.query_one("#mcp-schema-field-0", Input) is field
            assert field.value == "keep me" and field.has_focus
            assert inspector._test_preview.nonce == nonce
        else:
            assert not inspector.query("#mcp-inspector-test-panel")
            assert inspector._test_preview is None
            assert nonce in app.unified_mcp_service.revoked_nonces
            if change == "removed":
                assert inspector.current_tool is None
                assert not inspector.query_one("#mcp-inspector-tool").display
            else:
                assert inspector.current_tool == fresh
                assert (
                    str(
                        inspector.query_one(
                            "#mcp-inspector-tool-description", Static
                        ).renderable
                    )
                    == fresh.description
                )
                schema = inspector.query_one("#mcp-inspector-tool-schema", Static)
                assert str(schema.renderable) == (
                    "Parameters: raw JSON" if change == "schema" else "Parameters: form"
                )
                assert inspector.query_one("#mcp-inspector-test-tool", Button).has_focus
                note = inspector.query_one("#mcp-inspector-tool-refresh-note", Static)
                assert "changed" in str(note.renderable).lower()
        assert app.unified_mcp_service.test_calls == []
