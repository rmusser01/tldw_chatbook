"""Audit gestures must keep the selected execution and its drill target together."""

import pytest
from textual.widgets import DataTable, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_workbench import AuditApp, _audit_record, _select_audit_mode_row
from tldw_chatbook.UI.MCP_Modules.mcp_audit_mode import MCPAuditMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench


@pytest.mark.asyncio
@private_profile_test
async def test_filtering_out_selected_execution_clears_detail(request):
    app = AuditApp(
        [_audit_record(tool_name="fetch"), _audit_record(tool_name="search")]
    )
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 1)
        assert app.query_one("#mcp-inspector-audit").display
        field = app.query_one("#mcp-audit-filter-text", Input)
        field.focus()
        field.value = "fetch"
        await pilot.pause()
        assert app.query_one("#mcp-audit-table", DataTable).row_count == 1
        assert not app.query_one("#mcp-inspector-audit").display
        assert not app.query("#mcp-audit-open-tool")
        assert field.has_focus


@pytest.mark.asyncio
@private_profile_test
async def test_newer_execution_keeps_selected_event_and_drill_target(request):
    entries = [_audit_record(tool_name="fetch"), _audit_record(tool_name="search")]
    app = AuditApp(entries)
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 1)
        app.unified_mcp_service.execution_log._records.insert(
            0, _audit_record(tool_name="list_notes", server_key="local:notes")
        )
        await workbench._sync_audit_log_entries()
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.get_row_at(table.cursor_row)[1].plain == "local:docs::search"
        await pilot.press("enter")
        await pilot.pause()
        assert "search" in str(
            app.query_one("#mcp-inspector-audit-name", Static).render()
        )
        await pilot.click("#mcp-audit-open-tool")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert workbench.active_mode == "tools"
        assert "search" in str(
            app.query_one("#mcp-inspector-tool-name", Static).render()
        )


@pytest.mark.asyncio
@private_profile_test
async def test_execution_eviction_clears_detail_without_selecting_replacement(request):
    app = AuditApp(
        [_audit_record(tool_name="fetch"), _audit_record(tool_name="search")]
    )
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 1)
        app.unified_mcp_service.execution_log._records = [
            _audit_record(tool_name="fetch")
        ]
        await workbench._sync_audit_log_entries()
        await pilot.pause()
        assert not app.query_one("#mcp-inspector-audit").display
        assert not app.query("#mcp-audit-adjust-permission")


@pytest.mark.asyncio
@pytest.mark.parametrize("gesture", ["table", "canvas"])
@private_profile_test
async def test_delayed_activation_cannot_select_replacement(
    request, monkeypatch, gesture
):
    app = AuditApp([_audit_record(tool_name="search")])
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        canvas = app.query_one(MCPAuditMode)
        table = app.query_one("#mcp-audit-table", DataTable)
        key, _ = table.coordinate_to_cell_key((0, 0))
        old_gesture = DataTable.RowSelected(table, 0, key)
        held = []
        original = canvas.post_message

        def hold_selection(message):
            if isinstance(message, MCPAuditMode.EntrySelected):
                held.append(message)
                return True
            return original(message)

        if gesture == "canvas":
            monkeypatch.setattr(canvas, "post_message", hold_selection)
            canvas.on_data_table_row_selected(old_gesture)
            assert len(held) == 1
        app.unified_mcp_service.execution_log._records = [
            _audit_record(tool_name="fetch")
        ]
        await workbench._sync_audit_log_entries()
        await pilot.pause()
        if gesture == "table":
            canvas.on_data_table_row_selected(old_gesture)
        else:
            await workbench.on_mcp_audit_mode_entry_selected(held[0])
        await pilot.pause()
        assert not app.query_one("#mcp-inspector-audit").display
        assert not app.query("#mcp-audit-open-tool")


@pytest.mark.asyncio
@private_profile_test
async def test_ambiguous_identical_events_clear_selection_on_refresh(request):
    entry = _audit_record(tool_name="search")
    app = AuditApp([entry])
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 0)
        app.unified_mcp_service.execution_log._records = [dict(entry), dict(entry)]
        await workbench._sync_audit_log_entries()
        await pilot.pause()
        assert not app.query_one("#mcp-inspector-audit").display


@pytest.mark.asyncio
@private_profile_test
async def test_duplicate_event_keeps_its_row_within_same_filtered_snapshot(request):
    entry = _audit_record(tool_name="search")
    app = AuditApp([dict(entry), dict(entry)])
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 1)
        table = app.query_one("#mcp-audit-table", DataTable)
        key, _ = table.coordinate_to_cell_key((1, 0))
        app.query_one("#mcp-audit-filter-text", Input).value = "search"
        await pilot.pause()
        assert app.query_one("#mcp-inspector-audit").display
        assert table.coordinate_to_cell_key((table.cursor_row, 0))[0] == key


@pytest.mark.asyncio
@private_profile_test
async def test_refresh_cannot_identify_which_duplicate_execution_survived(request):
    entry = _audit_record(tool_name="search")
    app = AuditApp([dict(entry), dict(entry)])
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 1)
        app.unified_mcp_service.execution_log._records = [dict(entry)]
        await workbench._sync_audit_log_entries()
        await pilot.pause()
        assert not app.query_one("#mcp-inspector-audit").display
        assert not app.query("#mcp-audit-open-tool")


@pytest.mark.asyncio
@pytest.mark.parametrize("gesture", ["table", "canvas"])
@pytest.mark.parametrize("departure", ["mode", "subview"])
@private_profile_test
async def test_retired_execution_gesture_stays_retired_after_round_trip(
    request, monkeypatch, gesture, departure
):
    app = AuditApp([_audit_record(tool_name="search")])
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        canvas = app.query_one(MCPAuditMode)
        table = app.query_one("#mcp-audit-table", DataTable)
        key, _ = table.coordinate_to_cell_key((0, 0))
        old_gesture = DataTable.RowSelected(table, 0, key)
        held = []
        original = canvas.post_message

        def hold_selection(message):
            if isinstance(message, MCPAuditMode.EntrySelected):
                held.append(message)
                return True
            return original(message)

        if gesture == "canvas":
            monkeypatch.setattr(canvas, "post_message", hold_selection)
            canvas.on_data_table_row_selected(old_gesture)
            assert len(held) == 1
            monkeypatch.setattr(canvas, "post_message", original)
        if departure == "mode":
            workbench.set_mode("tools")
            await pilot.pause()
            workbench.set_mode("audit")
        else:
            await pilot.click("#mcp-audit-subview-findings")
            await pilot.click("#mcp-audit-subview-executions")
        await pilot.pause()
        assert not app.query_one("#mcp-inspector-audit").display
        if gesture == "table":
            canvas.on_data_table_row_selected(old_gesture)
        else:
            await workbench.on_mcp_audit_mode_entry_selected(held[0])
        await pilot.pause()
        assert not app.query_one("#mcp-inspector-audit").display
        assert not app.query("#mcp-audit-open-tool")
        table.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#mcp-inspector-audit").display


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_kind", ["text", "decision", "initiator"])
@private_profile_test
async def test_filters_retain_matching_detail_and_clear_zero_matches(
    request, filter_kind
):
    app = AuditApp(
        [
            _audit_record(tool_name="search", decision="allowed", initiator="test"),
            _audit_record(tool_name="fetch", decision="denied", initiator="agent"),
        ]
    )
    async with app.run_test(size=(170, 48)) as pilot:
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 0)
        control = app.query_one(f"#mcp-audit-filter-{filter_kind}")
        control.focus()
        control.value = {
            "text": " SEARCH ",
            "decision": "allowed",
            "initiator": "test",
        }[filter_kind]
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.row_count == 1
        assert "search" in str(
            app.query_one("#mcp-inspector-audit-name", Static).render()
        )
        app.query_one("#mcp-audit-filter-text", Input).value = "missing"
        await pilot.pause()
        assert table.row_count == 0
        assert not app.query_one("#mcp-inspector-audit").display
        app.query_one("#mcp-audit-filter-text", Input).value = ""
        if isinstance(control, Select):
            control.value = Select.NULL
        await pilot.pause()
        assert table.row_count == 2
        assert not app.query_one("#mcp-inspector-audit").display
        assert control.has_focus
