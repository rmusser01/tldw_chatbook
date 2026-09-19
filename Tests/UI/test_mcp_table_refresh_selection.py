"""Programmatic table highlights cannot become delayed user selections."""

import pytest
from textual.widgets import DataTable

from Tests.private_profile import private_profile_test
from Tests.UI.test_table_click_selects import _Harness
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.UI.MCP_Modules.mcp_audit_mode import MCPAuditMode
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode


@pytest.mark.parametrize(
    "view,gesture,cursor_type",
    [
        (view, gesture, "row")
        for view in ("tools", "executions", "findings")
        for gesture in ("enter", "down", "click")
    ]
    + [("tools", "down", "cell")],
)
@private_profile_test
async def test_queued_refresh_highlights_are_quiet_and_next_gesture_selects(
    request, monkeypatch, view, gesture, cursor_type
):
    pane = MCPToolsMode() if view == "tools" else MCPAuditMode()
    app = _Harness(pane)
    async with app.run_test(size=(140, 40)) as pilot:

        async def populate():
            if view == "tools":
                await pane.update_tools(
                    [
                        HubTool(
                            server_key="local:docs",
                            server_label="docs",
                            source="local",
                            name=name,
                            description=name,
                            input_schema=None,
                            tags=(),
                            stale=False,
                            executable=False,
                        )
                        for name in ("alpha", "beta", "gamma")
                    ]
                )
            elif view == "findings":
                await pane.update_findings(
                    [
                        {"severity": "high", "finding_type": "test", "message": name}
                        for name in ("alpha", "beta", "gamma")
                    ],
                    source="server",
                )
                pane._sub_view = "findings"
                pane._apply_subview_display()
            else:
                await pane.update_entries(
                    [
                        {
                            "server_key": "local:docs",
                            "tool_name": name,
                            "decision": "allow",
                        }
                        for name in ("alpha", "beta", "gamma")
                    ]
                )

        await populate()
        await pilot.pause()
        table = pane.query_one(
            {
                "tools": "#mcp-tools-table",
                "executions": "#mcp-audit-table",
                "findings": "#mcp-audit-findings-table",
            }[view],
            DataTable,
        )
        table.cursor_type = cursor_type
        table.focus()
        table.move_cursor(row=1)
        await pilot.pause()
        app.captured.clear()
        queued = []
        original = DataTable._on_message

        async def hold_published_highlights(control, message):
            if control is table and isinstance(
                message, (DataTable.RowHighlighted, DataTable.CellHighlighted)
            ):
                queued.append(message)
                return
            await original(control, message)

        monkeypatch.setattr(DataTable, "_on_message", hold_published_highlights)
        await populate()
        await pilot.pause()
        # Delivery can lag beyond all after-refresh callbacks. Only messages
        # already published by the real table are held; prevent() still runs.
        monkeypatch.setattr(DataTable, "_on_message", original)
        for message in queued:
            await original(table, message)
        await pilot.pause()
        selected_type = {
            "tools": "ToolSelected",
            "executions": "EntrySelected",
            "findings": "FindingSelected",
        }[view]
        selected = lambda: [
            m for m in app.captured if type(m).__name__ == selected_type
        ]
        assert table.row_count == 3 and table.cursor_row == 1 and table.has_focus
        assert selected() == [], "redrawing a focused table must not select its rows"
        if gesture == "click":
            await pilot.click(table, offset=(2, 2))
        else:
            await pilot.press(gesture)
        await pilot.pause()
        assert len(selected()) == 1, (
            "the next real gesture must still select exactly once"
        )


@private_profile_test
async def test_hidden_findings_refresh_preserves_active_execution_gesture(request):
    pane = MCPAuditMode()
    app = _Harness(pane)
    async with app.run_test(size=(140, 40)) as pilot:
        await pane.update_entries(
            [
                {"server_key": "local:docs", "tool_name": name, "decision": "allow"}
                for name in ("alpha", "beta")
            ]
        )
        await pilot.pause()
        table = pane.query_one("#mcp-audit-table", DataTable)
        table.focus()
        await pilot.pause()
        app.captured.clear()
        await pilot.press("down")
        await pilot.pause()
        assert (
            len([m for m in app.captured if type(m).__name__ == "EntrySelected"]) == 1
        )
        await pane.update_findings(
            [{"severity": "high", "finding_type": "test", "message": "finding"}],
            source="server",
        )
        await pilot.pause()
        assert table.has_focus
        await pilot.press("enter")
        await pilot.pause()
        assert (
            len([m for m in app.captured if type(m).__name__ == "EntrySelected"]) == 1
        )
        await pilot.press("enter")
        await pilot.pause()
        assert (
            len([m for m in app.captured if type(m).__name__ == "EntrySelected"]) == 2
        )


@pytest.mark.parametrize("view", ["tools", "permissions"])
@private_profile_test
async def test_focused_external_tool_drill_moves_without_selecting(request, view):
    from Tests.UI.test_mcp_permissions_mode import _tool_row
    from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode

    pane = MCPToolsMode() if view == "tools" else MCPPermissionsMode()
    app = _Harness(pane)
    async with app.run_test(size=(140, 40)) as pilot:
        if view == "tools":
            await pane.update_tools(
                [
                    HubTool(
                        server_key="local:docs",
                        server_label="docs",
                        source="local",
                        name=name,
                        description=name,
                        input_schema=None,
                        tags=(),
                        stale=False,
                        executable=False,
                    )
                    for name in ("alpha", "beta")
                ]
            )
        else:
            await pane.update_matrix(
                [
                    _tool_row(
                        server_key="local:docs", server_label="docs", tool_name=name
                    )
                    for name in ("alpha", "beta")
                ],
                kill_switch=False,
                preview="",
            )
        await pilot.pause()
        table = pane.query_one(DataTable)
        table.focus()
        await pilot.pause()
        app.captured.clear()
        if view == "tools":
            assert await pane.select_tool_row("local:docs::beta")
        else:
            assert pane.select_tool_row("local:docs", "beta")
        await pilot.pause()
        assert table.cursor_row == 1 and table.has_focus
        selected_type = "ToolSelected" if view == "tools" else "RowSelected"
        assert [m for m in app.captured if type(m).__name__ == selected_type] == []
        await pilot.press("enter")
        await pilot.pause()
        assert len([m for m in app.captured if type(m).__name__ == selected_type]) == 1


def test_identical_row_keys_in_different_tables_are_distinct_gestures():
    from types import SimpleNamespace

    from textual.widgets.data_table import RowKey

    from tldw_chatbook.UI.Widgets.table_click_select import DataTableClickSelectMixin

    class Pane(DataTableClickSelectMixin):
        def __init__(self):
            self.selected = []

        def on_data_table_row_selected(self, event):
            self.selected.append(event.data_table.id)

    pane = Pane()
    first = SimpleNamespace(id="executions", has_focus=True)
    second = SimpleNamespace(id="findings", has_focus=True)
    key = RowKey("1")
    pane.on_data_table_row_highlighted(DataTable.RowHighlighted(first, 1, key))
    pane.on_data_table_row_highlighted(DataTable.RowHighlighted(second, 1, key))
    pane.on_data_table_row_selected(DataTable.RowSelected(second, 1, key))
    assert pane.selected == ["executions", "findings"]
