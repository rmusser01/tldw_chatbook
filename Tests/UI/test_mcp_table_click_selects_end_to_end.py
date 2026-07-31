"""TASK-1180 AC#1: a click on the MCP Tools table reaches the Inspector.

`Tests/UI/test_table_click_selects.py` proves the pane posts its selection
message on highlight. This proves the other half — that the message reaches the
real `MCPWorkbench` and changes what the Inspector shows — using the real
screen, workbench and inspector rather than a bare pane.

The two halves matter separately: the pane could post correctly into a workbench
that never handled it, which is the shape of defect this session kept finding.
"""

from __future__ import annotations

import pytest
from textual.widgets import DataTable

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _wait_for_selector,
)
from Tests.UI.app_factory import _build_test_app

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_clicking_a_tool_row_updates_the_inspector_end_to_end():
    """A single click — i.e. a `RowHighlighted` — must select through to the
    workbench, not merely move the DataTable cursor."""
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    app = _build_test_app()
    host = DestinationHarness(app, "mcp")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#mcp-hub-rail")
        screen.action_mcp_mode("tools")
        await pilot.pause()

        tool = HubTool(
            name="search",
            server_key="local:docs",
            server_label="docs",
            source="local",
            description="Search docs",
            tags=(),
            input_schema=None,
            executable=True,
            stale=False,
        )
        screen.workbench._last_hub_tools = [tool]

        tools_mode = screen.query_one("MCPToolsMode")
        table = tools_mode.query_one(DataTable)
        row_key = table.add_row(tool.name, tool.server_label, key=tool.tool_id)
        # A click focuses the table before moving the cursor; the mixin gates on
        # that, because an unfocused table moving its own cursor is a
        # repopulation rather than a person.
        table.focus()
        await pilot.pause()

        # Exactly what a single click produces once the cursor lands on the row.
        tools_mode.post_message(DataTable.RowHighlighted(table, 0, row_key))
        await pilot.pause()
        await pilot.pause()

        # `current_tool` is the inspector's own public accessor for what its
        # tool-detail view currently describes -- the thing a user sees change.
        from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector

        inspector = screen.query_one(MCPInspector)
        shown = inspector.current_tool
        assert shown is not None and shown.tool_id == tool.tool_id, (
            "clicking a tool row did not reach the Inspector: the pane handles "
            f"only activation, so the tool view still shows {shown!r}"
        )


@pytest.mark.asyncio
async def test_opening_tools_mode_selects_nothing_on_its_own():
    """The regression the focus gate exists to prevent.

    A first draft of the mixin forwarded every `RowHighlighted`, including the
    ones a pane's own `clear()`/`add_row()` produce. On this screen a selection
    triggers an awaited remove/mount in `MCPInspector`, which re-syncs the mode,
    which repopulates the table, which highlights row 0 again: opening the Tools
    tab with no user input produced **157** selections and buried a real click
    under the repeats.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#mcp-hub-rail")

        selections: list[str] = []
        workbench_cls = type(screen.workbench)
        original = workbench_cls.on_mcp_tools_mode_tool_selected

        async def _record(self, event):
            selections.append(event.tool_id)
            return await original(self, event)

        workbench_cls.on_mcp_tools_mode_tool_selected = _record
        try:
            screen.action_mcp_mode("tools")
            for _ in range(6):
                await pilot.pause()
        finally:
            workbench_cls.on_mcp_tools_mode_tool_selected = original

        assert selections == [], (
            f"opening Tools mode selected {len(selections)} time(s) with no user "
            "input; a repopulating table is driving the Inspector"
        )
