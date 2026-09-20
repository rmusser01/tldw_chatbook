"""A header painted before idle measurement must follow the measured rows."""

from dataclasses import replace

import pytest
from textual.strip import Strip
from textual.widgets import DataTable, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_tools_mode import ToolsModeBundledCSSApp, _tool
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode


def _column_starts(strip: Strip) -> dict[int, int]:
    """Read column boundaries from the actual painted segment metadata."""
    starts = {}
    offset = 0
    for segment in strip:
        meta = segment.style.meta if segment.style else {}
        column = meta.get("column")
        if column is not None and not meta.get("out_of_bounds"):
            starts.setdefault(column, offset)
        offset += segment.cell_length
    return starts


def _assert_header_alignment(table: DataTable) -> None:
    # The table may render correctly while the composed screen still caches
    # its previous header. Inspect the strips actually painted on screen.
    region = table.content_region
    strips = table.screen._compositor.render_strips()
    header = strips[region.y].crop(region.x, region.right)
    body = strips[region.y + table.header_height].crop(region.x, region.right)
    assert _column_starts(body), body.text
    assert _column_starts(header) == _column_starts(body), (header.text, body.text)


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_header_painted_before_measurement_aligns_with_new_catalog(
    request, theme
):
    app = ToolsModeBundledCSSApp()
    app.theme = theme
    async with app.run_test(size=(170, 48)) as pilot:
        canvas = app.query_one(MCPToolsMode)
        table = canvas.query_one(DataTable)
        tool = _tool(
            server_key="local:docs",
            server_label="Documentation server",
            name="工具_read_catalog_document",
            tags=("read-only",),
        )
        state = EffectiveToolState(state="ask", origin="tool_override")
        await canvas.update_tools([tool], states={(tool.server_key, tool.name): state})
        await pilot.pause()
        # A refresh may paint before the table's queued idle measures its rows.
        # Render that real intermediate state without modifying any cache.
        await canvas.update_tools([tool], states={(tool.server_key, tool.name): state})
        table.render_line(0)
        await pilot.pause()
        region = table.content_region
        header = table.screen._compositor.render_strips()[region.y].crop(
            region.x, region.right
        )
        assert set(_column_starts(header)) == set(range(5))
        _assert_header_alignment(table)


@pytest.mark.parametrize("tags", [False, True])
@private_profile_test
async def test_catalog_filter_theme_and_resize_keep_header_and_selection(request, tags):
    app = ToolsModeBundledCSSApp()
    tools = [
        _tool(
            server_key="local:docs",
            server_label="Documentation server",
            name=f"工具_read_catalog_document_{index}",
            tags=("read-only",) if tags else (),
        )
        for index in range(3)
    ]
    target = tools[1]
    state = EffectiveToolState(state="ask", origin="tool_override", config_changed=True)
    states = {(tool.server_key, tool.name): state for tool in tools}
    async with app.run_test(size=(170, 48)) as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(tools, states=states)
        table = canvas.query_one(DataTable)
        table.focus()
        table.move_cursor(row=1)
        await pilot.pause()
        app.events.clear()
        for theme in ("textual-dark", "textual-light"):
            app.theme = theme
            for width in (170, 42, 120, 170):
                await pilot.resize_terminal(width, 40)
                await pilot.pause()
                # Same IDs get new labels/tags, with input order reversed.
                catalog = [
                    replace(tool, server_label="Updated 文档 server", tags=())
                    for tool in reversed(tools)
                ]
                for replacement in (catalog, tools):
                    await canvas.update_tools(replacement, states=states)
                    table.render_line(0)
                    await pilot.pause()
                    _assert_header_alignment(table)
                    assert table.has_focus
                    assert (
                        table.coordinate_to_cell_key((table.cursor_row, 0))[0].value
                        == target.tool_id
                    )
                    assert app.events == [], "repainting must not select a tool"
                # Check the right-hand columns when the table scrolls horizontally.
                await pilot.press("end")
                await pilot.pause()
                _assert_header_alignment(table)
                await pilot.press("home")
                await pilot.pause()
        field = canvas.query_one("#mcp-tools-filter-text", Input)
        field.focus()
        await pilot.press(*"document_1")
        await pilot.pause()
        assert table.row_count == 1
        _assert_header_alignment(table)
        assert app.events == []
        table.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert [event.tool_id for event in app.events] == [target.tool_id]
        field.focus()
        field.value = "no matching tool"
        await pilot.pause()
        assert table.row_count == 0
        field.value = ""
        await pilot.pause()
        assert table.row_count == 3
        _assert_header_alignment(table)
