# Tests/UI/test_mcp_tools_mode.py
from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Static

import tldw_chatbook
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode

_CSS_ROOT = Path(tldw_chatbook.__file__).parent / "css"
_AGENTIC_TERMINAL_TCSS = _CSS_ROOT / "components" / "_agentic_terminal.tcss"
_BUNDLED_STYLESHEET = _CSS_ROOT / "tldw_cli_modular.tcss"


def _tool(
    *,
    server_key: str,
    server_label: str,
    name: str,
    description: str = "",
    input_schema: dict | None = None,
    tags: tuple[str, ...] = (),
    stale: bool = False,
    executable: bool = True,
    source: str = "local",
) -> HubTool:
    return HubTool(
        server_key=server_key,
        server_label=server_label,
        source=source,
        name=name,
        description=description,
        input_schema=input_schema,
        tags=tags,
        stale=stale,
        executable=executable,
    )


class ToolsModeApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPToolsMode(id="mcp-mode-canvas-tools")

    def on_mcp_tools_mode_tool_selected(self, event) -> None:
        self.events.append(event)

    def on_mcp_tools_mode_empty_action_requested(self, event) -> None:
        self.events.append(event)


def _row_texts(table: DataTable, row_index: int) -> list[str]:
    row = table.get_row_at(row_index)
    return [cell.plain if hasattr(cell, "plain") else str(cell) for cell in row]


@pytest.mark.asyncio
async def test_rows_render_grouped_sorted_with_tags_and_schema_columns():
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        tools = [
            _tool(
                server_key="server:main", server_label="Main", name="web_search",
                tags=("high", "network"),
                input_schema={"type": "object", "properties": {}},
            ),
            _tool(
                server_key="local:docs", server_label="docs", name="search",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
            _tool(
                server_key="local:docs", server_label="docs", name="bare",
                input_schema=None,
            ),
        ]
        await canvas.update_tools(tools, empty_diagnosis=None)
        await pilot.pause()

        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.display is True
        assert table.row_count == 3
        # Grouped/sorted by (server_label, name): "Main" < "docs" (ASCII
        # uppercase sorts before lowercase), then within "docs": bare < search.
        assert _row_texts(table, 0)[:2] == ["web_search", "Main"]
        assert _row_texts(table, 1)[:2] == ["bare", "docs"]
        assert _row_texts(table, 2)[:2] == ["search", "docs"]

        # Tags cell.
        main_row = _row_texts(table, 0)
        assert main_row[2] == "high, network"
        bare_row = _row_texts(table, 1)
        assert bare_row[2] == "—"

        # Schema cell: a renderable object schema -> "form"; None/unrenderable -> "raw".
        assert main_row[3] == "form"
        assert bare_row[3] == "raw"
        search_row = _row_texts(table, 2)
        assert search_row[3] == "form"

        assert app.query_one("#mcp-tools-empty").display is False


@pytest.mark.asyncio
async def test_stale_tool_gets_stale_suffix_on_server_cell():
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(
            [_tool(server_key="local:docs", server_label="docs", name="search", stale=True)],
            empty_diagnosis=None,
        )
        await pilot.pause()
        table = app.query_one("#mcp-tools-table", DataTable)
        assert _row_texts(table, 0)[1] == "docs (stale)"


@pytest.mark.asyncio
async def test_row_keys_are_tool_id():
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(
            [_tool(server_key="local:docs", server_label="docs", name="search")],
            empty_diagnosis=None,
        )
        await pilot.pause()
        table = app.query_one("#mcp-tools-table", DataTable)
        row_key, _ = table.coordinate_to_cell_key((0, 0))
        assert row_key.value == "local:docs::search"


@pytest.mark.asyncio
async def test_filter_by_text_narrows_rows_without_touching_empty_state():
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(
            [
                _tool(server_key="local:docs", server_label="docs", name="search"),
                _tool(server_key="local:docs", server_label="docs", name="bare"),
            ],
            empty_diagnosis=None,
        )
        await pilot.pause()

        text_input = app.query_one("#mcp-tools-filter-text", Input)
        text_input.value = "sea"
        await pilot.pause()

        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1
        assert _row_texts(table, 0)[0] == "search"
        # A filter narrowing to a subset (or even zero) must not trigger the
        # diagnostic empty state -- that's reserved for a genuinely empty
        # catalog, not "no matches for this filter".
        assert app.query_one("#mcp-tools-empty").display is False
        assert table.display is True


@pytest.mark.asyncio
async def test_filter_by_server_select_narrows_rows():
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(
            [
                _tool(server_key="local:docs", server_label="docs", name="search"),
                _tool(server_key="server:main", server_label="Main", name="web_search"),
            ],
            empty_diagnosis=None,
        )
        await pilot.pause()

        select = app.query_one("#mcp-tools-filter-server", Select)
        select.value = "local:docs"
        await pilot.pause()

        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1
        assert _row_texts(table, 0)[:2] == ["search", "docs"]

        # Selecting back to "All servers" (Select.NULL) restores every row.
        select.value = Select.NULL
        await pilot.pause()
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_filter_server_select_mount_does_not_spuriously_refilter():
    """Verified Textual gotcha: mounting a Select posts a Changed event for
    its own constructor value. The filter Select is rebuilt on every
    `update_tools()` call (server option list can change) -- that rebuild's
    own mount-echo must not be mistaken for a real user re-filter (which
    here would be a no-op anyway, but this pins the guard down directly by
    asserting the event handler's derived state stays correct across
    several updates, not just "the app didn't crash")."""
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        tools = [_tool(server_key="local:docs", server_label="docs", name="search")]
        for _ in range(3):
            await canvas.update_tools(tools, empty_diagnosis=None)
            await pilot.pause()
        assert canvas._filter_server_key is None
        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1


@pytest.mark.asyncio
async def test_row_selection_posts_tool_selected_with_tool_id():
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(
            [_tool(server_key="local:docs", server_label="docs", name="search")],
            empty_diagnosis=None,
        )
        await pilot.pause()
        table = app.query_one("#mcp-tools-table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPToolsMode.ToolSelected)
        assert event.tool_id == "local:docs::search"


@pytest.mark.asyncio
async def test_empty_diagnosis_renders_message_and_action_button_posts():
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(
            [], empty_diagnosis=("No servers configured — add one to see its tools.", "add_server")
        )
        await pilot.pause()

        empty = app.query_one("#mcp-tools-empty")
        assert empty.display is True
        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.display is False
        message = str(app.query_one("#mcp-tools-empty-message", Static).renderable)
        assert message == "No servers configured — add one to see its tools."
        button = app.query_one("#mcp-tools-empty-action", Button)
        assert button.display is True

        await pilot.click("#mcp-tools-empty-action")
        await pilot.pause()
        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPToolsMode.EmptyActionRequested)
        assert event.action_key == "add_server"


@pytest.mark.asyncio
async def test_empty_diagnosis_connect_action_key_round_trips():
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(
            [], empty_diagnosis=("No tools discovered yet — connect or refresh a server.", "connect")
        )
        await pilot.pause()
        await pilot.click("#mcp-tools-empty-action")
        await pilot.pause()
        assert app.events[0].action_key == "connect"


@pytest.mark.asyncio
async def test_empty_without_diagnosis_hides_action_button():
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools([], empty_diagnosis=None)
        await pilot.pause()
        empty = app.query_one("#mcp-tools-empty")
        assert empty.display is True
        button = app.query_one("#mcp-tools-empty-action", Button)
        assert button.display is False


@pytest.mark.asyncio
async def test_going_from_empty_to_populated_hides_empty_state_again():
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools([], empty_diagnosis=("Nothing yet.", "refresh"))
        await pilot.pause()
        assert app.query_one("#mcp-tools-empty").display is True

        await canvas.update_tools(
            [_tool(server_key="local:docs", server_label="docs", name="search")],
            empty_diagnosis=None,
        )
        await pilot.pause()
        assert app.query_one("#mcp-tools-empty").display is False
        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.display is True
        assert table.row_count == 1


@pytest.mark.asyncio
async def test_server_select_options_reflect_current_catalog_and_prune_stale_filter():
    """When the catalog shrinks so the currently filtered server is gone,
    the filter must reset to "All servers" instead of leaving a dangling
    filter that would (at best) show nothing or (at worst) raise trying to
    reconstruct the Select with a value outside its new options."""
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(
            [
                _tool(server_key="local:docs", server_label="docs", name="search"),
                _tool(server_key="server:main", server_label="Main", name="web_search"),
            ],
            empty_diagnosis=None,
        )
        await pilot.pause()
        select = app.query_one("#mcp-tools-filter-server", Select)
        select.value = "local:docs"
        await pilot.pause()
        assert canvas._filter_server_key == "local:docs"

        # "docs" drops out of the catalog entirely.
        await canvas.update_tools(
            [_tool(server_key="server:main", server_label="Main", name="web_search")],
            empty_diagnosis=None,
        )
        await pilot.pause()
        assert canvas._filter_server_key is None
        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1


def test_tools_table_height_rule_pinned_in_bundle_source_and_bundle() -> None:
    """T7 (P3 UX batch) gave `#mcp-tools-table` `height: auto; max-height:
    70%;` in `MCPToolsMode.DEFAULT_CSS` alone -- no matching rule was ever
    added to the bundle-source component file (`_agentic_terminal.tcss`),
    unlike the established `#mcp-detail-builtin-toggles` / `#mcp-servers-
    form` / `#mcp-import-list` lockstep pairs there. Without a bundle-layer
    copy, app-loaded CSS (which cascades ON TOP of DEFAULT_CSS) could
    silently reintroduce the `height: 1fr` ballooning regression T7 fixed,
    with nothing here to catch it. Pins the rule in both the bundle-source
    file and the generated bundle (`tldw_cli_modular.tcss`) -- the latter
    also proves `build_css.py` was re-run after the source edit, mirroring
    `test_prompt_picker_css_blocks_pinned_in_source_and_bundle` in
    test_console_prompt_picker.py."""
    agentic_terminal = _AGENTIC_TERMINAL_TCSS.read_text(encoding="utf-8")
    bundled_stylesheet = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text, label in (
        (agentic_terminal, "_agentic_terminal.tcss"),
        (bundled_stylesheet, "tldw_cli_modular.tcss"),
    ):
        selector = "#mcp-tools-table {"
        start = text.find(selector)
        assert start != -1, f"{label} is missing {selector!r}"
        end = text.find("}", start)
        block = text[start:end]
        assert "height: auto;" in block, f"{label}'s {selector!r} block is missing 'height: auto;'"
        assert "max-height: 70%;" in block, f"{label}'s {selector!r} block is missing 'max-height: 70%;'"
