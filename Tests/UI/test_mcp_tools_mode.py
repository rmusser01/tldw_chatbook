# Tests/UI/test_mcp_tools_mode.py
from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Static

import tldw_chatbook
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState
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
        # Columns are Tool | State | Server | Tags | Schema -- no `states`
        # was passed, so every State cell is the "—" absent default.
        assert _row_texts(table, 0)[:3] == ["web_search", "—", "Main"]
        assert _row_texts(table, 1)[:3] == ["bare", "—", "docs"]
        assert _row_texts(table, 2)[:3] == ["search", "—", "docs"]

        # Tags cell.
        main_row = _row_texts(table, 0)
        assert main_row[3] == "high, network"
        bare_row = _row_texts(table, 1)
        assert bare_row[3] == "—"

        # Schema cell: a renderable object schema -> "form"; None/unrenderable -> "raw".
        assert main_row[4] == "form"
        assert bare_row[4] == "raw"
        search_row = _row_texts(table, 2)
        assert search_row[4] == "form"

        assert app.query_one("#mcp-tools-empty").display is False


@pytest.mark.asyncio
async def test_state_column_renders_marker_labels_from_states_dict():
    """T8: the State column reuses the exact same label+marker rendering as
    the Permissions-mode matrix (`format_tool_state_label()`) -- pinned here
    against every marker variant, keyed `(server_key, name)` same as
    `effective_tool_states()`."""
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        tools = [
            _tool(server_key="local:docs", server_label="docs", name="allowed"),
            _tool(server_key="local:docs", server_label="docs", name="asked"),
            _tool(server_key="local:docs", server_label="docs", name="rug_pulled"),
            _tool(server_key="local:docs", server_label="docs", name="risk_floored"),
        ]
        states = {
            ("local:docs", "allowed"): EffectiveToolState(state="allow", origin="tool_override"),
            ("local:docs", "asked"): EffectiveToolState(state="ask", origin="global_default"),
            ("local:docs", "rug_pulled"): EffectiveToolState(
                state="ask", origin="tool_override", config_changed=True
            ),
            ("local:docs", "risk_floored"): EffectiveToolState(
                state="ask", origin="server_default", risk_floored=True
            ),
        }
        await canvas.update_tools(tools, empty_diagnosis=None, states=states)
        await pilot.pause()

        table = app.query_one("#mcp-tools-table", DataTable)
        rows_by_tool = {_row_texts(table, i)[0]: _row_texts(table, i)[1] for i in range(table.row_count)}
        assert rows_by_tool["allowed"] == "Allow •"
        assert rows_by_tool["asked"] == "Ask"
        assert rows_by_tool["rug_pulled"] == "Ask ⚠"
        assert rows_by_tool["risk_floored"] == "Ask ⚑"


@pytest.mark.asyncio
async def test_state_column_renders_em_dash_when_states_none_or_missing():
    """`states=None` (default) and a tool absent from a non-empty `states`
    dict must both fall back to the "—" absent marker -- never guess a
    default verdict for a tool the workbench couldn't resolve one for."""
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        tools = [
            _tool(server_key="local:docs", server_label="docs", name="known"),
            _tool(server_key="local:docs", server_label="docs", name="unknown"),
        ]

        # states omitted entirely.
        await canvas.update_tools(tools, empty_diagnosis=None)
        await pilot.pause()
        table = app.query_one("#mcp-tools-table", DataTable)
        assert _row_texts(table, 0)[1] == "—"
        assert _row_texts(table, 1)[1] == "—"

        # states present but missing an entry for "unknown".
        states = {("local:docs", "known"): EffectiveToolState(state="allow", origin="tool_override")}
        await canvas.update_tools(tools, empty_diagnosis=None, states=states)
        await pilot.pause()
        rows_by_tool = {_row_texts(table, i)[0]: _row_texts(table, i)[1] for i in range(table.row_count)}
        assert rows_by_tool["known"] == "Allow •"
        assert rows_by_tool["unknown"] == "—"


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
        assert _row_texts(table, 0)[2] == "docs (stale)"


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
async def test_duplicate_tool_ids_do_not_crash_update_tools():
    """C1: a duplicate `HubTool.tool_id` (e.g. a persisted discovery snapshot
    with two same-named tools that upstream dedup somehow missed) must not
    raise Textual's `DuplicateKey` when rebuilding the DataTable -- only one
    row should ever render for that key. Mirrors the whole-branch review's
    end-to-end probe (mounting the canvas with a hand-crafted duplicate list,
    bypassing hub_tool_catalog's own dedup so this exercises the table's own
    defense in depth)."""
    app = ToolsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(
            [
                _tool(server_key="local:docs", server_label="docs", name="search"),
                _tool(server_key="local:docs", server_label="docs", name="search"),
            ],
            empty_diagnosis=None,
        )
        await pilot.pause()
        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1
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
        assert _row_texts(table, 0)[0] == "search"
        assert _row_texts(table, 0)[2] == "docs"

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


def test_filter_server_select_width_rule_pinned_in_bundle_source_and_bundle() -> None:
    """Defect 1 (QA round mcp-hub-phase3-2026-07): `MCPToolsMode.DEFAULT_CSS`
    gives `#mcp-tools-filter-server-slot Select` a fixed `width: 28`, but
    Textual's cascade always treats every CSS_PATH-sourced rule (the app
    bundle) as higher priority than any widget's own DEFAULT_CSS regardless
    of selector specificity -- so `_conversations.tcss`'s global, unscoped
    `Select { width: 100%; }` rule (compiled into the same bundle, intended
    for a completely different screen's sidebar forms) silently won,
    collapsing the filter-server Select to a 0x0 region in the real running
    app. Fix has to live in the bundle itself, with equal-or-higher
    specificity than that bare rule, to actually win -- mirrors
    `test_tools_table_height_rule_pinned_in_bundle_source_and_bundle`
    immediately above and `test_prompt_picker_css_blocks_pinned_in_source_
    and_bundle` in test_console_prompt_picker.py."""
    agentic_terminal = _AGENTIC_TERMINAL_TCSS.read_text(encoding="utf-8")
    bundled_stylesheet = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text, label in (
        (agentic_terminal, "_agentic_terminal.tcss"),
        (bundled_stylesheet, "tldw_cli_modular.tcss"),
    ):
        selector = "#mcp-tools-filter-server-slot Select {"
        start = text.find(selector)
        assert start != -1, f"{label} is missing {selector!r}"
        end = text.find("}", start)
        block = text[start:end]
        assert "width: 28;" in block, f"{label}'s {selector!r} block is missing 'width: 28;'"


class ToolsModeAppWithBundledCSS(App):
    """Same harness as `ToolsModeApp` above but with the real bundled
    stylesheet loaded, so the filter-server Select contests its actual CSS
    priority battle against `_conversations.tcss`'s global
    `Select { width: 100%; }` rule exactly as it does in the live app --
    mirrors `CanvasAppWithBundledCSS` in test_mcp_servers_mode.py and
    `RailAppWithBundledCSS` in test_mcp_rail.py. Regression coverage for
    Defect 1 (QA round mcp-hub-phase3-2026-07): before the bundle-layer fix
    above, this Select rendered at 0x0 under the real app stylesheet even
    though `MCPToolsMode.DEFAULT_CSS` alone (no CSS_PATH) already asked for
    the correct `width: 28`."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        yield MCPToolsMode(id="mcp-mode-canvas-tools")


@pytest.mark.asyncio
async def test_filter_server_select_has_nonzero_geometry_with_bundled_css():
    app = ToolsModeAppWithBundledCSS()
    async with app.run_test(size=(120, 40)) as pilot:
        canvas = app.query_one(MCPToolsMode)
        await canvas.update_tools(
            [
                _tool(server_key="local:docs", server_label="docs-server", name="search_docs"),
                _tool(server_key="server:weather", server_label="weather-api", name="get_forecast"),
            ],
            empty_diagnosis=None,
        )
        await pilot.pause()

        select = app.query_one("#mcp-tools-filter-server", Select)
        assert select.size.width > 0, (
            "filter-server Select collapsed to zero width under the real "
            "bundled stylesheet (Defect 1, QA round mcp-hub-phase3-2026-07) "
            "-- _conversations.tcss's global `Select { width: 100%; }` rule "
            "is clobbering MCPToolsMode's own DEFAULT_CSS override again."
        )
        assert select.size.height > 0, "filter-server Select collapsed to zero height"
