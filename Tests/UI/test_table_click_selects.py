"""TASK-1180: a single click on a table row must select it, not just move the cursor.

Textual fires `RowSelected` on **activation** — Enter, or a second click — while a
single click produces `RowHighlighted`. A pane that handles only `RowSelected`
therefore highlights on click and selects nothing: the Inspector stays empty and
row actions stay disabled.

TASK-1105 fixed this for the Watchlists Sources pane, where it had made
`Preview` / `Check now` / `Delete` unreachable by mouse and ultimately hid a dead
scrape path (TASK-1100). Every other table pane still had it.

These tests drive the panes the way a mouse user does — post a
`RowHighlighted`, which is what a click produces — and assert the pane's own
selection message comes back out.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

pytestmark = pytest.mark.unit


class _Harness(App):
    """Mount one pane and capture every message it posts."""

    def __init__(self, pane) -> None:
        super().__init__()
        self._pane = pane
        self.captured: list[object] = []

    def compose(self) -> ComposeResult:
        yield self._pane

    async def _on_message(self, message) -> None:  # noqa: D401 - Textual hook
        # Capture rather than intercept: the message still reaches its normal
        # handlers, so this observes the pane instead of standing in for it.
        self.captured.append(message)
        await super()._on_message(message)


def _click_row(table: DataTable, row_key) -> DataTable.RowHighlighted:
    """The event a single click produces once the cursor lands on a row."""
    return DataTable.RowHighlighted(table, 0, row_key)


@pytest.mark.asyncio
async def test_clicking_a_tool_row_selects_it():
    """MCP Tools: confirmed affected in the task report."""
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode

    pane = MCPToolsMode()
    app = _Harness(pane)
    async with app.run_test(size=(120, 40)) as pilot:
        table = pane.query_one(DataTable)
        row_key = table.add_row("echo", "server-a", key="tool-echo")
        table.focus()
        await pilot.pause()

        pane.post_message(_click_row(table, row_key))
        await pilot.pause()
        await pilot.pause()

        selected = [m for m in app.captured if type(m).__name__ == "ToolSelected"]
        assert selected, (
            "clicking a tool row posted no ToolSelected: the pane handles only "
            "RowSelected, which a single click does not produce"
        )


@pytest.mark.asyncio
async def test_clicking_a_permission_row_selects_it():
    """MCP Permissions: the other confirmed pane."""
    from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode

    pane = MCPPermissionsMode()
    app = _Harness(pane)
    async with app.run_test(size=(120, 40)) as pilot:
        table = pane.query_one(DataTable)
        row_key = table.add_row("server", "srv", "ask", key="perm-1")
        table.focus()
        # The pane resolves a click back through the map update_matrix() builds.
        pane._rows_by_key["perm-1"] = type(
            "PermRow", (), {"kind": "server", "server_key": "srv", "tool_name": None}
        )()
        await pilot.pause()

        pane.post_message(_click_row(table, row_key))
        await pilot.pause()
        await pilot.pause()

        selected = [m for m in app.captured if type(m).__name__ == "RowSelected"]
        assert selected, (
            "clicking a permission row posted no RowSelected identity message"
        )


@pytest.mark.asyncio
async def test_an_unfocused_table_moving_its_own_cursor_selects_nothing():
    """The focus gate, tested on its own.

    `clear()`/`add_row()` move the cursor back to row 0 and emit
    `RowHighlighted` exactly as a click does. The distinguishing fact — measured
    on the real MCP screen — is that a repopulating table is not focused, while
    a click focuses it before the cursor moves.

    Without this gate, opening the MCP Tools tab produced 157 selections from no
    user input, because each one remounted the Inspector, which re-synced the
    mode, which repopulated the table.
    """
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode

    pane = MCPToolsMode()
    app = _Harness(pane)
    async with app.run_test(size=(120, 40)) as pilot:
        table = pane.query_one(DataTable)
        row_key = table.add_row("echo", "server-a", key="tool-echo")
        # Deliberately NOT focused: this is a repopulation, not a person.
        await pilot.pause()

        pane.post_message(_click_row(table, row_key))
        await pilot.pause()
        await pilot.pause()

        selected = [m for m in app.captured if type(m).__name__ == "ToolSelected"]
        assert not selected, (
            "an unfocused table's cursor movement selected a row; a pane "
            "rebuilding its own rows will drive the inspector"
        )


@pytest.mark.asyncio
async def test_the_same_row_highlighted_twice_selects_once():
    """The dedup guard, tested on its own.

    Belt to the focus gate's braces: a background refresh landing on the row the
    user is already sitting in must not re-post.
    """
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode

    pane = MCPToolsMode()
    app = _Harness(pane)
    async with app.run_test(size=(120, 40)) as pilot:
        table = pane.query_one(DataTable)
        row_key = table.add_row("echo", "server-a", key="tool-echo")
        table.focus()
        await pilot.pause()

        for _ in range(4):
            pane.post_message(_click_row(table, row_key))
            await pilot.pause()
        await pilot.pause()

        selected = [m for m in app.captured if type(m).__name__ == "ToolSelected"]
        assert len(selected) == 1, (
            f"the same row selected {len(selected)} times; repeated highlights "
            "of the current row should be ignored"
        )


@pytest.mark.asyncio
async def test_arrowing_then_pressing_enter_selects_once_not_twice():
    """The gesture the highlight-forwarding makes ambiguous.

    Once a highlight selects, a user who arrows onto a row and *then* presses
    Enter would be processed twice — once by the forwarded highlight, once by
    Textual's native `RowSelected`. Harmless for an idempotent handler, wrong
    for one that posts a message: `test_row_selection_posts_entry_selected_with_
    synthetic_index` asserts exactly one event for exactly this.

    Raised by review, and missed by my own run because `-q` suppressed the
    summary lines the failures appeared in.
    """
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode

    pane = MCPToolsMode()
    app = _Harness(pane)
    async with app.run_test(size=(120, 40)) as pilot:
        table = pane.query_one(DataTable)
        table.add_row("echo", "server-a", key="tool-echo")
        row_key = table.add_row("fetch", "server-b", key="tool-fetch")
        table.focus()
        await pilot.pause()

        # Arrow onto the row: the highlight selects it.
        pane.post_message(_click_row(table, row_key))
        await pilot.pause()
        # Then activate it, exactly as `pilot.press("enter")` would.
        pane.post_message(DataTable.RowSelected(table, 1, row_key))
        await pilot.pause()
        await pilot.pause()

        selected = [m for m in app.captured if type(m).__name__ == "ToolSelected"]
        assert len(selected) == 1, (
            f"one gesture produced {len(selected)} selections; the native "
            "activation was not deduped against the highlight that preceded it"
        )


@pytest.mark.asyncio
async def test_reselecting_the_same_row_later_is_not_swallowed():
    """The dedup must be one-shot, scoped to a single gesture.

    A persistent "last selected key" looks equivalent and is not: re-selecting
    the same row later — which `goto_permission_row()` and the workbench's
    sub-view triggers both do — would be silently dropped. That broke
    `test_goto_permission_row_is_the_single_shared_implementation_for_all_three_
    triggers` and `test_both_sub_view_selections_do_not_stack_visible_detail_
    panels`, but only in the full-file run, because those two need a prior test
    to leave the key set. This asserts the property directly so it fails alone.
    """
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode

    pane = MCPToolsMode()
    app = _Harness(pane)
    async with app.run_test(size=(120, 40)) as pilot:
        table = pane.query_one(DataTable)
        row_key = table.add_row("echo", "server-a", key="tool-echo")
        table.focus()
        await pilot.pause()

        # Gesture one: highlight selects, the activation completing it is
        # swallowed.
        pane.post_message(_click_row(table, row_key))
        await pilot.pause()
        pane.post_message(DataTable.RowSelected(table, 0, row_key))
        await pilot.pause()

        # Later, something re-selects the same row outright. That is a new
        # gesture and must reach the pane.
        pane.post_message(DataTable.RowSelected(table, 0, row_key))
        await pilot.pause()
        await pilot.pause()

        selected = [m for m in app.captured if type(m).__name__ == "ToolSelected"]
        assert len(selected) == 2, (
            f"expected the gesture plus the later re-selection, got "
            f"{len(selected)}; the dedup is persistent rather than one-shot"
        )


@pytest.mark.asyncio
async def test_pressing_enter_without_a_preceding_highlight_still_selects():
    """The dedup must not swallow a genuine activation.

    Guards the obvious over-correction: a `RowSelected` for a row the mixin did
    not just forward has to reach the pane.
    """
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode

    pane = MCPToolsMode()
    app = _Harness(pane)
    async with app.run_test(size=(120, 40)) as pilot:
        table = pane.query_one(DataTable)
        row_key = table.add_row("echo", "server-a", key="tool-echo")
        await pilot.pause()

        pane.post_message(DataTable.RowSelected(table, 0, row_key))
        await pilot.pause()
        await pilot.pause()

        selected = [m for m in app.captured if type(m).__name__ == "ToolSelected"]
        assert len(selected) == 1, (
            "a native activation with no preceding highlight was swallowed"
        )


@pytest.mark.asyncio
async def test_keyboard_cursor_movement_selects_the_same_way(monkeypatch):
    """AC#3: arrow keys and a click must agree.

    Both produce `RowHighlighted`, so this asserts the shared mechanism does not
    special-case pointer input.
    """
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode

    pane = MCPToolsMode()
    app = _Harness(pane)
    async with app.run_test(size=(120, 40)) as pilot:
        table = pane.query_one(DataTable)
        table.add_row("echo", "server-a", key="tool-echo")
        second = table.add_row("fetch", "server-b", key="tool-fetch")
        table.focus()
        await pilot.pause()

        pane.post_message(_click_row(table, second))
        await pilot.pause()
        await pilot.pause()

        selected = [m for m in app.captured if type(m).__name__ == "ToolSelected"]
        assert selected, "cursor movement posted no selection"
        assert selected[-1].tool_id == "tool-fetch", (
            "the selection did not follow the cursor onto the second row"
        )
