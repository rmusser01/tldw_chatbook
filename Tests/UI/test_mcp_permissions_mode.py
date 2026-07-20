# Tests/UI/test_mcp_permissions_mode.py
from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Static

import tldw_chatbook
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import (
    MCPPermissionsMode,
    PermRow,
    format_tool_state_label,
    state_text,
    tool_state_kind,
)

_CSS_ROOT = Path(tldw_chatbook.__file__).parent / "css"
_AGENTIC_TERMINAL_TCSS = _CSS_ROOT / "components" / "_agentic_terminal.tcss"
_BUNDLED_STYLESHEET = _CSS_ROOT / "tldw_cli_modular.tcss"


def _row(
    *,
    kind: str,
    server_key: str = "",
    server_label: str = "",
    tool_name: str | None = None,
    state_label: str = "Ask",
    tags_label: str = "—",
    cycle_current: str | None = None,
) -> PermRow:
    return PermRow(
        kind=kind,
        server_key=server_key,
        server_label=server_label,
        tool_name=tool_name,
        state_label=state_label,
        tags_label=tags_label,
        cycle_current=cycle_current,
    )


def _global_row(
    *, state_label: str = "Ask", cycle_current: str | None = "ask"
) -> PermRow:
    return _row(kind="global", state_label=state_label, cycle_current=cycle_current)


def _server_row(
    *,
    server_key: str,
    server_label: str,
    state_label: str = "Ask",
    cycle_current: str | None = None,
) -> PermRow:
    return _row(
        kind="server",
        server_key=server_key,
        server_label=server_label,
        state_label=state_label,
        cycle_current=cycle_current,
    )


def _tool_row(
    *,
    server_key: str,
    server_label: str,
    tool_name: str,
    state_label: str = "Ask",
    tags_label: str = "—",
    cycle_current: str | None = None,
) -> PermRow:
    return _row(
        kind="tool",
        server_key=server_key,
        server_label=server_label,
        tool_name=tool_name,
        state_label=state_label,
        tags_label=tags_label,
        cycle_current=cycle_current,
    )


class PermissionsModeApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPPermissionsMode(id="mcp-mode-canvas-permissions")

    def on_mcp_permissions_mode_state_cycle_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_permissions_mode_kill_switch_toggled(self, event) -> None:
        self.events.append(event)

    def on_mcp_permissions_mode_row_selected(self, event) -> None:
        self.events.append(event)


def _row_texts(table: DataTable, row_index: int) -> list[str]:
    row = table.get_row_at(row_index)
    return [cell.plain if hasattr(cell, "plain") else str(cell) for cell in row]


# -- rendering ----------------------------------------------------------


@pytest.mark.asyncio
async def test_rows_render_in_given_order_with_pinned_row_keys():
    """The widget is render-only: it renders `PermRow`s in the exact order
    given (grouping/sorting is the workbench's job) but the ROW KEYS it
    assigns must follow the spec-verbatim formats: `__global__`,
    `__server__::<server_key>`, and a tool's `tool_id`
    (`<server_key>::<name>`).
    """
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(state_label="Ask", cycle_current="ask"),
            _server_row(
                server_key="local:docs", server_label="docs", state_label="Ask"
            ),
            _tool_row(
                server_key="local:docs",
                server_label="docs",
                tool_name="fetch",
                state_label="Ask",
                tags_label="—",
            ),
            _tool_row(
                server_key="local:docs",
                server_label="docs",
                tool_name="search",
                state_label="Allow •",
                tags_label="network",
            ),
            _server_row(
                server_key="local:notes", server_label="notes", state_label="Off •"
            ),
            _tool_row(
                server_key="local:notes",
                server_label="notes",
                tool_name="list_notes",
                state_label="Off •",
            ),
        ]
        await canvas.update_matrix(
            rows, kill_switch=False, preview="Global default: Ask."
        )
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count == 6

        # UX batch item 10: tool-row labels get a two-space indent under
        # their "Server default — X" row; the pinned global/server rows
        # themselves are never indented.
        assert _row_texts(table, 0) == ["Global default", "Ask", "—"]
        assert _row_texts(table, 1) == ["Server default — docs", "Ask", "—"]
        assert _row_texts(table, 2) == ["  fetch", "Ask", "—"]
        assert _row_texts(table, 3) == ["  search", "Allow •", "network"]
        assert _row_texts(table, 4) == ["Server default — notes", "Off •", "—"]
        assert _row_texts(table, 5) == ["  list_notes", "Off •", "—"]

        expected_keys = [
            "__global__",
            "__server__::local:docs",
            "local:docs::fetch",
            "local:docs::search",
            "__server__::local:notes",
            "local:notes::list_notes",
        ]
        for index, expected_key in enumerate(expected_keys):
            row_key, _ = table.coordinate_to_cell_key((index, 0))
            assert row_key.value == expected_key

        assert (
            str(app.query_one("#mcp-perm-preview", Static).renderable)
            == "Global default: Ask."
        )


@pytest.mark.asyncio
async def test_update_matrix_skips_duplicate_row_keys_instead_of_crashing():
    """Minor 7 (DuplicateKey guard parity): two `PermRow`s sharing the same
    identity (same `_row_key()` -- here, two `tool` rows for the same
    `(server_key, tool_name)`) would raise Textual's `DuplicateKey` out of
    `table.add_row()` and crash every future resync of this canvas --
    `MCPToolsMode._apply_filter()` already guards the same crash-loop class
    with its own `seen_keys` skip. `update_matrix()` must render the FIRST
    occurrence and silently skip the rest, not raise.
    """
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _tool_row(
                server_key="local:docs",
                server_label="docs",
                tool_name="search",
                state_label="Ask",
            ),
            _tool_row(
                server_key="local:docs",
                server_label="docs",
                tool_name="search",
                state_label="Allow •",
            ),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count == 2
        # No row in this batch carries a tag -- Tags column omitted (item
        # 11); the surviving tool row is indented (item 10).
        assert _row_texts(table, 1) == ["  search", "Ask"]


@pytest.mark.asyncio
async def test_state_label_renders_verbatim_and_markup_safe():
    """`state_label` may embed a bullet/warning/flag marker already baked in
    by the workbench -- the widget must render it literally as plain `Text`,
    not parse it as Rich markup (a server label a user typed could otherwise
    inject styling)."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(
                server_key="local:[bold red]x",
                server_label="[bold red]x",
                state_label="Ask ⚠",
            ),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        # No row in this batch carries a tag -- Tags column omitted (item 11).
        assert _row_texts(table, 1) == ["Server default — [bold red]x", "Ask ⚠"]


# -- kill switch ----------------------------------------------------------


@pytest.mark.asyncio
async def test_mount_alone_posts_no_kill_switch_event():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.events == []


@pytest.mark.asyncio
async def test_update_matrix_sets_kill_switch_label_without_posting_a_toggle():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix([_global_row()], kill_switch=True, preview="")
        await pilot.pause()
        button = app.query_one("#mcp-perm-kill-switch", Button)
        assert str(button.label) == "block MCP tools in chat: yes ▸"
        assert app.events == []


@pytest.mark.asyncio
async def test_kill_switch_button_default_label_reads_no():
    app = PermissionsModeApp()
    async with app.run_test():
        button = app.query_one("#mcp-perm-kill-switch", Button)
        assert str(button.label) == "block MCP tools in chat: no ▸"


@pytest.mark.asyncio
async def test_user_press_posts_kill_switch_toggled_exactly_once():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix([_global_row()], kill_switch=False, preview="")
        await pilot.pause()
        await pilot.click("#mcp-perm-kill-switch")
        await pilot.pause()
        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPPermissionsMode.KillSwitchToggled)
        assert event.value is True


@pytest.mark.asyncio
async def test_user_press_toggles_off_when_currently_on():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix([_global_row()], kill_switch=True, preview="")
        await pilot.pause()
        await pilot.click("#mcp-perm-kill-switch")
        await pilot.pause()
        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPPermissionsMode.KillSwitchToggled)
        assert event.value is False


# -- Space cycling ----------------------------------------------------------


@pytest.mark.asyncio
async def test_space_on_tool_row_posts_next_state_per_cycle_helper():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(server_key="local:docs", server_label="docs"),
            _tool_row(
                server_key="local:docs",
                server_label="docs",
                tool_name="search",
                cycle_current=None,
            ),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=2)
        await pilot.press("space")
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPPermissionsMode.StateCycleRequested)
        assert event.row_kind == "tool"
        assert event.server_key == "local:docs"
        assert event.tool_name == "search"
        # cycle_ui_state(None) == "allow"
        assert event.new_state == "allow"


@pytest.mark.asyncio
async def test_space_on_server_row_allows_cycling_back_to_inherit():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(
                server_key="local:docs", server_label="docs", cycle_current="deny"
            ),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=1)
        await pilot.press("space")
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert event.row_kind == "server"
        assert event.server_key == "local:docs"
        assert event.tool_name is None
        # cycle_ui_state("deny") == None (Inherit)
        assert event.new_state is None


@pytest.mark.asyncio
async def test_space_on_global_row_never_posts_none():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(
            [_global_row(state_label="Off", cycle_current="deny")],
            kill_switch=False,
            preview="",
        )
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.press("space")
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert event.row_kind == "global"
        assert event.new_state is not None
        # cycle_global("deny") == "allow"
        assert event.new_state == "allow"


@pytest.mark.asyncio
async def test_space_with_no_rows_is_a_noop():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        await pilot.press("space")
        await pilot.pause()
        assert app.events == []


# -- Task 7: row selection ----------------------------------------------


@pytest.mark.asyncio
async def test_enter_on_tool_row_posts_row_selected_with_tool_fields():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(server_key="local:docs", server_label="docs"),
            _tool_row(server_key="local:docs", server_label="docs", tool_name="search"),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=2)
        await pilot.press("enter")
        await pilot.pause()

        events = [
            e for e in app.events if isinstance(e, MCPPermissionsMode.RowSelected)
        ]
        assert len(events) == 1
        assert events[0].row_kind == "tool"
        assert events[0].server_key == "local:docs"
        assert events[0].tool_name == "search"


@pytest.mark.asyncio
async def test_enter_on_global_row_posts_row_selected_with_no_tool_name():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix([_global_row()], kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.press("enter")
        await pilot.pause()

        events = [
            e for e in app.events if isinstance(e, MCPPermissionsMode.RowSelected)
        ]
        assert len(events) == 1
        assert events[0].row_kind == "global"
        assert events[0].tool_name is None


@pytest.mark.asyncio
async def test_enter_on_server_row_posts_row_selected_with_no_tool_name():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(server_key="local:docs", server_label="docs"),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=1)
        await pilot.press("enter")
        await pilot.pause()

        events = [
            e for e in app.events if isinstance(e, MCPPermissionsMode.RowSelected)
        ]
        assert len(events) == 1
        assert events[0].row_kind == "server"
        assert events[0].server_key == "local:docs"
        assert events[0].tool_name is None


# -- T7 (MCP Hub Phase 5): select_tool_row() external-drill entry point -----


@pytest.mark.asyncio
async def test_select_tool_row_moves_cursor_to_matching_tool_row():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(server_key="local:docs", server_label="docs"),
            _tool_row(server_key="local:docs", server_label="docs", tool_name="fetch"),
            _tool_row(server_key="local:docs", server_label="docs", tool_name="search"),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()

        found = canvas.select_tool_row("local:docs", "search")
        await pilot.pause()
        assert found is True
        table = app.query_one("#mcp-perm-table", DataTable)
        row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        assert row_key.value == "local:docs::search"


@pytest.mark.asyncio
async def test_select_tool_row_returns_false_for_tool_not_in_matrix():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix([_global_row()], kill_switch=False, preview="")
        await pilot.pause()
        found = canvas.select_tool_row("local:docs", "gone")
        assert found is False


# -- preview ----------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_text_renders_verbatim():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        preview = "docs-server: 2 allow · 1 ask · 1 off — global default: ask"
        await canvas.update_matrix([_global_row()], kill_switch=False, preview=preview)
        await pilot.pause()
        assert str(app.query_one("#mcp-perm-preview", Static).renderable) == preview


# -- Task 3 (MCP Hub Phase 6): mutation echo ---------------------------------


@pytest.mark.asyncio
async def test_update_matrix_without_echo_renders_preview_unprefixed():
    """`echo` defaults to `None` -- every pre-Task-3 `update_matrix()` call
    site (every full `_sync_children()` pass) must render the preview
    unchanged, exactly as before this parameter existed."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        preview = "global default: ask"
        await canvas.update_matrix([_global_row()], kill_switch=False, preview=preview)
        await pilot.pause()
        assert str(app.query_one("#mcp-perm-preview", Static).renderable) == preview


@pytest.mark.asyncio
async def test_update_matrix_with_echo_prefixes_preview():
    """A standalone mutation resync (Space-cycle/kill-switch/re-allow) passes
    `echo` -- the workbench's own transient "{tool_name} → {ui_label} · "
    (or kill-switch) copy -- and it is prepended verbatim to the preview
    sentence, with no separator of its own (the echo string already carries
    its trailing " · ")."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        preview = "global default: ask"
        await canvas.update_matrix(
            [_global_row()], kill_switch=False, preview=preview, echo="search → Allow · "
        )
        await pilot.pause()
        assert (
            str(app.query_one("#mcp-perm-preview", Static).renderable)
            == "search → Allow · global default: ask"
        )


@pytest.mark.asyncio
async def test_update_matrix_echo_none_clears_a_previously_shown_echo():
    """The NEXT `update_matrix()` call that passes `echo=None` (a full
    resync that isn't itself a standalone mutation resync) must clear
    whatever transient echo a previous call rendered -- no separate "clear"
    call, just the natural next render."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(
            [_global_row()], kill_switch=False, preview="global default: ask",
            echo="search → Allow · ",
        )
        await pilot.pause()
        await canvas.update_matrix(
            [_global_row()], kill_switch=False, preview="global default: ask", echo=None,
        )
        await pilot.pause()
        assert str(app.query_one("#mcp-perm-preview", Static).renderable) == "global default: ask"


# -- UX batch item 4: marker legend ------------------------------------------


@pytest.mark.asyncio
async def test_legend_line_renders_fixed_marker_key():
    """The legend is a fixed, dimmed hint line -- not derived from any
    `PermRow`/preview text -- explaining the matrix's own State-column
    glyphs and giving Space-cycling minimal discoverability."""
    app = PermissionsModeApp()
    async with app.run_test():
        legend = str(app.query_one("#mcp-perm-legend", Static).renderable)
        assert legend == (
            "• override · ⚠ definition changed · ⚑ high-risk floor · "
            "Space cycles Inherit → Allow → Ask → Off"
        )


# -- UX batch item 11: adaptive Tags column ----------------------------------


@pytest.mark.asyncio
async def test_tags_column_omitted_when_no_row_has_tags():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _tool_row(server_key="local:docs", server_label="docs", tool_name="search"),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        columns = [str(col.label) for col in table.ordered_columns]
        assert columns == ["Tool", "State"]


@pytest.mark.asyncio
async def test_tags_column_shown_when_any_row_has_tags():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _tool_row(
                server_key="local:docs",
                server_label="docs",
                tool_name="search",
                tags_label="network",
            ),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        columns = [str(col.label) for col in table.ordered_columns]
        assert columns == ["Tool", "State", "Tags"]


# -- T8: shared state-label rendering helper -----------------------------


def test_format_tool_state_label_marker_precedence():
    """Module-level helper (imported by `MCPWorkbench._tool_state_label` in
    mcp_workbench.py and by `MCPToolsMode`'s own State column) -- pinned
    directly here, independent of how a real `EffectiveToolState` gets
    constructed. Mirrors `test_tool_state_label_marker_precedence` in
    test_mcp_workbench.py, which pins the same behavior through the
    workbench's delegating staticmethod."""
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    assert (
        format_tool_state_label(
            EffectiveToolState(state="allow", origin="tool_override")
        )
        == "Allow •"
    )
    assert (
        format_tool_state_label(
            EffectiveToolState(state="ask", origin="server_default")
        )
        == "Ask"
    )
    assert (
        format_tool_state_label(
            EffectiveToolState(state="ask", origin="global_default")
        )
        == "Ask"
    )
    assert (
        format_tool_state_label(
            EffectiveToolState(state="ask", origin="tool_override", config_changed=True)
        )
        == "Ask ⚠"
    )
    assert (
        format_tool_state_label(
            EffectiveToolState(state="ask", origin="server_default", risk_floored=True)
        )
        == "Ask ⚑"
    )


# -- T8: server-source governance listing (read-only) ---------------------


@pytest.mark.asyncio
async def test_update_server_profiles_renders_pointer_and_profile_names():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_server_profiles(
            [
                {"name": "Docs writers", "id": "prof-1"},
                {"label": "Analysts", "profile_id": "prof-2"},
            ]
        )
        await pilot.pause()

        section = app.query_one("#mcp-perm-server-profiles")
        assert section.display is True

        pointer = str(
            app.query_one("#mcp-perm-server-profiles-pointer", Static).renderable
        )
        assert pointer == (
            "Server-side profiles are managed in the tldw_server webui. The "
            "matrix above is chatbook's client-side gate and still applies."
        )

        rows = [str(s.renderable) for s in app.query(".mcp-perm-server-profile-row")]
        assert rows == ["Docs writers (prof-1)", "Analysts (prof-2)"]


@pytest.mark.asyncio
async def test_update_server_profiles_none_leaves_section_absent():
    """Local/builtin sources (or a guarded fetch failure) pass `None` --
    the section is entirely absent, not merely hidden."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_server_profiles(None)
        await pilot.pause()
        assert len(app.query("#mcp-perm-server-profiles")) == 0


@pytest.mark.asyncio
async def test_update_server_profiles_empty_list_still_shows_pointer_with_no_rows():
    """Server source, fetch succeeded, zero profiles configured -- a
    distinct case from `None`: the section (and its pointer text) still
    renders, just with no profile rows."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_server_profiles([])
        await pilot.pause()
        assert len(app.query("#mcp-perm-server-profiles")) == 1
        assert len(app.query(".mcp-perm-server-profile-row")) == 0


@pytest.mark.asyncio
async def test_update_server_profiles_defensive_reads_handle_missing_and_malformed_entries():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_server_profiles(
            [
                {"name": "Only a name"},
                {"id": "prof-only-id"},
                {},
                "not-a-dict",
                42,
            ]
        )
        await pilot.pause()
        rows = [str(s.renderable) for s in app.query(".mcp-perm-server-profile-row")]
        # Non-dict entries are skipped entirely, mirroring
        # `server_tools_from_inventory()`'s own defensive style.
        assert rows == ["Only a name", "prof-only-id", "Unnamed profile"]


@pytest.mark.asyncio
async def test_update_server_profiles_transitions_from_present_to_absent():
    """A source switch (server -> local) must tear the whole section back
    down, not just clear its rows -- `update_server_profiles(None)` after a
    populated call removes the section entirely."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_server_profiles([{"name": "Docs writers", "id": "prof-1"}])
        await pilot.pause()
        assert len(app.query("#mcp-perm-server-profiles")) == 1

        await canvas.update_server_profiles(None)
        await pilot.pause()
        assert len(app.query("#mcp-perm-server-profiles")) == 0


@pytest.mark.asyncio
async def test_update_server_profiles_markup_safe():
    """Raw profile name/id text must render literally, not parsed as Rich
    markup -- same rationale as `test_state_label_renders_verbatim_and_
    markup_safe` above (a server-supplied name could otherwise inject
    styling)."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_server_profiles([{"name": "[bold red]x", "id": "p1"}])
        await pilot.pause()
        rows = [str(s.renderable) for s in app.query(".mcp-perm-server-profile-row")]
        assert rows == ["[bold red]x (p1)"]


# -- T9: bundle-parity (dual-layer CSS) -----------------------------------


def test_perm_table_height_rule_pinned_in_bundle_source_and_bundle() -> None:
    """`MCPPermissionsMode.DEFAULT_CSS` gives `#mcp-perm-table` the same
    `height: auto; max-height: 70%;` discipline T7 (P3 UX batch) gave
    `#mcp-tools-table`/`#mcp-servers-table` -- so it hugs its own row count
    instead of ballooning to fill the canvas and stranding the
    policy-preview strip below it. Pins the matching bundle-layer copy
    (added in lockstep, T9) in both the bundle-source file and the
    generated bundle (`tldw_cli_modular.tcss`) -- the latter also proves
    `build_css.py` was re-run after the source edit, mirroring
    `test_tools_table_height_rule_pinned_in_bundle_source_and_bundle` in
    test_mcp_tools_mode.py and `test_servers_table_height_rule_pinned_in_
    bundle_source_and_bundle` in test_mcp_servers_mode.py."""
    agentic_terminal = _AGENTIC_TERMINAL_TCSS.read_text(encoding="utf-8")
    bundled_stylesheet = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text, label in (
        (agentic_terminal, "_agentic_terminal.tcss"),
        (bundled_stylesheet, "tldw_cli_modular.tcss"),
    ):
        selector = "#mcp-perm-table {"
        start = text.find(selector)
        assert start != -1, f"{label} is missing {selector!r}"
        end = text.find("}", start)
        block = text[start:end]
        assert "height: auto;" in block, (
            f"{label}'s {selector!r} block is missing 'height: auto;'"
        )
        assert "max-height: 70%;" in block, (
            f"{label}'s {selector!r} block is missing 'max-height: 70%;'"
        )


def test_perm_preview_height_rule_pinned_in_bundle_source_and_bundle() -> None:
    """`MCPPermissionsMode.DEFAULT_CSS` gives `#mcp-perm-preview`
    `height: auto;` so the policy-preview Static hugs its own (one- or
    two-sentence) content instead of competing with the matrix table above
    for the canvas's remaining space. Pins the matching bundle-layer copy
    (T9) in both layers -- same rationale as `test_perm_table_height_rule_
    pinned_in_bundle_source_and_bundle` above."""
    agentic_terminal = _AGENTIC_TERMINAL_TCSS.read_text(encoding="utf-8")
    bundled_stylesheet = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text, label in (
        (agentic_terminal, "_agentic_terminal.tcss"),
        (bundled_stylesheet, "tldw_cli_modular.tcss"),
    ):
        selector = "#mcp-perm-preview {"
        start = text.find(selector)
        assert start != -1, f"{label} is missing {selector!r}"
        end = text.find("}", start)
        block = text[start:end]
        assert "height: auto;" in block, (
            f"{label}'s {selector!r} block is missing 'height: auto;'"
        )


# UX batch items 2+3: the kill-switch Checkbox is gone -- the whole
# defensive-bundle-rule class `test_kill_switch_height_rule_pinned_in_
# bundle_source_and_bundle` existed for (a bare, unscoped `Checkbox {
# height: 2; }` rule elsewhere in the bundle clobbering a compact
# Checkbox's own focus-border DEFAULT_CSS) is moot for a Button, which
# needs no such floor -- the rule itself was removed from
# `_agentic_terminal.tcss`, so this pinned-bundle test is retired along
# with it rather than repointed at a Button that doesn't need the guard.


# -- real bundled CSS (Phase 3 lesson: DEFAULT_CSS alone can still collapse
# to 0x0 under the real app stylesheet's global widget rules) ---------------


class PermissionsModeAppWithBundledCSS(App):
    """Mirrors `ToolsModeAppWithBundledCSS` (test_mcp_tools_mode.py) --
    loads the real generated bundle as CSS_PATH so the matrix table and
    kill-switch row contest their actual CSS priority battle exactly as
    they do in the live app."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        yield MCPPermissionsMode(id="mcp-mode-canvas-permissions")


@pytest.mark.asyncio
async def test_matrix_and_kill_switch_have_nonzero_geometry_with_bundled_css():
    app = PermissionsModeAppWithBundledCSS()
    async with app.run_test(size=(120, 40)) as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(server_key="local:docs", server_label="docs"),
            _tool_row(server_key="local:docs", server_label="docs", tool_name="search"),
        ]
        await canvas.update_matrix(
            rows, kill_switch=True, preview="Global default: Ask."
        )
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.size.width > 0, (
            "matrix table collapsed to zero width under bundled CSS"
        )
        assert table.size.height > 0, (
            "matrix table collapsed to zero height under bundled CSS"
        )

        # UX batch items 2+3: the kill switch is now a Button, not a
        # Checkbox -- it needs no `min-height: 3` floor (the defect class
        # T6/T9 defended against was specific to a compact Checkbox's own
        # focus-border redraw), so this only checks ordinary non-zero
        # geometry, same shape as every other Button check in this suite.
        kill_button = app.query_one("#mcp-perm-kill-switch", Button)
        assert kill_button.size.width > 0, (
            "kill-switch button collapsed to zero width under bundled CSS"
        )
        assert kill_button.size.height > 0, (
            "kill-switch button collapsed to zero height under bundled CSS"
        )

        # T8's server-source governance listing is its own dedicated slot
        # (`#mcp-perm-server-profiles-slot`, `height: auto; min-height: 0;`)
        # -- exercise it under the real bundle too, same rationale as the
        # matrix table/kill-switch checks above (Phase 3 lesson: DEFAULT_CSS
        # alone can still collapse to 0x0 once the bundle's global widget
        # rules are also in play).
        await canvas.update_server_profiles([{"name": "Docs writers", "id": "prof-1"}])
        await pilot.pause()

        section = app.query_one("#mcp-perm-server-profiles")
        assert section.size.width > 0, (
            "server-profiles section collapsed to zero width under bundled CSS"
        )
        assert section.size.height > 0, (
            "server-profiles section collapsed to zero height under bundled CSS"
        )

        pointer = app.query_one("#mcp-perm-server-profiles-pointer", Static)
        assert pointer.size.width > 0, (
            "server-profiles pointer collapsed to zero width under bundled CSS"
        )
        assert pointer.size.height > 0, (
            "server-profiles pointer collapsed to zero height under bundled CSS"
        )
        # Re-check after the server-profiles mount above.
        assert kill_button.size.height > 0, (
            "kill-switch button collapsed to zero height under bundled CSS after the "
            "server-profiles mount"
        )


@pytest.mark.asyncio
async def test_filter_text_input_has_nonzero_geometry_with_bundled_css():
    """Task 6 (MCP Hub Phase 6) dual-layer CSS audit: `#mcp-perm-filter-
    text`'s own DEFAULT_CSS comment (Task 4) explicitly deferred verifying
    it against the REAL bundled stylesheet to this task ("T6 note") rather
    than assuming Input's own sane defaults survive the bundle's global
    widget rules -- same Phase 3 lesson as the matrix table/kill-switch
    checks above. No bare `Input { width/height: ... }` rule exists
    anywhere in the bundle (grepped) to contest it, and this test confirms
    that empirically rather than by inspection: a non-zero, single-line-
    bordered (height 3) box, not a 0x0 collapse. No bundle-layer rule was
    added for it -- this test is the verification T4 asked for, not a fix."""
    app = PermissionsModeAppWithBundledCSS()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        filter_input = app.query_one("#mcp-perm-filter-text", Input)
        assert filter_input.size.width > 0, "filter Input collapsed to zero width under bundled CSS"
        assert filter_input.size.height > 0, "filter Input collapsed to zero height under bundled CSS"
        # The single-line bordered box Input's own DEFAULT_CSS/comment
        # promises (region includes the border, hence 3 not 1).
        assert filter_input.region.height == 3, (
            f"expected the bordered single-line Input's 3-row region, got "
            f"{filter_input.region.height}"
        )


# -- Task 1 (MCP Hub Phase 6): semantic state colors ------------------------


def test_state_text_styles_each_kind_and_falls_back_safely():
    """`state_text()`'s own contract: a concrete, kind-specific Rich style,
    verbatim label text (no markup parsing), and a safe fallback (no style
    at all, not a crash) for an unrecognized kind."""
    assert state_text("Allow", "ready").style == "green"
    assert state_text("Ask", "warning").style == "yellow"
    assert state_text("Off", "error").style == "red"
    assert state_text("Checking", "info").style == "cyan"
    assert state_text("—", "muted").style == "dim"
    assert state_text("Allow", "not-a-real-kind").style == ""
    # Markup-unsafe input must render literally, exactly like every other
    # `Text(...)` cell in this canvas family (see
    # `test_state_label_renders_verbatim_and_markup_safe` above) -- the
    # style parameter does not change that discipline.
    unsafe = state_text("[bold red]x[/bold red]", "ready")
    assert unsafe.plain == "[bold red]x[/bold red]"


def test_tool_state_kind_maps_allow_ask_deny_to_ready_warning_error():
    """`tool_state_kind()` keys purely off `EffectiveToolState.state` --
    `config_changed`/`risk_floored` both already resolve `state` to `"ask"`
    themselves (see `permission_store.resolve_effective_state()`), so a
    downgraded verdict still lands on `warning`, matching its `ui_label`
    ("Ask ⚠"/"Ask ⚑")."""
    assert tool_state_kind(EffectiveToolState(state="allow", origin="tool_override")) == "ready"
    assert tool_state_kind(EffectiveToolState(state="ask", origin="global_default")) == "warning"
    assert tool_state_kind(EffectiveToolState(state="deny", origin="tool_override")) == "error"
    assert (
        tool_state_kind(
            EffectiveToolState(state="ask", origin="tool_override", config_changed=True)
        )
        == "warning"
    )
    assert (
        tool_state_kind(
            EffectiveToolState(state="ask", origin="server_default", risk_floored=True)
        )
        == "warning"
    )


@pytest.mark.asyncio
async def test_state_column_cells_carry_semantic_color_by_resolved_word():
    """The matrix's State cell is now colored by the resolved verdict its
    own leading word names -- Allow -> ready/green, Ask -> warning/amber,
    Off -> error/red -- regardless of row kind (global/server/tool) or
    whatever origin marker (·/⚠/⚑) trails it. A DataTable cell can't carry a
    CSS class, so `state_text()`'s Rich `Text.style` is the only per-cell
    coloring mechanism available here (see that helper's own docstring)."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(state_label="Allow", cycle_current="allow"),
            _server_row(server_key="local:docs", server_label="docs", state_label="Ask"),
            _tool_row(
                server_key="local:docs", server_label="docs", tool_name="search",
                state_label="Off •",
            ),
            _tool_row(
                server_key="local:docs", server_label="docs", tool_name="rug_pulled",
                state_label="Ask ⚠",
            ),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.get_cell_at((0, 1)).style == state_text("Allow", "ready").style
        assert table.get_cell_at((1, 1)).style == state_text("Ask", "warning").style
        assert table.get_cell_at((2, 1)).style == state_text("Off •", "error").style
        assert table.get_cell_at((3, 1)).style == state_text("Ask ⚠", "warning").style
        # The marker glyph is still part of the SAME plain text -- coloring
        # never strips or re-derives it.
        assert str(table.get_cell_at((2, 1))) == "Off •"
        assert str(table.get_cell_at((3, 1))) == "Ask ⚠"


# -- Task 4 (MCP Hub Phase 6): permissions matrix text filter ---------------


def _two_server_rows() -> list[PermRow]:
    """Fixture matrix shared by the filter tests below: two servers, each
    with two tools, one of which is named "beta" so a "beta" filter spans
    both servers."""
    return [
        _global_row(state_label="Ask", cycle_current="ask"),
        _server_row(server_key="local:docs", server_label="docs", state_label="Ask"),
        _tool_row(
            server_key="local:docs", server_label="docs", tool_name="alpha_tool",
            state_label="Ask",
        ),
        _tool_row(
            server_key="local:docs", server_label="docs", tool_name="beta_tool",
            state_label="Allow •",
        ),
        _server_row(server_key="local:notes", server_label="notes", state_label="Off •"),
        _tool_row(
            server_key="local:notes", server_label="notes", tool_name="beta_other",
            state_label="Off •",
        ),
        _tool_row(
            server_key="local:notes", server_label="notes", tool_name="gamma_tool",
            state_label="Off •",
        ),
    ]


async def _type_filter(pilot, text: str) -> None:
    filter_input = pilot.app.query_one("#mcp-perm-filter-text", Input)
    filter_input.value = text
    await pilot.pause()


@pytest.mark.asyncio
async def test_filter_input_is_present_above_the_matrix_with_placeholder():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        filter_input = app.query_one("#mcp-perm-filter-text", Input)
        assert filter_input.placeholder == "filter tools"


@pytest.mark.asyncio
async def test_empty_filter_shows_every_row():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count == 7


@pytest.mark.asyncio
async def test_filter_narrows_to_matching_tool_rows_and_hides_unrelated_pinned_rows():
    """Task 4: a filter matching exactly one tool must still show the
    global row and that tool's OWN server-default row (its server has >=1
    visible tool row), but hides the OTHER server's default row and tools
    entirely -- none of them match and that server has zero visible tools
    left."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        await _type_filter(pilot, "alpha")

        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count == 3
        # No row in this fixture carries a tag -- Tags column omitted (UX
        # batch item 11), same as every other tagless-batch test above.
        assert _row_texts(table, 0) == ["Global default", "Ask"]
        assert _row_texts(table, 1) == ["Server default — docs", "Ask"]
        assert _row_texts(table, 2) == ["  alpha_tool", "Ask"]

        rendered_keys = {
            table.coordinate_to_cell_key((i, 0))[0].value for i in range(table.row_count)
        }
        assert "__server__::local:notes" not in rendered_keys
        assert "local:notes::beta_other" not in rendered_keys
        assert "local:notes::gamma_tool" not in rendered_keys
        assert "local:docs::beta_tool" not in rendered_keys


@pytest.mark.asyncio
async def test_filter_matches_server_label_reveals_every_tool_under_that_server():
    """Filter text matching a TOOL row's `server_label` field (not just its
    own name) is a match too -- so a filter equal to a server's label
    reveals every tool under that server, per the spec's "name +
    server_label" match fields."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        await _type_filter(pilot, "docs")

        table = app.query_one("#mcp-perm-table", DataTable)
        # global + docs server-default + alpha_tool + beta_tool
        assert table.row_count == 4
        rendered_keys = [
            table.coordinate_to_cell_key((i, 0))[0].value for i in range(table.row_count)
        ]
        assert rendered_keys == [
            "__global__",
            "__server__::local:docs",
            "local:docs::alpha_tool",
            "local:docs::beta_tool",
        ]


@pytest.mark.asyncio
async def test_filter_is_case_insensitive():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        await _type_filter(pilot, "ALPHA")
        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count == 3


@pytest.mark.asyncio
async def test_filter_matching_across_two_servers_keeps_both_pinned_server_rows():
    """A filter that matches a tool in EACH server keeps both servers'
    pinned default rows (plus global) and both matching tools -- the
    non-matching third tool ("gamma_tool") is dropped."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        await _type_filter(pilot, "beta")

        table = app.query_one("#mcp-perm-table", DataTable)
        rendered_keys = [
            table.coordinate_to_cell_key((i, 0))[0].value for i in range(table.row_count)
        ]
        assert rendered_keys == [
            "__global__",
            "__server__::local:docs",
            "local:docs::beta_tool",
            "__server__::local:notes",
            "local:notes::beta_other",
        ]


@pytest.mark.asyncio
async def test_clearing_the_filter_after_narrowing_restores_every_row():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        await _type_filter(pilot, "alpha")
        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count == 3
        await _type_filter(pilot, "")
        assert table.row_count == 7


@pytest.mark.asyncio
async def test_cursor_row_key_survives_a_refilter_that_still_shows_it():
    """Task 4: typing into the filter re-renders the table (`DataTable.
    clear()` unconditionally resets the cursor to (0, 0)) -- the cursor's
    ROW KEY, not its numeric position, must survive that rebuild exactly
    like an ordinary `update_matrix()` resync already preserves it (Minor
    7 precedent). Before the filter, "beta_tool" sits at unfiltered index
    3; after filtering to "beta", it moves to index 2 -- proving the
    restore is key-based, not a stale index carried over.
    """
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # "  beta_tool" in the unfiltered matrix
        row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        assert row_key.value == "local:docs::beta_tool"

        await _type_filter(pilot, "beta")

        assert table.cursor_row == 2
        row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        assert row_key.value == "local:docs::beta_tool"


@pytest.mark.asyncio
async def test_cursor_falls_back_to_row_zero_when_its_row_is_filtered_out():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=6)  # "  gamma_tool" -- will be filtered out below
        await _type_filter(pilot, "alpha")
        assert table.cursor_row == 0


@pytest.mark.asyncio
async def test_space_after_filter_cycles_the_row_actually_under_the_cursor():
    """Task 4: Space must resolve whatever row is ACTUALLY rendered at the
    cursor's current table position post-filter, not some row that would
    have been there before filtering narrowed the table. Filtering to
    "beta" renders (global, docs-default, beta_tool, notes-default,
    beta_other) -- moving the cursor to the filtered table's row 4 must
    cycle "beta_other" (server "local:notes"), which sat at unfiltered
    index 5, not whatever row 4 held in the ORIGINAL unfiltered matrix
    (there, index 4 was the notes server-default row).
    """
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        await _type_filter(pilot, "beta")

        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count == 5
        table.focus()
        table.move_cursor(row=4)
        await pilot.press("space")
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPPermissionsMode.StateCycleRequested)
        assert event.row_kind == "tool"
        assert event.server_key == "local:notes"
        assert event.tool_name == "beta_other"


@pytest.mark.asyncio
async def test_echo_prefixed_preview_survives_a_refilter():
    """T3's mutation echo is part of the preview Static's own text --
    typing into the filter only re-renders the TABLE, never touches
    `#mcp-perm-preview`, so a previously shown echo must still read
    verbatim after the user narrows the matrix with a filter."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(
            _two_server_rows(), kill_switch=False, preview="global default: ask",
            echo="beta_tool → Allow · ",
        )
        await pilot.pause()
        await _type_filter(pilot, "alpha")

        preview = str(app.query_one("#mcp-perm-preview", Static).renderable)
        assert preview == "beta_tool → Allow · global default: ask"


@pytest.mark.asyncio
async def test_filter_persists_across_a_full_matrix_resync():
    """A background `update_matrix()` resync (e.g. after a Space-cycle
    elsewhere resolves a fresh matrix) must not silently clear whatever
    filter text the user was mid-typing -- mirrors `MCPToolsMode`'s own
    cached-filter-survives-`update_tools()` contract."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        await _type_filter(pilot, "alpha")
        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count == 3

        # A second update_matrix() call (a fresh resync) with the SAME
        # rows -- the filter Input's own value is untouched by this call.
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        assert table.row_count == 3
        filter_input = app.query_one("#mcp-perm-filter-text", Input)
        assert filter_input.value == "alpha"


@pytest.mark.asyncio
async def test_select_tool_row_clears_an_active_filter_to_reveal_a_hidden_target():
    """T7's external-drill entry point must not be silently defeated by an
    active filter typed by the user in the meantime -- mirrors
    `MCPToolsMode.select_tool_row()`'s own "an external drill's target row
    must never be swallowed by an active filter" discipline."""
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()
        await _type_filter(pilot, "alpha")
        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count == 3

        found = canvas.select_tool_row("local:notes", "gamma_tool")
        await pilot.pause()

        assert found is True
        assert table.row_count == 7
        row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        assert row_key.value == "local:notes::gamma_tool"
        filter_input = app.query_one("#mcp-perm-filter-text", Input)
        assert filter_input.value == ""


# -- real bundled CSS: filter Input geometry ---------------------------------


@pytest.mark.asyncio
async def test_filter_input_has_nonzero_geometry_with_bundled_css():
    app = PermissionsModeAppWithBundledCSS()
    async with app.run_test(size=(120, 40)) as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        await canvas.update_matrix(_two_server_rows(), kill_switch=False, preview="")
        await pilot.pause()

        filter_input = app.query_one("#mcp-perm-filter-text", Input)
        assert filter_input.outer_size.width > 0, (
            "filter Input collapsed to zero width under bundled CSS"
        )
        assert filter_input.outer_size.height > 0, (
            "filter Input collapsed to zero height under bundled CSS"
        )
