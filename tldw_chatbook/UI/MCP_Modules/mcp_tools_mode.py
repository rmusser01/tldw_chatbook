# tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py
"""Tools mode canvas: a cross-server tool catalog with filters and a
diagnostic empty state.

`MCPToolsMode` renders whatever `HubTool` list the workbench hands it (T2's
`hub_tool_catalog` derivation) -- it never fetches anything itself. Filtering
by free text and by server is client-side, against a cached copy of the last
full list `update_tools()` was given (`filter_tools()`, also T2), so typing
in the filter Input or picking a server from the Select never round-trips
through the workbench.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Input, Select, Static

from tldw_chatbook.MCP.hub_tool_catalog import HubTool, filter_tools
from tldw_chatbook.UI.MCP_Modules.mcp_schema_form import parse_schema

_TABLE_COLUMNS = ("Tool", "Server", "Tags", "Schema")

# Button copy for the diagnostic empty state, keyed by the workbench's
# `empty_diagnosis` action_key. Falls back to a title-cased version of the
# key itself for anything unrecognized, so an unexpected key still renders
# something clickable rather than a blank button.
_EMPTY_ACTION_LABELS: dict[str, str] = {
    "add_server": "Add server",
    "connect": "Connect a server",
    "refresh": "Refresh",
}

# Tooltip copy for the same button, keyed the same way -- explains the
# outcome of the press (audited by
# test_destination_shells.py::test_destination_action_buttons_explain_their_outcome).
# "connect" and "refresh" share copy: both just hand off to Servers mode,
# same as the EmptyActionRequested handler that routes them.
_EMPTY_ACTION_TOOLTIPS: dict[str, str] = {
    "add_server": "Open the add-server form.",
    "connect": "Go to Servers mode to connect or refresh its tools.",
    "refresh": "Go to Servers mode to connect or refresh its tools.",
}

# One-shot mount-echo consumption sentinel -- mirrors mcp_rail.py's
# `_ECHO_CONSUMED`. Textual 8.2.7 posts a `Select.Changed` for a freshly
# mounted Select's own constructor value as part of mounting it; the filter
# Select is rebuilt (remove + mount, not `set_options()`) every
# `update_tools()` call, so this echo would otherwise re-trigger
# `_apply_filter()` on every background resync even when the user never
# touched the control. `on_select_changed()` compares the incoming event
# against the value the Select was actually (re)mounted with
# (`_displayed_server_value`) to swallow exactly that one echo -- once
# consumed, the sentinel flips to this unique object so a later REAL user
# selection landing back on the same value is never mistaken for a second
# echo.
_ECHO_CONSUMED = object()


class MCPToolsMode(Vertical):
    """Canvas for the Tools mode: cross-server catalog, filters, empty state."""

    DEFAULT_CSS = """
    MCPToolsMode {
        width: 1fr;
        height: 100%;
        min-height: 0;
    }
    #mcp-tools-filter-bar {
        height: auto;
        min-height: 0;
    }
    #mcp-tools-filter-text {
        width: 1fr;
    }
    #mcp-tools-filter-server-slot {
        width: auto;
        height: auto;
    }
    #mcp-tools-filter-server-slot Select {
        width: 28;
    }
    /* T7 (P3 UX batch): same fix as MCPServersMode.DEFAULT_CSS's
    #mcp-servers-table -- height: auto + max-height: 70% instead of height:
    1fr, so the table hugs its own row count instead of ballooning to fill
    the canvas. */
    #mcp-tools-table {
        height: auto;
        max-height: 70%;
        min-height: 4;
    }
    #mcp-tools-empty {
        height: auto;
        min-height: 0;
    }
    """

    class ToolSelected(Message, namespace="mcp_tools_mode"):
        """Posted when a catalog row is selected. `tool_id` is
        `HubTool.tool_id` (`"{server_key}::{name}"`), the DataTable's row key.
        """

        def __init__(self, tool_id: str) -> None:
            super().__init__()
            self.tool_id = tool_id

    class EmptyActionRequested(Message, namespace="mcp_tools_mode"):
        """Posted when the diagnostic empty state's primary Button is
        pressed. `action_key` is whatever `update_tools()`'s
        `empty_diagnosis` supplied (`"add_server"|"connect"|"refresh"`)."""

        def __init__(self, action_key: str) -> None:
            super().__init__()
            self.action_key = action_key

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tools: list[HubTool] = []
        self._filter_text: str = ""
        self._filter_server_key: str | None = None
        self._empty_diagnosis: tuple[str, str] | None = None
        self._empty_action_key: str | None = None
        # Mount-echo guard state for the filter Select -- see _ECHO_CONSUMED.
        self._displayed_server_value: Any = Select.NULL

    def compose(self) -> ComposeResult:
        with Horizontal(id="mcp-tools-filter-bar", classes="ds-toolbar"):
            yield Input(placeholder="Filter tools…", id="mcp-tools-filter-text")
            # The filter Select is mounted dynamically (see
            # `_rebuild_server_select()`) -- its option list depends on the
            # servers actually present in the last `update_tools()` call,
            # which compose() (run once, at construction time) can't know.
            yield Vertical(id="mcp-tools-filter-server-slot")
        table = DataTable(id="mcp-tools-table")
        table.cursor_type = "row"
        yield table
        with Vertical(id="mcp-tools-empty", classes="ds-recovery-callout"):
            yield Static("", id="mcp-tools-empty-message", markup=False)
            yield Button(
                "",
                id="mcp-tools-empty-action",
                classes="console-action-primary",
                compact=True,
            )

    async def on_mount(self) -> None:
        table = self.query_one("#mcp-tools-table", DataTable)
        table.add_columns(*_TABLE_COLUMNS)
        await self._rebuild_server_select()
        self._apply_filter()

    # -- data ---------------------------------------------------------------

    async def update_tools(
        self,
        tools: list[HubTool],
        *,
        empty_diagnosis: tuple[str, str] | None = None,
    ) -> None:
        """Rebuild the catalog from a fresh `HubTool` list.

        Args:
            tools: The full cross-server tool catalog (unfiltered). Cached
                on the widget so subsequent filter Input/Select changes can
                re-filter client-side without another call here.
            empty_diagnosis: `(message, action_key)` to render in the
                diagnostic empty state when `tools` is empty. `action_key` is
                one of `"add_server"|"connect"|"refresh"`. `None` renders a
                generic fallback message with no action button.
        """
        self._tools = list(tools)
        self._empty_diagnosis = empty_diagnosis
        await self._rebuild_server_select()
        self._apply_filter()

    def _server_options(self) -> list[tuple[str, str]]:
        """Unique `(server_label, server_key)` options, one per server
        actually present in the current (unfiltered) tool list, sorted by
        label -- the "All servers" choice is `Select.NULL`, not a row here.
        """
        labels_by_key: dict[str, str] = {}
        for tool in self._tools:
            labels_by_key.setdefault(tool.server_key, tool.server_label)
        return sorted(
            ((label, key) for key, label in labels_by_key.items()),
            key=lambda pair: pair[0],
        )

    async def _rebuild_server_select(self) -> None:
        """Remount `#mcp-tools-filter-server` with the current server list.

        Remove + mount (not `Select.set_options()`) into a dedicated slot
        container, mirroring the awaited remove-then-mount discipline used
        throughout this canvas family (`MCPServersMode._rebuild_detail_
        toolbar()`, `MCPRail`'s per-compose() scope selects) -- simpler to
        reason about here than `set_options()`'s own selection-reset
        semantics, and it reuses the exact mount-echo guard pattern
        `MCPRail` already established for this Textual version.
        """
        options = self._server_options()
        valid_keys = {key for _, key in options}
        if self._filter_server_key not in valid_keys:
            # The previously filtered server is no longer in the catalog
            # (disconnected, deleted) -- fall back to "All servers" rather
            # than keep a dangling filter that would raise
            # InvalidSelectValueError when the Select is constructed below.
            self._filter_server_key = None
        value: Any = (
            self._filter_server_key if self._filter_server_key is not None else Select.NULL
        )
        # Select's `value` is a `var` with `init=False` -- mounting it only
        # actually FIRES a `Changed` echo when the constructor value differs
        # from the var's own default (`Select.NULL`); constructing with
        # `Select.NULL` itself (the common "All servers" case) is a no-op
        # assignment that never echoes at all. Arming the one-shot guard
        # with `Select.NULL` in that case would leave it loaded forever,
        # ready to wrongly swallow the next REAL user selection back to "All
        # servers" (there is no second, later mount to re-arm it) -- so only
        # arm the guard when an echo is actually coming; otherwise mark it
        # pre-consumed.
        self._displayed_server_value = value if value is not Select.NULL else _ECHO_CONSUMED
        slot = self.query_one("#mcp-tools-filter-server-slot", Vertical)
        await slot.remove_children()
        await slot.mount(
            Select(options, id="mcp-tools-filter-server", prompt="All servers", value=value)
        )

    def _apply_filter(self) -> None:
        """Re-render the DataTable from `self._tools` under the current
        text/server filters, and toggle the table/empty-state visibility.

        The diagnostic empty state is driven by whether the catalog has ANY
        tools at all (`self._tools`), not by whether the current filter
        happens to match zero rows -- a text/server filter narrowing to zero
        rows is just an empty table (self-evident), not a "no servers
        configured" diagnosis, which would be actively misleading if servers
        with tools genuinely exist.
        """
        filtered = filter_tools(
            self._tools, server_key=self._filter_server_key, text=self._filter_text
        )
        ordered = sorted(filtered, key=lambda tool: (tool.server_label, tool.name))
        table = self.query_one("#mcp-tools-table", DataTable)
        table.clear()
        for tool in ordered:
            server_cell = f"{tool.server_label} (stale)" if tool.stale else tool.server_label
            tags_cell = ", ".join(tool.tags) if tool.tags else "—"
            schema_cell = "form" if parse_schema(tool.input_schema) is not None else "raw"
            table.add_row(
                Text(tool.name),
                Text(server_cell),
                Text(tags_cell),
                Text(schema_cell),
                key=tool.tool_id,
            )
        has_any_tools = bool(self._tools)
        table.display = has_any_tools
        self._update_empty_state(show=not has_any_tools)

    def _update_empty_state(self, *, show: bool) -> None:
        container = self.query_one("#mcp-tools-empty", Vertical)
        container.display = show
        if not show:
            return
        if self._empty_diagnosis is not None:
            message, action_key = self._empty_diagnosis
        else:
            message, action_key = "No tools available.", None
        self._empty_action_key = action_key
        self.query_one("#mcp-tools-empty-message", Static).update(message)
        button = self.query_one("#mcp-tools-empty-action", Button)
        if action_key is None:
            button.display = False
            # Hidden (no action to take), but still audited by
            # test_destination_action_buttons_explain_their_outcome, which
            # queries every Button regardless of `display` -- keep it a
            # truthy, honest tooltip rather than leaving stale copy behind.
            button.tooltip = "No action available."
        else:
            button.display = True
            button.label = _EMPTY_ACTION_LABELS.get(
                action_key, action_key.replace("_", " ").title()
            )
            button.tooltip = _EMPTY_ACTION_TOOLTIPS.get(
                action_key, f"Go to Servers mode to {action_key.replace('_', ' ')}."
            )

    # -- events ---------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "mcp-tools-filter-text":
            return
        event.stop()
        self._filter_text = event.value
        self._apply_filter()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "mcp-tools-filter-server":
            return
        event.stop()
        if (
            event.value == self._displayed_server_value
            and self._displayed_server_value is not _ECHO_CONSUMED
        ):
            self._displayed_server_value = _ECHO_CONSUMED
            return
        self._filter_server_key = None if event.value is Select.NULL else str(event.value)
        self._apply_filter()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            self.post_message(self.ToolSelected(str(event.row_key.value)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "mcp-tools-empty-action":
            return
        event.stop()
        if self._empty_action_key:
            self.post_message(self.EmptyActionRequested(self._empty_action_key))
