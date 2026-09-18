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

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click, DescendantFocus, Resize
from textual.message import Message
from textual.widgets import Button, DataTable, Input, OptionList, Select, Static
from textual.widgets._select import SelectCurrent, SelectOverlay
from textual.widgets.data_table import RowDoesNotExist

from tldw_chatbook.MCP.hub_tool_catalog import HubTool, filter_tools
from tldw_chatbook.MCP.local_config_saves import ConfigSaveState
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.UI.MCP_Modules.mcp_local_master_button import MCPLocalMasterButton
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import (
    format_tool_state_label,
    state_text,
    tool_state_kind,
)
from tldw_chatbook.UI.MCP_Modules.mcp_schema_form import parse_schema
from tldw_chatbook.UI.Widgets.table_click_select import DataTableClickSelectMixin

_TABLE_COLUMNS = ("Tool", "State", "Server", "Tags", "Schema")
# UX batch item 11: the Tags column is omitted entirely when no tool in the
# current (unfiltered) catalog carries a real tag -- mirrors
# `mcp_servers_mode.py`'s own per-source `_TABLE_COLUMNS_NO_SCOPE`
# precedent (Task 11 there) and `mcp_permissions_mode.py`'s matching
# `_TABLE_COLUMNS_NO_TAGS` (same UX batch item, sibling table).
_TABLE_COLUMNS_NO_TAGS = ("Tool", "State", "Server", "Schema")

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

# task-32286: the master control's title text, reused by the toggle Button's
# label (see `_local_tools_toggle_label()`) so the two never drift apart.
_LOCAL_TOOLS_TITLE = "Local workspace, web, and Watchlists tools"


def _local_tools_toggle_label(enabled: bool) -> str:
    """Render the local-tools master switch Button's label.

    task-32286: a Checkbox + separate "Enabled"/"Disabled" Static used to
    render this row -- the bundle's `MCPToolsMode #mcp-tools-local-enabled
    { width: 8; }` escape hatch (needed back when an app-wide unscoped
    `Checkbox { width: 100%; height: 2; }` rule, since retired in
    TASK-18960, would have collapsed it otherwise) clamped the Checkbox to
    a bordered 7-cell frame, truncating its own "On" label to a lone "…".
    Reusing `mcp_servers_mode._gate_button()`'s toggle-Button idiom (same
    `[console] local_tools_enabled` gate, surfaced a second time in the
    Servers-mode Tool gates group as of task-32284) retires the escape
    hatch entirely -- a Button sizes to its own label (`width: auto`) --
    and spells the state out in text rather than a glyph that looks
    identical in both states.
    """
    return f"{_LOCAL_TOOLS_TITLE}: {'on' if enabled else 'off'} ▸"


# Wave A (F9): rendered-width cap for the Server column. DataTable clips
# overflowing cell text silently -- a group label like "Local workspace,
# web, and Watchlists" rendered as "Local workspace, web, and" with no
# marker, which reads as a DIFFERENT label rather than a truncated one.
# A fixed budget with an explicit ellipsis keeps the truncation honest;
# the full label remains visible in the server-filter Select's options.
_SERVER_CELL_BUDGET = 24


def _ellipsize(text: str, budget: int) -> str:
    """Truncate `text` to `budget` rendered COLUMNS with an explicit "…".

    Qodo #2620 #7: server labels are user-controlled and may contain wide
    Unicode -- measuring/slicing by Python character count lets a
    24-character label occupy far more than 24 cells and bypass the
    budget entirely. Rich's `Text.truncate` measures and cuts by cell
    width (Rich is already a hard dependency of every Textual app).
    """
    rendered = Text(text)
    # Rich's Text.truncate mutates in place and returns None on this
    # version -- read .plain back off the object.
    rendered.truncate(budget, overflow="ellipsis", pad=False)
    return rendered.plain


class _MCPToolsServerOverlay(SelectOverlay):
    """Commit menu activation before queued indices can outlive the options."""

    def post_message(self, message: Message) -> bool:
        if isinstance(message, Click):
            index = message.style.meta.get("option")
            if isinstance(index, int):
                # Pointer style metadata also carries an index. Apply the
                # stock click action at admission, before it can queue across
                # an option replacement (including removal of the last row).
                if (
                    0 <= index < len(self._options)
                    and not self._options[index].disabled
                ):
                    self.highlighted = index
                    self.action_select()
                return True
        if (
            isinstance(message, OptionList.OptionSelected)
            and message.option_list is self
        ):
            select = self.parent
            if isinstance(select, MCPToolsServerSelect) and self.is_attached:
                # Textual normally queues OptionSelected, then UpdateSelection.
                # Both carry indices. Admit this real gesture synchronously so
                # set_options cannot reinterpret it between either queue stage.
                select._update_selection(self.UpdateSelection(message.option_index))
            return True
        return super().post_message(message)


class MCPToolsServerSelect(Select):
    """A catalog filter whose committed choice survives background refresh."""

    def compose(self) -> ComposeResult:
        yield SelectCurrent(self.prompt)
        yield _MCPToolsServerOverlay(type_to_search=self._type_to_search).data_bind(
            compact=Select.compact
        )


class MCPToolsTable(DataTable):
    """Report the final catalog viewport, including parent scrollbar changes."""

    class Resized(Message, namespace="mcp_tools_table"):
        """The table's geometry changed independently of its outer canvas."""

    def on_resize(self, event: Resize) -> None:
        self.post_message(self.Resized())


class MCPToolsMode(DataTableClickSelectMixin, VerticalScroll):
    """Canvas for the Tools mode: cross-server catalog, filters, empty state."""

    BUNDLED_CSS = """
    MCPToolsMode {
        width: 1fr;
        height: 100%;
        min-height: 0;
    }
    #mcp-tools-filter-bar {
        height: auto;
        min-height: 0;
    }
    #mcp-tools-local-config {
        height: auto;
        min-height: 0;
        padding: 0 1 1 1;
        background: $surface;
    }
    #mcp-tools-workspace-row {
        height: auto;
        min-height: 0;
    }
    #mcp-tools-local-config-help,
    #mcp-tools-local-config-status {
        height: auto;
        color: $text-muted;
    }
    #mcp-tools-local-config-status.is-error {
        color: $error;
        text-style: bold;
    }
    #mcp-tools-workspace-root {
        width: 1fr;
    }
    #mcp-tools-workspace-save {
        width: 14;
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
    /* T7 (P3 UX batch): same fix as MCPServersMode.BUNDLED_CSS's
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

    class LocalToolsEnabledChanged(Message, namespace="mcp_tools_mode"):
        """Persist the workspace, web, and Watchlists provider master switch."""

        def __init__(self, enabled: bool, *, config_path: Path | None = None) -> None:
            super().__init__()
            self.enabled = enabled
            self.config_path = config_path

    class WorkspaceRootSaveRequested(Message, namespace="mcp_tools_mode"):
        """Request validation and persistence of the workspace root."""

        def __init__(
            self,
            workspace_root: str,
            *,
            draft_identity: tuple[object, int] | None = None,
            config_path: Path | None = None,
        ) -> None:
            super().__init__()
            self.workspace_root = workspace_root
            self.draft_identity = draft_identity
            self.config_path = config_path

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tools: list[HubTool] = []
        # T8: the last `states` dict `update_tools()` was given, cached
        # alongside `self._tools` -- `_apply_filter()` re-renders client-
        # side on every filter Input/Select change without another
        # `update_tools()` call, so the State column has to survive that
        # same re-render loop rather than being consumed once.
        self._states: dict[tuple[str, str], EffectiveToolState] = {}
        self._filter_text: str = ""
        self._filter_server_key: str | None = None
        # task-32283: the RAIL's selected server (not the filter Select --
        # that one is `_filter_server_key`). Its group is ordered first in
        # `_apply_filter()` so drilling into a server whose label sorts
        # late can't leave its rows below the fold.
        self._selected_server_key: str | None = None
        self._empty_diagnosis: tuple[str, str] | None = None
        self._empty_action_key: str | None = None
        # UX batch item 11: whether ANY tool in the current (unfiltered)
        # catalog carries a real tag -- set once per `update_tools()` call
        # (a "rebuild", per the item's own wording) and reused by
        # `_apply_filter()` so the Tags column doesn't flicker in/out as a
        # text/server filter narrows the visible rows to a tagless subset.
        self._has_tags: bool = False
        self._tool_column_width: int | None = None
        self._server_filter_options: list[tuple[str, str]] = []
        # task-32286: the last `enabled` value `update_local_config()` was
        # given -- the toggle Button posts the OPPOSITE of this on press
        # (mirrors `mcp_servers_mode._tool_gates_by_id`'s same read-not-
        # widget-state pattern; a Button carries no `.value` of its own).
        self._local_tools_enabled: bool = False
        self.submit_local_master: Callable[[bool, Path | None], None] | None = None
        self._workspace_root_saved: str | None = None
        self._workspace_root_draft = ""
        self._workspace_root_dirty = False
        self._workspace_root_revision = 0
        self._workspace_root_origin = object()
        self._workspace_root_config: Path | None = None
        self._workspace_root_pending = False

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-tools-local-config"):
            yield MCPLocalMasterButton(
                _local_tools_toggle_label(False),
                id="mcp-tools-local-enabled",
                classes="console-action-secondary",
                compact=True,
                tooltip=(
                    "Toggle the workspace, web, and Watchlists tool master "
                    "switch. Calls still follow Ask, Allow, or Off permissions."
                ),
            )
            yield Static(
                "",
                id="mcp-tools-local-config-status",
                classes="h-auto ds-text-muted",
                markup=False,
            )
            yield Static(
                "Available by default. Calls still follow Ask, Allow, or Off permissions.",
                id="mcp-tools-local-config-help",
                markup=False,
            )
            with Horizontal(id="mcp-tools-workspace-row"):
                yield Input(
                    placeholder="Local MCP / Hub test root",
                    id="mcp-tools-workspace-root",
                )
                yield Button(
                    "Save root",
                    id="mcp-tools-workspace-save",
                    classes="console-action-primary",
                    compact=True,
                    tooltip=(
                        "Save the root for local MCP serving and Hub tool tests. "
                        "Blank uses the serving process's current folder."
                    ),
                )
            yield Static(
                "",
                id="mcp-tools-workspace-status",
                classes="h-auto ds-text-muted",
                markup=False,
            )
            yield Static(
                "Root: local MCP serving and Hub tests. Blank uses the serving "
                "process's current folder. Console uses Chat scratch and "
                "admitted Workspace folders.",
                id="mcp-tools-workspace-help",
                classes="h-auto ds-text-muted",
                markup=False,
            )
        with Horizontal(id="mcp-tools-filter-bar", classes="ds-toolbar"):
            yield Input(placeholder="Filter tools…", id="mcp-tools-filter-text")
            with Vertical(id="mcp-tools-filter-server-slot"):
                yield MCPToolsServerSelect(
                    [], id="mcp-tools-filter-server", prompt="All servers"
                )
        table = MCPToolsTable(id="mcp-tools-table")
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
        await self._sync_server_select()
        self._apply_filter()

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        self.call_after_refresh(self.reveal_focused_control)

    def on_resize(self, event: Resize) -> None:
        self.call_after_refresh(self._reflow_table)
        self.call_after_refresh(self.reveal_focused_control)

    def on_mcp_tools_table_resized(self, event: MCPToolsTable.Resized) -> None:
        event.stop()
        self.call_after_refresh(self._reflow_table)

    def _measured_tool_width(self, table: DataTable) -> int | None:
        """Reserve readable State cells before allowing Tool names to wrap."""
        if table.content_region.width <= 0:
            return None
        states = (
            self._states.get((tool.server_key, tool.name)) for tool in self._tools
        )
        state_width = max(
            [Text("State").cell_len]
            + [
                Text(format_tool_state_label(state)).cell_len
                for state in states
                if state is not None
            ]
        )
        tool_width = max(
            [Text("Tool").cell_len] + [Text(tool.name).cell_len for tool in self._tools]
        )
        # ds-runtime: available table cells minus measured State text, both
        # columns' native padding and the scrollbar that wrapping may reveal.
        budget = max(
            1,
            table.content_region.width
            - table.styles.scrollbar_size_vertical
            - state_width
            - 4 * table.cell_padding,
        )
        return budget if tool_width > budget else None

    def _reflow_table(self) -> None:
        """Rebuild only when resize changes wrapping, retaining tool identity."""
        if not self.is_attached or not self._tools:
            return
        table = self.query_one("#mcp-tools-table", DataTable)
        if self._measured_tool_width(table) != self._tool_column_width:
            self._apply_filter()
            self.call_after_refresh(self.reveal_focused_control)

    def reveal_focused_control(self) -> None:
        """Reveal the current child after layout without overriding newer focus."""
        if (
            not self.is_attached
            or not self.display
            or self.app.screen is not self.screen
        ):
            return
        focused = self.app.focused
        if focused is not None and self in focused.ancestors:
            focused.scroll_visible(animate=False, immediate=True)
            if focused.id == "mcp-tools-empty-action":
                # The recovery callout has nested padding/borders. Measure the
                # action in this scroll viewport: ancestor-relative scrolling
                # can leave it just below the compact canvas's bottom edge.
                self.scroll_to_region(
                    focused.region.translate(
                        self.scroll_offset - self.content_region.offset
                    ),
                    animate=False,
                    immediate=True,
                )
            if isinstance(focused, DataTable) and focused.row_count:
                focused._scroll_cursor_into_view(animate=False)

    # -- data ---------------------------------------------------------------

    def _read_filter_controls(self) -> None:
        """Read drafts that may be newer than their queued Changed messages."""
        self._filter_text = self.query_one("#mcp-tools-filter-text", Input).value
        value = self.query_one("#mcp-tools-filter-server", Select).value
        self._filter_server_key = None if value is Select.NULL else str(value)

    async def update_tools(
        self,
        tools: list[HubTool],
        *,
        empty_diagnosis: tuple[str, str] | None = None,
        states: dict[tuple[str, str], EffectiveToolState] | None = None,
        selected_server_key: str | None = None,
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
            states: T8 -- keyed `(server_key, name)` (same key shape as
                `UnifiedMCPControlPlaneService.effective_tool_states()`),
                one `EffectiveToolState` per tool the workbench was able to
                resolve a permission verdict for. Rendered in the State
                column with the SAME label+marker formatting as the
                Permissions-mode matrix
                (`mcp_permissions_mode.format_tool_state_label()`) -- a
                tool absent from this dict (or `states=None` entirely, e.g.
                a service without the Phase 4 permission seams yet) renders
                "—" rather than guessing a default.
            selected_server_key: task-32283 -- the RAIL's currently selected
                server. Its group is ordered first (see `_apply_filter()`);
                `None` ("All servers") keeps the plain label order.
        """
        self._read_filter_controls()
        self._tools = list(tools)
        self._states = dict(states) if states else {}
        self._selected_server_key = selected_server_key
        self._empty_diagnosis = empty_diagnosis
        self._has_tags = any(tool.tags for tool in self._tools)
        await self._sync_server_select()
        self._apply_filter()

    async def focus_server(self, server_key: str | None) -> None:
        """Scope the catalog to one server, as the filter Select would.

        task-32283: the Servers-mode inspector's "Open tool catalog" drill
        lands here, so the tools of the server the user drilled from are
        the whole visible table rather than a screenful of some other
        server's. A `server_key` with no tools in the current catalog falls
        back to "All servers" (`_sync_server_select()`'s own dangling-
        filter guard).

        Args:
            server_key: The server to scope to, or `None` for all servers.
        """
        self._filter_server_key = server_key
        await self._sync_server_select()
        self._apply_filter()

    def update_local_config(
        self,
        *,
        enabled: bool,
        workspace_root: str,
        visible: bool,
        config_path: Path | None = None,
    ) -> None:
        """Refresh local-tool controls from persisted configuration truth.

        Args:
            enabled: State of the workspace, web, and Watchlists master switch.
            workspace_root: Saved local MCP/Hub root; blank selects process cwd.
            visible: Whether the local-source configuration panel is visible.
            config_path: Configuration identity owning this field and receipt.
        """
        panel = self.query_one("#mcp-tools-local-config", Vertical)
        panel.display = visible
        self._local_tools_enabled = bool(enabled)
        button = self.query_one("#mcp-tools-local-enabled", MCPLocalMasterButton)
        button.enabled, button.config_path = bool(enabled), config_path
        button.label = _local_tools_toggle_label(self._local_tools_enabled)
        root_input = self.query_one("#mcp-tools-workspace-root", Input)
        if config_path != self._workspace_root_config:
            self._workspace_root_saved = None
            self._workspace_root_dirty = False
            self._workspace_root_origin = object()
            self._workspace_root_revision = 0
            self._workspace_root_config = config_path
            self._workspace_root_pending = False
            self.set_workspace_root_status("", error=False)
        if self._workspace_root_saved is None or not self._workspace_root_dirty:
            with root_input.prevent(Input.Changed):
                root_input.value = workspace_root
            self._workspace_root_draft = workspace_root
        self._workspace_root_saved = workspace_root

    @property
    def workspace_root_draft_identity(self) -> tuple[object, int]:
        """Identify the current draft without keeping its widget in a receipt."""
        return self._workspace_root_origin, self._workspace_root_revision

    def set_workspace_root_status(self, message: str, *, error: bool) -> None:
        """Keep root outcomes independent of the local-tool master switch."""
        status = self.query_one("#mcp-tools-workspace-status", Static)
        status.update(message)
        status.set_class(error, "is-error", "ds-text-error")
        status.set_class(not error, "ds-text-muted")

    def project_workspace_root_save(
        self,
        state: ConfigSaveState,
        *,
        generation: int | None = None,
        file_revision: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Project an owned outcome without overwriting a newer local draft."""
        if state.request.config_path != self._workspace_root_config:
            return
        field = self.query_one("#mcp-tools-workspace-root", Input)
        same_draft = state.request.draft_identity == self.workspace_root_draft_identity
        newer_draft = not same_draft and (
            state.request.draft_identity[0] is self._workspace_root_origin
            or self._workspace_root_dirty
        )
        self._workspace_root_pending = state.result is None
        if state.result is None:
            text = "Saving local MCP / Hub test root…"
            error = False
        else:
            result = state.result
            error = result.phase in {"invalid", "failed", "changed"}
            text = {
                "saved": "Saved local MCP / Hub test root.",
                "cache_refresh": "Root saved to file. Restart to refresh live settings.",
                "invalid": "Root not saved: choose an existing directory. Your edits are kept.",
                "failed": "Root save failed. Your edits are kept; choose Save root to retry.",
                "changed": "Configuration changed. Root not saved; reopen MCP before retrying.",
            }[result.phase]
            same_origin = state.request.draft_identity[0] is self._workspace_root_origin
            if result.phase in {"failed", "invalid"} and not same_origin:
                text = "An earlier root save failed. Review the current root and choose Save root to retry."
            superseded = (
                result.cache_generation is not None
                and result.cache_generation != generation
            ) or (
                result.file_revision is not None
                and result.file_revision != file_revision
            )
            if result.phase in {"saved", "cache_refresh"} and superseded:
                text = "Root saved earlier; live settings have since changed."
                newer_draft = self._workspace_root_dirty
            if (
                result.phase in {"saved", "cache_refresh"}
                and same_draft
                and not superseded
            ):
                with field.prevent(Input.Changed):
                    field.value = result.stored or ""
                self._workspace_root_draft = field.value
                self._workspace_root_saved = field.value
                self._workspace_root_dirty = False
        if newer_draft:
            text += " Current edits are not saved."
        self.set_workspace_root_status(text, error=error)

    def _record_workspace_root_edit(self, value: str) -> None:
        if value == self._workspace_root_draft:
            return
        self._workspace_root_draft = value
        self._workspace_root_dirty = True
        self._workspace_root_revision += 1
        prefix = "Saving submitted root. " if self._workspace_root_pending else ""
        self.set_workspace_root_status(
            prefix + "Current edits are not saved. Choose Save root to apply them.",
            error=False,
        )

    def set_local_config_status(self, message: str, *, error: bool) -> None:
        """Render persistence or validation feedback beside the controls.

        Args:
            message: User-facing status text.
            error: Whether to apply the error-state presentation.
        """
        status = self.query_one("#mcp-tools-local-config-status", Static)
        status.update(message)
        status.set_class(error, "is-error", "ds-text-error")
        status.set_class(not error, "ds-text-muted")

    def _request_workspace_root_save(self) -> None:
        value = self.query_one("#mcp-tools-workspace-root", Input).value
        self._record_workspace_root_edit(value)
        self.post_message(
            self.WorkspaceRootSaveRequested(
                value,
                draft_identity=self.workspace_root_draft_identity,
                config_path=self._workspace_root_config,
            )
        )

    def update_states(self, states: dict[tuple[str, str], EffectiveToolState]) -> None:
        """Refresh the cached State-column data in place and re-render rows,
        without touching the cached tool list or rebuilding the server
        filter Select.

        Defect 1 fix (MCP Hub Phase 4 live QA, 2026-07-16): the three
        standalone permission-mutation handlers in `mcp_workbench.py`
        (Space-cycle, kill-switch toggle, Re-allow) deliberately resync
        ONLY the Permissions matrix for latency (see
        `MCPWorkbench._sync_permissions_mode()`'s docstring) -- but each of
        them already resolves a fresh `EffectiveToolState` batch to do
        that. This narrow setter lets those handlers hand that SAME dict to
        this widget too, so its State column reflects the mutation without
        the caller needing a second `effective_tool_states()` call, a
        governance fetch, or a full `update_tools()` catalog refresh.
        """
        self._states = dict(states) if states else {}
        self._apply_filter()

    async def select_tool_row(self, tool_id: str) -> bool:
        """Move the table cursor to `tool_id`'s row for an external drill
        (T7, MCP Hub Phase 5: `MCPWorkbench`'s Audit-mode "Open tool"
        routing) -- does NOT post `ToolSelected` itself (the caller already
        knows the resolved `HubTool` and populates the inspector directly;
        posting here would be a redundant, indirect round trip).

        Clears any active text/server filter first when `tool_id` isn't
        currently a rendered row -- an active filter must not silently
        swallow an external drill's target row. Returns whether `tool_id`
        exists in the current (unfiltered) catalog at all; `False` means
        the caller should fall back to a "tool no longer available" toast
        rather than assume the cursor moved.
        """
        if not any(tool.tool_id == tool_id for tool in self._tools):
            return False
        self._read_filter_controls()
        self._apply_filter()
        table = self.query_one("#mcp-tools-table", DataTable)
        try:
            table.get_row_index(tool_id)
        except RowDoesNotExist:
            # Hidden by the active filter -- clear it so the row renders,
            # then retry the lookup below.
            if self._filter_text:
                self._filter_text = ""
                text_input = self.query_one("#mcp-tools-filter-text", Input)
                with text_input.prevent(Input.Changed):
                    text_input.value = ""
            if self._filter_server_key is not None:
                self._filter_server_key = None
                select = self.query_one("#mcp-tools-filter-server", Select)
                with select.prevent(Select.Changed):
                    select.value = Select.NULL
            self._apply_filter()
        try:
            index = table.get_row_index(tool_id)
        except RowDoesNotExist:
            return False
        table.move_cursor(row=index)
        return True

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

    async def _sync_server_select(self) -> None:
        """Reconcile options without replacing the live filter or its open menu."""
        options = self._server_options()
        valid_keys = {key for _, key in options}
        if self._filter_server_key not in valid_keys:
            self._filter_server_key = None
        value = self._filter_server_key or Select.NULL
        select = self.query_one("#mcp-tools-filter-server", Select)
        overlay = select.query_one(SelectOverlay)
        old_values = [Select.NULL, *(key for _, key in self._server_filter_options)]
        highlighted = overlay.highlighted
        highlighted_value = (
            old_values[highlighted]
            if highlighted is not None and 0 <= highlighted < len(old_values)
            else Select.NULL
        )
        options_changed = options != self._server_filter_options
        with select.prevent(Select.Changed):
            if options_changed:
                select.set_options(options)
                self._server_filter_options = options
            select.value = value
        if options_changed and select.expanded:
            # Labels can reorder while the user has highlighted (but has not
            # committed) a choice. Preserve identity, not the old row number.
            values = [Select.NULL, *(key for _, key in options)]
            overlay.select(
                values.index(highlighted_value) if highlighted_value in values else 0
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
        # task-32283: the rail-selected server's group leads, then the
        # existing `(server_label, name)` order. With no selection the
        # first term is constant and the order is exactly what it was.
        selected = self._selected_server_key
        ordered = sorted(
            filtered,
            key=lambda tool: (
                tool.server_key != selected,
                tool.server_label,
                tool.name,
            ),
        )
        table = self.query_one("#mcp-tools-table", DataTable)
        cursor_key = None
        if table.columns and 0 <= table.cursor_row < table.row_count:
            cursor_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        # UX batch item 11: the Tags column tuple is decided by
        # `self._has_tags` (the FULL unfiltered catalog, set once per
        # `update_tools()` call), never recomputed against `ordered`
        # (the filtered subset) -- a text/server filter that happens to
        # narrow the visible rows to an all-tagless subset must not make
        # the column flicker away mid-typing.
        # Rebuilding the rows moves the cursor back to row 0, which emits the
        # same RowHighlighted a click does. Declaring the rebuild stops that
        # being read as a selection -- see DataTableClickSelectMixin.
        self.repopulating_table()
        # A retained key is still the same tool, but an activation after this
        # redraw is a fresh gesture, not the Enter paired with an old highlight.
        self._pending_activation_key = None
        table.clear(columns=True)
        self._tool_column_width = self._measured_tool_width(table)
        table.add_column("Tool", width=self._tool_column_width)
        table.add_columns(
            *(_TABLE_COLUMNS[1:] if self._has_tags else _TABLE_COLUMNS_NO_TAGS[1:])
        )
        seen_keys: set[str] = set()
        for tool in ordered:
            if tool.tool_id in seen_keys:
                # Defense in depth: hub_tool_catalog's derivation functions
                # already dedupe by (server_key, name), but a row key
                # collision here would raise Textual's `DuplicateKey` and
                # crash every mount that renders this table -- skip rather
                # than trust every current and future upstream caller to
                # have deduped first.
                continue
            seen_keys.add(tool.tool_id)
            tool_state = self._states.get((tool.server_key, tool.name))
            # Task 1 (MCP Hub Phase 6): the State cell's word is colored by
            # the resolved verdict it names -- a tool absent from `states`
            # renders the plain "—" placeholder at the `muted` weight (no
            # verdict to color), same visual tier as every other "not
            # resolved yet" dash in this canvas family.
            if tool_state is not None:
                state_cell = state_text(
                    format_tool_state_label(tool_state), tool_state_kind(tool_state)
                )
            else:
                state_cell = state_text("—", "muted")
            # Qodo #2620 #6: when the label alone consumes the budget, the
            # "(stale)" suffix -- the only table-level signal that a
            # discovered local tool is currently disconnected -- was being
            # truncated away. Reserve its width up front so the marker
            # always survives the ellipsis.
            suffix = " (stale)" if tool.stale else ""
            budget = _SERVER_CELL_BUDGET - len(suffix)
            server_cell = _ellipsize(tool.server_label, budget) + suffix
            schema_cell = (
                "form" if parse_schema(tool.input_schema) is not None else "raw"
            )
            row_cells: list[Any] = [Text(tool.name), state_cell, Text(server_cell)]
            if self._has_tags:
                tags_cell = ", ".join(tool.tags) if tool.tags else "—"
                row_cells.append(Text(tags_cell))
            row_cells.append(Text(schema_cell))
            table.add_row(*row_cells, key=tool.tool_id, height=None)
        if cursor_key is not None:
            try:
                cursor_row = table.get_row_index(cursor_key)
            except RowDoesNotExist:
                # A removed or filtered-out tool leaves clear()'s first-row
                # fallback. Resolve by key because duplicate IDs are skipped.
                pass
            else:
                table.move_cursor(row=cursor_row)
        has_any_tools = bool(self._tools)
        if not has_any_tools and self.app.focused is table:
            self.screen.set_focus(self.query_one("#mcp-tools-filter-text", Input))
        table.display = has_any_tools
        self._update_empty_state(show=not has_any_tools)

    def _update_empty_state(self, *, show: bool) -> None:
        container = self.query_one("#mcp-tools-empty", Vertical)
        button = self.query_one("#mcp-tools-empty-action", Button)
        has_action = self._empty_diagnosis is not None and bool(
            self._empty_diagnosis[1]
        )
        if self.app.focused is button and (not show or not has_action):
            self.screen.set_focus(self.query_one("#mcp-tools-filter-text", Input))
        container.display = show
        if not show:
            return
        if self._empty_diagnosis is not None:
            message, action_key = self._empty_diagnosis
        else:
            message, action_key = "No tools available.", None
        self._empty_action_key = action_key
        self.query_one("#mcp-tools-empty-message", Static).update(message)
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
        if event.input.id == "mcp-tools-workspace-root":
            event.stop()
            if event.value == event.input.value:
                self._record_workspace_root_edit(event.value)
            return
        if event.input.id != "mcp-tools-filter-text":
            return
        event.stop()
        current = self.query_one("#mcp-tools-filter-text", Input)
        if event.input is not current or event.value != current.value:
            return
        self._filter_text = event.value
        self._apply_filter()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "mcp-tools-filter-server":
            return
        event.stop()
        current = self.query_one("#mcp-tools-filter-server", Select)
        if event.select is not current or event.value != current.value:
            return
        if event.value is not Select.NULL and event.value not in {
            key for _, key in self._server_filter_options
        }:
            return
        self._filter_server_key = (
            None if event.value is Select.NULL else str(event.value)
        )
        self._apply_filter()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "mcp-tools-workspace-root":
            return
        event.stop()
        self._request_workspace_root_save()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            self.post_message(self.ToolSelected(str(event.row_key.value)))

    def project_local_master(
        self,
        enabled: bool,
        message: str,
        *,
        error: bool,
        config_path: Path,
        pending: bool = False,
    ) -> None:
        """Refresh only the master control; root input events may still be queued."""
        self._local_tools_enabled = enabled
        button = self.query_one("#mcp-tools-local-enabled", MCPLocalMasterButton)
        button.enabled, button.config_path = enabled, config_path
        button.label = _local_tools_toggle_label(enabled)
        self.set_local_config_status(message, error=error)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-tools-local-enabled":
            event.stop()
            requested, config_path = getattr(
                event,
                "mcp_local_master_choice",
                (not self._local_tools_enabled, self._workspace_root_config),
            )
            self._local_tools_enabled = requested
            event.button.enabled = requested
            event.button.label = _local_tools_toggle_label(requested)
            if self.submit_local_master is not None:
                self.submit_local_master(requested, config_path)
            else:
                self.post_message(
                    self.LocalToolsEnabledChanged(requested, config_path=config_path)
                )
            return
        if event.button.id == "mcp-tools-workspace-save":
            event.stop()
            self._request_workspace_root_save()
            return
        if event.button.id != "mcp-tools-empty-action":
            return
        event.stop()
        if self._empty_action_key:
            self.post_message(self.EmptyActionRequested(self._empty_action_key))
