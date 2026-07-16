# tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py
"""Permissions mode canvas: the allow/ask/deny matrix, kill switch, and
policy preview for the MCP Hub (Phase 4).

`MCPPermissionsMode` renders whatever `PermRow` list + kill-switch value +
preview sentence the workbench hands it -- it never touches the permission
store (T1's `MCPPermissionStore`) itself. The workbench is the SINGLE
writer: a Space press or a kill-switch toggle here only posts a message;
the workbench resolves the next state (T2's `cycle_ui_state`/`cycle_global`
already ran, client-side, to compute it) and mutates the store via T4's
typed `UnifiedMCPControlPlaneService` methods, then resyncs this canvas
with the freshly resolved matrix. Mirrors `mcp_tools_mode.py`'s "render
whatever the workbench hands you" contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Checkbox, DataTable, Static

from tldw_chatbook.MCP.permission_store import (
    DEFAULT_GLOBAL,
    EffectiveToolState,
    cycle_global,
    cycle_ui_state,
)

_TABLE_COLUMNS = ("Tool", "State", "Tags")

# T8: exact copy pinned by the server-source governance section below --
# tests assert this verbatim, so any copy change must happen here (single
# source of truth), not by editing a literal string at each call site.
_SERVER_PROFILES_POINTER = (
    "Server-side profiles are managed in the tldw_server webui. The matrix "
    "above is chatbook's client-side gate and still applies."
)


def format_tool_state_label(effective: EffectiveToolState) -> str:
    """Format a tool row's State cell: the UI label plus its origin marker.

    T8: module-level (was `MCPWorkbench._tool_state_label`, a staticmethod
    in mcp_workbench.py) so both `MCPToolsMode` (the cross-server catalog's
    own State column) and `MCPWorkbench` (this canvas's own matrix rows,
    `PermRow.state_label`) render an `EffectiveToolState` identically
    without either duplicating the marker-precedence logic or importing
    from the other (mcp_workbench.py composes both mode canvases, so a
    mode canvas importing back from it would be circular) -- this widget
    module has no dependents of its own, so it's the natural shared home.

    Precedence mirrors `resolve_effective_state()`'s own downgrade order
    (T2) -- `config_changed` and `risk_floored` are mutually exclusive (the
    former only ever fires for an explicit `tool_override`, the latter only
    for an *inherited* origin), so checking both ahead of the plain
    override bullet is unambiguous: "Ask ⚠" (rug-pull downgrade), "Ask ⚑"
    (high-risk floor), "Allow •" (a plain, undowngraded explicit override),
    or a bare label (plain inherited value, no marker at all).
    """
    if effective.config_changed:
        return f"{effective.ui_label} ⚠"
    if effective.risk_floored:
        return f"{effective.ui_label} ⚑"
    if effective.origin == "tool_override":
        return f"{effective.ui_label} •"
    return effective.ui_label


def _profile_text(value: Any) -> str:
    """Defensive string coercion for one raw governance-profile field --
    mirrors `hub_tool_catalog.py`'s own `_text()` helper (not imported
    directly: that module is tool-catalog-specific and this is a
    permissions-only concern)."""
    return str(value).strip() if value is not None else ""


def _profile_display_text(profile: Mapping[str, Any]) -> str:
    """Render one raw `permission_profiles` entry's name/id defensively --
    the wire shape isn't pinned down (server-side product, versioned
    independently), so every key is optional, mirroring
    `hub_tool_catalog.server_tools_from_inventory()`'s own tolerant-of-
    missing-keys reads."""
    name = _profile_text(profile.get("name")) or _profile_text(profile.get("label"))
    profile_id = _profile_text(profile.get("id")) or _profile_text(profile.get("profile_id"))
    if name and profile_id:
        return f"{name} ({profile_id})"
    if name:
        return name
    if profile_id:
        return profile_id
    return "Unnamed profile"


# Row-key formats (spec-verbatim, see task-6-brief.md): the pinned global
# row is a fixed literal key; a pinned server-default row is
# "__server__::<server_key>"; a tool row's key is `HubTool.tool_id`
# ("<server_key>::<name>") -- built inline below from `PermRow.server_key`/
# `tool_name` rather than importing `HubTool` just for its property.
_GLOBAL_ROW_KEY = "__global__"


def _server_row_key(server_key: str) -> str:
    return f"__server__::{server_key}"


@dataclass(frozen=True)
class PermRow:
    """One rendered matrix row -- the workbench's render-ready projection
    of the pinned global/server-default row or a tool row.

    Attributes:
        kind: One of `"global"|"server"|"tool"`.
        server_key: The owning server's stable key (`""` for the pinned
            global row, which owns no server).
        server_label: Human-readable server label (`""` for the global
            row).
        tool_name: Tool name for a `"tool"` row; `None` for `"global"`/
            `"server"` rows.
        state_label: Fully formatted State-cell text -- the origin marker
            (override bullet `"•"`, rug-pull `"⚠"`, risk-floor `"⚑"`, or no
            marker at all for a plain inherited value) is already baked in
            by the workbench (T2's `EffectiveToolState`); this widget
            renders it verbatim, never re-derives it.
        tags_label: Tags-cell text (comma-joined tags, or `"—"` for a
            tagless tool or a pinned global/server row).
        cycle_current: The raw STORE value this row's Space press cycles
            FROM -- `None`|`"allow"`|`"ask"`|`"deny"`, i.e. exactly what
            `cycle_ui_state()`/`cycle_global()` (T2) expect as input. This
            is NOT the same as the resolved state `state_label` describes:
            an inherited tool row's `cycle_current` is `None` (nothing set
            at the tool level yet) even though its `state_label` shows the
            resolved "Ask"/"Allow"/"Off" it inherited.
    """

    kind: str
    server_key: str
    server_label: str
    tool_name: str | None
    state_label: str
    tags_label: str
    cycle_current: str | None


def _row_key(row: PermRow) -> str:
    if row.kind == "global":
        return _GLOBAL_ROW_KEY
    if row.kind == "server":
        return _server_row_key(row.server_key)
    return f"{row.server_key}::{row.tool_name}"


def _tool_column_text(row: PermRow) -> str:
    if row.kind == "global":
        return "Global default"
    if row.kind == "server":
        return f"Server default — {row.server_label}"
    return row.tool_name or ""


class MCPPermissionsMode(Vertical):
    """Canvas for the Permissions mode: kill switch, matrix, policy preview."""

    DEFAULT_CSS = """
    MCPPermissionsMode {
        width: 1fr;
        height: 100%;
        min-height: 0;
    }
    /* Defensive against a PRE-EXISTING bundle defect (verified empirically,
    same family as Phase 3's Select-collapse Defect 1): the app bundle's
    bare, unscoped `Checkbox { height: 2; }` rule (a different component's
    global rule) wins over even a compact Checkbox's OWN `!important`
    border-removal DEFAULT_CSS -- a fixed 2-row Checkbox still renders its
    (non-removable, from this override's perspective) 1-row-top + 1-row-
    bottom border, leaving ZERO rows for content. `height` itself can't be
    reclaimed here without editing the bundle (Task 9's scope, not this
    one) -- but `min-height`, a DIFFERENT property the bundle rule never
    touches, still applies per-declaration and forces enough total room for
    one visible content row. */
    #mcp-perm-kill-switch {
        height: auto;
        min-height: 3;
    }
    /* Mirrors MCPToolsMode/MCPServersMode's own #mcp-tools-table /
    #mcp-servers-table rule (T7, P3 UX batch): height: auto + max-height so
    the table hugs its own row count instead of ballooning to fill the
    canvas, leaving room for the preview strip below it. */
    #mcp-perm-table {
        height: auto;
        max-height: 70%;
        min-height: 4;
    }
    #mcp-perm-preview {
        height: auto;
        min-height: 0;
    }
    /* T8: the server-source governance section is a plain read-only
    listing (a handful of Static rows at most) -- hugs its own content,
    same rationale as #mcp-perm-preview immediately above. */
    #mcp-perm-server-profiles-slot {
        height: auto;
        min-height: 0;
    }
    """

    BINDINGS = [
        Binding("space", "cycle_state", "Cycle permission", show=False),
    ]

    class StateCycleRequested(Message, namespace="mcp_permissions_mode"):
        """Posted by a Space press on the matrix's cursor row.

        `row_kind` is `"global"|"server"|"tool"`; `new_state` is whatever
        `cycle_ui_state()`/`cycle_global()` (T2) resolved from the row's own
        `cycle_current` -- `None` is legal for a `"server"`/`"tool"` row
        (Inherit) but never for `"global"` (T2's `cycle_global` has no
        Inherit rung). The WORKBENCH mutates the store via T4's typed
        methods and resyncs this canvas; this widget never touches the
        store itself.
        """

        def __init__(
            self,
            row_kind: str,
            server_key: str,
            tool_name: str | None,
            new_state: str | None,
        ) -> None:
            super().__init__()
            self.row_kind = row_kind
            self.server_key = server_key
            self.tool_name = tool_name
            self.new_state = new_state

    class KillSwitchToggled(Message, namespace="mcp_permissions_mode"):
        """Posted once per genuine user toggle of `#mcp-perm-kill-switch` --
        never for `update_matrix()`'s own programmatic sync (see that
        method's `checkbox.prevent(Checkbox.Changed)` guard)."""

        def __init__(self, value: bool) -> None:
            super().__init__()
            self.value = value

    class RowSelected(Message, namespace="mcp_permissions_mode"):
        """Posted when the user selects a matrix row (Enter/click on the
        DataTable's cursor row) -- mirrors `MCPToolsMode.on_data_table_row_
        selected()`'s own Enter/click precedent (not every cursor move, see
        `action_cycle_state()`'s Space-only bare-cursor-move handling
        above). Carries the resolved `PermRow`'s identity so `MCPWorkbench`
        can route it without re-parsing the row key: `row_kind` is
        `"global"|"server"|"tool"`; `server_key`/`tool_name` are `""`/`None`
        for the pinned global row, the owning server key (`tool_name=None`)
        for a pinned server-default row, and the tool's own fields for a
        `"tool"` row.
        """

        def __init__(self, row_kind: str, server_key: str, tool_name: str | None) -> None:
            super().__init__()
            self.row_kind = row_kind
            self.server_key = server_key
            self.tool_name = tool_name

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rows: list[PermRow] = []
        self._rows_by_key: dict[str, PermRow] = {}

    def compose(self) -> ComposeResult:
        yield Checkbox(
            "MCP tools in chat",
            value=False,
            id="mcp-perm-kill-switch",
            compact=True,
            tooltip=(
                "Master switch for chat tool calls (arrives with the chat "
                "bridge). Does not affect Hub tool tests."
            ),
        )
        table = DataTable(id="mcp-perm-table")
        table.cursor_type = "row"
        yield table
        yield Static("", id="mcp-perm-preview", markup=False)
        # T8: dedicated slot container for the server-source governance
        # listing -- mirrors `MCPToolsMode`'s own
        # `#mcp-tools-filter-server-slot` pattern (a persistent empty
        # container that `update_server_profiles()` remove+remounts into),
        # so the section can be entirely ABSENT (not merely hidden) for
        # local/builtin sources without `compose()` itself needing to know
        # the source.
        yield Vertical(id="mcp-perm-server-profiles-slot")

    async def on_mount(self) -> None:
        table = self.query_one("#mcp-perm-table", DataTable)
        table.add_columns(*_TABLE_COLUMNS)

    # -- data -----------------------------------------------------------

    async def update_matrix(
        self, rows: list[PermRow], *, kill_switch: bool, preview: str
    ) -> None:
        """Rebuild the matrix from a fresh `PermRow` list.

        `rows` is rendered in the exact order given -- pinning/grouping/
        sorting is the workbench's job (it derives rows from
        `_last_hub_tools` + `effective_tool_states()`); this widget is
        render-only, same division of labor as `MCPToolsMode.update_tools()`
        versus `filter_tools()`.

        Minor 7 (DuplicateKey guard parity): two `PermRow`s sharing the
        same `_row_key()` identity would raise Textual's `DuplicateKey`
        out of `table.add_row()` below and crash every future resync of
        this canvas -- the same persistent-crash-loop class
        `MCPToolsMode._apply_filter()`'s own `seen_keys` guard exists for.
        Deduped ONCE, first occurrence wins, before anything (`self._rows`,
        `self._rows_by_key`, the table itself) is built from it, so all
        three stay consistent with each other.
        """
        deduped: list[PermRow] = []
        seen_keys: set[str] = set()
        for row in rows:
            key = _row_key(row)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(row)
        rows = deduped

        self._rows = rows
        self._rows_by_key = {_row_key(row): row for row in rows}

        checkbox = self.query_one("#mcp-perm-kill-switch", Checkbox)
        # Programmatic sync from the workbench, never a user toggle -- must
        # not echo back as a KillSwitchToggled. `ToggleButton.__init__`
        # already suppresses its OWN constructor-value echo the same way
        # (see mcp_servers_mode.py's `on_checkbox_changed` docstring for the
        # verified-against-source precedent); this guards the same class's
        # post-mount `.value =` assignment, which is not otherwise silent.
        with checkbox.prevent(Checkbox.Changed):
            checkbox.value = kill_switch

        table = self.query_one("#mcp-perm-table", DataTable)

        # Preserve the cursor's ROW KEY (not its numeric position) across
        # the rebuild below -- `DataTable.clear()` unconditionally resets
        # `cursor_coordinate` to (0, 0), and this canvas resyncs after
        # EVERY Space press (`mcp_workbench.py`'s
        # `on_mcp_permissions_mode_state_cycle_requested`), so an unguarded
        # rebuild would silently redirect a SECOND press onto row 0 (Global
        # default) even though the user is still looking at their tool
        # row. Mirrors `mcp_servers_mode.py`'s `_restore_overview_cursor()`
        # -- same bug class, same fix shape: a key lookup survives a row
        # reorder/insertion/removal; a bare saved index would not.
        cursor_key: str | None = None
        if table.row_count > 0 and table.cursor_row >= 0:
            try:
                row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
            except Exception:
                # Defensive, same rationale as `action_cycle_state()`: a
                # cursor position that doesn't resolve to a live cell is
                # simply not restored, not a crash.
                row_key = None
            if row_key is not None and row_key.value is not None:
                cursor_key = str(row_key.value)

        table.clear()
        for row in rows:
            table.add_row(
                Text(_tool_column_text(row)),
                Text(row.state_label),
                Text(row.tags_label),
                key=_row_key(row),
            )

        if cursor_key is not None:
            for index, row in enumerate(rows):
                if _row_key(row) == cursor_key:
                    table.move_cursor(row=index)
                    break
            # A key that no longer has a row (e.g. its tool/server vanished
            # from this resync) leaves the cursor at `clear()`'s own
            # default -- row 0 -- the same graceful fallback
            # `_restore_overview_cursor()` uses.

        self.query_one("#mcp-perm-preview", Static).update(preview)

    async def update_server_profiles(self, profiles: list[Mapping[str, Any]] | None) -> None:
        """Rebuild the read-only server-source governance listing.

        `profiles` is the raw `permission_profiles` list from the
        workbench's `get_governance()` fetch (T8) when the active source is
        a tldw_server target AND that fetch succeeded -- `None` means
        either a local/builtin source or a failed/guarded fetch, and the
        whole section (not just its rows) is absent. An empty list is a
        distinct, legitimate case (server source, fetch succeeded, zero
        profiles configured) -- still renders the section with its pointer
        text and no profile rows.

        Removes and remounts the section every call -- same discipline as
        `MCPToolsMode._rebuild_server_select()` (a dedicated slot container,
        awaited remove_children before remount) -- simpler to reason about
        than incrementally diffing a rarely-changing read-only list.
        """
        slot = self.query_one("#mcp-perm-server-profiles-slot", Vertical)
        await slot.remove_children()
        if profiles is None:
            return
        section = Vertical(id="mcp-perm-server-profiles")
        await slot.mount(section)
        await section.mount(
            Static(
                _SERVER_PROFILES_POINTER,
                id="mcp-perm-server-profiles-pointer",
                classes="ds-info-callout",
                markup=False,
            )
        )
        for profile in profiles:
            if not isinstance(profile, Mapping):
                # Defensive, mirrors `server_tools_from_inventory()`'s own
                # "skip non-dict entries entirely" style for a raw,
                # wire-derived list.
                continue
            await section.mount(
                Static(
                    _profile_display_text(profile),
                    classes="mcp-perm-server-profile-row",
                    markup=False,
                )
            )

    # -- events -----------------------------------------------------------

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id != "mcp-perm-kill-switch":
            return
        event.stop()
        self.post_message(self.KillSwitchToggled(event.value))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """T7: resolve the selected row back to the `PermRow` it renders
        (via the row-key map `update_matrix()` last built) and post
        `RowSelected` with its identity -- mirrors
        `MCPToolsMode.on_data_table_row_selected()`'s own bare-forward
        shape."""
        event.stop()
        if event.row_key is None or event.row_key.value is None:
            return
        row = self._rows_by_key.get(str(event.row_key.value))
        if row is None:
            return
        self.post_message(self.RowSelected(row.kind, row.server_key, row.tool_name))

    def action_cycle_state(self) -> None:
        """Cycle the matrix's CURSOR row one Space-press (T2's cycle rules).

        Reads `table.cursor_row`, resolves it back to the `PermRow` it
        renders (via the row-key map `update_matrix()` last built), and
        posts `StateCycleRequested` with the next state already computed --
        the workbench applies it, it never re-derives it.
        """
        table = self.query_one("#mcp-perm-table", DataTable)
        if table.row_count == 0:
            return
        cursor_row = table.cursor_row
        if cursor_row < 0:
            return
        try:
            row_key, _ = table.coordinate_to_cell_key((cursor_row, 0))
        except Exception:
            # Defensive: a cursor position that doesn't resolve to a live
            # cell (e.g. a table mid-rebuild) is a no-op, not a crash.
            return
        key_value = row_key.value
        if key_value is None:
            return
        row = self._rows_by_key.get(str(key_value))
        if row is None:
            return
        if row.kind == "global":
            # T2's cycle_global has no Inherit rung -- it never returns
            # None. `cycle_current` is always a concrete state for the
            # global row (the workbench always resolves SOME
            # global_default), but the `or DEFAULT_GLOBAL` fallback keeps
            # this safe even against a malformed row.
            new_state: str | None = cycle_global(row.cycle_current or DEFAULT_GLOBAL)
        else:
            new_state = cycle_ui_state(row.cycle_current)
        self.post_message(
            self.StateCycleRequested(
                row_kind=row.kind,
                server_key=row.server_key,
                tool_name=row.tool_name,
                new_state=new_state,
            )
        )
