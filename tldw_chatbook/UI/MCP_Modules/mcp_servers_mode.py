"""Servers-mode canvas: readiness overview table and per-server detail."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Checkbox, DataTable, Static

from tldw_chatbook.Agents.builtin_tool_gate import (
    LOCAL_TOOLS_MASTER_KEY,
    ToolGate,
    all_tool_gates,
)
from tldw_chatbook.MCP.readiness import (
    STATE_CSS_CLASSES,
    STATE_GLYPHS,
    STATE_LABELS,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    aggregate_summary,
    env_placeholder_names,
    is_off_opt_in,
    worst_state,
)
from tldw_chatbook.MCP.redaction import redact_args, redact_url
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import state_text
from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel, MCPProfileForm
from tldw_chatbook.UI.MCP_Modules.mcp_server_mutations import MCPServerMutationsPanel
from tldw_chatbook.UI.Widgets.table_click_select import DataTableClickSelectMixin

_MUTATIONS_GATED_TOOLTIP = "Requires team, org, or system-admin scope."
# I3: Import always writes to the LOCAL profile store (`_apply_import()` in
# mcp_workbench.py calls `save_local_profile()` unconditionally) -- under
# server source that write would be invisible in the current view (a
# different source/table entirely), so the button is gated off there rather
# than silently landing writes nobody looking at this screen would see.
_IMPORT_GATED_TOOLTIP = "Import creates LOCAL server profiles — switch Source to Local."
_IMPORT_LOCAL_TOOLTIP = (
    "Import servers from a Claude-Desktop-style mcpServers JSON file or paste."
)

_TABLE_COLUMNS = ("Name", "Connection", "Status", "Tools", "Auth", "Scope")
# Task 11: the Local source never has a meaningful Scope (built-in is
# stdio-only; local profiles are always "Personal") -- the overview table
# omits the column entirely there instead of rendering a column of dashes.
_TABLE_COLUMNS_NO_SCOPE = _TABLE_COLUMNS[:-1]

# Task 11: at most this many actionable recovery callouts render below the
# table -- beyond that, a single "+N more" Static points back at the table
# rather than growing the callout list without bound.
_CALLOUT_CAP = 4

# F-057: column drop order when the overview table is too narrow for its
# full column set -- lowest priority first. Name/Status are never dropped
# (identity + readiness are the table's primary content); the dropped
# columns' facts remain one click away in the detail pane.
_COLUMN_DROP_PRIORITY = ("Auth", "Connection")

# F-058: readiness-glyph legend for the Servers overview -- one quiet dim
# line, derived from STATE_GLYPHS/STATE_LABELS so the legend can never
# drift from the statuses the table/rail/inspector actually render, plus
# the ⌂ built-in marker (mcp_rail.py's row prefix), which had no
# explanation anywhere. Mirrors Permissions mode's `#mcp-perm-legend`
# (mcp_permissions_mode.py) in placement (after the content) and styling.
_SERVERS_LEGEND_TEXT = (
    " · ".join(
        f"{glyph} {STATE_LABELS[state].lower()}"
        for state, glyph in STATE_GLYPHS.items()
    )
    + " · ⌂ built-in"
)


def _fit_columns(
    snapshots: list[ReadinessSnapshot], *, show_scope: bool, available: int
) -> list[str]:
    """Choose the overview table's column set for its rendered width (F-057).

    Estimates each column's rendered width as its longest content string
    (header or cell, the same plain strings the cells are built from) plus
    DataTable's per-cell padding, and drops columns in
    `_COLUMN_DROP_PRIORITY` order until the set fits `available`. Name and
    Status are always kept. An unknown width (`available <= 0`, e.g. before
    the first layout) keeps the full per-source set.
    """
    columns = list(_TABLE_COLUMNS if show_scope else _TABLE_COLUMNS_NO_SCOPE)
    if available <= 0:
        return columns

    cell_text: dict[str, Any] = {
        "Name": lambda s: s.label,
        "Connection": lambda s: s.transport,
        "Status": lambda s: s.badge_text(),
        "Tools": lambda s: "—" if s.tool_count is None else str(s.tool_count),
        "Auth": lambda s: s.auth_display,
        "Scope": lambda s: s.scope_display,
    }

    def fits(candidate: list[str]) -> bool:
        total = 0
        for column in candidate:
            longest = max(
                [len(column)] + [len(cell_text[column](snap)) for snap in snapshots]
            )
            total += longest + 2  # DataTable per-cell padding
        return total <= available

    while not fits(columns) and len(columns) > 2:
        droppable = next((c for c in _COLUMN_DROP_PRIORITY if c in columns), None)
        if droppable is None:
            break
        columns.remove(droppable)
    return columns


# `_named_items_text()`'s "show at most this many names, then '+N more'"
# cap -- pulled out to a named constant (was three inline `8` literals) so
# the truncation point and the "how many are left" arithmetic can't drift
# apart from each other.
_NAMED_ITEMS_CAP = 8


# Task 1 (MCP Hub Phase 6): the overview Status cell's `state_text()` kind,
# derived from `STATE_CSS_CLASSES` (readiness.py) rather than a second,
# parallel `ReadinessState -> kind` table -- every class name in that dict is
# already exactly `"mcp-status-{kind}"` (Task 3/11's own CSS-class
# precedent, reused verbatim by `mcp_rail.py`'s rows and `mcp_inspector.py`'s
# readiness Static), so stripping the shared prefix reuses that single
# source of truth instead of duplicating it.
def _readiness_kind(state: ReadinessState) -> str:
    return STATE_CSS_CLASSES[state].removeprefix("mcp-status-")


def _callout_tooltip(snap: ReadinessSnapshot) -> str:
    """ "Open {label}." for a callout, prefixed by any technical detail the
    snapshot carries (F-050 -- e.g. the disabled built-in's config syntax,
    which no longer appears in the one-line callout label itself)."""
    technical = str((snap.detail or {}).get("technical_detail") or "").strip()
    open_line = f"Open {snap.label}."
    return escape_markup(f"{technical} {open_line}" if technical else open_line)


def _count_display(value: int | None) -> str:
    """ "—" for an unreported count, else the plain integer as a string.

    Mirrors `update_overview()`'s own inline `"—" if snap.tool_count is
    None else str(snap.tool_count)` ternary for the overview table's Tools
    cell -- pulled out to a shared helper now that `_detail_text()`'s
    server-source branch (Task 5, MCP Hub Phase 6) needs the identical
    "unreported vs. zero" distinction for resource/prompt counts too.
    """
    return "—" if value is None else str(value)


def _named_items_text(items: Any, *, key: str) -> str:
    """ "{count}: {comma-joined names}" for a Servers-mode detail line, or
    the literal "none" when there's nothing to show (Task 5, MCP Hub Phase
    6 -- see `_detail_text()`'s local-source Tools/Resources/Prompts
    lines).

    Defensive reads: a missing/malformed `discovery_snapshot` field (not a
    list or tuple at all) is treated as empty rather than raising; a
    non-Mapping entry within an otherwise-list/tuple field falls back to
    `str(item)` instead of assuming `.get()` exists. `key` is tried first
    (`"uri"` for resources, `"name"` for tools/prompts), falling back to
    whichever of `name`/`uri` the entry actually carries, then a literal
    `"?"` -- mirrors the pre-Task-5 inline join this replaces.
    """
    if not isinstance(items, (list, tuple)) or not items:
        return "none"
    names: list[str] = []
    for item in items[:_NAMED_ITEMS_CAP]:
        if isinstance(item, Mapping):
            names.append(
                str(item.get(key) or item.get("name") or item.get("uri") or "?")
            )
        else:
            names.append(str(item))
    text = ", ".join(names)
    if len(items) > _NAMED_ITEMS_CAP:
        text += f", … +{len(items) - _NAMED_ITEMS_CAP} more"
    return f"{len(items)}: {text}"


# Task 10: the built-in detail view's Checkbox ids -> the `[mcp]` config key
# (and `BuiltinFlagChanged.key`) each one edits.
_BUILTIN_CHECKBOX_KEYS: dict[str, str] = {
    "mcp-builtin-enabled": "enabled",
    "mcp-builtin-expose-tools": "expose_tools",
    "mcp-builtin-expose-resources": "expose_resources",
    "mcp-builtin-expose-prompts": "expose_prompts",
}

# task-3240: prefix for a [tools]/[console] gate Checkbox's id -- the
# `(section, key)` each one edits is looked up from `_tool_gate_ids`
# (instance state, rebuilt every `_tool_gate_widgets()` call from
# `all_tool_gates()` -- see that method) rather than a second static table,
# so it cannot drift from the enumerator that built the checkboxes.
_TOOL_GATE_ID_PREFIX = "mcp-gate-"

# task-3240 (spec §5): the adapted restart note for gate checkboxes -- NOT
# `_builtin_toggle_widgets()`'s "next client launch" wording. These gates
# affect the in-process AGENT tool catalog (BuiltinToolProvider/
# LocalToolProvider construction), not the MCP client/server handshake --
# no MCP client is involved at all.
_TOOL_GATE_NOTE_TEXT = (
    "Applies on next app restart — tool providers build their catalogs at startup."
)

# task-3240 fix round 1 (Important 1c): the master switch's raw config key
# ("local_tools_enabled") is unreadable as a checkbox label -- humanized
# here, display-only. The Checkbox's `id` is still built from `gate.key`
# (LOCAL_TOOLS_MASTER_KEY), never from this label, so the save/reload path
# is untouched.
_LOCAL_TOOLS_MASTER_LABEL = "Local workspace, web, and Watchlists tools (master switch)"

_LOCAL_TOOLS_INCLUDED_TEXT = (
    "Includes web_search, web_fetch, and web_crawl, plus workspace file, "
    "Git, Watchlists search/detail, and session todo tools. web_deep_search "
    "is separately gated."
)

# task-3240 fix round 1 (Important 1a): shown under the local-tools subheading
# whenever the master switch is off -- without it, an
# enabled web_deep_search with the master off LOOKS live (both checkboxes
# read as independent toggles) but stays unreachable until the master is
# also turned on.
_LOCAL_TOOLS_MASTER_OFF_NOTE_TEXT = (
    "Master switch is off — workspace, web, and Watchlists tools are "
    "unavailable to Console agents."
)


class MCPServersMode(DataTableClickSelectMixin, Vertical):
    """Canvas for the Servers mode."""

    # F-056: Escape disarms a pending delete confirmation (same path as the
    # arm-then-confirm pair's "Keep" button) -- a destructive confirm must
    # never require the mouse to back out of. No-op when unarmed.
    BINDINGS = [Binding("escape", "disarm_delete", "Cancel delete", show=False)]

    BUNDLED_CSS = """
    MCPServersMode {
        width: 1fr;
        height: 100%;
        min-height: 0;
    }
    /* T7 (P3 UX batch): height: auto + max-height: 70% instead of height:
    1fr -- a 1fr table balloons to fill the entire overview pane no matter
    how few servers are configured, stranding #mcp-overview-callouts dozens
    of rows below the table it explains on a tall canvas. Auto-sizing lets
    the table hug its own row count so the callouts render directly under
    its last row; max-height still caps it so a large server list leaves
    room for the callouts/summary above the fold instead of pushing them
    off-screen. */
    #mcp-servers-table {
        height: auto;
        max-height: 70%;
        min-height: 4;
    }
    #mcp-detail-scroll {
        height: 1fr;
        min-height: 0;
    }
    #mcp-servers-form {
        height: auto;
        min-height: 0;
    }
    #mcp-detail-header {
        height: auto;
        min-height: 1;
    }
    #mcp-detail-header #mcp-detail-title {
        width: 1fr;
    }
    #mcp-detail-builtin-toggles {
        height: auto;
        min-height: 0;
    }
    #mcp-detail-tool-gates {
        height: auto;
        min-height: 0;
    }
    #mcp-overview-summary-glyph {
        width: 2;
    }
    /* F-057: let the aggregate sentence WRAP at narrow widths instead of
    clipping mid-sentence -- the shared `.ds-status-badge` rule pins
    `height: 1`. This override covers the bare test harnesses that never
    load the app bundle; the REAL app gets the identical rule from
    _agentic_terminal.tcss (app-tier CSS beats widget DEFAULT_CSS on ties
    in this Textual version, so the bundle carries its own copy -- the
    established lockstep pattern documented there). */
    #mcp-overview-summary {
        width: 1fr;
        height: auto;
        min-height: 1;
    }
    /* F-058: the legend is a single dimmed hint line under the overview
    content -- mirrors `#mcp-perm-legend` (mcp_permissions_mode.py), same
    raw `$text-muted` token rationale: this widget's unit tests mount it
    without the app bundle where the `$ds-*` aliases are defined. */
    #mcp-servers-legend {
        height: auto;
        min-height: 0;
        color: $text-muted;
    }
    """

    class ServerRowSelected(Message, namespace="mcp_servers_mode"):
        """Posted on a table row click, a callout click, or the detail
        breadcrumb (Task 11). `server_key=None` means "clear the
        selection" -- the workbench's `_select_server_key()` already
        treats a `None` key that way (same path `MCPRail.ServerSelected`
        uses for its "All servers" row)."""

        def __init__(self, server_key: str | None) -> None:
            super().__init__()
            self.server_key = server_key

    class AddServerRequested(Message, namespace="mcp_servers_mode"):
        pass

    class ImportServersRequested(Message, namespace="mcp_servers_mode"):
        pass

    class DisconnectRequested(Message, namespace="mcp_servers_mode"):
        """Posted when Disconnect is pressed in the detail toolbar. The
        workbench owns the actual lifecycle worker (`_start_lifecycle`,
        same dispatch T5 wired for connect/test/refresh) -- this pane only
        knows which server the button belongs to."""

        def __init__(self, server_key: str) -> None:
            super().__init__()
            self.server_key = server_key

    class DeleteConfirmed(Message, namespace="mcp_servers_mode"):
        """Posted once the arm-then-confirm sequence completes (Delete,
        then Confirm delete). The workbench owns the actual delete worker
        against `delete_local_profile`."""

        def __init__(self, server_key: str) -> None:
            super().__init__()
            self.server_key = server_key

    class BuiltinFlagChanged(Message, namespace="mcp_servers_mode"):
        """Posted when a built-in server enable/expose Checkbox is toggled.

        `key` is one of `enabled|expose_tools|expose_resources|
        expose_prompts` -- the workbench owns writing it via
        `save_setting_to_cli_config("mcp", key, value)` (a thread-offloaded
        config write; see `MCPWorkbench._save_builtin_flag()`) and then
        reloading the catalog so the built-in row's readiness reflects the
        change (Phase 1 derivation: `enabled=False` -> the muted OFF_OPT_IN
        display state, task-2239).
        """

        def __init__(self, key: str, value: bool) -> None:
            super().__init__()
            self.key = key
            self.value = value

    class ToolGateChanged(Message, namespace="mcp_servers_mode"):
        """Posted when a `[tools]`/`[console]` registration-gate Checkbox
        (task-3240) is toggled.

        Unlike `BuiltinFlagChanged` (hardcoded to the `[mcp]` section),
        `section` is explicit here: task-3240's gates span both `[tools]`
        (the `_GATEABLE_BUILTINS` rows plus `web_deep_search`) and
        `[console]` (the local group's master switch,
        `local_tools_enabled`). The workbench owns writing it via
        `save_setting_to_cli_config(section, key, value)` (see
        `MCPWorkbench._save_tool_gate()`, which mirrors
        `_save_builtin_flag()`) and then reloading so the checkbox --
        rebuilt fresh from `all_tool_gates()` -- reflects the round trip.
        """

        def __init__(self, section: str, key: str, value: bool) -> None:
            super().__init__()
            self.section = section
            self.key = key
            self.value = value

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshots: list[ReadinessSnapshot] = []
        self._detail_snapshot: ReadinessSnapshot | None = None
        # Maps the (possibly `#N`-suffixed) DataTable row key back to the
        # canonical `ReadinessSnapshot.server_key` it represents -- see F3
        # in `update_overview()`/`on_data_table_row_selected()`.
        self._row_key_to_server_key: dict[str, str] = {}
        # T7: True once the first Delete press has armed the inline
        # confirm/keep pair in the detail toolbar. Reset whenever a new
        # snapshot is shown (`show_detail()`) so navigating away silently
        # disarms rather than leaving a stale "Confirm delete" button armed
        # for whatever server happens to be selected next.
        self._delete_armed: bool = False
        # T9: mirrors the workbench's own `_source`/`_server_mutations_available`
        # -- kept here purely for rendering the Add-server button's
        # disabled/tooltip state (`_update_add_server_button()`), set by
        # `update_overview()` (full resync) and `set_mutations_available()`
        # (cheap scope-only update, no resync). `_mutation_target_label`
        # (review fix) is the human label of the target a create would
        # implicitly attach to -- None means no target is active at all,
        # which disables Add-server even when scope allows mutations.
        self._source: str = "local"
        self._mutations_available: bool = False
        self._mutation_target_label: str | None = None
        # Task 11: the last server_key a non-None `show_detail()` call
        # rendered -- `_restore_overview_cursor()` uses it to put the
        # DataTable cursor back on that row when the user returns to the
        # overview (breadcrumb, or any other path that clears the
        # selection), instead of resetting to the top of the table.
        self._last_selected_key: str | None = None
        # Task 11: row_key of each currently-mounted callout Button, indexed
        # by the numeric suffix of its `mcp-callout-{index}` id -- lets
        # `on_button_pressed` translate a callout click back to the
        # server_key to select.
        self._callout_keys: list[str] = []
        # F-057: the table width the current column set was fitted to
        # (0 = never fitted) -- `on_resize` refits only on real changes.
        self._table_width: int = 0
        # task-3240: id -> (section, key) for the currently-mounted gate
        # Checkboxes, rebuilt fresh by every `_tool_gate_widgets()` call --
        # `on_checkbox_changed` reads it to route a toggle to the right
        # `ToolGateChanged`. Empty whenever the detail isn't builtin-source
        # (mirrors `_tool_gate_widgets()` returning `[]` there).
        self._tool_gate_ids: dict[str, tuple[str, str]] = {}

    def on_resize(self) -> None:
        """F-057: refit the overview's column set when the table's rendered
        width changes (terminal resize, compact-mode rebalance). The first
        fit at mount ran pre-layout (width 0 = full set); this is what
        corrects it once the real width is known."""
        table = self.query_one("#mcp-servers-table", DataTable)
        width = table.region.width
        if width > 0 and width != self._table_width:
            self._table_width = width
            self.run_worker(
                self.update_overview(
                    self._snapshots,
                    source=self._source,
                    mutations_available=self._mutations_available,
                    mutation_target_label=self._mutation_target_label,
                ),
                group="mcp-overview-refit",
                exclusive=True,
            )

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-servers-overview"):
            with Horizontal(classes="ds-toolbar"):
                yield Button(
                    "Add server",
                    id="mcp-add-server",
                    classes="console-action-primary",
                    compact=True,
                    tooltip="Create a new local stdio server profile.",
                )
                yield Button(
                    "Import…",
                    id="mcp-import-server",
                    classes="console-action-secondary",
                    compact=True,
                    tooltip=_IMPORT_LOCAL_TOOLTIP,
                )
            # A5: the aggregate line is a neutral sentence with a small
            # colored glyph in front of it, not the whole sentence taking on
            # the worst-state color -- coloring an entire line red/orange
            # reads as more alarming than the underlying signal warrants. The
            # glyph Static carries the worst-state STATE_CSS_CLASSES class;
            # the sentence Static stays plain (`ds-status-badge` only).
            with Horizontal(id="mcp-overview-summary-row"):
                yield Static(
                    "",
                    id="mcp-overview-summary-glyph",
                    classes="ds-status-badge",
                    markup=False,
                )
                yield Static(
                    "",
                    id="mcp-overview-summary",
                    classes="ds-status-badge",
                    markup=False,
                )
            table = DataTable(id="mcp-servers-table")
            table.cursor_type = "row"
            yield table
            yield Vertical(id="mcp-overview-callouts")
            # F-058: quiet glyph legend under the overview content (mirrors
            # Permissions mode's legend placement).
            yield Static(_SERVERS_LEGEND_TEXT, id="mcp-servers-legend", markup=False)
        with Vertical(id="mcp-servers-detail"):
            with Horizontal(id="mcp-detail-header", classes="ds-toolbar"):
                yield Button(
                    "← All servers",
                    id="mcp-detail-back",
                    classes="console-action-subdued",
                    compact=True,
                    tooltip="Return to the overview table.",
                )
                yield Static(
                    "",
                    id="mcp-detail-title",
                    classes="destination-section",
                    markup=False,
                )
            yield Horizontal(id="mcp-detail-toolbar", classes="ds-toolbar")
            with VerticalScroll(id="mcp-detail-scroll"):
                yield Static(
                    "", id="mcp-detail-body", classes="ds-field-row", markup=False
                )
                yield Vertical(id="mcp-detail-builtin-toggles")
                yield Vertical(id="mcp-detail-tool-gates")
                yield Button(
                    "Copy client config",
                    id="mcp-detail-copy-snippet",
                    classes="console-action-secondary",
                    compact=True,
                    tooltip="Copy this built-in server's client config snippet to the clipboard.",
                )
        yield Vertical(id="mcp-servers-form")

    def on_mount(self) -> None:
        table = self.query_one("#mcp-servers-table", DataTable)
        table.add_columns(*_TABLE_COLUMNS)
        self.query_one("#mcp-servers-form").display = False
        self._show_overview_container(True)
        self._update_add_server_button()
        self._update_import_server_button()

    def _show_overview_container(self, show_overview: bool) -> None:
        self.query_one("#mcp-servers-overview").display = show_overview
        self.query_one("#mcp-servers-detail").display = not show_overview

    def _form_visible(self) -> bool:
        """Whether `#mcp-servers-form` (add/edit, import, or mutations panel)
        is currently the visible pane.

        I1 fix: `update_overview()` and `show_detail()` are both called from
        `_sync_children()`, which runs on every background resync (lifecycle
        completion, the `r` keybinding, a runtime-backend refresh) -- not
        just on an explicit navigation. Previously both unconditionally
        flipped the overview/detail container visibility, so a resync while
        a form was open re-showed the overview (or detail) UNDERNEATH the
        still-mounted form, stacking two views and silently discarding
        whatever the user had typed the next time the form closed. Callers
        that check this must still apply their DATA updates as normal (the
        table/detail text underneath should stay current) -- only the
        container-visibility flip is skipped while the form has the floor.
        """
        try:
            return bool(self.query_one("#mcp-servers-form").display)
        except Exception:
            return False

    def set_mutations_available(
        self, mutations_available: bool, *, mutation_target_label: str | None = None
    ) -> None:
        """Cheap, no-resync update of the Add-server gating (T9).

        Called by the workbench's scope-change handler, which deliberately
        avoids a full `_sync_children()` resync (see that handler's comment)
        -- this only touches the Add-server button, not the table/detail/rail.
        """
        self._mutations_available = mutations_available
        self._mutation_target_label = mutation_target_label
        self._update_add_server_button()

    def _update_add_server_button(self) -> None:
        """Render the Add-server button's gate state.

        Server source, in precedence order (review fix): scope gate first
        (mutations not offered at all), then the no-active-target gate
        (nothing for a create to attach to), then enabled -- with the
        tooltip NAMING the implicit target, because Add-server runs from
        the overview where no selection is visible and the create would
        otherwise silently attach to whatever target the service remembers.
        """
        button = self.query_one("#mcp-add-server", Button)
        if self._source == "server":
            if not self._mutations_available:
                button.disabled = True
                button.tooltip = _MUTATIONS_GATED_TOOLTIP
            elif self._mutation_target_label is None:
                button.disabled = True
                button.tooltip = "Select a server target first."
            else:
                button.disabled = False
                # Target labels are user/remote-configured -- escape before
                # the markup-interpreting tooltip (mcp_rail.py precedent).
                button.tooltip = (
                    f"Adds to server: {escape_markup(self._mutation_target_label)}."
                )
        else:
            button.disabled = False
            button.tooltip = "Create a new local stdio server profile."

    def _update_import_server_button(self) -> None:
        """Render the Import button's source gate (I3).

        Import always writes LOCAL profiles (`MCPWorkbench._apply_import()`
        calls `save_local_profile()` unconditionally) -- offering it under
        server source would silently write somewhere invisible in the
        current view. Mirrors `_update_add_server_button()`'s
        disabled+tooltip pattern, gated purely on source (no scope/target
        gating applies here -- Import never touches server-side records).
        """
        button = self.query_one("#mcp-import-server", Button)
        if self._source == "server":
            button.disabled = True
            button.tooltip = _IMPORT_GATED_TOOLTIP
        else:
            button.disabled = False
            button.tooltip = _IMPORT_LOCAL_TOOLTIP

    async def show_form(self, profile: dict[str, Any] | None) -> None:
        """Show the add/edit form, hiding overview and detail while it is up."""
        self.query_one("#mcp-servers-overview").display = False
        self.query_one("#mcp-servers-detail").display = False
        form_container = self.query_one("#mcp-servers-form", Vertical)
        await form_container.remove_children()
        await form_container.mount(MCPProfileForm(profile=profile))
        form_container.display = True

    async def show_server_mutations(
        self, record: dict[str, Any] | None, slots: list[dict[str, Any]]
    ) -> None:
        """Show the external-server add/edit + credential-slot panel (T9).

        Hosted in the same `#mcp-servers-form` container as `show_form()`'s
        local-profile form -- Servers mode only ever has one of
        overview/detail/form visible at a time, so `hide_form()` also closes
        this. `record=None` is add mode; a populated record is edit mode
        with `slots` (already fetched by the workbench via
        `external_server.slots.list`) rendered as manageable rows.
        """
        self.query_one("#mcp-servers-overview").display = False
        self.query_one("#mcp-servers-detail").display = False
        form_container = self.query_one("#mcp-servers-form", Vertical)
        await form_container.remove_children()
        await form_container.mount(MCPServerMutationsPanel(record=record, slots=slots))
        form_container.display = True

    async def hide_form(self) -> None:
        """Hide the form and restore whichever view (overview or detail) was active."""
        form_container = self.query_one("#mcp-servers-form", Vertical)
        await form_container.remove_children()
        form_container.display = False
        self._show_overview_container(self._detail_snapshot is None)

    async def show_import(self, existing_ids: set[str] | None = None) -> None:
        """Show the mcpServers import panel, hiding overview and detail while it is up.

        Hosted in the same `#mcp-servers-form` container as `show_form()`'s
        add/edit form -- Servers mode only ever has one of overview/detail/
        form/import visible at a time, so `hide_form()` also closes this.
        """
        self.query_one("#mcp-servers-overview").display = False
        self.query_one("#mcp-servers-detail").display = False
        form_container = self.query_one("#mcp-servers-form", Vertical)
        await form_container.remove_children()
        await form_container.mount(MCPImportPanel(existing_ids=existing_ids))
        form_container.display = True

    async def update_overview(
        self,
        snapshots: list[ReadinessSnapshot],
        *,
        source: str = "local",
        mutations_available: bool = False,
        mutation_target_label: str | None = None,
    ) -> None:
        """Rebuild the overview table, summary, and recovery callouts.

        The callouts container is refreshed the same awaited way
        `MCPInspector.update_readiness()` rebuilds its action buttons (see
        the P0 fix in mcp_inspector.py): `remove_children()` is awaited
        before mounting, and the new callouts are mounted in a single
        batched `mount_all()` call rather than one `mount()` call per
        callout in a loop, so a second `update_overview()` call queued
        right behind this one cannot interleave its own removal/mount with
        this call's -- and the canvas takes one layout pass instead of one
        per callout.

        Args:
            snapshots: Readiness snapshots for every server currently
                visible under the active source (local or server).
            source: The active source ("local" or "server") -- drives the
                Add-server button's label/gating (T9).
            mutations_available: Whether `external_server.*` mutation
                actions are currently usable (server source only; see
                `MCPWorkbench._compute_server_mutations_available()`).
            mutation_target_label: Human label of the target a create would
                implicitly attach to (None disables Add-server with a
                "Select a server target first." tooltip).
        """
        self._source = source
        self._mutations_available = mutations_available
        self._mutation_target_label = mutation_target_label
        self._update_add_server_button()
        self._update_import_server_button()
        self._snapshots = list(snapshots)
        summary = self.query_one("#mcp-overview-summary", Static)
        summary.update(aggregate_summary(self._snapshots))
        # A5: the sentence itself stays neutral (no status class ever added
        # here) -- only the small glyph in front of it carries the CSS class
        # for the WORST state present (READY -- no extra class beyond the
        # base ds-status-badge look -- when every server is ready, or when
        # there are none at all). Both Statics persist across calls (neither
        # is ever removed/remounted), so the previous call's class must be
        # dropped from the glyph before possibly adding a different one.
        worst = worst_state(self._snapshots)
        glyph = self.query_one("#mcp-overview-summary-glyph", Static)
        for css_class in STATE_CSS_CLASSES.values():
            glyph.remove_class(css_class)
        glyph.add_class(STATE_CSS_CLASSES[worst])
        glyph.update(STATE_GLYPHS[worst])
        table = self.query_one("#mcp-servers-table", DataTable)
        # F-057: remember the width this call fitted its columns to, so
        # `on_resize` only refits when the rendered width actually changed.
        self._table_width = table.region.width
        # Task 11: per-source columns -- Local (built-in + local profiles)
        # has no meaningful Scope (stdio-only / always "Personal"), so the
        # column is omitted there rather than rendering a column of dashes.
        # Columns are rebuilt from scratch every call (not just when the
        # set actually changes) -- simpler than tracking the previously
        # rendered column set, and this only runs on an actual overview
        # resync, not per keystroke.
        # Task 11: per-source columns -- Local (built-in + local profiles)
        # has no meaningful Scope (stdio-only / always "Personal"), so the
        # column is omitted there rather than rendering a column of dashes.
        # Columns are rebuilt from scratch every call (not just when the
        # set actually changes) -- simpler than tracking the previously
        # rendered column set, and this only runs on an actual overview
        # resync, not per keystroke.
        show_scope = source != "local"
        # F-057: at narrow widths the full column set overflows the
        # viewport and the right-most columns silently vanish behind the
        # DataTable's horizontal scroll. Drop the lowest-priority columns
        # (Auth, then Connection) until the estimated content fits the
        # table's current rendered width -- Name/Status always stay, and
        # the dropped facts remain one click away in the detail pane
        # (env/credential lines, `Connection · ...`). Unknown width (0,
        # pre-layout) keeps the full set, matching pre-F-057 behavior.
        # Rebuilding moves the cursor to row 0 before the key-based restore
        # below puts it back; declaring the rebuild keeps that transient from
        # being read as a selection (DataTableClickSelectMixin).
        self.repopulating_table()
        table.clear(columns=True)
        columns = _fit_columns(
            self._snapshots, show_scope=show_scope, available=table.region.width
        )
        table.add_columns(*columns)
        seen_keys: set[str] = set()
        self._row_key_to_server_key = {}
        for snap in self._snapshots:
            row_key = snap.server_key
            if row_key in seen_keys:
                # Two malformed records can both fall back to the same
                # server_key (e.g. two local profiles missing profile_id
                # both become "local:unknown" -- see
                # local_profile_readiness()). DataTable.add_row(key=...)
                # raises DuplicateKey for a repeat; de-dupe with a suffix
                # instead of crashing the whole canvas over bad data.
                suffix = 2
                candidate = f"{row_key}#{suffix}"
                while candidate in seen_keys:
                    suffix += 1
                    candidate = f"{row_key}#{suffix}"
                row_key = candidate
            seen_keys.add(row_key)
            # The suffixed row_key is a table-internal de-dupe identifier,
            # not a real server_key -- remember the canonical key so
            # `on_data_table_row_selected()` can translate it back.
            self._row_key_to_server_key[row_key] = snap.server_key
            # label/auth_display/scope_display are user-controlled (local
            # profile ids, server-reported names) and DataTable parses
            # plain str cells as Rich markup -- wrap in Text so a value like
            # "[/bold]docs" can't crash the app (MarkupError) and
            # "[red]x[/red]" can't inject styling. Status cells now carry
            # the readiness state's color too (Task 1, MCP Hub Phase 6,
            # supersedes the old Task 11 "stays plain" decision above this
            # comment used to describe) -- `state_text()` colors the WHOLE
            # cell (glyph + word together, one string), mirroring
            # `mcp_rail.py`'s row Buttons, which already color both the
            # same way via `STATE_CSS_CLASSES`.
            row_cells_by_name: dict[str, Any] = {
                "Name": Text(snap.label),
                "Connection": snap.transport,
                "Status": state_text(snap.badge_text(), _readiness_kind(snap.state)),
                "Tools": "—" if snap.tool_count is None else str(snap.tool_count),
                "Auth": Text(snap.auth_display),
                "Scope": Text(snap.scope_display),
            }
            table.add_row(
                *(row_cells_by_name[column] for column in columns), key=row_key
            )
        callouts = self.query_one("#mcp-overview-callouts", Vertical)
        await callouts.remove_children()
        callout_widgets: list[Widget] = []
        # F-051: the disabled built-in is an OFF/opt-in state, not a
        # problem -- it never files a recovery callout. Instead it gets a
        # calm Enable affordance whose click performs the fix directly
        # (BuiltinFlagChanged("enabled", True), the same message the detail
        # view's Enabled checkbox posts), rendered in the same one-line
        # callout row style.
        for snap in self._snapshots:
            if not is_off_opt_in(snap):
                continue
            technical = str((snap.detail or {}).get("technical_detail") or "").strip()
            enable_tooltip = (
                f"{technical} Enable the built-in MCP server so an MCP client "
                "can launch it."
            ).strip()
            callout_widgets.append(
                Button(
                    escape_markup(f"{snap.label} is turned off — Enable"),
                    id="mcp-builtin-enable",
                    classes="mcp-callout mcp-optin console-action-subdued",
                    compact=True,
                    tooltip=escape_markup(enable_tooltip),
                )
            )
        # Task 11: callouts are now actionable one-line Buttons (posting
        # ServerRowSelected straight to the problem row) instead of inert
        # Statics -- capped at _CALLOUT_CAP with a final "+N more" Static
        # pointing back at the table so a source with many problem servers
        # doesn't grow the callout list without bound.
        problem_snapshots = [
            snap
            for snap in self._snapshots
            if snap.state not in (ReadinessState.READY, ReadinessState.CHECKING)
            and not is_off_opt_in(snap)
        ]
        visible = problem_snapshots[:_CALLOUT_CAP]
        overflow = len(problem_snapshots) - len(visible)
        self._callout_keys = [snap.server_key for snap in visible]
        callout_widgets.extend(
            Button(
                escape_markup(
                    f"{STATE_GLYPHS[snap.state]} {snap.label}: {snap.message}"
                ),
                id=f"mcp-callout-{index}",
                classes="mcp-callout console-action-subdued",
                compact=True,
                tooltip=_callout_tooltip(snap),
            )
            for index, snap in enumerate(visible)
        )
        if overflow > 0:
            callout_widgets.append(
                Static(
                    f"+{overflow} more — see the table above.",
                    classes="ds-recovery-callout",
                    markup=False,
                )
            )
        if callout_widgets:
            await callouts.mount_all(callout_widgets)
        # I1: data (table/summary/callouts, all above) always refreshes: only
        # the container-visibility flip is skipped while the form is open,
        # so a background resync can never re-show the overview underneath
        # an in-progress add/edit/import/mutations form.
        if self._detail_snapshot is None and not self._form_visible():
            self._show_overview_container(True)

    async def show_detail(
        self, snapshot: ReadinessSnapshot | None, *, mutations_available: bool = False
    ) -> None:
        """Render `snapshot` into the detail pane.

        I1 fix: `_sync_children()` calls this on every resync, including
        background ones (lifecycle completion, the `r` keybinding, a
        runtime-backend refresh) that can fire while the add/edit/import/
        mutations form is open. Data (`_detail_snapshot`, title, body,
        toggles, toolbar) always updates so the pane underneath the form is
        current the moment it closes -- only the overview/detail
        container-visibility flip is skipped while the form has the floor,
        so a resync can never re-show detail (or overview) stacked
        underneath a still-open form and silently discard typed input.
        """
        self._detail_snapshot = snapshot
        # Any new snapshot -- including re-showing the same server after a
        # lifecycle resync -- disarms a pending delete confirmation rather
        # than leaving it armed against whatever is selected next.
        self._delete_armed = False
        form_visible = self._form_visible()
        if snapshot is None:
            if not form_visible:
                self._show_overview_container(True)
            await self._rebuild_toggle_groups()
            await self._rebuild_detail_toolbar()
            # Task 11: selection restoration -- returning to the overview
            # (breadcrumb, or any other path that clears the selection)
            # moves the DataTable cursor back to the row for the
            # last-selected server so keyboard users resume where they
            # left instead of landing back at the top of the table.
            self._restore_overview_cursor()
            return
        self._last_selected_key = snapshot.server_key
        if not form_visible:
            self._show_overview_container(False)
        self.query_one("#mcp-detail-title", Static).update(
            f"{snapshot.badge_text()}  {snapshot.label}"
        )
        self.query_one("#mcp-detail-body", Static).update(
            self._detail_text(snapshot, mutations_available=mutations_available)
        )
        self.query_one("#mcp-detail-copy-snippet", Button).display = (
            snapshot.source == "builtin"
        )
        await self._rebuild_toggle_groups()
        await self._rebuild_detail_toolbar()

    def _restore_overview_cursor(self) -> None:
        """Move the overview DataTable's cursor onto `_last_selected_key`'s
        row, if it still has one.

        Called from `show_detail(None)` -- i.e. every path that returns to
        the overview (breadcrumb click, `ServerRowSelected(None)` from a
        callout-cleared parent, etc). `self._snapshots` is already the
        table's current row order (each snapshot produces exactly one row,
        in order -- the dedupe suffix in `update_overview()` only changes a
        row's *key*, never its position), so a plain index lookup is
        enough; no separate row-order bookkeeping needed. A key that no
        longer has a row (e.g. the server was deleted) leaves the cursor
        wherever it already was.
        """
        if self._last_selected_key is None:
            return
        table = self.query_one("#mcp-servers-table", DataTable)
        for index, snap in enumerate(self._snapshots):
            if snap.server_key == self._last_selected_key:
                table.move_cursor(row=index)
                return

    def _detail_toolbar_widgets(self) -> list[Button]:
        """Build the local-profile toolbar (Edit/Disconnect/Delete), or the
        arm-then-confirm pair once Delete has been pressed.

        Local-source snapshots only: built-in is edited via config.toml, and
        server-source profiles are mutated server-side (Advanced), so both
        render no toolbar at all here.
        """
        snapshot = self._detail_snapshot
        if snapshot is None or snapshot.source != "local":
            return []
        if self._delete_armed:
            return [
                Button(
                    "Confirm delete",
                    id="mcp-detail-delete-confirm",
                    classes="console-action-primary",
                    compact=True,
                    tooltip="Confirm permanent deletion.",
                ),
                Button(
                    "Keep",
                    id="mcp-detail-delete-cancel",
                    classes="console-action-secondary",
                    compact=True,
                    tooltip="Keep the profile.",
                ),
            ]
        widgets = [
            Button(
                "Edit",
                id="mcp-detail-edit",
                classes="console-action-secondary",
                compact=True,
                tooltip="Edit this profile.",
            ),
        ]
        if snapshot.is_connected:
            widgets.append(
                Button(
                    "Disconnect",
                    id="mcp-detail-disconnect",
                    classes="console-action-secondary",
                    compact=True,
                    tooltip="Disconnect the running server.",
                )
            )
        widgets.append(
            Button(
                "Delete",
                id="mcp-detail-delete",
                classes="console-action-secondary",
                compact=True,
                tooltip="Delete this profile — asks to confirm.",
            )
        )
        return widgets

    async def disarm_delete(self) -> None:
        """Disarm a pending delete confirmation (no-op when unarmed).

        The arm-then-confirm contract is "any other interaction disarms".
        `show_detail()` already resets the arm state for every interaction
        that flows through it (selecting another server, lifecycle resyncs),
        but a mode switch does not: the ContentSwitcher hides this canvas
        without unmounting it, so a live "Confirm delete" button would
        otherwise survive a Servers -> Tools -> Servers round-trip.
        `MCPWorkbench.set_mode()` calls this on every actual mode change.
        """
        if not self._delete_armed:
            return
        self._delete_armed = False
        await self._rebuild_detail_toolbar()

    async def action_disarm_delete(self) -> None:
        """F-056: Escape -- disarm exactly like the "Keep" button (no-op
        when nothing is armed)."""
        await self.disarm_delete()

    async def _rebuild_detail_toolbar(self) -> None:
        """Rebuild `#mcp-detail-toolbar` from `_detail_toolbar_widgets()`.

        Mirrors the awaited remove-then-mount discipline used elsewhere in
        this canvas (`update_overview()`'s callouts, `show_form()`/
        `hide_form()`'s form container) so a second `show_detail()` (or a
        button press) queued right behind this one cannot interleave its
        own removal/mount with this call's and produce DuplicateIds.
        """
        toolbar = self.query_one("#mcp-detail-toolbar", Horizontal)
        await toolbar.remove_children()
        widgets = self._detail_toolbar_widgets()
        # Built-in/server-source detail views (and no snapshot at all) get
        # no toolbar -- hide the row itself rather than leaving an empty
        # padded `.ds-toolbar` band under the title.
        toolbar.display = bool(widgets)
        if widgets:
            await toolbar.mount_all(widgets)
            if self._delete_armed:
                # F-056: arming the delete confirmation moves keyboard focus
                # onto the SAFE option ("Keep") -- a keyboard user can back
                # out with Enter or Escape immediately, and an accidental
                # Enter never confirms the delete. Only the arm pair gets
                # this (the plain Edit/Disconnect/Delete toolbar never
                # steals focus).
                self.call_after_refresh(
                    self.query_one("#mcp-detail-delete-cancel", Button).focus
                )

    def _focused_toggle_id(self) -> str | None:
        """Id of this pane's [mcp]/gate Checkbox that currently has focus.

        task-3240 fix round 1 (Critical 1). Mirrors `sources_pane.py`'s
        `_focused_create_field_id()`: `self.screen.focused` (never
        `self.app.focused` -- the same `ScreenStackError` risk that
        precedent documents), guarded so a screen-less or mid-teardown
        widget never raises out of a resync.
        """
        try:
            focused = self.screen.focused if self.is_mounted else None
        except Exception:
            return None
        if focused is None:
            return None
        fid = focused.id or ""
        if fid not in _BUILTIN_CHECKBOX_KEYS and not fid.startswith(
            _TOOL_GATE_ID_PREFIX
        ):
            return None
        try:
            return fid if self in focused.ancestors_with_self else None
        except Exception:
            return None

    def _restore_toggle_focus(self, focused_id: str | None) -> None:
        """Restore focus to `focused_id` after a toggle-group rebuild.

        task-3240 fix round 1 (Critical 1). Falls back to the inert
        `#mcp-detail-scroll` container -- never to whatever the DOM
        happens to leave focus on -- when `focused_id` no longer exists
        post-rebuild (e.g. the detail pane changed source entirely). That
        fallback matches this code's OWN pre-fix behavior for the plain
        `[mcp]`-toggles-only case: a Space there landed on the scroll
        container, which does nothing on a further Space. The bug this
        fixes is a DIFFERENT, live Checkbox inheriting focus instead of
        that inert container -- restoring by id (or falling back to the
        same inert spot) closes that regardless of which group toggled.
        """
        if focused_id is None:
            return
        try:
            target = self.query_one(f"#{focused_id}", Checkbox)
        except NoMatches:
            target = None
        if target is not None:
            target.focus()
            return
        try:
            self.query_one("#mcp-detail-scroll").focus()
        except NoMatches:
            pass

    async def _rebuild_toggle_groups(self) -> None:
        """Rebuild the `[mcp]` toggles AND the `[tools]`/`[console]` gate
        checkboxes, preserving keyboard focus across the remount.

        task-3240 fix round 1 (Critical 1, reviewer-caught regression).
        `remove_children()` (in either `_rebuild_builtin_toggles()` or
        `_rebuild_tool_gate_checkboxes()`) destroys whichever Checkbox
        currently holds focus; Textual does not itself relocate focus when
        the focused widget is removed this way (identical mechanism to
        `sources_pane.py`'s `recompose()`, which documents it at length).
        Before `#mcp-detail-tool-gates` existed, the nearest surviving
        focusable was `#mcp-detail-scroll` -- an inert container, so a
        stray Space there did nothing. Adding that sibling AFTER the
        `[mcp]` toggles changed the nearest survivor for a gate-checkbox
        toggle to the LAST `[mcp]` checkbox (`mcp-builtin-expose-prompts`)
        -- a live, actionable Checkbox. The save+resync this SAME toggle
        triggers therefore left focus parked there, so the user's very
        next Space silently wrote an unrelated `[mcp]` key instead of
        toggling the gate again. Capturing/restoring by id fixes both
        rebuilds -- including the pre-existing `[mcp]`-only case, which
        was merely lucky before, not correct.
        """
        focused_id = self._focused_toggle_id()
        await self._rebuild_builtin_toggles()
        await self._rebuild_tool_gate_checkboxes()
        self._restore_toggle_focus(focused_id)

    def _builtin_toggle_widgets(self) -> list[Widget]:
        """Build the built-in detail's enable/expose Checkbox rows + note.

        Builtin-source snapshots only -- local and server-source detail
        views render no toggles at all (empty list, mirrors
        `_detail_toolbar_widgets()`'s source gate).
        """
        snapshot = self._detail_snapshot
        if snapshot is None or snapshot.source != "builtin":
            return []
        detail = snapshot.detail or {}
        # `enabled` is read directly off `detail["enabled"]` (populated by
        # `builtin_readiness()`, Task 10) rather than re-derived from
        # `snapshot.state is not ReadinessState.OFF_OPT_IN` -- see the
        # comment on that call site for why. The `True` fallback only
        # matters for a hypothetical builtin-source snapshot built without
        # going through `builtin_readiness()` at all (none do today).
        enabled = bool(detail.get("enabled", True))
        return [
            Checkbox(
                "Enabled",
                value=enabled,
                id="mcp-builtin-enabled",
                compact=True,
                tooltip="Enable the built-in MCP server so an MCP client can launch it.",
            ),
            Checkbox(
                "Expose tools",
                value=bool(detail.get("expose_tools", True)),
                id="mcp-builtin-expose-tools",
                compact=True,
                tooltip="Expose tldw_chatbook's tools to MCP clients.",
            ),
            Checkbox(
                "Expose resources",
                value=bool(detail.get("expose_resources", True)),
                id="mcp-builtin-expose-resources",
                compact=True,
                tooltip="Expose tldw_chatbook's resources to MCP clients.",
            ),
            Checkbox(
                "Expose prompts",
                value=bool(detail.get("expose_prompts", True)),
                id="mcp-builtin-expose-prompts",
                compact=True,
                tooltip="Expose tldw_chatbook's prompts to MCP clients.",
            ),
            Static(
                "Applies to the next client launch — the built-in server "
                "reads config at start.",
                id="mcp-builtin-toggles-note",
                classes="ds-field-row",
                markup=False,
            ),
        ]

    async def _rebuild_builtin_toggles(self) -> None:
        """Rebuild `#mcp-detail-builtin-toggles` from `_builtin_toggle_widgets()`.

        Mirrors `_rebuild_detail_toolbar()`'s awaited remove-then-mount
        discipline so a second `show_detail()` queued right behind this one
        cannot interleave its own removal/mount with this call's.
        """
        container = self.query_one("#mcp-detail-builtin-toggles", Vertical)
        await container.remove_children()
        widgets = self._builtin_toggle_widgets()
        container.display = bool(widgets)
        if widgets:
            await container.mount_all(widgets)

    def _gate_checkbox(self, gate: ToolGate, *, disabled: bool = False) -> Checkbox:
        """Build one gate's Checkbox.

        task-3240 fix round 1: `disabled` (Important 1b) is True for a
        LOCAL-group gate other than the master switch itself while that
        master is off -- re-evaluated fresh on every rebuild, so switching
        the master back on re-enables its dependents the very next resync
        with no separate wiring. The master row's own label is humanized
        (Important 1c) via `_LOCAL_TOOLS_MASTER_LABEL`; its `id` still
        comes from `gate.key`, never from the label, so the save/reload
        path is unaffected.
        """
        label = (
            _LOCAL_TOOLS_MASTER_LABEL
            if gate.key == LOCAL_TOOLS_MASTER_KEY
            else gate.tool_name
        )
        return Checkbox(
            label,
            value=gate.enabled,
            id=f"{_TOOL_GATE_ID_PREFIX}{gate.key}",
            compact=True,
            tooltip=gate.description,
            disabled=disabled,
        )

    def _tool_gate_widgets(self) -> list[Widget]:
        """Build the `[tools]`/`[console]` gate Checkbox rows (task-3240).

        Builtin-source snapshots only, same gate as `_builtin_toggle_
        widgets()` -- spec review finding 5 (branch (b)): `_collect_
        snapshots()` never produces a `local:__local__` row, so this is the
        only reachable place for ALL of them. Rendered under two
        subheadings ("Agent built-ins" / "Local workspace, web, and Watchlists tools") so the
        accepted UX trade-off stays visible rather than papered over: this
        pane is badged as the built-in MCP SERVER, but these checkboxes
        control the in-process AGENT tool catalog -- a different subsystem.

        Also rebuilds `self._tool_gate_ids` (id -> (section, key)) fresh
        from the same `all_tool_gates()` batch the checkboxes were built
        from, so `on_checkbox_changed` can never route a toggle against a
        stale mapping.
        """
        snapshot = self._detail_snapshot
        if snapshot is None or snapshot.source != "builtin":
            self._tool_gate_ids = {}
            return []
        gates = all_tool_gates()
        self._tool_gate_ids = {
            f"{_TOOL_GATE_ID_PREFIX}{gate.key}": (gate.section, gate.key)
            for gate in gates
        }
        builtin_gates = [gate for gate in gates if gate.group == "builtin"]
        local_gates = [gate for gate in gates if gate.group == "local"]

        widgets: list[Widget] = [
            Static(
                "Tool gates",
                id="mcp-gate-section-title",
                classes="destination-section",
                markup=False,
            ),
        ]
        if builtin_gates:
            widgets.append(
                Static(
                    "Agent built-ins",
                    id="mcp-gate-heading-builtin",
                    classes="ds-field-row",
                    markup=False,
                )
            )
            widgets.extend(self._gate_checkbox(gate) for gate in builtin_gates)
        if local_gates:
            widgets.append(
                Static(
                    "Local workspace, web, and Watchlists tools",
                    id="mcp-gate-heading-local",
                    classes="ds-field-row",
                    markup=False,
                )
            )
            widgets.append(
                Static(
                    _LOCAL_TOOLS_INCLUDED_TEXT,
                    id="mcp-gate-local-includes",
                    classes="ds-field-row",
                    markup=False,
                )
            )
            # task-3240 fix round 1 (Important 1): the master switch is
            # always enabled itself (only ITS OWN checkbox toggles it) --
            # every OTHER local-group gate is disabled while it's off, and
            # a dependency note explains why rather than leaving two
            # apparently-independent checkboxes with no visible link.
            master_gate = next(
                (g for g in local_gates if g.key == LOCAL_TOOLS_MASTER_KEY), None
            )
            master_enabled = master_gate.enabled if master_gate is not None else True
            if not master_enabled:
                widgets.append(
                    Static(
                        _LOCAL_TOOLS_MASTER_OFF_NOTE_TEXT,
                        id="mcp-gate-local-master-off-note",
                        classes="ds-field-row",
                        markup=False,
                    )
                )
            widgets.extend(
                self._gate_checkbox(
                    gate,
                    disabled=(
                        not master_enabled and gate.key != LOCAL_TOOLS_MASTER_KEY
                    ),
                )
                for gate in local_gates
            )
        widgets.append(
            Static(
                _TOOL_GATE_NOTE_TEXT,
                id="mcp-gate-toggles-note",
                classes="ds-field-row",
                markup=False,
            )
        )
        return widgets

    async def _rebuild_tool_gate_checkboxes(self) -> None:
        """Rebuild `#mcp-detail-tool-gates` from `_tool_gate_widgets()`.

        Mirrors `_rebuild_builtin_toggles()`'s awaited remove-then-mount
        discipline for the identical reason.
        """
        container = self.query_one("#mcp-detail-tool-gates", Vertical)
        await container.remove_children()
        widgets = self._tool_gate_widgets()
        container.display = bool(widgets)
        if widgets:
            await container.mount_all(widgets)

    def _detail_text(
        self, snapshot: ReadinessSnapshot, *, mutations_available: bool = False
    ) -> str:
        detail = snapshot.detail or {}
        lines: list[str] = [snapshot.message, ""]
        if snapshot.source == "server" and isinstance(detail.get("raw"), dict):
            # T9: an external-server record (server_external_record_readiness
            # sets detail["raw"] to the raw record; a plain server-target
            # snapshot never does). Reached here only when mutations are
            # gated off -- the workbench routes an available edit straight to
            # `show_server_mutations()` instead (see
            # `MCPWorkbench._show_selected_detail()`).
            raw = detail["raw"]
            lines.append(f"Connection · {snapshot.transport}")
            lines.append(f"Enabled · {'yes' if raw.get('enabled', True) else 'no'}")
            lines.append(f"Credentials · {snapshot.auth_display}")
            # Task 5 (MCP Hub Phase 6, §14 Advanced-opt-in compensation):
            # server-source records show resource/prompt COUNTS only --
            # straight off the snapshot fields `server_external_record_
            # readiness()` derives from the record's own reported counts (or
            # raw lists), the same existing payload the overview table's
            # Tools column already reads. No names/URIs: the server owns
            # those, and the read-only listing lives on the LOCAL discovery
            # snapshot below; test-read/test-get remain via opt-in Advanced.
            lines.append(f"Resources · {_count_display(snapshot.resource_count)}")
            lines.append(f"Prompts · {_count_display(snapshot.prompt_count)}")
            if not mutations_available:
                lines.append("")
                lines.append(_MUTATIONS_GATED_TOOLTIP)
        elif snapshot.source == "local":
            args = redact_args([str(a) for a in detail.get("args") or []])
            lines.append(
                f"Command · {detail.get('command') or '—'} {' '.join(args)}".rstrip()
            )
            placeholders = detail.get("env_placeholders") or {}
            missing = set(detail.get("missing_env") or [])
            for env_key, raw in placeholders.items():
                # Reuse the same canonicalization `missing` was computed
                # with (env_placeholder_names() strips whitespace *then*
                # the $/${} wrapper) instead of a local ad hoc
                # `str(raw).strip("${}")`, which leaves surrounding
                # whitespace intact and so never matches `missing` for a
                # value like " $MY_KEY " (F5).
                names = env_placeholder_names({env_key: raw})
                is_missing = bool(names) and names[0] in missing
                marker = "missing" if is_missing else "set"
                lines.append(f"Env · {env_key} ({marker})")
            # Task 5 (MCP Hub Phase 6, §14 Advanced-opt-in compensation):
            # resources and prompts get their own compact, always-present
            # lines here -- with Advanced now opt-in (see mcp_inspector.py),
            # this detail body is the only place a user sees a local
            # server's resource URIs or prompt names without deliberately
            # revealing the legacy runner. Tools keeps the same "Kind · N:
            # names" shape it always had; `_named_items_text()` now backs
            # all three uniformly (defensive reads, explicit "none" empty
            # copy) rather than duplicating the join/truncate logic per kind.
            discovery = detail.get("discovery_snapshot") or {}
            lines.append(
                f"Tools · {_named_items_text(discovery.get('tools'), key='name')}"
            )
            lines.append("")
            lines.append(
                f"Resources · {_named_items_text(discovery.get('resources'), key='uri')}"
            )
            lines.append(
                f"Prompts · {_named_items_text(discovery.get('prompts'), key='name')}"
            )
        elif snapshot.source == "server":
            base_url = str(detail.get("base_url") or "")
            lines.append(f"Base URL · {redact_url(base_url) if base_url else '—'}")
            lines.append(f"Auth · {snapshot.auth_display}")
            # Task 5 (MCP Hub Phase 6): server-source records don't carry a
            # local discovery_snapshot to list names/URIs from -- counts
            # only, straight off the snapshot's own tool_count/
            # resource_count/prompt_count fields (the same "existing
            # payload" tool_count already reads for the overview table's
            # Tools column; server_external_record_readiness() populates
            # resource_count/prompt_count the identical record.get(...)-or-
            # len(list) way).
            lines.append(f"Resources · {_count_display(snapshot.resource_count)}")
            lines.append(f"Prompts · {_count_display(snapshot.prompt_count)}")
            lines.append("External server records: see Advanced ▸ External Servers.")
        else:  # builtin
            lines.append("Runs over stdio when an MCP client launches it:")
            lines.append("  python3 -m tldw_chatbook.MCP")
            # A3c/Task 10: the old "Exposes · tools, resources" prose line
            # (a human-readable summary of the expose_* flags) is now the
            # four Checkbox rows built by `_builtin_toggle_widgets()` --
            # this body text no longer dumps flags at all, raw or humanized.
        return "\n".join(lines)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            raw_key = str(event.row_key.value)
            # Translate a de-duped table key (e.g. "local:unknown#2") back
            # to the canonical server_key -- see F3 in update_overview().
            server_key = self._row_key_to_server_key.get(raw_key, raw_key)
            self.post_message(self.ServerRowSelected(server_key))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Forward a detail-pane Checkbox toggle as `BuiltinFlagChanged` or
        `ToolGateChanged` (task-3240), whichever id table matches.

        Mount-echo note (verified against `textual.widgets._toggle_button.
        ToggleButton`, Task 10): the base class wraps its constructor's
        initial `value` set in `self.prevent(self.Changed)` AND declares the
        `value` reactive with `init=False` -- unlike `Select` (see
        mcp_rail.py's `_ECHO_CONSUMED`/`_displayed_scope_value` sentinels),
        constructing/mounting a Checkbox with a non-default initial value
        does NOT itself fire `Changed`. No compare-before-post guard is
        needed here; `test_showing_builtin_detail_does_not_post_builtin_
        flag_changed` in test_mcp_servers_mode.py pins this down, and its
        task-3240 sibling pins the same for gate checkboxes.
        """
        checkbox_id = event.checkbox.id or ""
        key = _BUILTIN_CHECKBOX_KEYS.get(checkbox_id)
        if key is not None:
            event.stop()
            self.post_message(self.BuiltinFlagChanged(key, event.value))
            return
        gate_target = self._tool_gate_ids.get(checkbox_id)
        if gate_target is not None:
            event.stop()
            section, gate_key = gate_target
            self.post_message(self.ToolGateChanged(section, gate_key, event.value))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "mcp-add-server":
            event.stop()
            self.post_message(self.AddServerRequested())
            return
        if button_id == "mcp-import-server":
            event.stop()
            self.post_message(self.ImportServersRequested())
            return
        if button_id == "mcp-detail-back":
            # Task 11: breadcrumb -- reuses ServerRowSelected(None), the
            # same "clear the selection" path the rail's "All servers" row
            # drives via MCPRail.ServerSelected(None).
            event.stop()
            self.post_message(self.ServerRowSelected(None))
            return
        if button_id == "mcp-builtin-enable":
            # F-051: the off/opt-in affordance performs the fix itself --
            # same message the detail view's Enabled checkbox posts; the
            # workbench persists [mcp].enabled and resyncs.
            event.stop()
            self.post_message(self.BuiltinFlagChanged("enabled", True))
            return
        if button_id.startswith("mcp-callout-"):
            # Task 11: actionable callout -- jump straight to the problem
            # server's detail view, same destination a table-row click for
            # that server would reach.
            event.stop()
            index = int(button_id.removeprefix("mcp-callout-"))
            if 0 <= index < len(self._callout_keys):
                self.post_message(self.ServerRowSelected(self._callout_keys[index]))
            return
        if button_id == "mcp-detail-copy-snippet":
            event.stop()
            snippet = ""
            if self._detail_snapshot is not None:
                snippet = str(
                    (self._detail_snapshot.detail or {}).get("client_snippet") or ""
                )
            if snippet:
                self.app.copy_to_clipboard(snippet)
                self.app.notify("Client config copied to clipboard.")
            return
        if button_id == "mcp-detail-edit":
            event.stop()
            if self._detail_snapshot is not None:
                # Reuses the existing EDIT_CONFIG path (Task 6's
                # `show_form(record)` via the workbench's
                # `on_mcp_inspector_hub_action_requested` handler) instead
                # of duplicating the catalog record lookup here.
                self.post_message(
                    MCPInspector.HubActionRequested(
                        HubAction.EDIT_CONFIG, self._detail_snapshot.server_key
                    )
                )
            return
        if button_id == "mcp-detail-disconnect":
            event.stop()
            if self._detail_snapshot is not None:
                self.post_message(
                    self.DisconnectRequested(self._detail_snapshot.server_key)
                )
            return
        if button_id == "mcp-detail-delete":
            event.stop()
            self._delete_armed = True
            await self._rebuild_detail_toolbar()
            return
        if button_id == "mcp-detail-delete-cancel":
            event.stop()
            self._delete_armed = False
            await self._rebuild_detail_toolbar()
            return
        if button_id == "mcp-detail-delete-confirm":
            event.stop()
            self._delete_armed = False
            await self._rebuild_detail_toolbar()
            if self._detail_snapshot is not None:
                self.post_message(
                    self.DeleteConfirmed(self._detail_snapshot.server_key)
                )
            return
