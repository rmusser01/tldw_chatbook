"""Servers-mode canvas: readiness overview table and per-server detail."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Checkbox, DataTable, Static

from tldw_chatbook.MCP.readiness import (
    STATE_CSS_CLASSES,
    STATE_GLYPHS,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    aggregate_summary,
    env_placeholder_names,
    worst_state,
)
from tldw_chatbook.MCP.redaction import redact_args, redact_url
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel, MCPProfileForm
from tldw_chatbook.UI.MCP_Modules.mcp_server_mutations import MCPServerMutationsPanel

_MUTATIONS_GATED_TOOLTIP = "Requires team, org, or system-admin scope."
# I3: Import always writes to the LOCAL profile store (`_apply_import()` in
# mcp_workbench.py calls `save_local_profile()` unconditionally) -- under
# server source that write would be invisible in the current view (a
# different source/table entirely), so the button is gated off there rather
# than silently landing writes nobody looking at this screen would see.
_IMPORT_GATED_TOOLTIP = "Import creates LOCAL server profiles — switch Source to Local."
_IMPORT_LOCAL_TOOLTIP = "Import servers from a Claude-Desktop-style mcpServers JSON file or paste."

_TABLE_COLUMNS = ("Name", "Transport", "Status", "Tools", "Auth", "Scope")
# Task 11: the Local source never has a meaningful Scope (built-in is
# stdio-only; local profiles are always "Personal") -- the overview table
# omits the column entirely there instead of rendering a column of dashes.
_TABLE_COLUMNS_NO_SCOPE = _TABLE_COLUMNS[:-1]

# Task 11: at most this many actionable recovery callouts render below the
# table -- beyond that, a single "+N more" Static points back at the table
# rather than growing the callout list without bound.
_CALLOUT_CAP = 4

# Task 10: the built-in detail view's Checkbox ids -> the `[mcp]` config key
# (and `BuiltinFlagChanged.key`) each one edits.
_BUILTIN_CHECKBOX_KEYS: dict[str, str] = {
    "mcp-builtin-enabled": "enabled",
    "mcp-builtin-expose-tools": "expose_tools",
    "mcp-builtin-expose-resources": "expose_resources",
    "mcp-builtin-expose-prompts": "expose_prompts",
}


class MCPServersMode(Vertical):
    """Canvas for the Servers mode. Read-only in Phase 1."""

    DEFAULT_CSS = """
    MCPServersMode {
        width: 1fr;
        height: 100%;
        min-height: 0;
    }
    #mcp-servers-table {
        height: 1fr;
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
    #mcp-overview-summary-glyph {
        width: 2;
    }
    #mcp-overview-summary {
        width: 1fr;
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
        change (Phase 1 derivation: `enabled=False` -> NEEDS_SETUP).
        """

        def __init__(self, key: str, value: bool) -> None:
            super().__init__()
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
                    "", id="mcp-overview-summary-glyph", classes="ds-status-badge", markup=False,
                )
                yield Static("", id="mcp-overview-summary", classes="ds-status-badge", markup=False)
            table = DataTable(id="mcp-servers-table")
            table.cursor_type = "row"
            yield table
            yield Vertical(id="mcp-overview-callouts")
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
                    "", id="mcp-detail-title", classes="destination-section", markup=False
                )
            yield Horizontal(id="mcp-detail-toolbar", classes="ds-toolbar")
            with VerticalScroll(id="mcp-detail-scroll"):
                yield Static("", id="mcp-detail-body", classes="ds-field-row", markup=False)
                yield Vertical(id="mcp-detail-builtin-toggles")
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
        # Task 11: per-source columns -- Local (built-in + local profiles)
        # has no meaningful Scope (stdio-only / always "Personal"), so the
        # column is omitted there rather than rendering a column of dashes.
        # Columns are rebuilt from scratch every call (not just when the
        # set actually changes) -- simpler than tracking the previously
        # rendered column set, and this only runs on an actual overview
        # resync, not per keystroke.
        show_scope = source != "local"
        table.clear(columns=True)
        table.add_columns(*(_TABLE_COLUMNS if show_scope else _TABLE_COLUMNS_NO_SCOPE))
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
            # "[red]x[/red]" can't inject styling. Status cells stay plain
            # (theme-token colors aren't addressable per-cell in a
            # DataTable -- Task 11 documented decision; the rail row and
            # inspector badge carry the status color instead).
            row_cells: list[Any] = [
                Text(snap.label),
                snap.transport,
                snap.badge_text(),
                "—" if snap.tool_count is None else str(snap.tool_count),
                Text(snap.auth_display),
            ]
            if show_scope:
                row_cells.append(Text(snap.scope_display))
            table.add_row(*row_cells, key=row_key)
        callouts = self.query_one("#mcp-overview-callouts", Vertical)
        await callouts.remove_children()
        # Task 11: callouts are now actionable one-line Buttons (posting
        # ServerRowSelected straight to the problem row) instead of inert
        # Statics -- capped at _CALLOUT_CAP with a final "+N more" Static
        # pointing back at the table so a source with many problem servers
        # doesn't grow the callout list without bound.
        problem_snapshots = [
            snap
            for snap in self._snapshots
            if snap.state not in (ReadinessState.READY, ReadinessState.CHECKING)
        ]
        visible = problem_snapshots[:_CALLOUT_CAP]
        overflow = len(problem_snapshots) - len(visible)
        self._callout_keys = [snap.server_key for snap in visible]
        callout_widgets: list[Widget] = [
            Button(
                escape_markup(f"{STATE_GLYPHS[snap.state]} {snap.label}: {snap.message}"),
                id=f"mcp-callout-{index}",
                classes="mcp-callout console-action-subdued",
                compact=True,
                tooltip=f"Open {escape_markup(snap.label)}.",
            )
            for index, snap in enumerate(visible)
        ]
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
            await self._rebuild_builtin_toggles()
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
        await self._rebuild_builtin_toggles()
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
        # `snapshot.state is not ReadinessState.NEEDS_SETUP` -- see the
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

    def _detail_text(self, snapshot: ReadinessSnapshot, *, mutations_available: bool = False) -> str:
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
            lines.append(f"Transport · {snapshot.transport}")
            lines.append(f"Enabled · {'yes' if raw.get('enabled', True) else 'no'}")
            lines.append(f"Credentials · {snapshot.auth_display}")
            if not mutations_available:
                lines.append("")
                lines.append(_MUTATIONS_GATED_TOOLTIP)
        elif snapshot.source == "local":
            args = redact_args([str(a) for a in detail.get("args") or []])
            lines.append(f"Command · {detail.get('command') or '—'} {' '.join(args)}".rstrip())
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
            discovery = detail.get("discovery_snapshot") or {}
            for kind in ("tools", "resources", "prompts"):
                items = discovery.get(kind) or []
                names = ", ".join(str(item.get("name") or item.get("uri") or "?") for item in items[:8])
                suffix = f": {names}" if names else ""
                lines.append(f"{kind.title()} · {len(items)}{suffix}")
        elif snapshot.source == "server":
            base_url = str(detail.get("base_url") or "")
            lines.append(f"Base URL · {redact_url(base_url) if base_url else '—'}")
            lines.append(f"Auth · {snapshot.auth_display}")
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
        """Forward a built-in enable/expose Checkbox toggle as `BuiltinFlagChanged`.

        Mount-echo note (verified against `textual.widgets._toggle_button.
        ToggleButton`, Task 10): the base class wraps its constructor's
        initial `value` set in `self.prevent(self.Changed)` AND declares the
        `value` reactive with `init=False` -- unlike `Select` (see
        mcp_rail.py's `_ECHO_CONSUMED`/`_displayed_scope_value` sentinels),
        constructing/mounting a Checkbox with a non-default initial value
        does NOT itself fire `Changed`. No compare-before-post guard is
        needed here; `test_showing_builtin_detail_does_not_post_builtin_
        flag_changed` in test_mcp_servers_mode.py pins this down.
        """
        key = _BUILTIN_CHECKBOX_KEYS.get(event.checkbox.id or "")
        if key is None:
            return
        event.stop()
        self.post_message(self.BuiltinFlagChanged(key, event.value))

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
                snippet = str((self._detail_snapshot.detail or {}).get("client_snippet") or "")
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
                self.post_message(self.DisconnectRequested(self._detail_snapshot.server_key))
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
                self.post_message(self.DeleteConfirmed(self._detail_snapshot.server_key))
            return
