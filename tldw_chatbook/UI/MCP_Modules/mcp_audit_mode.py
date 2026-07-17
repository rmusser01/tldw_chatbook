# tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py
"""Audit mode canvas: the JSONL execution-log workbench view, with
decision/initiator/text filters and drill-through row selection.

`MCPAuditMode` renders whatever raw record dicts the workbench hands it
(`MCPExecutionLog.read_recent()`, execution_log.py -- newest first) -- it
never touches the log itself. Filtering by decision, initiator, and free
text is client-side, against a cached copy of the last full list
`update_entries()` was given, mirroring `mcp_tools_mode.py`'s own "render
whatever the workbench hands you" contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Input, Select, Static

from tldw_chatbook.UI.MCP_Modules.mcp_inspector import format_duration_ms

_TABLE_COLUMNS = ("When", "Tool", "Initiator", "Decision", "Duration", "Outcome")

_EMPTY_MESSAGE = "No tool executions recorded yet."

# T8 (MCP Hub Phase 5): the Findings sub-view's own table shape and
# empty-state copy. Findings are a SERVER-SOURCE-ONLY concept (governance
# audit findings live on the tldw_server side; local/builtin profiles have
# no such thing) -- three distinct empty states cover the three reasons
# the table can be absent, each with its own copy so the user can tell
# "not applicable" from "something went wrong":
#   - not server source (or nothing fetched yet -- the same copy a fresh
#     mount shows before the first `_sync_children()` pass, mirroring
#     `MCPWorkbench._source`'s own "local" default)
#   - server source, but the fetch itself failed (fail-soft -- a retry
#     hint, not a crash)
#   - server source, fetch succeeded, zero findings (a legitimately clean
#     connection, not a failure)
_FINDINGS_TABLE_COLUMNS = ("Severity", "Type", "Message")

# Spec-verbatim (task-8-brief.md) -- tests assert this exact copy.
_FINDINGS_LOCAL_EMPTY_MESSAGE = "Findings come from a tldw_server target."
_FINDINGS_FETCH_FAILED_MESSAGE = (
    "Findings could not be loaded. Try refreshing this server's connection."
)
_FINDINGS_NONE_FOUND_MESSAGE = "No governance findings for this connection."


def _finding_field(finding: Mapping[str, Any], key: str) -> str:
    """Defensive raw-dict read for one Findings-table cell.

    Mirrors `hub_tool_catalog.server_tools_from_inventory()`'s own
    tolerant-of-missing-keys style: a finding comes straight off the wire
    (a server-side product, versioned independently), so every field is
    optional -- a missing/blank value renders the same em dash the
    Initiator/Decision columns already use for "not present" rather than
    the literal string "None".
    """
    value = finding.get(key)
    if value in (None, ""):
        return "—"
    return str(value)

# Decision vocabulary (spec-verbatim, task-7-brief.md): the permission
# decision under which one execution-log entry ran or stopped.
# "allowed"/"approved" reached the tool (`execute_hub_tool()`/
# `test_hub_tool()`, unified_control_plane_service.py); "denied"/
# "denied-timeout" never did (`record_tool_decision()`, the Phase 5 agent
# bridge's approval flow -- mcp_tool_provider.py); "downgraded" is a system
# audit note for a rug-pull-invalidated tool-level allow
# (`_audit_downgrade_if_fresh()`), not a call outcome at all -- see
# `_outcome_text()` below.
_DECISION_OPTIONS: list[tuple[str, str]] = [
    ("Allowed", "allowed"),
    ("Approved", "approved"),
    ("Denied", "denied"),
    ("Denied (timeout)", "denied-timeout"),
    ("Downgraded", "downgraded"),
]

_INITIATOR_OPTIONS: list[tuple[str, str]] = [
    ("Test", "test"),
    ("Agent", "agent"),
    ("System", "system"),
]

_BLOCKED_DECISIONS = {"denied", "denied-timeout"}


def _format_when(ts: Any) -> str:
    """Render an `ExecutionRecord.ts` ISO-8601 string for the When column.

    Defensive against a missing/unparsable value (a torn JSONL line that
    happened to still decode as JSON, or a future record shape) -- falls
    back to the raw value's string form rather than raising out of a
    DataTable rebuild.
    """
    if not ts:
        return "—"
    try:
        parsed = datetime.fromisoformat(str(ts))
    except ValueError:
        return str(ts)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _outcome_text(entry: dict[str, Any]) -> str:
    """Render the Outcome column for one execution-log entry.

    "denied"/"denied-timeout" never reached the tool at all (`ok`/
    `duration_ms` are always `False`/`0` for these -- `record_tool_
    decision()`) -- "Blocked" reads more honestly than routing them through
    the ok/duration failure template, which would misleadingly imply an
    attempted, timed run (mirrors `MCPInspector.show_tool_result()`'s own
    `blocked` status-line branch). "downgraded" is a system audit note, not
    a call outcome -- distinct copy so it doesn't read as a failed
    execution. Everything else (allowed/approved) reflects the real `ok`
    outcome of an attempted call.
    """
    decision = str(entry.get("decision") or "")
    if decision in _BLOCKED_DECISIONS:
        return "Blocked"
    if decision == "downgraded":
        return "Downgraded"
    return "OK" if entry.get("ok") else "Failed"


class MCPAuditMode(Vertical):
    """Canvas for the Audit mode: execution-log table, filters, empty state."""

    DEFAULT_CSS = """
    MCPAuditMode {
        width: 1fr;
        height: 100%;
        min-height: 0;
    }
    #mcp-audit-filter-bar {
        height: auto;
        min-height: 0;
    }
    #mcp-audit-filter-text {
        width: 1fr;
    }
    /* T7 (MCP Hub Phase 5) note for T9's bundle pass: the Phase 3/4
    Select-width lesson (`#mcp-tools-filter-server-slot Select`,
    mcp_tools_mode.py / _agentic_terminal.tcss) applies here too --
    `_conversations.tcss`'s bare `Select { width: 100%; }` rule always wins
    over ANY rule targeting a `Select` in this widget's own DEFAULT_CSS
    once the real app bundle is loaded (CSS_PATH always beats DEFAULT_CSS,
    regardless of selector specificity) -- so styling
    `#mcp-audit-filter-decision`/`#mcp-audit-filter-initiator` directly
    would be silently overridden under the real app exactly like Tools
    mode's server Select was (Defect 1, QA round mcp-hub-phase3-2026-07).
    Unlike that fix (a bundle-layer override, T9's job here, not this
    task's), these two Selects are each wrapped in their own small
    FIXED-width slot container instead -- the bundle has no competing rule
    for an arbitrarily-IDed Vertical, so `width: 24`/`width: 20` below
    fully controls each slot's own size, and the bundle's `Select { width:
    100%; }` then resolves against THAT definite width (never `width:
    auto`, which is what caused the original 0x0 collapse -- a percentage
    child inside an auto-sized parent) instead of the whole filter bar,
    leaving real, non-overlapping space for the sibling text Input's own
    `1fr`. Verified empirically against the bundled-CSS harness below
    (`test_table_and_filter_bar_have_nonzero_geometry_with_bundled_css`,
    test_mcp_audit_mode.py) before landing this shape -- an earlier version
    without the slots left the Input at literal 0 width under the real
    bundle. */
    #mcp-audit-filter-decision-slot {
        width: 24;
        height: auto;
    }
    #mcp-audit-filter-initiator-slot {
        width: 20;
        height: auto;
    }
    #mcp-audit-table {
        height: auto;
        max-height: 70%;
        min-height: 4;
    }
    #mcp-audit-empty {
        height: auto;
        min-height: 0;
    }
    /* T8 (MCP Hub Phase 5): the sub-view strip -- two plain Buttons
    ("Executions"/"Findings"), the same "no RadioSet/TabbedContent, a
    Button per option plus an `is-active` marker" house idiom
    `library_skills_canvas.py`'s toggle Buttons and `mcp_rail.py`'s row
    Buttons both already use (`set_class(condition, "is-active")` --
    picked up by the bundle's already-global `.is-active` rule, so no
    bundle edit is needed here at all). Sized to content, not `1fr` --
    unlike the filter bar's Select slots above, a bare Button auto-sizes
    to its label and never collapses to 0 the way a `width: 100%` child
    of an auto-height parent does. */
    #mcp-audit-subview-strip {
        height: auto;
        min-height: 0;
    }
    /* The two sub-view panes below split the canvas by CLIENT-SIDE Python
    state (`self._sub_view`), not CSS `display: none` -- mirrors this
    widget's own `_apply_filter()` precedent for `#mcp-audit-table`/
    `#mcp-audit-empty` (no CSS-only hidden default there either): a single
    source of truth, toggled explicitly by `_apply_subview_display()` at
    `on_mount()` and on every sub-view Button press, rather than splitting
    the "what's visible" answer across both a stylesheet rule and Python
    code. `height: 1fr` (not `auto`) so `#mcp-audit-findings-table`'s own
    `max-height: 70%` below resolves against a definite height, same
    percentage-inside-a-definite-not-auto-parent discipline the Select
    slot comment above documents for the filter bar. */
    #mcp-audit-executions-view,
    #mcp-audit-findings-view {
        width: 1fr;
        height: 1fr;
        min-height: 0;
    }
    #mcp-audit-findings-table {
        height: auto;
        max-height: 70%;
        min-height: 4;
    }
    #mcp-audit-findings-empty {
        height: auto;
        min-height: 0;
    }
    """

    class EntrySelected(Message, namespace="mcp_audit_mode"):
        """Posted when an Executions-table row is selected. `index` is the
        entry's position in the FULL cached list `update_entries()` was
        last given (the row key, a stable synthetic index -- stable across
        a client-side re-filter, since that re-renders from the same
        cached list)."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    class FindingSelected(Message, namespace="mcp_audit_mode"):
        """Posted when a Findings-table row is selected (T8, MCP Hub
        Phase 5). `index` is the finding's position in the full list
        `update_findings()` was last given -- mirrors `EntrySelected`'s own
        row-key contract exactly (Findings has no client-side filtering to
        re-render against, but the same stable-synthetic-index shape keeps
        both selection paths symmetric for the workbench's routing code)."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    class SubViewChanged(Message, namespace="mcp_audit_mode"):
        """Posted when the sub-view toggle switches between Executions and
        Findings (Critical fix, MCP Hub Phase 5 T8 review). Flipping
        `self._sub_view` and re-rendering via `_apply_subview_display()`
        alone never told the workbench anything changed -- a still-mounted
        `#mcp-inspector-audit` detail (WITH its own live "Open tool"/
        "Adjust permission" drill buttons, from a prior Executions-row
        selection) survived a toggle to Findings, and selecting a finding
        there then left BOTH `#mcp-inspector-audit` and `#mcp-inspector-
        finding` visible at once (and the same in reverse). `sub_view` is
        the NEW value ("executions" or "findings") -- the one now visible
        -- so the workbench's handler knows which of the two inspector
        panels is now stale and must be cleared."""

        def __init__(self, sub_view: str) -> None:
            super().__init__()
            self.sub_view = sub_view

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entries: list[dict[str, Any]] = []
        self._filter_text: str = ""
        self._filter_decision: str | None = None
        self._filter_initiator: str | None = None
        # T8 (MCP Hub Phase 5): Findings sub-view state. `_sub_view`
        # defaults to "executions" per spec. `_findings_source` defaults to
        # "local" -- mirrors `MCPWorkbench._source`'s own default -- so a
        # fresh mount, before the workbench's first `_sync_children()` pass
        # ever calls `update_findings()`, already renders the correct
        # server-only empty copy rather than a blank Findings pane.
        self._sub_view: str = "executions"
        self._findings: list[dict[str, Any]] | None = None
        self._findings_source: str = "local"

    def compose(self) -> ComposeResult:
        with Horizontal(id="mcp-audit-subview-strip", classes="ds-toolbar"):
            yield Button(
                "Executions", id="mcp-audit-subview-executions",
                classes="mcp-audit-subview-btn", compact=True,
                tooltip="Show the tool execution log.",
            )
            yield Button(
                "Findings", id="mcp-audit-subview-findings",
                classes="mcp-audit-subview-btn", compact=True,
                tooltip="Show governance findings for the active tldw_server target.",
            )
        with Vertical(id="mcp-audit-executions-view"):
            with Horizontal(id="mcp-audit-filter-bar", classes="ds-toolbar"):
                yield Input(placeholder="Filter tool or server…", id="mcp-audit-filter-text")
                with Vertical(id="mcp-audit-filter-decision-slot"):
                    yield Select(
                        _DECISION_OPTIONS, id="mcp-audit-filter-decision",
                        prompt="All decisions", value=Select.NULL,
                    )
                with Vertical(id="mcp-audit-filter-initiator-slot"):
                    yield Select(
                        _INITIATOR_OPTIONS, id="mcp-audit-filter-initiator",
                        prompt="All initiators", value=Select.NULL,
                    )
            table = DataTable(id="mcp-audit-table")
            table.cursor_type = "row"
            yield table
            with Vertical(id="mcp-audit-empty", classes="ds-recovery-callout"):
                yield Static(_EMPTY_MESSAGE, id="mcp-audit-empty-message", markup=False)
        with Vertical(id="mcp-audit-findings-view"):
            findings_table = DataTable(id="mcp-audit-findings-table")
            findings_table.cursor_type = "row"
            yield findings_table
            with Vertical(id="mcp-audit-findings-empty", classes="ds-recovery-callout"):
                yield Static("", id="mcp-audit-findings-empty-message", markup=False)

    async def on_mount(self) -> None:
        table = self.query_one("#mcp-audit-table", DataTable)
        table.add_columns(*_TABLE_COLUMNS)
        self._apply_filter()
        findings_table = self.query_one("#mcp-audit-findings-table", DataTable)
        findings_table.add_columns(*_FINDINGS_TABLE_COLUMNS)
        self._render_findings()
        self._apply_subview_display()

    # -- data ---------------------------------------------------------------

    async def update_entries(self, entries: list[dict[str, Any]]) -> None:
        """Rebuild the table from a fresh execution-log record list.

        Args:
            entries: The full (unfiltered) `read_recent()` result, newest
                first. Cached on the widget so subsequent filter Input/
                Select changes can re-filter client-side without another
                call here. Rendered in the exact order given -- this widget
                never re-sorts (the log is already newest-first).
        """
        self._entries = list(entries)
        self._apply_filter()

    def _matches(self, entry: dict[str, Any]) -> bool:
        if self._filter_decision and str(entry.get("decision") or "") != self._filter_decision:
            return False
        if self._filter_initiator and str(entry.get("initiator") or "") != self._filter_initiator:
            return False
        if self._filter_text:
            needle = self._filter_text.strip().lower()
            if needle:
                haystack = f"{entry.get('tool_name', '')} {entry.get('server_key', '')}".lower()
                if needle not in haystack:
                    return False
        return True

    def _apply_filter(self) -> None:
        """Re-render the DataTable from `self._entries` under the current
        decision/initiator/text filters, and toggle the table/empty-state
        visibility.

        The empty state is driven by whether the log has ANY entries at all
        (`self._entries`), not by whether the current filter happens to
        match zero rows -- mirrors `mcp_tools_mode.py`'s own diagnostic-
        empty-state rationale: a filter narrowing to zero rows is just an
        empty table, not "nothing has ever been recorded".

        Preserves the cursor's ROW KEY (the synthetic index, stable across
        a re-filter against the same cached list) across the rebuild below
        -- `DataTable.clear()` unconditionally resets `cursor_coordinate` to
        (0, 0), and this canvas re-renders on every keystroke in the filter
        Input/Select. Mirrors `mcp_permissions_mode.py`'s `update_matrix()`
        cursor-key restore (same bug class, same fix shape: a key lookup
        survives a row reorder/insertion/removal from filtering; a bare
        saved index would not).
        """
        table = self.query_one("#mcp-audit-table", DataTable)

        cursor_key: str | None = None
        if table.row_count > 0 and table.cursor_row >= 0:
            try:
                row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
            except Exception:
                # Defensive: a cursor position that doesn't resolve to a
                # live cell is simply not restored, not a crash.
                row_key = None
            if row_key is not None and row_key.value is not None:
                cursor_key = str(row_key.value)

        table.clear(columns=True)
        table.add_columns(*_TABLE_COLUMNS)
        restored_index: int | None = None
        for index, entry in enumerate(self._entries):
            if not self._matches(entry):
                continue
            key = str(index)
            table.add_row(
                Text(_format_when(entry.get("ts"))),
                Text(f"{entry.get('server_key', '')}::{entry.get('tool_name', '')}"),
                Text(str(entry.get("initiator") or "—")),
                Text(str(entry.get("decision") or "—")),
                Text(format_duration_ms(int(entry.get("duration_ms") or 0))),
                Text(_outcome_text(entry)),
                key=key,
            )
            if cursor_key is not None and key == cursor_key:
                restored_index = table.row_count - 1

        if restored_index is not None:
            table.move_cursor(row=restored_index)

        has_any = bool(self._entries)
        table.display = has_any
        self.query_one("#mcp-audit-empty", Vertical).display = not has_any

    # -- T8 (MCP Hub Phase 5): Findings sub-view -----------------------------

    async def update_findings(
        self, findings: list[dict[str, Any]] | None, *, source: str
    ) -> None:
        """Rebuild the Findings sub-view from the workbench's latest fetch.

        Args:
            findings: The raw `governance_audit_findings.items` list from
                the workbench's `(source, target)`-cached "advanced"
                section fetch (mirrors `_load_server_governance_profiles()`'s
                own T11 cache/fetch contract in mcp_workbench.py). `None`
                means either `source` isn't `"server"` (findings are a
                server-only concept) or the fetch itself failed --
                `_render_findings()` below tells those two apart via
                `source` itself, not by inspecting `findings`. An empty
                list is a distinct, legitimate third case: a successful
                fetch that genuinely found zero findings.
            source: The workbench's current `_source` for THIS call, read
                fresh every call rather than cached on first mount -- a
                source switch is reflected the same pass its
                `_sync_children()` resync runs.
        """
        self._findings = findings
        self._findings_source = source
        self._render_findings()

    def _render_findings(self) -> None:
        """Re-render `#mcp-audit-findings-table` from `self._findings`/
        `self._findings_source`, and toggle the table/empty-state
        visibility -- mirrors `_apply_filter()`'s own table/empty-state
        toggle shape, but with three empty-state branches instead of one
        (see `update_findings()`'s docstring for what each means).

        Preserves the cursor's ROW KEY across the rebuild -- same
        `DataTable.clear()`-resets-cursor-to-(0,0) rationale as
        `_apply_filter()`'s own cursor-key restore: `update_findings()` can
        be called again with an unchanged list on a later
        `_sync_children()` pass that didn't touch this table's rows at
        all, and an unguarded rebuild would still silently redirect the
        cursor back to row 0.
        """
        table = self.query_one("#mcp-audit-findings-table", DataTable)
        empty = self.query_one("#mcp-audit-findings-empty", Vertical)
        message = self.query_one("#mcp-audit-findings-empty-message", Static)

        cursor_key: str | None = None
        if table.row_count > 0 and table.cursor_row >= 0:
            try:
                row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
            except Exception:
                # Defensive, same rationale as `_apply_filter()`'s own
                # cursor-key read: a cursor position that doesn't resolve
                # to a live cell is simply not restored, not a crash.
                row_key = None
            if row_key is not None and row_key.value is not None:
                cursor_key = str(row_key.value)

        table.clear(columns=True)
        table.add_columns(*_FINDINGS_TABLE_COLUMNS)

        if self._findings_source != "server":
            table.display = False
            empty.display = True
            message.update(_FINDINGS_LOCAL_EMPTY_MESSAGE)
            return
        if self._findings is None:
            table.display = False
            empty.display = True
            message.update(_FINDINGS_FETCH_FAILED_MESSAGE)
            return
        if not self._findings:
            table.display = False
            empty.display = True
            message.update(_FINDINGS_NONE_FOUND_MESSAGE)
            return

        restored_index: int | None = None
        for index, finding in enumerate(self._findings):
            if not isinstance(finding, Mapping):
                # Defensive, mirrors `server_tools_from_inventory()`'s own
                # "skip non-dict entries entirely" style for a raw,
                # wire-derived list.
                continue
            key = str(index)
            table.add_row(
                Text(_finding_field(finding, "severity")),
                Text(_finding_field(finding, "finding_type")),
                Text(_finding_field(finding, "message")),
                key=key,
            )
            if cursor_key is not None and key == cursor_key:
                restored_index = table.row_count - 1

        if restored_index is not None:
            table.move_cursor(row=restored_index)

        table.display = True
        empty.display = False

    def _apply_subview_display(self) -> None:
        """Toggle the Executions/Findings pane visibility and the sub-view
        Buttons' `is-active` marker from `self._sub_view` -- the single
        source of truth (see the DEFAULT_CSS comment above
        `#mcp-audit-executions-view`/`#mcp-audit-findings-view` for why
        this isn't split across a CSS default too)."""
        executions_active = self._sub_view == "executions"
        self.query_one("#mcp-audit-executions-view", Vertical).display = executions_active
        self.query_one("#mcp-audit-findings-view", Vertical).display = not executions_active
        self.query_one("#mcp-audit-subview-executions", Button).set_class(
            executions_active, "is-active"
        )
        self.query_one("#mcp-audit-subview-findings", Button).set_class(
            not executions_active, "is-active"
        )

    # -- events ---------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-audit-subview-executions":
            event.stop()
            self._sub_view = "executions"
            self._apply_subview_display()
            self.post_message(self.SubViewChanged(self._sub_view))
        elif event.button.id == "mcp-audit-subview-findings":
            event.stop()
            self._sub_view = "findings"
            self._apply_subview_display()
            self.post_message(self.SubViewChanged(self._sub_view))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "mcp-audit-filter-text":
            return
        event.stop()
        self._filter_text = event.value
        self._apply_filter()

    def on_select_changed(self, event: Select.Changed) -> None:
        # No mount-echo guard needed here (unlike mcp_tools_mode.py's
        # per-catalog server Select): both filter Selects are constructed
        # once, in compose(), with `value=Select.NULL` -- the SAME value as
        # the reactive `value` var's own default, so Textual never fires a
        # `Changed` echo for their own mount (`Select.value` only echoes
        # when the constructor value differs from that default). They are
        # also never remounted (their option lists are a fixed vocabulary,
        # not derived from data), so there is no later remount that could
        # echo either.
        if event.select.id == "mcp-audit-filter-decision":
            event.stop()
            self._filter_decision = None if event.value is Select.NULL else str(event.value)
            self._apply_filter()
        elif event.select.id == "mcp-audit-filter-initiator":
            event.stop()
            self._filter_initiator = None if event.value is Select.NULL else str(event.value)
            self._apply_filter()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Route a row selection from EITHER table (both post the same
        `DataTable.RowSelected` message type, distinguished here by
        `event.data_table.id`) to its own message class -- `EntrySelected`
        for the Executions table, `FindingSelected` (T8) for the Findings
        table."""
        event.stop()
        if event.row_key is None or event.row_key.value is None:
            return
        try:
            index = int(str(event.row_key.value))
        except ValueError:
            # Defensive: row keys are always synthetic ints assigned by
            # `_apply_filter()`/`_render_findings()` above, but a
            # mismatched cell key from a mid-rebuild race is a no-op, not
            # a crash.
            return
        if event.data_table.id == "mcp-audit-findings-table":
            self.post_message(self.FindingSelected(index))
        else:
            self.post_message(self.EntrySelected(index))
