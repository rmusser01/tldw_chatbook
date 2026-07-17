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

from datetime import datetime
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import DataTable, Input, Select, Static

from tldw_chatbook.UI.MCP_Modules.mcp_inspector import format_duration_ms

_TABLE_COLUMNS = ("When", "Tool", "Initiator", "Decision", "Duration", "Outcome")

_EMPTY_MESSAGE = "No tool executions recorded yet."

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
    """

    class EntrySelected(Message, namespace="mcp_audit_mode"):
        """Posted when a table row is selected. `index` is the entry's
        position in the FULL cached list `update_entries()` was last given
        (the row key, a stable synthetic index -- stable across a client-
        side re-filter, since that re-renders from the same cached list)."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entries: list[dict[str, Any]] = []
        self._filter_text: str = ""
        self._filter_decision: str | None = None
        self._filter_initiator: str | None = None

    def compose(self) -> ComposeResult:
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

    async def on_mount(self) -> None:
        table = self.query_one("#mcp-audit-table", DataTable)
        table.add_columns(*_TABLE_COLUMNS)
        self._apply_filter()

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

    # -- events ---------------------------------------------------------------

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
        event.stop()
        if event.row_key is None or event.row_key.value is None:
            return
        try:
            index = int(str(event.row_key.value))
        except ValueError:
            # Defensive: row keys are always synthetic ints assigned by
            # `_apply_filter()` above, but a mismatched cell key from a
            # mid-rebuild race is a no-op, not a crash.
            return
        self.post_message(self.EntrySelected(index))
