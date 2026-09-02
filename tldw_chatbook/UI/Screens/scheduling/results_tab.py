"""Results tab for the Schedules workbench (schedules-handoff PR-6 Task 3).

Minimal-honest inbox (plan ruling 1): a `DataTable` of `automation_results`
rows spanning every owner (Task 1's `list_automation_results(owner_id=None)`)
plus a read-only detail pane. Row mutations (read/dismiss/mark-solved/mark
all read) are keybindings owned by `SchedulesWorkbench` -- this widget is a
pure renderer, the same split the Automations tab uses for its `m`/`M`/`y`/
`k` actions (no per-row detail widget there either). Unlike `ConflictsTab`
(whose "Use server"/"Use local" buttons resolve locally, synchronously),
the actions here (`SchedulingService.review_automation_result`/
`resolve_definition`) are async server-aware calls, so they live on the
screen that already owns `run_worker` plumbing for exactly that shape
(`_begin_automation_transfer`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

_KIND_GLYPHS = {"finding": "●", "failure": "✕"}  # ● / ✕

#: Failure rows get the same red-toned Rich style `status_badge_text`
#: (task_detail.py) uses for BLOCKED/CONFLICT -- there is no Rich-usable
#: `$error` CSS token, so this reuses that established literal-style idiom
#: rather than inventing a second one.
_FAILURE_KIND_STYLE = "bold white on red"


def _result_kind_cell(kind: str) -> Text:
    """Render the Kind column: a glyph, red-styled for a failure row."""
    glyph = _KIND_GLYPHS.get(kind, "?")
    if kind == "failure":
        return Text(f" {glyph} ", style=_FAILURE_KIND_STYLE)
    return Text(f" {glyph} ")


def _result_owner_suffix(result: dict[str, Any]) -> str:
    """Row-title owner suffix (ruling 4's `_transfer_row_suffix` idiom).

    `""` for a local row -- only a server-scoped owner gets a visible
    suffix, matching `_transfer_row_suffix`'s own "nothing to say" case.
    """
    from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

    owner_id = result.get("owner_id")
    if not is_server_scoped_owner(owner_id):
        return ""
    owner_id = str(owner_id)
    label = owner_id.split(":", 1)[1] if ":" in owner_id else owner_id
    return f" (server: {label})"


def _format_result_created(
    created_at: str | None, *, now: datetime | None = None
) -> str:
    """Relative "created" timestamp for a results-tab row.

    Same distance-math shape as `task_detail._format_relative`, but a
    result is a past event, not a future run -- "overdue" would be a
    category error, so this uses "X ago" instead. Naive timestamps are
    treated as UTC, matching `_format_relative`'s own convention.
    """
    if not created_at:
        return "-"
    try:
        created = datetime.fromisoformat(str(created_at))
    except ValueError:
        return str(created_at)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    reference = now if now is not None else datetime.now(timezone.utc)
    seconds = (reference - created).total_seconds()
    future = seconds < 0
    seconds = abs(seconds)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        amount = f"{int(seconds // 60)}m"
    elif seconds < 2 * 86400:
        amount = f"{int(seconds // 3600)}h"
    else:
        amount = f"{int(seconds // 86400)}d"
    return f"in {amount}" if future else f"{amount} ago"


def _review_state_cell(review_state: str) -> Text:
    """Render the State column: unread is bold with a leading dot."""
    if review_state == "unread":
        return Text(f"● {review_state}", style="bold")
    return Text(review_state or "-")


def solved_eligibility(
    result: dict[str, Any], definitions_by_id: dict[str, dict[str, Any]]
) -> tuple[bool, str | None]:
    """Whether `result` can currently drive a mark-solved action.

    Plan ruling 1's gate: only a `kind="finding"` result can resolve its
    definition, and only while that definition is still unresolved --
    resolution is definition-level (plan ruling 2), so a definition
    already `"solved"` reads as a confusing no-op if offered again here,
    even though the server itself treats a repeat mark-solved as an
    idempotent no-op (Task 2's report). Server-connectivity/transfer-lock
    refusals are NOT checked here: those only surface once the action
    actually runs, from `ResolveOutcome.reason` (UX-073) -- this is a
    client-side "does this even make sense" gate, not a connectivity
    probe.
    """
    if result.get("kind") != "finding":
        return False, "Only findings can be marked solved."
    definition = definitions_by_id.get(result.get("definition_id"))
    if definition is None:
        return False, "This result's automation definition could not be found."
    if definition.get("resolution_state") == "solved":
        return False, "This automation is already marked solved."
    return True, None


class ResultsTab(Vertical):
    """DataTable + detail pane for the `automation_results` inbox."""

    BUNDLED_CSS = """
    ResultsTab {
        height: 1fr;
    }
    #scheduling-results-table {
        height: 1fr;
    }
    #scheduling-results-empty {
        color: $text-muted;
        padding: 2 1;
        display: none;
    }
    #scheduling-results-detail {
        height: auto;
        max-height: 14;
        padding: 0 1;
        color: $text;
        border-top: solid $surface-lighten-2;
    }
    .results-detail-muted {
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the results tab."""
        super().__init__(**kwargs)
        self._results_by_id: dict[str, dict[str, Any]] = {}
        self._definitions_by_id: dict[str, dict[str, Any]] = {}
        self._selected_result_id: str | None = None

    def compose(self) -> ComposeResult:
        """Build the tab layout."""
        yield Static("Automation results")
        table = DataTable(id="scheduling-results-table")
        table.add_columns("Kind", "Result", "Created", "State")
        yield table
        yield Static(
            "Select a result to see details.",
            id="scheduling-results-detail",
            classes="results-detail-muted",
        )
        yield Static("No results yet.", id="scheduling-results-empty")

    def on_mount(self) -> None:
        """Configure the table cursor."""
        table = self.query_one("#scheduling-results-table", DataTable)
        table.cursor_type = "row"

    @property
    def definitions_by_id(self) -> dict[str, dict[str, Any]]:
        """The definition rows the last `populate()` call was given, by id."""
        return self._definitions_by_id

    def results(self) -> list[dict[str, Any]]:
        """Every result row currently loaded in the tab."""
        return list(self._results_by_id.values())

    def populate(
        self,
        results: list[dict[str, Any]],
        definitions_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Rebuild the table from a fresh `list_automation_results` page.

        Args:
            results: Rows from `ScheduledTasksDB.list_automation_results`.
            definitions_by_id: The owning definitions, by id, for the
                mark-solved eligibility gate (`solved_eligibility`).
        """
        self._definitions_by_id = dict(definitions_by_id or {})
        previous_selection = self._selected_result_id
        self._results_by_id = {result["id"]: result for result in results}

        table = self.query_one("#scheduling-results-table", DataTable)
        table.clear()
        for result in results:
            table.add_row(
                _result_kind_cell(result.get("kind", "")),
                (result.get("title") or result.get("definition_id") or "Untitled")
                + _result_owner_suffix(result),
                _format_result_created(result.get("created_at")),
                _review_state_cell(result.get("review_state", "")),
                key=result["id"],
            )

        has_rows = bool(results)
        table.display = has_rows
        detail = self.query_one("#scheduling-results-detail", Static)
        empty_state = self.query_one("#scheduling-results-empty", Static)
        empty_state.display = "none" if has_rows else "block"
        if not has_rows:
            detail.display = False
            self._selected_result_id = None
            return
        detail.display = True

        row_keys = [result["id"] for result in results]
        if previous_selection in row_keys:
            # Restoring the cursor fires RowHighlighted, which re-records
            # the same id -- belt and braces, set both explicitly (matches
            # the Automations tab's load_automations reconciliation).
            table.cursor_coordinate = (row_keys.index(previous_selection), 0)
            self._selected_result_id = previous_selection
            self._show_detail(self._results_by_id[previous_selection])
        else:
            self._selected_result_id = None
            detail.update("Select a result to see details.")

    @on(DataTable.RowHighlighted, "#scheduling-results-table")
    def _on_result_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show the highlighted result's detail."""
        result_id, result = self._selected_result()
        if result_id is None or result is None:
            return
        self._selected_result_id = result_id
        self._show_detail(result)

    def _selected_result(self) -> tuple[str, dict[str, Any]] | tuple[None, None]:
        """Return the (id, result) pair at the current cursor row."""
        table = self.query_one("#scheduling-results-table", DataTable)
        if table.cursor_row is None or not table.ordered_rows:
            return None, None
        row = table.ordered_rows[table.cursor_row]
        result_id = row.key.value
        return result_id, self._results_by_id.get(result_id)

    def selected_result(self) -> dict[str, Any] | None:
        """Return the result row under the tab's cursor, if any."""
        _, result = self._selected_result()
        return result

    def selected_result_id(self) -> str | None:
        """Return the id of the result row under the tab's cursor, if any."""
        result_id, _ = self._selected_result()
        return result_id

    def _show_detail(self, result: dict[str, Any]) -> None:
        """Render answer / evidence / source_refs / review metadata (spec)."""
        lines: list[str] = []
        title = result.get("title") or "(untitled)"
        lines.append(f"{title}{_result_owner_suffix(result)}")
        lines.append(
            f"Kind: {result.get('kind', '?')}  ·  "
            f"Definition: {result.get('definition_id', '?')}"
        )
        answer = result.get("answer")
        lines.append(f"Answer: {answer}" if answer else "Answer: (none)")

        source_refs = result.get("source_refs") or []
        if source_refs:
            lines.append("Evidence:")
            for ref in source_refs:
                if isinstance(ref, dict):
                    source_type = ref.get("source_type", "?")
                    source_id = ref.get("source_id", "?")
                else:
                    source_type, source_id = "?", ref
                lines.append(f"  - {source_type}: {source_id}")
        else:
            lines.append("Evidence: (none)")

        review_state = result.get("review_state", "unread")
        review_line = f"Review: {review_state}"
        if result.get("reviewed_at"):
            review_line += (
                f" ({result.get('reviewed_by') or '?'} at {result['reviewed_at']})"
            )
        if result.get("review_note"):
            review_line += f" -- {result['review_note']}"
        lines.append(review_line)

        if result.get("kind") == "finding":
            eligible, reason = solved_eligibility(result, self._definitions_by_id)
            lines.append(
                "Solve: eligible"
                if eligible
                else f"Solve: not eligible — {reason}"
            )

        self.query_one("#scheduling-results-detail", Static).update("\n".join(lines))
