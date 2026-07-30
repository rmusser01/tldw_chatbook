"""Runs pane for the watchlists screen."""

from __future__ import annotations

import asyncio
from typing import Any

from rich.text import Text
from textual import work
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Static
from textual.worker import get_current_worker

from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .table_selection import highlight_is_user_driven


class RunSelected(Message):
    """Posted when the user selects a run in the runs table."""

    def __init__(self, run: dict[str, Any] | None) -> None:
        self.run = run
        super().__init__()


class CancelRunRequested(Message):
    """Posted when the user requests cancellation of a run."""

    def __init__(self, run_id: Any) -> None:
        self.run_id = run_id
        super().__init__()


class RerunRunRequested(Message):
    """Posted when the user requests re-running a source/job."""

    def __init__(self, source_id: Any) -> None:
        self.source_id = source_id
        super().__init__()


class RunsPane(RecomposeCaptureGuard, Vertical):
    """Run list and run inspector for watchlists."""

    #: task-876: same Rich terminal-agnostic "current item" idiom as
    #: `SourcesPane._SELECTED_ROW_STYLE` -- see that attribute's docstring.
    _SELECTED_ROW_STYLE = "reverse bold"

    runs = reactive[list[dict[str, Any]]]([], recompose=True)
    selected_run = reactive[dict[str, Any] | None](None)
    run_items = reactive[list[dict[str, Any]]]([], recompose=True)
    run_logs = reactive("", recompose=True)
    runtime_backend = reactive("local")

    # Plain attribute, not a reactive: mirrors SourcesPane's
    # `_highlighted_source_key` for the identical reason -- see that
    # attribute's docstring.
    _highlighted_run_key: str | None = None

    def compose(self):
        with Horizontal(id="runs-toolbar", classes="destination-filter-strip"):
            yield Button("Refresh", id="runs-refresh-button", variant="primary")
            yield Button("Cancel run", id="runs-cancel-button", disabled=True)
            yield Button("Re-run source", id="runs-rerun-button", disabled=True)

        selected_key = str(self.selected_run.get("id")) if self.selected_run else None
        table = DataTable(id="runs-table")
        table.add_columns(
            "Source / Job", "Status", "Started", "Duration", "Found", "Processed", "Filtered", "Errors"
        )
        for run in self.runs:
            row_key = str(run.get("id") or id(run))
            table.add_row(
                *self._run_row_cells(run, row_key == selected_key),
                key=row_key,
            )
        # See `SourcesPane.compose()`'s identical assignment for why this is
        # authoritative going forward.
        self._highlighted_run_key = selected_key
        yield table

        selected_run = self.selected_run
        with Vertical(id="runs-detail-pane"):
            yield Static("Run detail", classes="pane-title")
            yield Static(
                self._stats_text(selected_run),
                id="runs-detail-stats",
            )
            yield Static("Items", classes="pane-title")
            items_table = DataTable(id="runs-detail-items")
            items_table.add_columns("Title", "Status", "Alerts")
            for item in self.run_items:
                items_table.add_row(
                    str(item.get("title") or "Untitled"),
                    str(item.get("status") or "-"),
                    str(item.get("alert_count") or "0"),
                )
            yield items_table
            yield Static("Logs", classes="pane-title")
            yield Static(self.run_logs, id="runs-detail-logs")

    @staticmethod
    def _run_row_cells(run: dict[str, Any], highlighted: bool) -> tuple[Text, ...]:
        """One row's cell values, styled if `highlighted` (task-876).

        Shared between `compose()` and `_update_selection_highlight` so both
        draw an identical row -- see `SourcesPane._source_row_cells`.
        """
        style = RunsPane._SELECTED_ROW_STYLE if highlighted else ""
        return (
            Text(str(run.get("source_title") or run.get("job_name") or "Untitled"), style=style),
            Text(str(run.get("status") or "-"), style=style),
            Text(str(run.get("started_at") or "-"), style=style),
            Text(str(run.get("duration") or "-"), style=style),
            Text(str(run.get("found_count") or "0"), style=style),
            Text(str(run.get("processed_count") or "0"), style=style),
            Text(str(run.get("filtered_count") or "0"), style=style),
            Text(str(run.get("error_count") or "0"), style=style),
        )

    @staticmethod
    def _stats_text(run: dict[str, Any] | None) -> str:
        if not run:
            return "No run selected."
        base = (
            f"Status: {run.get('status', '-')}\n"
            f"Started: {run.get('started_at', '-')}\n"
            f"Duration: {run.get('duration', '-')}\n"
            f"Found: {run.get('found_count', 0)} | "
            f"Processed: {run.get('processed_count', 0)} | "
            f"Filtered: {run.get('filtered_count', 0)} | "
            f"Errors: {run.get('error_count', 0)}"
        )
        # TASK-1362 Task 7 (spec §4): a url-family run's check dispositions,
        # so a silent run finally says WHY it was silent (unchanged? withheld
        # under threshold? re-baselined?) instead of just "Found: 0". Absent
        # entirely for feed/API runs, which have no dispositions at all (see
        # `normalize_watchlist_run`) -- `dispositions` is only ever `{}` or
        # missing for those, so no empty "Checks:" line is added.
        dispositions = run.get("dispositions") or {}
        if dispositions:
            # Whole-branch review, Critical 1. `baseline` and `rebaselined` are
            # rendered separately because they mean opposite things: a first
            # check discarded nothing, while a settings-change re-baseline
            # threw away a real diff window in which a change could have been
            # lost. Spec §3 accepts that lost window only on the strength of
            # this line saying so -- one `baseline` count could not, which left
            # the disposition's `reason` with no consumer anywhere in the
            # product.
            withheld = dispositions.get("withheld", 0)
            withheld_text = f"{withheld} withheld"
            max_withheld = run.get("max_withheld_pct")
            if withheld and isinstance(max_withheld, (int, float)):
                # Spec §1: say what is being withheld, not merely that
                # something was. Without the number the user cannot tell a
                # threshold that is slightly too high from one that is
                # swallowing everything.
                withheld_text += f" (largest {float(max_withheld):.1f}%)"
            base += (
                f"\nChecks: {dispositions.get('changed', 0)} changed | "
                f"{dispositions.get('unchanged', 0)} unchanged | "
                f"{withheld_text} | "
                f"{dispositions.get('baseline', 0)} baseline | "
                f"{dispositions.get('rebaselined', 0)} re-baselined "
                "(settings changed)"
            )
        return base

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if event.data_table.id != "runs-table":
            return
        self.select_run_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        event.stop()
        if event.data_table.id != "runs-table":
            return
        self.select_run_by_id(str(event.cell_key.row_key.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Select on cursor movement, which is what a mouse click produces.

        TASK-1105, matching `SourcesPane`. Scoped to `#runs-table`: this pane
        also owns `#runs-detail-items`, whose rows are the *content* of the
        selected run, not runs -- highlighting one of those must not try to
        re-select a run by an item's key (and would resolve to `None`,
        clearing the very selection that produced the detail table).
        """
        event.stop()
        if event.data_table.id != "runs-table":
            return
        if not highlight_is_user_driven(event):
            return
        if event.row_key is not None and event.row_key.value is not None:
            self.select_run_by_id(str(event.row_key.value))

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Same, for a table whose cursor is cell-shaped rather than row-shaped."""
        event.stop()
        if event.data_table.id != "runs-table":
            return
        if not highlight_is_user_driven(event):
            return
        row_key = getattr(event.cell_key, "row_key", None)
        if row_key is not None and row_key.value is not None:
            self.select_run_by_id(str(row_key.value))

    def select_run_by_id(self, run_id: str) -> None:
        """Select the run with the given id and notify listeners."""
        run = None
        for candidate in self.runs:
            if str(candidate.get("id") or "") == run_id:
                run = candidate
                break
        self.selected_run = run

    def watch_selected_run(self, run: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(RunSelected(run))
        self._update_action_buttons()
        self._update_selection_highlight(run)
        if run and str(run.get("status", "")).lower() == "running":
            self._start_run_poll(run)

    def _update_selection_highlight(self, run: dict[str, Any] | None) -> None:
        """Move the table's selected-row highlight without rebuilding it.

        Mirrors `SourcesPane._update_selection_highlight` -- see that
        method's docstring; `selected_run` is not `recompose=True` for the
        same reason `selected_source` is not.
        """
        new_key = str(run.get("id")) if run else None
        old_key = self._highlighted_run_key
        if new_key == old_key:
            return
        try:
            table = self.query_one("#runs-table", DataTable)
        except Exception:
            self._highlighted_run_key = new_key
            return
        try:
            column_keys = list(table.columns.keys())
        except Exception:
            column_keys = []
        for row_key, highlighted in ((old_key, False), (new_key, True)):
            if row_key is None:
                continue
            candidate = next(
                (r for r in self.runs if str(r.get("id") or "") == row_key), None
            )
            if candidate is None:
                continue
            cells = self._run_row_cells(candidate, highlighted)
            for column_key, value in zip(column_keys, cells):
                try:
                    table.update_cell(row_key, column_key, value, update_width=False)
                except Exception:
                    pass
        self._highlighted_run_key = new_key

    def _update_action_buttons(self) -> None:
        try:
            cancel_button = self.query_one("#runs-cancel-button", Button)
            rerun_button = self.query_one("#runs-rerun-button", Button)
        except Exception:
            return
        run = self.selected_run
        can_cancel = run is not None and str(run.get("status", "")).lower() == "running"
        can_rerun = run is not None
        cancel_button.disabled = not can_cancel
        rerun_button.disabled = not can_rerun

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        run = self.selected_run
        if button_id == "runs-cancel-button" and run:
            self.post_message(CancelRunRequested(run.get("id")))
        elif button_id == "runs-rerun-button" and run:
            self.post_message(RerunRunRequested(run.get("source_id")))
        elif button_id == "runs-refresh-button":
            self._update_action_buttons()
        event.stop()

    def _start_run_poll(self, run: dict[str, Any]) -> None:
        self.run_poll(run)

    @work(exclusive=True)
    async def run_poll(self, run: dict[str, Any]) -> None:
        """Poll the selected run while it is running."""
        worker = get_current_worker()
        run_id = run.get("id")
        for _ in range(60):
            if worker.is_cancelled:
                return
            await asyncio.sleep(1)
            current = self.selected_run
            if current is None or str(current.get("id")) != str(run_id):
                return
            if str(current.get("status", "")).lower() != "running":
                return
            self.post_message(RunSelected(current))
