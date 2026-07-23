"""Runs pane for the watchlists screen."""

from __future__ import annotations

import asyncio
from typing import Any

from textual import work
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Static
from textual.worker import get_current_worker


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


class RunsPane(Vertical):
    """Run list and run inspector for watchlists."""

    runs = reactive[list[dict[str, Any]]]([], recompose=True)
    selected_run = reactive[dict[str, Any] | None](None)
    run_items = reactive[list[dict[str, Any]]]([], recompose=True)
    run_logs = reactive("", recompose=True)
    runtime_backend = reactive("local")

    def compose(self):
        with Horizontal(id="runs-toolbar", classes="destination-filter-strip"):
            yield Button("Refresh", id="runs-refresh-button", variant="primary")
            yield Button("Cancel run", id="runs-cancel-button", disabled=True)
            yield Button("Re-run source", id="runs-rerun-button", disabled=True)

        table = DataTable(id="runs-table")
        table.add_columns(
            "Source / Job", "Status", "Started", "Duration", "Found", "Processed", "Filtered", "Errors"
        )
        for run in self.runs:
            table.add_row(
                str(run.get("source_title") or run.get("job_name") or "Untitled"),
                str(run.get("status") or "-"),
                str(run.get("started_at") or "-"),
                str(run.get("duration") or "-"),
                str(run.get("found_count") or "0"),
                str(run.get("processed_count") or "0"),
                str(run.get("filtered_count") or "0"),
                str(run.get("error_count") or "0"),
                key=str(run.get("id") or id(run)),
            )
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
    def _stats_text(run: dict[str, Any] | None) -> str:
        if not run:
            return "No run selected."
        return (
            f"Status: {run.get('status', '-')}\n"
            f"Started: {run.get('started_at', '-')}\n"
            f"Duration: {run.get('duration', '-')}\n"
            f"Found: {run.get('found_count', 0)} | "
            f"Processed: {run.get('processed_count', 0)} | "
            f"Filtered: {run.get('filtered_count', 0)} | "
            f"Errors: {run.get('error_count', 0)}"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.select_run_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        event.stop()
        self.select_run_by_id(str(event.cell_key.row_key.value))

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
        if run and str(run.get("status", "")).lower() == "running":
            self._start_run_poll(run)

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
