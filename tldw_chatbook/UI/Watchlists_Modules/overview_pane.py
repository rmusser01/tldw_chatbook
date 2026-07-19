"""Overview pane for the watchlists screen."""

from __future__ import annotations

from textual.containers import Grid, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Static


class OverviewPane(Vertical):
    """Dashboard cards and recent failed runs for watchlists."""

    data = reactive({}, recompose=True)

    def compose(self):
        feed_count = self.data.get("feed_count", "-")
        update_count = self.data.get("update_count", "-")
        activity_status = self.data.get("activity_status", "-")
        with Grid(id="watchlists-overview-grid"):
            yield Static(f"Feeds: {feed_count}", id="overview-feeds")
            yield Static(f"Updates: {update_count}", id="overview-updates")
            yield Static(f"Activity: {activity_status}", id="overview-activity")
        yield Static("Recent failed runs", classes="pane-title")
        table = DataTable(id="overview-failed-runs")
        table.add_columns("Source", "Status", "Error")
        for run in self.data.get("failed_runs", []):
            table.add_row(
                run.get("source_title", ""),
                run.get("status", ""),
                run.get("error_msg", ""),
            )
        yield table

