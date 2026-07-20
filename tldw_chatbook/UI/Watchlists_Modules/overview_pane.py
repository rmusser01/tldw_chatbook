"""Overview pane for the watchlists screen."""

from __future__ import annotations

from textual.containers import Grid, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Static


class OverviewPane(Vertical):
    """Dashboard cards and recent failed runs for watchlists."""

    data = reactive({}, recompose=True)

    _CARD_IDS = {
        "total_sources": "overview-total-sources",
        "active_sources": "overview-active-sources",
        "sources_in_error": "overview-sources-in-error",
        "total_items": "overview-total-items",
        "new_items": "overview-new-items",
        "latest_run_status": "overview-latest-run-status",
        "active_alert_rules": "overview-active-alert-rules",
    }

    def _card_value(self, key: str, label: str) -> str:
        value = self.data.get(key, "-")
        return f"{label}\n{value}"

    def compose(self):
        with Grid(id="watchlists-overview-grid"):
            yield Static(
                self._card_value("total_sources", "Total sources"),
                id=self._CARD_IDS["total_sources"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("active_sources", "Active sources"),
                id=self._CARD_IDS["active_sources"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("sources_in_error", "Sources in error"),
                id=self._CARD_IDS["sources_in_error"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("total_items", "Total items"),
                id=self._CARD_IDS["total_items"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("new_items", "New items"),
                id=self._CARD_IDS["new_items"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("latest_run_status", "Latest run status"),
                id=self._CARD_IDS["latest_run_status"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("active_alert_rules", "Active alert rules"),
                id=self._CARD_IDS["active_alert_rules"],
                classes="overview-card",
            )

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
