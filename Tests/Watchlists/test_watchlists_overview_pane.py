"""Tests for the Watchlists overview pane."""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Watchlists_Modules.overview_pane import OverviewPane


class OverviewPaneHarness(App):
    def compose(self) -> ComposeResult:
        yield OverviewPane()


@pytest.mark.asyncio
async def test_overview_pane_renders_summary_cards():
    app = OverviewPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(OverviewPane)
        pane.data = {
            "total_sources": 3,
            "active_sources": 2,
            "sources_in_error": 1,
            "total_items": 12,
            "new_items": 5,
            "latest_run_status": "completed",
            "active_alert_rules": 2,
            "failed_runs": [],
        }
        await pilot.pause()

        assert pane.query_one("#watchlists-overview-grid")
        assert "Total sources\n3" in str(pane.query_one("#overview-total-sources").renderable)
        assert "Active sources\n2" in str(pane.query_one("#overview-active-sources").renderable)
        assert "Sources in error\n1" in str(pane.query_one("#overview-sources-in-error").renderable)
        assert "Total items\n12" in str(pane.query_one("#overview-total-items").renderable)
        assert "New items\n5" in str(pane.query_one("#overview-new-items").renderable)
        assert "Latest run status\ncompleted" in str(
            pane.query_one("#overview-latest-run-status").renderable
        )
        assert "Active alert rules\n2" in str(
            pane.query_one("#overview-active-alert-rules").renderable
        )
        table = pane.query_one("#overview-failed-runs")
        assert table.row_count == 0


@pytest.mark.asyncio
async def test_overview_pane_renders_failed_runs():
    app = OverviewPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(OverviewPane)
        pane.data = {
            "total_sources": 1,
            "active_sources": 1,
            "sources_in_error": 0,
            "total_items": 5,
            "new_items": 1,
            "latest_run_status": "failed",
            "active_alert_rules": 0,
            "failed_runs": [
                {"source_title": "RSS A", "status": "timeout", "error_msg": "slow"},
            ],
        }
        await pilot.pause()

        table = pane.query_one("#overview-failed-runs")
        assert table.row_count == 1
        assert list(table.get_row_at(0)) == ["RSS A", "timeout", "slow"]
