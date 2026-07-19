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
            "feed_count": 3,
            "update_count": 12,
            "activity_status": "healthy",
            "failed_runs": [],
        }
        await pilot.pause()

        assert pane.query_one("#watchlists-overview-grid")
        feeds = pane.query_one("#overview-feeds")
        updates = pane.query_one("#overview-updates")
        activity = pane.query_one("#overview-activity")
        table = pane.query_one("#overview-failed-runs")

        assert "Feeds: 3" in str(feeds.renderable)
        assert "Updates: 12" in str(updates.renderable)
        assert "Activity: healthy" in str(activity.renderable)
        assert table.row_count == 0


@pytest.mark.asyncio
async def test_overview_pane_renders_failed_runs():
    app = OverviewPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(OverviewPane)
        pane.data = {
            "feed_count": 1,
            "update_count": 5,
            "activity_status": "degraded",
            "failed_runs": [
                {"source_title": "RSS A", "status": "timeout", "error_msg": "slow"},
            ],
        }
        await pilot.pause()

        table = pane.query_one("#overview-failed-runs")
        assert table.row_count == 1
        assert list(table.get_row_at(0)) == ["RSS A", "timeout", "slow"]
