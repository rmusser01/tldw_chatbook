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


# -- task-670: RecomposeCaptureGuard extended to OverviewPane -------------
# OverviewPane.data is a `recompose=True` reactive; before this fix the pane
# carried no guard against task-637's bug class (a capture landing in the
# window between the reactive-driven `refresh(recompose=True)` and the
# deferred teardown it schedules leaks `App.mouse_captured` onto a removed
# widget forever, silently swallowing every mouse event anywhere in the app).


@pytest.mark.asyncio
async def test_data_recompose_releases_a_capture_that_lands_in_the_deferred_teardown_window():
    """A capture that lands after ``pane.data = ...`` schedules the pane's
    recompose (via the `recompose=True` reactive's own `refresh(recompose=True)`
    call) but before the deferred teardown actually runs must not survive it.

    Mirrors ``test_sync_state_recompose_releases_a_capture_that_lands_in_the_
    deferred_teardown_window`` in ``Tests/UI/test_mcp_rail.py`` (task-637):
    ``Widget.refresh(recompose=True)`` only *schedules* the real teardown via
    ``self.call_next(self._check_recompose)`` -- it runs on a LATER
    message-loop iteration, not synchronously. This simulates a MouseDown
    capturing a pane descendant (one of the summary cards) landing in that
    same window, the same way a real one arriving over a laggy transport
    would.
    """
    app = OverviewPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(OverviewPane)
        # TASK-1020: a pane whose `data` is still `{}` is in the LOADING
        # state and renders no cards, so the victim widget this test needs to
        # capture has to be brought into existence first. The recompose being
        # exercised is the second assignment below, exactly as before.
        pane.data = {"total_sources": 1, "failed_runs": []}
        await pilot.pause()
        victim = pane.query_one("#overview-total-sources")

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
        # Same synchronous stack, no `await` yet: simulate a MouseDown
        # capturing a widget the just-scheduled (but not yet run) recompose
        # is about to tear down.
        pilot.app.capture_mouse(victim)
        assert pilot.app.mouse_captured is victim, (
            "test setup didn't actually capture the victim widget"
        )

        # Let the deferred recompose (and everything else queued) run.
        await pilot.pause()
        await pilot.pause()

        assert pilot.app.mouse_captured is None, (
            "mouse_captured is still referencing a widget OverviewPane's "
            "deferred recompose already tore down -- every mouse click "
            "anywhere in the app is now silently swallowed (task-670)"
        )
