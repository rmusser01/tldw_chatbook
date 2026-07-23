"""Tests for the Watchlists runs pane."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Static

from tldw_chatbook.UI.Watchlists_Modules.runs_pane import (
    CancelRunRequested,
    RerunRunRequested,
    RunSelected,
    RunsPane,
)


class RunsPaneHarness(App):
    def __init__(self):
        super().__init__()
        self.captured_messages = []

    def compose(self) -> ComposeResult:
        yield RunsPane()

    def on_run_selected(self, message: RunSelected) -> None:
        self.captured_messages.append(("run_selected", message.run))

    def on_cancel_run_requested(self, message: CancelRunRequested) -> None:
        self.captured_messages.append(("cancel_run_requested", message.run_id))

    def on_rerun_run_requested(self, message: RerunRunRequested) -> None:
        self.captured_messages.append(("rerun_run_requested", message.source_id))


@pytest.fixture
def sample_runs():
    return [
        {
            "id": "run-1",
            "source_title": "AI News RSS",
            "status": "completed",
            "started_at": "2026-07-18 10:00",
            "duration": "5m",
            "found_count": 12,
            "processed_count": 10,
            "filtered_count": 2,
            "error_count": 0,
            "source_id": "source-1",
        },
        {
            "id": "run-2",
            "source_title": "Tech Atom Feed",
            "status": "running",
            "started_at": "2026-07-18 11:00",
            "duration": "-",
            "found_count": 5,
            "processed_count": 2,
            "filtered_count": 0,
            "error_count": 1,
            "source_id": "source-2",
        },
    ]


@pytest.mark.asyncio
async def test_runs_pane_renders_table_and_toolbar():
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        assert pane.query_one("#runs-table", DataTable)
        assert pane.query_one("#runs-refresh-button", Button)
        assert pane.query_one("#runs-cancel-button", Button)
        assert pane.query_one("#runs-rerun-button", Button)


@pytest.mark.asyncio
async def test_runs_pane_populates_table(sample_runs):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        await pilot.pause()

        table = pane.query_one("#runs-table", DataTable)
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_runs_pane_selects_run_and_posts_message(sample_runs):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        await pilot.pause()

        table = pane.query_one("#runs-table", DataTable)
        assert "run-1" in [str(key.value) for key in table.rows]

        pane.select_run_by_id("run-1")
        await pilot.pause()

        assert pane.selected_run == sample_runs[0]
        assert app.captured_messages == [("run_selected", sample_runs[0])]


@pytest.mark.asyncio
async def test_runs_pane_disables_cancel_for_non_running_run(sample_runs):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        await pilot.pause()

        pane.select_run_by_id("run-1")
        await pilot.pause()

        cancel_button = pane.query_one("#runs-cancel-button", Button)
        rerun_button = pane.query_one("#runs-rerun-button", Button)
        assert cancel_button.disabled is True
        assert rerun_button.disabled is False


@pytest.mark.asyncio
async def test_runs_pane_enables_cancel_for_running_run(sample_runs):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        await pilot.pause()

        pane.select_run_by_id("run-2")
        await pilot.pause()

        cancel_button = pane.query_one("#runs-cancel-button", Button)
        rerun_button = pane.query_one("#runs-rerun-button", Button)
        assert cancel_button.disabled is False
        assert rerun_button.disabled is False


@pytest.mark.asyncio
async def test_runs_pane_cancel_button_posts_request(sample_runs):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        await pilot.pause()

        pane.select_run_by_id("run-2")
        await pilot.pause()

        pane.query_one("#runs-cancel-button", Button).press()
        await pilot.pause()

        assert ("cancel_run_requested", "run-2") in app.captured_messages


@pytest.mark.asyncio
async def test_runs_pane_rerun_button_posts_request(sample_runs):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        await pilot.pause()

        pane.select_run_by_id("run-1")
        await pilot.pause()

        pane.query_one("#runs-rerun-button", Button).press()
        await pilot.pause()

        assert ("rerun_run_requested", "source-1") in app.captured_messages


@pytest.mark.asyncio
async def test_runs_pane_renders_run_detail():
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.selected_run = {
            "id": "run-1",
            "source_title": "AI News RSS",
            "status": "completed",
            "started_at": "2026-07-18 10:00",
            "duration": "5m",
            "found_count": 12,
            "processed_count": 10,
            "filtered_count": 2,
            "error_count": 0,
        }
        pane.run_items = [
            {"title": "Item A", "status": "new", "alert_count": 1},
            {"title": "Item B", "status": "filtered", "alert_count": 0},
        ]
        pane.run_logs = "Scrape started\nDone"
        await pilot.pause()

        stats = pane.query_one("#runs-detail-stats", Static)
        assert "Status: completed" in str(stats.renderable)
        assert "Found: 12" in str(stats.renderable)

        items_table = pane.query_one("#runs-detail-items", DataTable)
        assert items_table.row_count == 2

        logs = pane.query_one("#runs-detail-logs", Static)
        assert "Scrape started" in str(logs.renderable)
