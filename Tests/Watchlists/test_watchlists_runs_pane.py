"""Tests for the Watchlists runs pane."""

import pytest
from rich.style import Style
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


# --- TASK-1362 Task 7: check dispositions in the run detail block ----------


def test_stats_text_shows_check_dispositions_when_present():
    """A url-family run's `dispositions` counts render as a `Checks:` line.

    Spec §4: a silent run must say what actually happened. The four counts
    come straight through from the service's `stats["dispositions"]`
    aggregate (Task 3), lifted onto the run dict (this task).
    """
    run = {
        "id": "run-1",
        "status": "completed",
        "started_at": "2026-07-29 10:00",
        "duration": "5m",
        "found_count": 5,
        "processed_count": 5,
        "filtered_count": 0,
        "error_count": 0,
        "dispositions": {"changed": 1, "unchanged": 3, "withheld": 0, "baseline": 1},
    }
    text = RunsPane._stats_text(run)
    assert "1 changed" in text
    assert "3 unchanged" in text
    assert "1 baseline" in text
    assert "0 withheld" in text


def test_stats_text_without_dispositions_key_is_unchanged():
    """A feed run (no `dispositions` key at all) must render exactly the
    pre-Task-7 text -- no empty `Checks:` line tacked on.
    """
    run = {
        "id": "run-2",
        "source_title": "AI News RSS",
        "status": "completed",
        "started_at": "2026-07-18 10:00",
        "duration": "5m",
        "found_count": 12,
        "processed_count": 10,
        "filtered_count": 2,
        "error_count": 0,
    }
    assert RunsPane._stats_text(run) == (
        "Status: completed\n"
        "Started: 2026-07-18 10:00\n"
        "Duration: 5m\n"
        "Found: 12 | Processed: 10 | Filtered: 2 | Errors: 0"
    )


# --- task-876: selected row is distinguishable from a merely-focused one ---
# See the identical section in test_watchlists_sources_pane.py for context.


def _cell_style(table: DataTable, row_key: str, column_index: int) -> Style:
    column_key = list(table.columns.keys())[column_index]
    raw_style = table.get_cell(row_key, column_key).style
    return Style.parse(raw_style) if isinstance(raw_style, str) else raw_style


@pytest.mark.asyncio
async def test_selected_run_row_is_styled_distinctly_from_others(sample_runs):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        pane.select_run_by_id("run-1")
        await pilot.pause()

        table = pane.query_one("#runs-table", DataTable)
        assert _cell_style(table, "run-1", 0).reverse
        assert not _cell_style(table, "run-2", 0).reverse


@pytest.mark.asyncio
async def test_run_selection_highlight_moves_without_rebuilding_the_table(sample_runs):
    """Mirrors `SourcesPane`'s identical test: `selected_run` is not
    `recompose=True`, so the highlight moves via a targeted `update_cell`.
    """
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        pane.select_run_by_id("run-1")
        await pilot.pause()

        table = pane.query_one("#runs-table", DataTable)
        assert _cell_style(table, "run-1", 0).reverse

        pane.select_run_by_id("run-2")
        await pilot.pause()

        assert pane.query_one("#runs-table", DataTable) is table
        assert not _cell_style(table, "run-1", 0).reverse
        assert _cell_style(table, "run-2", 0).reverse
