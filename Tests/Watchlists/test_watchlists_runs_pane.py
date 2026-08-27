"""Tests for the Watchlists runs pane."""

import pytest
from rich.style import Style
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Static

from tldw_chatbook.UI.Watchlists_Modules.runs_pane import (
    CancelRunRequested,
    RefreshRunsRequested,
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

    def on_refresh_runs_requested(self, message: RefreshRunsRequested) -> None:
        self.captured_messages.append(("refresh_runs_requested",))

    def on_rerun_run_requested(self, message: RerunRunRequested) -> None:
        self.captured_messages.append(
            (
                "rerun_run_requested",
                message.runtime_backend,
                message.target_id,
                message.name,
            )
        )


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
    async with app.run_test(size=(120, 40)):
        pane = app.query_one(RunsPane)
        assert pane.query_one("#runs-table", DataTable)
        assert pane.query_one("#runs-refresh-button", Button)
        assert pane.query_one("#runs-cancel-button", Button)
        assert pane.query_one("#runs-rerun-button", Button)


@pytest.mark.asyncio
async def test_runs_pane_carries_one_line_of_guidance_when_empty():
    """TASK-2313, AC#4: a bare empty table read as broken next to
    Overview's rich first-run guidance."""
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)):
        pane = app.query_one(RunsPane)
        assert pane.runs == [], "precondition: nothing seeded"
        hint = pane.query_one("#runs-empty-state", Static)
        assert "No runs yet" in str(hint.renderable)


@pytest.mark.asyncio
async def test_runs_pane_hides_the_guidance_once_rows_exist(sample_runs):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        await pilot.pause()
        assert not pane.query("#runs-empty-state")


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

        assert (
            "rerun_run_requested",
            "local",
            "source-1",
            "AI News RSS",
        ) in app.captured_messages


@pytest.mark.asyncio
async def test_runs_pane_refresh_button_posts_refresh_request(sample_runs):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        pane.select_run_by_id("run-1")
        await pilot.pause()

        pane.query_one("#runs-refresh-button", Button).press()
        await pilot.pause()

        assert ("refresh_runs_requested",) in app.captured_messages


@pytest.mark.asyncio
async def test_local_rerun_uses_source_id_and_inert_name():
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = [
            {
                "id": "run-1",
                "source_id": "source-1",
                "source_title": "[not markup]",
                "watchlist_names": ["Morning read", "Security"],
            }
        ]
        pane.select_run_by_id("run-1")
        await pilot.pause()

        pane.query_one("#runs-rerun-button", Button).press()
        await pilot.pause()

        assert app.captured_messages[-1] == (
            "rerun_run_requested",
            "local",
            "source-1",
            "[not markup] · Morning read +1",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_title", "expected_name"),
    [("Server source", "Server source"), (None, "Job job-7")],
)
async def test_server_rerun_uses_job_id_and_display_name(source_title, expected_name):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runtime_backend = "server"
        pane.runs = [{"id": "run-1", "job_id": "job-7", "source_title": source_title}]
        pane.select_run_by_id("run-1")
        await pilot.pause()

        pane.query_one("#runs-rerun-button", Button).press()
        await pilot.pause()

        assert app.captured_messages[-1] == (
            "rerun_run_requested",
            "server",
            "job-7",
            expected_name,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_backend", "run"),
    [
        ("local", {"id": "run-1", "source_title": "Missing source"}),
        ("server", {"id": "run-1", "source_title": "Missing job"}),
    ],
)
async def test_runs_pane_disables_rerun_without_backend_target(runtime_backend, run):
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runtime_backend = runtime_backend
        pane.runs = [run]
        pane.select_run_by_id("run-1")
        await pilot.pause()

        assert pane.query_one("#runs-rerun-button", Button).disabled is True


@pytest.mark.asyncio
async def test_runs_pane_paints_busy_rerun_and_check_now_in_place():
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = [{"id": "run-1", "source_id": "source-1"}]
        pane.select_run_by_id("run-1")
        await pilot.pause()
        table = pane.query_one("#runs-table", DataTable)
        button = pane.query_one("#runs-rerun-button", Button)
        pane.selected_operation_key = "operation-1"

        pane.rerun_operation_keys = {"operation-1"}
        await pilot.pause()
        assert button.disabled is True
        assert str(button.label) == "Re-running..."
        assert pane.query_one("#runs-table", DataTable) is table

        pane.rerun_operation_keys = set()
        pane.busy_operation_keys = {"operation-1"}
        await pilot.pause()
        assert button.disabled is True
        assert str(button.label) == "Checking..."
        assert pane.query_one("#runs-table", DataTable) is table

        pane.busy_operation_keys = set()
        await pilot.pause()
        assert button.disabled is False
        assert str(button.label) == "Re-run source"
        assert pane.query_one("#runs-table", DataTable) is table


@pytest.mark.asyncio
async def test_runs_pane_busy_state_for_another_target_does_not_disable_selection():
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = [{"id": "run-1", "source_id": "source-1"}]
        pane.select_run_by_id("run-1")
        await pilot.pause()

        pane.selected_operation_key = "operation-1"
        pane.busy_operation_keys = {"operation-2"}
        pane.rerun_operation_keys = {"operation-2"}
        await pilot.pause()

        button = pane.query_one("#runs-rerun-button", Button)
        assert button.disabled is False
        assert str(button.label) == "Re-run source"


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


def _disposition_run(**dispositions: int) -> dict[str, object]:
    """A completed url-family run carrying exactly `dispositions`."""
    return {
        "id": "run-1",
        "status": "completed",
        "started_at": "2026-07-29 10:00",
        "duration": "5m",
        "found_count": 5,
        "processed_count": 5,
        "filtered_count": 0,
        "error_count": 0,
        "dispositions": {
            "changed": 0,
            "unchanged": 0,
            "withheld": 0,
            "baseline": 0,
            "rebaselined": 0,
            # task-1394: a URL that raised instead of completing `check_url`.
            "error": 0,
            **dispositions,
        },
    }


def test_stats_text_shows_check_dispositions_when_present():
    """A url-family run's `dispositions` counts render as a `Checks:` line.

    Spec §4: a silent run must say what actually happened. The counts come
    straight through from the service's `stats["dispositions"]` aggregate
    (Task 3), lifted onto the run dict (Task 7).
    """
    text = RunsPane._stats_text(
        _disposition_run(changed=1, unchanged=3, baseline=1, rebaselined=2)
    )
    assert "1 changed" in text
    assert "3 unchanged" in text
    assert "1 baseline" in text
    assert "0 withheld" in text
    assert "2 re-baselined" in text


def test_stats_text_shows_the_error_count_for_a_partially_failed_run():
    """task-1394, AC#2: a partially-failed run must say so, not read clean.

    `url_list`/`sitemap` runs now isolate per-URL failures instead of
    aborting the whole run, so a run that had one bad URL among several
    completes normally with the OTHER urls' items intact. Without a rendered
    error count, that run would look indistinguishable from one where every
    URL was checked cleanly and simply had nothing to report -- the same
    silence spec §4 was written to remove for the other four dispositions.
    """
    text = RunsPane._stats_text(_disposition_run(changed=2, error=1))
    assert "1 error" in text

    # And a clean run explicitly reports zero rather than omitting the count,
    # matching every other counter on this line (`0 withheld`, `0 baseline`).
    clean = RunsPane._stats_text(_disposition_run(changed=2))
    assert "0 error" in clean


def test_stats_text_shows_the_skipped_count_only_when_a_check_was_skipped():
    """task-16838: a URL skipped by the in-flight guard is named, not silent.

    A run that landed while another check of the same source was mid-flight
    completes with a `skipped` disposition instead of double-checking. That
    must be visible here, or the run reads like a clean check that found
    nothing -- the exact ambiguity the disposition line exists to remove.
    Unlike `error`, the segment is conditional: a zero is omitted (the counts
    are zero-filled at write time, so absence always means a true zero), and
    runs recorded before the counter existed render exactly as before.
    """
    text = RunsPane._stats_text(_disposition_run(unchanged=2, skipped=1))
    assert "1 skipped (check already running)" in text

    # No skip: the segment is absent entirely -- both for a new run with a
    # zero-filled counter and for an old row with no `skipped` key at all.
    assert "skipped" not in RunsPane._stats_text(
        _disposition_run(changed=2, skipped=0)
    )
    assert "skipped" not in RunsPane._stats_text(_disposition_run(changed=2))


def test_stats_text_distinguishes_a_first_check_from_a_settings_rebaseline():
    """Whole-branch review, Critical 1: the two must not read alike.

    Spec §3 accepts that a settings-change re-baseline throws away one diff
    window -- a change the page made in it is never reported -- and accepts it
    only because the Runs pane says so. Before this, `_disposition_counts`
    aggregated both causes into one `baseline` count and the disposition's
    `reason` had no consumer anywhere in the product, so the sanctioned lost
    window was silent after all.

    Asserted as an inequality between two renders rather than as a substring,
    because a substring check passes if a collapsed counter happens to print
    the same digits: with one shared counter these two runs render the SAME
    text.
    """
    first_check = RunsPane._stats_text(_disposition_run(baseline=1))
    settings_changed = RunsPane._stats_text(_disposition_run(rebaselined=1))

    assert first_check != settings_changed, (
        "a first check discarded nothing and a re-baseline discarded a real "
        "diff window; one line for both tells the user neither"
    )
    assert "1 baseline" in first_check
    assert "0 re-baselined" in first_check
    assert "1 re-baselined" in settings_changed
    assert "0 baseline" in settings_changed
    # And the line has to say WHY a re-baseline happened, not just count it:
    # "re-baselined" alone is jargon for a state the user did not ask for.
    assert "settings changed" in settings_changed


def test_stats_text_names_the_largest_withheld_change():
    """Spec §1: say what is being withheld, not merely that something was.

    `withheld_percentage` was computed per check, aggregated to
    `max_withheld_pct` and had no reader (whole-branch review, Critical 1). A
    bare "2 withheld" gives the user no way to tell a threshold that is
    slightly too high from one swallowing every change the page makes.
    """
    run = _disposition_run(withheld=2)
    run["max_withheld_pct"] = 3.42
    text = RunsPane._stats_text(run)
    assert "2 withheld" in text
    assert "3.4%" in text


def test_stats_text_omits_the_withheld_percentage_when_nothing_was_withheld():
    """A run that withheld nothing must not print a percentage.

    Both halves matter: a run with a zero count and no recorded percentage
    (the feed-like case) and a run whose count is zero while a stale
    percentage is somehow present. Neither may render a number, because
    "0 withheld (largest 0.0%)" reads as a measurement that was taken.
    """
    assert "largest" not in RunsPane._stats_text(_disposition_run(changed=1))

    stale = _disposition_run(changed=1)
    stale["max_withheld_pct"] = 7.5
    assert "largest" not in RunsPane._stats_text(stale)


def test_stats_text_without_dispositions_key_is_unchanged():
    """A feed run (no `dispositions` key at all) must render exactly the
    pre-Task-7 text -- no empty `Checks:` line tacked on.

    TASK-2308: `Started:` used to interpolate `run.get('started_at')`
    verbatim; it now goes through `humane_timestamp`, so this asserts via
    that formatter rather than pinning one of its outputs -- see the
    identical note on `test_source_row_cells_render_the_normalizer_status_
    summary` in `Tests/UI/test_watchlists_check_now_failure.py`, and
    `Tests/Watchlists/test_humane_time.py` for that formatter's own,
    clock-controlled coverage.
    """
    from tldw_chatbook.UI.Watchlists_Modules.humane_time import humane_timestamp

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
    # The `Source:` line is task-2305's -- a run that could not name its
    # source was the other half of the same UAT finding. What this test pins
    # is that no empty `Checks:` line is tacked on for a run with no
    # dispositions at all.
    assert RunsPane._stats_text(run) == (
        "Source: AI News RSS\n"
        "Status: completed\n"
        f"Started: {humane_timestamp('2026-07-18 10:00')}\n"
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


# --- TASK-2306: the run-detail region follows the selection -----------------


@pytest.mark.asyncio
async def test_selecting_a_run_repaints_the_detail_stats_in_place(sample_runs):
    """F34's render half, at the unit.

    `selected_run` is not `recompose=True` (and must not become one -- a
    recompose rebuilds `#runs-table` under the cursor the click just moved), so
    the detail block is only ever written by `compose()` unless the watcher
    pushes it. It did not, so `#runs-detail-stats` kept the "No run selected."
    the FIRST compose wrote and the Runs tab had a permanently dead detail.
    """
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        await pilot.pause()
        stats = pane.query_one("#runs-detail-stats", Static)
        table = pane.query_one("#runs-table", DataTable)
        assert "No run selected" in str(stats.renderable)

        pane.select_run_by_id("run-2")
        await pilot.pause()

        assert "No run selected" not in str(stats.renderable)
        assert "Status: running" in str(stats.renderable)
        assert "Found: 5" in str(stats.renderable)
        assert pane.query_one("#runs-detail-stats", Static) is stats
        assert pane.query_one("#runs-table", DataTable) is table, (
            "the repaint must not rebuild the table the user just clicked"
        )


@pytest.mark.asyncio
async def test_run_items_and_logs_land_in_the_mounted_widgets(sample_runs):
    """Both detail reactives are pushed in place, not composed."""
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        pane.select_run_by_id("run-1")
        await pilot.pause()
        items_table = pane.query_one("#runs-detail-items", DataTable)
        logs = pane.query_one("#runs-detail-logs", Static)

        pane.run_items = [{"title": "Item A", "status": "new", "alert_count": 3}]
        pane.run_logs = "Scrape started"
        await pilot.pause()

        assert pane.query_one("#runs-detail-items", DataTable) is items_table
        assert items_table.row_count == 1
        assert str(items_table.get_cell_at((0, 2))) == "3"
        assert pane.query_one("#runs-detail-logs", Static) is logs
        assert "Scrape started" in str(logs.renderable)


@pytest.mark.asyncio
async def test_changing_the_selection_drops_the_previous_runs_detail(sample_runs):
    """A run's items and log must never outlive the run they describe."""
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = sample_runs
        pane.select_run_by_id("run-1")
        pane.run_items = [{"title": "Item A", "status": "new", "alert_count": 0}]
        pane.run_logs = "Run one log"
        await pilot.pause()
        assert pane.query_one("#runs-detail-items", DataTable).row_count == 1

        pane.select_run_by_id("run-2")
        await pilot.pause()

        assert pane.run_items == []
        assert pane.run_logs == ""
        assert pane.query_one("#runs-detail-items", DataTable).row_count == 0
        assert "Run one log" not in str(
            pane.query_one("#runs-detail-logs", Static).renderable
        )


# --- TASK-2305: a run row names its source (and watchlist) ------------------


@pytest.mark.asyncio
async def test_a_run_row_names_its_source_and_its_watchlist():
    """F32: a history of "Untitled" rows is unusable.

    The row is the only place a run is identified, and `local_watchlist_runs`
    stores nothing but a `source_id` -- so the name has to arrive on the
    record (TASK-2305's `_RUN_SELECT`) and be rendered here.
    """
    app = RunsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = [
            {
                "id": "run-9",
                "source_title": "Hacker News",
                "watchlist_names": ["Morning read"],
                "status": "completed",
                "found_count": 30,
            }
        ]
        await pilot.pause()

        table = pane.query_one("#runs-table", DataTable)
        assert str(table.get_cell_at((0, 0))) == "Hacker News · Morning read"
        assert str(table.get_cell_at((0, 4))) == "30"


def test_a_run_row_abbreviates_extra_watchlists():
    """`DataTable` sizes a column to its widest cell, so the join is bounded."""
    cell = RunsPane._run_identity(
        {
            "source_title": "Hacker News",
            "watchlist_names": ["Morning read", "Security", "Ops"],
        }
    )

    assert cell == "Hacker News · Morning read +2"


def test_a_run_whose_source_cannot_be_resolved_still_renders():
    assert RunsPane._run_identity({}) == "Untitled"
    assert RunsPane._run_identity({"source_title": "Feed"}) == "Feed"


def test_the_detail_block_lists_every_watchlist_not_just_the_first():
    """The row abbreviates for width; the detail block has no such constraint."""
    text = RunsPane._stats_text(
        {
            "source_title": "Hacker News",
            "watchlist_names": ["Morning read", "Security"],
            "status": "completed",
            "found_count": 30,
        }
    )

    assert text.startswith(
        "Source: Hacker News\nWatchlists: Morning read, Security\n"
    )


@pytest.mark.asyncio
async def test_a_run_identity_reaches_the_table_inert():
    """Source and watchlist names are user-typed; `DataTable` parses markup."""
    app = RunsPaneHarness()
    hostile = "[bold red]Feed[/]"
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RunsPane)
        pane.runs = [{"id": "run-x", "source_title": hostile, "status": "completed"}]
        await pilot.pause()

        cell = pane.query_one("#runs-table", DataTable).get_cell_at((0, 0))
        assert cell.plain == hostile
        assert cell.spans == []


def test_run_identity_strips_control_characters_from_source_and_watchlist_names():
    """Batch-4 review, I1. `_run_identity` wraps its result in a `Text(...)`,
    which stops Rich markup but not a raw control byte -- `source_title` is
    remote-derived (an imported source's name) the same way an item's title
    is, and `watchlist_names` is user-typed free text with the same gap.
    """
    from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane

    cell = RunsPane._run_identity(
        {
            "source_title": "Evil\x9b31mFeed",
            "watchlist_names": ["Morning\x1bread"],
        }
    )
    assert "\x9b" not in cell and "\x1b" not in cell
    assert "Evil" in cell and "31mFeed" in cell
    assert "Morning" in cell and "read" in cell


def test_run_item_row_title_strips_control_characters():
    """Batch-4 review, I1. An item title is remote content (a feed entry's
    own `<title>`); the same stripping the reader applies must reach this
    cell too.
    """
    from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane

    cells = RunsPane._run_item_row_cells({"title": "Evil\x9b31mTitle"})
    assert "\x9b" not in cells[0].plain
    assert "Evil" in cells[0].plain and "31mTitle" in cells[0].plain
