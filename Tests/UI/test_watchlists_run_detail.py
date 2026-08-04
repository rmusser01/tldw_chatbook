"""Selecting a run must populate Run detail, its Items and its Logs — TASK-2306.

UAT finding F34: clicking a run row (and click+Enter) left "Run detail" reading
"No run selected" forever, with the Items and Logs sub-regions unreachable.

Two stacked defects, both proven here against the production screen:

1. **Nothing repainted the detail block.** `RunsPane.selected_run` is
   deliberately not `recompose=True` (a recompose would rebuild `#runs-table`
   under the cursor the user just moved), and its watcher moved the row
   highlight and armed the toolbar but never touched `#runs-detail-stats`. That
   `Static` therefore kept whatever the *first* `compose()` wrote, which ran
   before anything was selected.
2. **Nothing ever produced the data.** `RunsPane.run_items` / `run_logs` had no
   writer anywhere in the product -- only the pane's own unit test set them --
   so the Items and Logs sub-regions were structurally empty in the running app
   whatever was selected.

Every assertion below reads the MOUNTED widgets, never the reactives: the
reactives were not the thing that was broken.
"""

from __future__ import annotations

from typing import Any

import pytest
from rich.text import Text
from textual.widgets import DataTable, Static

from Tests.UI.full_app_destination_context import (
    StaticWatchlistsScopeService,
    active_destination_screen as _active_destination_screen,
    full_app_destination_context as _visual_destination_harness,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane

# TWO runs, and every click targets the SECOND row -- the same discipline as
# `test_watchlists_source_row_click_selects`: with one run, a default row-0
# selection would stand in for the click and the assertions would pass over a
# fully regressed selection path.
RUNS: list[dict[str, Any]] = [
    {
        "id": "local:watchlist_run:1",
        "run_id": 1,
        "backend": "local",
        "entity_kind": "watchlist_run",
        "source_id": 1,
        "source_title": "Summit Route",
        "status": "completed",
        "started_at": "2026-08-04T10:00:00+00:00",
        "duration": "1.2s",
        "found_count": 3,
        "processed_count": 3,
        "filtered_count": 0,
        "error_count": 0,
        "log_text": "fetched 3 items",
    },
    {
        "id": "local:watchlist_run:2",
        "run_id": 2,
        "backend": "local",
        "entity_kind": "watchlist_run",
        "source_id": 2,
        "source_title": "Darknet Diaries",
        "status": "completed",
        "started_at": "2026-08-04T11:00:00+00:00",
        "duration": "4.8s",
        "found_count": 7,
        "processed_count": 5,
        "filtered_count": 2,
        "error_count": 0,
        "log_text": "fetched 7 items",
    },
]

# Run 2 produced items; run 1 produced none. Selecting run 1 after run 2 must
# therefore EMPTY the table -- a stale-detail bug would leave run 2's rows
# standing under run 1's name.
RUN_ITEMS: dict[int, list[dict[str, Any]]] = {
    1: [],
    2: [
        {
            "id": "local:watchlist_item:10",
            "item_id": 10,
            "run_id": 2,
            "title": "Ep. 141",
            "status": "new",
            "alert_count": 2,
        },
        {
            "id": "local:watchlist_item:11",
            "item_id": 11,
            "run_id": 2,
            "title": "Ep. 142",
            "status": "new",
            "alert_count": 0,
        },
    ],
}

# Row 0 sits one line below the header; the second row is at y-offset 2.
FIRST_ROW_OFFSET = (4, 1)
SECOND_ROW_OFFSET = (4, 2)


def _install_item_source(screen) -> list[dict[str, Any]]:
    """Answer the run-detail item query from `RUN_ITEMS`; record every call."""
    calls: list[dict[str, Any]] = []

    async def fake_list_items(**kwargs):
        calls.append(kwargs)
        return [dict(item) for item in RUN_ITEMS.get(int(kwargs.get("run_id") or 0), [])]

    screen._controller.list_items = fake_list_items
    return calls


async def _runs_pane(pilot, host):
    screen = _active_destination_screen(host)
    screen.active_section = "runs"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-runs-pane", RunsPane)
    pane.runs = [dict(run) for run in RUNS]
    await pilot.pause(0.2)
    return screen, pane


async def _settle_until(pilot, predicate, tries: int = 80) -> bool:
    for _ in range(tries):
        await pilot.pause(0.05)
        if predicate():
            return True
    return False


def _stats_text(pane: RunsPane) -> str:
    return str(pane.query_one("#runs-detail-stats", Static).renderable)


def _logs_text(pane: RunsPane) -> str:
    return str(pane.query_one("#runs-detail-logs", Static).renderable)


@pytest.mark.asyncio
async def test_clicking_a_run_row_fills_run_detail_items_and_logs():
    """AC#1/AC#2 (mouse): the whole UAT gesture, against the real table."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        _install_item_source(screen)

        assert "No run selected" in _stats_text(pane), (
            "precondition: nothing is selected before the click"
        )
        table_before = pane.query_one("#runs-table", DataTable)

        await pilot.click("#runs-table", offset=SECOND_ROW_OFFSET)
        filled = await _settle_until(
            pilot,
            lambda: pane.query_one("#runs-detail-items", DataTable).row_count == 2,
        )

        assert pane.selected_run is not None
        assert pane.selected_run["id"] == RUNS[1]["id"]
        stats = _stats_text(pane)
        assert "No run selected" not in stats, (
            "F34: the detail block must stop saying 'No run selected' once a "
            "run IS selected"
        )
        assert "Status: completed" in stats
        assert "Found: 7" in stats, (
            "the detail must describe the run that was clicked, not row 0"
        )
        assert filled, "the clicked run's items must reach #runs-detail-items"
        assert "fetched 7 items" in _logs_text(pane)
        assert pane.query_one("#runs-table", DataTable) is table_before, (
            "the detail must be pushed in place: rebuilding the runs table "
            "would discard the cursor the click just moved"
        )


@pytest.mark.asyncio
async def test_keyboard_selection_fills_run_detail_items_and_logs():
    """AC#1 (keyboard): the same, driven entirely from the cursor keys."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        _install_item_source(screen)

        pane.query_one("#runs-table", DataTable).focus()
        await pilot.pause(0.1)
        await pilot.press("down")
        filled = await _settle_until(
            pilot,
            lambda: pane.query_one("#runs-detail-items", DataTable).row_count == 2,
        )

        assert pane.selected_run is not None
        assert pane.selected_run["id"] == RUNS[1]["id"]
        assert "Found: 7" in _stats_text(pane)
        assert filled, "keyboard selection must reach the detail region too"
        assert "fetched 7 items" in _logs_text(pane)


@pytest.mark.asyncio
async def test_a_runs_items_never_outlive_the_run_they_belong_to():
    """Selecting a run with no items must EMPTY the table, not keep the last."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        _install_item_source(screen)

        await pilot.click("#runs-table", offset=SECOND_ROW_OFFSET)
        assert await _settle_until(
            pilot,
            lambda: pane.query_one("#runs-detail-items", DataTable).row_count == 2,
        ), "precondition: run 2's items are on screen"

        await pilot.click("#runs-table", offset=FIRST_ROW_OFFSET)
        emptied = await _settle_until(
            pilot,
            lambda: pane.query_one("#runs-detail-items", DataTable).row_count == 0,
        )

        assert pane.selected_run["id"] == RUNS[0]["id"]
        assert emptied, (
            "run 2's items must not be left standing under run 1's name"
        )
        assert "Found: 3" in _stats_text(pane)
        assert "fetched 3 items" in _logs_text(pane)


@pytest.mark.asyncio
async def test_run_detail_survives_a_workbench_rebuild():
    """A rebuilt `RunsPane` is re-seeded with the selection AND its detail."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        _install_item_source(screen)

        await pilot.click("#runs-table", offset=SECOND_ROW_OFFSET)
        assert await _settle_until(
            pilot,
            lambda: pane.query_one("#runs-detail-items", DataTable).row_count == 2,
        ), "precondition: the detail is populated"

        # `[` toggles a rail, which rebuilds the workbench and with it the pane.
        await pilot.press("[")
        rebuilt = await _settle_until(
            pilot,
            lambda: screen.query_one("#watchlists-runs-pane", RunsPane) is not pane,
        )
        assert rebuilt, "precondition: the pane really was reconstructed"
        fresh = screen.query_one("#watchlists-runs-pane", RunsPane)

        assert "Found: 7" in _stats_text(fresh)
        assert fresh.query_one("#runs-detail-items", DataTable).row_count == 2, (
            "a rebuilt pane seeded with a selection but no detail renders the "
            "exact blank this task exists to remove"
        )
        assert "fetched 7 items" in _logs_text(fresh)


@pytest.mark.asyncio
async def test_run_detail_lands_when_the_push_happens_in_the_mount_window():
    """TASK-2200's window: `_is_mounted` is False while the DOM is queryable.

    `MessagePump._pre_process` sets `_is_mounted` in its `finally`, AFTER
    dispatching both `Compose` and `Mount`, so a loader that finishes inside
    `on_mount` (which the Watchlists run deep link does, on a cold database)
    runs with `is_mounted` False and every widget already present. This
    reconstructs that state rather than racing for it.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        _install_item_source(screen)

        await pilot.click("#runs-table", offset=SECOND_ROW_OFFSET)
        assert await _settle_until(
            pilot,
            lambda: pane.query_one("#runs-detail-items", DataTable).row_count == 2,
        ), "precondition: the ordinary path fills the detail"

        # Rewind the detail to what a pane looks like the instant before its
        # loader answers, with the selection (which the deep link arms
        # pre-mount) already standing.
        pane.run_items = []
        pane.run_logs = ""
        await pilot.pause(0.1)
        assert pane.query_one("#runs-detail-items", DataTable).row_count == 0, (
            "precondition: rewound to an empty detail"
        )
        assert screen.is_attached, "precondition: the DOM is live throughout"

        screen._is_mounted = False
        pane._is_mounted = False
        try:
            await screen._load_run_detail(dict(RUNS[1]))
            await pilot.pause(0.1)
            rows_in_window = pane.query_one("#runs-detail-items", DataTable).row_count
            logs_in_window = _logs_text(pane)
        finally:
            screen._is_mounted = True
            pane._is_mounted = True

        assert rows_in_window == 2, (
            "an `is_mounted` guard on this push would drop the deep link's "
            "detail on the floor with nothing to re-request it"
        )
        assert "fetched 7 items" in logs_in_window


@pytest.mark.asyncio
async def test_a_deep_linked_run_arrives_with_its_detail():
    """The deep link cannot rely on `RunSelected` to trigger the detail load.

    `RunsPane.watch_selected_run` posts that message only `if self.is_mounted`,
    and `_load_runs` is started by `on_mount` -- inside the window where
    `is_mounted` is still False (TASK-2200). So the loader has to ask for the
    detail itself, or a deep-linked run lands selected with a blank detail.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        _install_item_source(screen)

        async def fake_list_runs(**_kwargs):
            return [dict(run) for run in RUNS]

        screen._controller.list_runs = fake_list_runs
        screen._pending_navigation_run_id = "local:watchlist_run:2"
        screen._pending_navigation_run_backend = "local"

        # THE WINDOW. `_load_runs` is started by `on_mount`, and a pane whose
        # `_is_mounted` has not been flipped yet refuses to post `RunSelected`
        # -- so the message path that serves a mouse click is simply absent
        # here. Reconstructed rather than raced for, the same technique
        # TASK-2200's own mount-window test uses.
        pane._is_mounted = False
        try:
            await screen._load_runs()
            await pilot.pause(0.1)
        finally:
            pane._is_mounted = True

        live = screen.query_one("#watchlists-runs-pane", RunsPane)
        assert live.selected_run is not None
        assert live.selected_run["id"] == RUNS[1]["id"]
        assert live.query_one("#runs-detail-items", DataTable).row_count == 2, (
            "a deep-linked run must arrive with its Items, not a blank pane"
        )
        assert "fetched 7 items" in _logs_text(live)


@pytest.mark.asyncio
async def test_clearing_the_selection_off_the_runs_tab_drops_the_mirrored_detail():
    """The screen's mirror is keyed to `selected_run` and must follow it.

    `_apply_tree_scope`, the backend switch and `_delete_run` all clear
    `selected_run` wherever the user happens to be. With the `RunsPane`
    mounted, the pane's own `RunSelected(None)` reaches the loader and cleans
    up; with it NOT mounted -- any other tab -- nothing does, and the next
    visit to Runs would seed a fresh pane with no selection and the departed
    run's items still in its table.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        _install_item_source(screen)

        await pilot.click("#runs-table", offset=SECOND_ROW_OFFSET)
        assert await _settle_until(
            pilot,
            lambda: pane.query_one("#runs-detail-items", DataTable).row_count == 2,
        ), "precondition: the detail is populated"

        screen.active_section = "sources"
        await pilot.pause(0.3)
        assert not screen.query("#watchlists-runs-pane"), (
            "precondition: the runs pane is gone, so it cannot self-correct"
        )
        # What `watch_runtime_backend` and `_apply_tree_scope` do, from any tab.
        screen.selected_run = None

        screen.active_section = "runs"
        await pilot.pause(0.4)

        live = screen.query_one("#watchlists-runs-pane", RunsPane)
        assert live.selected_run is None
        assert live.query_one("#runs-detail-items", DataTable).row_count == 0, (
            "a pane with nothing selected must not be seeded with the items of "
            "the run that WAS selected"
        )
        assert "No run selected" in _stats_text(live)


def test_a_run_with_no_log_says_so_rather_than_rendering_blank():
    """An empty Logs box reads as "never ran"; the absence is said out loud."""
    from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
        WatchlistsCollectionsScreen,
    )

    assert WatchlistsCollectionsScreen._run_log_text({"log_text": "hello"}) == "hello"
    assert (
        WatchlistsCollectionsScreen._run_log_text({"error_msg": "404 Not Found"})
        == "404 Not Found"
    )
    assert (
        WatchlistsCollectionsScreen._run_log_text({})
        == "No log was recorded for this run."
    )


@pytest.mark.asyncio
async def test_run_item_titles_reach_the_table_inert():
    """Item titles are remote content; `DataTable` markup-parses bare strings.

    `default_cell_formatter` runs `Text.from_markup` over any `str` cell, so a
    feed entry titled `[bold red]...[/]` would be interpreted rather than shown.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    hostile = "[bold red]Ep. 143[/] [link=file:///etc/passwd]x[/link]"
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)

        pane.run_items = [{"title": hostile, "status": "new", "alert_count": 0}]
        await pilot.pause(0.1)

        table = pane.query_one("#runs-detail-items", DataTable)
        assert table.row_count == 1
        cell = table.get_cell_at((0, 0))
        assert isinstance(cell, Text), (
            "a bare `str` cell would be markup-parsed by DataTable"
        )
        assert cell.plain == hostile
        assert cell.spans == [], "no markup may have been applied"
