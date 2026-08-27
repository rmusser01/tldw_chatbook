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

import asyncio
from copy import deepcopy
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
from tldw_chatbook.UI.Watchlists_Modules.runs_pane import (
    RefreshRunsRequested,
    RunsPane,
)
from tldw_chatbook.tldw_api.exceptions import APIResponseError

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


async def _pump_until(pilot, predicate, tries: int = 80) -> bool:
    """Yield to Textual until an event-driven condition lands, without sleeps."""
    for _ in range(tries):
        if predicate():
            return True
        await pilot.pause()
    return bool(predicate())


async def _wait_worker_group_idle(pilot, screen, group: str) -> bool:
    """Wait only for the worker group owned by the behavior under test."""
    return await _pump_until(
        pilot,
        lambda: not any(
            worker.group == group and not worker.is_finished
            for worker in screen.workers
        ),
    )


async def _refresh_test_pane(pilot, host):
    """Mount Runs and let its initial loader finish before installing fakes."""
    screen = _active_destination_screen(host)
    screen.active_section = "runs"
    assert await _pump_until(
        pilot, lambda: bool(screen.query("#watchlists-runs-pane"))
    )
    assert await _wait_worker_group_idle(pilot, screen, "wc_runs")
    await pilot.pause()
    return screen, screen.query_one("#watchlists-runs-pane", RunsPane)


async def _seed_refresh_snapshot(screen, pane, pilot, *, run=None) -> None:
    """Install one complete mounted Runs snapshot through the selection path."""
    selected = dict(run or RUNS[1])
    rows = [dict(RUNS[0]), selected]

    async def initial_items(**kwargs):
        run_id = int(kwargs.get("run_id") or 0)
        return [dict(item) for item in RUN_ITEMS.get(run_id, [])]

    screen._controller.list_items = initial_items
    screen._loaded_runs = rows
    pane.runs = rows
    await pilot.pause()
    pane.select_run_by_id(str(selected["id"]))
    assert await _wait_worker_group_idle(pilot, screen, "wc_run_detail")
    await pilot.pause()
    assert screen.selected_run is not None
    assert pane.selected_run is not None


def _mounted_run_snapshot(screen, pane) -> dict[str, Any]:
    """The full state a failed refresh is required to leave byte-for-byte."""
    return deepcopy(
        {
            "loaded_runs": screen._loaded_runs,
            "screen_selection": screen.selected_run,
            "pane_rows": pane.runs,
            "pane_selection": pane.selected_run,
            "screen_items": screen._run_detail_items,
            "screen_logs": screen._run_detail_logs,
            "screen_note": screen._run_detail_items_note,
            "pane_items": pane.run_items,
            "pane_logs": pane.run_logs,
            "pane_note": pane.run_items_note,
            "cancel": (
                str(pane.query_one("#runs-cancel-button").label),
                pane.query_one("#runs-cancel-button").disabled,
            ),
            "rerun": (
                str(pane.query_one("#runs-rerun-button").label),
                pane.query_one("#runs-rerun-button").disabled,
            ),
        }
    )


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

        # Rebuild the region the pane lives in, through the production
        # primitive that does it: `WatchlistsWorkbench.refresh_region_content`
        # calls the region factory again and swaps the result in. Every
        # rebuild route reaches this -- a section switch (`_swap_active_
        # section`), a collapsed region being expanded again, the surface
        # drain -- so it is the seam, not a test-only shortcut.
        #
        # This used to be a single `[`, back when any layout key recomposed
        # the whole workbench. task-15461 scoped that to the region whose form
        # moved, so a rail toggle deliberately leaves this pane's instance
        # alone (`test_a_rail_toggle_rebuilds_only_the_toggled_region` pins
        # it), and `z` is refused outright off the Read tab -- which the Runs
        # section is. The contract under test -- a REBUILT pane is re-seeded
        # with its selection AND its detail -- is unchanged; only the trigger
        # moves.
        from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
        from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
            WatchlistsWorkbench,
        )

        await screen.query_one(WatchlistsWorkbench).refresh_region_content(
            Region.ITEMS
        )
        rebuilt = await _settle_until(
            pilot,
            lambda: bool(screen.query("#watchlists-runs-pane"))
            and screen.query_one("#watchlists-runs-pane", RunsPane) is not pane,
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


# --- Review wave, Important 1 / 2 + Minor 2: the Items region says why ------
#
# An empty `#runs-detail-items` renders identically for four unrelated causes,
# directly beneath a stats block that may say `Found: 20`. Each of these tests
# drives ONE cause and asserts the note names THAT cause, so a single generic
# "no items" string would fail all but one of them.


def _note_text(pane: RunsPane) -> str:
    return str(pane.query_one("#runs-detail-items-note", Static).renderable)


async def _select_second_run(pilot, pane) -> None:
    await pilot.click("#runs-table", offset=SECOND_ROW_OFFSET)
    await _settle_until(pilot, lambda: pane.selected_run is not None)


@pytest.mark.asyncio
async def test_a_run_whose_items_a_later_check_reclaimed_says_so():
    """The commonest blank, and the one that contradicts the counts above it.

    `persist_subscription_item`'s `ON CONFLICT … run_id = excluded.run_id`
    re-attributes unchanged items to the newest run, so after ANY re-check
    every older run renders `Found: 7` over an empty table. The storage rule
    stays as it is; the label is what this fixes.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)

        async def no_rows(**_kwargs):
            return []

        screen._controller.list_items = no_rows

        await _select_second_run(pilot, pane)
        assert await _settle_until(pilot, lambda: bool(_note_text(pane)))

        assert pane.query_one("#runs-detail-items", DataTable).row_count == 0
        assert "Processed: 5" in _stats_text(pane), (
            "precondition: the run persisted rows, so their absence IS "
            "re-attribution and not filtering"
        )
        assert "re-claimed" in _note_text(pane), (
            f"the note must name the cause; got {_note_text(pane)!r}"
        )
        assert pane.query_one("#runs-detail-items-note", Static).display is True


@pytest.mark.asyncio
async def test_a_run_that_genuinely_found_nothing_is_not_blamed_on_a_later_check():
    """The discriminating half: `Found: 0` + no rows is not re-attribution."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)

        async def no_rows(**_kwargs):
            return []

        screen._controller.list_items = no_rows
        empty_run = dict(RUNS[1])
        empty_run["found_count"] = 0
        empty_run["processed_count"] = 0
        pane.runs = [dict(RUNS[0]), empty_run]
        await pilot.pause(0.2)

        await _select_second_run(pilot, pane)
        assert await _settle_until(pilot, lambda: bool(_note_text(pane)))

        note = _note_text(pane)
        assert "produced no items" in note, f"got {note!r}"
        assert "re-claimed" not in note, (
            "a run that found nothing must not be told a later check took its "
            "items"
        )


@pytest.mark.asyncio
async def test_a_server_backend_run_says_items_are_not_listed():
    """`WatchlistScopeService.list_items` refuses the server backend outright.

    There is no query to fail, so the region must not draw the same blank a
    local run with no items draws.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        calls = _install_item_source(screen)
        server_run = dict(RUNS[1])
        server_run["backend"] = "server"
        pane.runs = [dict(RUNS[0]), server_run]
        await pilot.pause(0.2)

        await _select_second_run(pilot, pane)
        assert await _settle_until(pilot, lambda: bool(_note_text(pane)))

        assert "server-backend" in _note_text(pane), f"got {_note_text(pane)!r}"
        assert calls == [], (
            "a server run must not even attempt the local-only item query"
        )


@pytest.mark.asyncio
async def test_a_failed_items_query_says_so_and_raises_a_toast():
    """Review wave, Important 2.

    `_load_run_detail` takes the "loaders may log at debug" exemption that
    `test_watchlists_check_now_failure.py` documents, and that exemption is
    paid for with a visible toast. Without one, a denied `items.list` policy
    or a locked database renders byte-identically to "this run produced no
    items".
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    toasts: list[tuple[str, dict]] = []
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        app.notify = lambda message, **kwargs: toasts.append((str(message), kwargs))

        async def denied(**_kwargs):
            raise PermissionError("watchlists.items.list.local is denied")

        screen._controller.list_items = denied

        await _select_second_run(pilot, pane)
        assert await _settle_until(pilot, lambda: bool(_note_text(pane)))

        assert "Could not load" in _note_text(pane), f"got {_note_text(pane)!r}"
        assert toasts, "a failed background read must raise a toast, not just a log"
        message, kwargs = toasts[-1]
        assert "items" in message.lower()
        assert kwargs.get("severity") == "error"
        assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_a_truncated_items_table_says_how_many_it_is_showing():
    """Review wave, Minor 2: `Found: 500` over exactly 200 rows, silently."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        limit = screen._RUN_ITEMS_LIMIT

        async def a_full_page(**kwargs):
            assert kwargs["limit"] == limit
            return [
                {"title": f"Item {index}", "status": "new", "alert_count": 0}
                for index in range(limit)
            ]

        screen._controller.list_items = a_full_page
        big_run = dict(RUNS[1])
        big_run["found_count"] = 500
        big_run["processed_count"] = 500
        pane.runs = [dict(RUNS[0]), big_run]
        await pilot.pause(0.2)

        await _select_second_run(pilot, pane)
        assert await _settle_until(pilot, lambda: bool(_note_text(pane)))

        assert _note_text(pane) == f"Showing the first {limit} of 500 items."


@pytest.mark.asyncio
async def test_a_complete_items_table_carries_no_note():
    """The note is hidden, not merely empty, when there is nothing to say."""
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
        )

        assert _note_text(pane) == ""
        assert pane.query_one("#runs-detail-items-note", Static).display is False


@pytest.mark.asyncio
async def test_the_items_note_does_not_outlive_the_run_it_describes():
    """Selecting another run must not leave the previous run's excuse on screen."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)

        async def no_rows(**_kwargs):
            return []

        screen._controller.list_items = no_rows
        await _select_second_run(pilot, pane)
        assert await _settle_until(pilot, lambda: "re-claimed" in _note_text(pane))

        # Run 1 DOES still own a row, so its note must be absent entirely --
        # not merely replaced by another excuse.
        async def one_row(**_kwargs):
            return [{"title": "Still mine", "status": "new", "alert_count": 0}]

        screen._controller.list_items = one_row
        await pilot.click("#runs-table", offset=FIRST_ROW_OFFSET)
        assert await _settle_until(
            pilot,
            lambda: pane.query_one("#runs-detail-items", DataTable).row_count == 1,
        )

        assert pane.selected_run["id"] == RUNS[0]["id"]
        assert _note_text(pane) == "", (
            f"the previous run's excuse is still on screen: {_note_text(pane)!r}"
        )
        assert pane.query_one("#runs-detail-items-note", Static).display is False


def test_every_empty_items_cause_has_its_own_words():
    """Five roads to an empty table; five different things to say."""
    from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
        WatchlistsCollectionsScreen as Screen,
    )

    notes = {
        Screen._RUN_ITEMS_SERVER_NOTE,
        Screen._RUN_ITEMS_FAILED_NOTE,
        Screen._RUN_ITEMS_REATTRIBUTED_NOTE,
        Screen._RUN_ITEMS_ALL_FILTERED_NOTE,
        Screen._RUN_ITEMS_EMPTY_NOTE,
    }
    assert len(notes) == 5, "each cause must be distinguishable from the others"
    assert not hasattr(Screen, "_RUN_ITEMS_UNIDENTIFIED_NOTE"), (
        "the 'unidentified run' label was unreachable -- `normalize_watchlist_"
        "run` reads `payload['id']` unsubscripted, so every run has a run_id "
        "(re-review, m6)"
    )


@pytest.mark.asyncio
async def test_the_previous_runs_note_is_gone_before_the_loader_answers():
    """The pane clears its own note on selection, not just when told to.

    Same rationale as the `run_items`/`run_logs` clear beside it: between the
    click and the query returning, the pane would otherwise show the PREVIOUS
    run's excuse under the newly selected run's stats. Driven with a
    deliberately slow query so the assertions land inside that window.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)

        async def no_rows(**_kwargs):
            return []

        screen._controller.list_items = no_rows
        await _select_second_run(pilot, pane)
        assert await _settle_until(pilot, lambda: "re-claimed" in _note_text(pane)), (
            "precondition: run 2 carries a note"
        )

        import asyncio

        async def slow(**_kwargs):
            await asyncio.sleep(5)
            return []

        screen._controller.list_items = slow

        await pilot.click("#runs-table", offset=FIRST_ROW_OFFSET)
        assert await _settle_until(
            pilot, lambda: pane.selected_run["id"] == RUNS[0]["id"]
        )
        # THE WINDOW: the new selection has landed, the query has not returned.
        in_window = _note_text(pane)

        assert in_window == "", (
            "run 2's excuse is showing under run 1's stats while run 1's own "
            f"items are still loading: {in_window!r}"
        )


# --- Re-review, I1-b: the discriminator is `processed`, never `found` -------


@pytest.mark.asyncio
async def test_a_run_that_filtered_everything_is_not_blamed_on_a_later_check():
    """The re-review's measured scenario, at the UI.

    A source with an exclude filter, checked ONCE: `found 5 · processed 0 ·
    filtered 5`, zero rows. The first cut of this feature discriminated on
    `found_count` and told the user "a later check re-claimed the items that
    had not changed" — when there was no later check, and nothing had been
    re-claimed. `Found` is what the FETCH saw; `Processed` is what the run
    actually stored, and rows are the only thing this table can show.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)

        async def no_rows(**_kwargs):
            return []

        screen._controller.list_items = no_rows
        filtered_run = dict(RUNS[1])
        filtered_run.update(
            found_count=5, processed_count=0, filtered_count=5, error_count=0
        )
        pane.runs = [dict(RUNS[0]), filtered_run]
        await pilot.pause(0.2)

        await _select_second_run(pilot, pane)
        assert await _settle_until(pilot, lambda: bool(_note_text(pane)))

        note = _note_text(pane)
        assert "excluded by a filter" in note, f"got {note!r}"
        assert "re-claimed" not in note, (
            "a run checked once cannot have had its items re-claimed by a "
            "later check that does not exist"
        )
        assert "produced no items" not in note, (
            "it DID find five; it stored none of them, which is a different "
            "thing to say"
        )


@pytest.mark.asyncio
async def test_a_full_page_of_everything_the_run_stored_is_not_called_truncated():
    """The same bug in reverse (re-review, I1-b).

    `found 500 · processed 200`, returning exactly 200 rows: every row the run
    ever stored is on screen. Keying the truncation line off `found` claimed
    300 more were hidden when the missing 300 were filtered out and never
    stored at all.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        limit = screen._RUN_ITEMS_LIMIT

        async def a_full_page(**_kwargs):
            return [
                {"title": f"Item {index}", "status": "new", "alert_count": 0}
                for index in range(limit)
            ]

        screen._controller.list_items = a_full_page
        run = dict(RUNS[1])
        run.update(found_count=500, processed_count=limit, filtered_count=300)
        pane.runs = [dict(RUNS[0]), run]
        await pilot.pause(0.2)

        await _select_second_run(pilot, pane)
        assert await _settle_until(
            pilot,
            lambda: pane.query_one("#runs-detail-items", DataTable).row_count == limit,
        )

        assert _note_text(pane) == "", (
            "nothing is hidden: the run stored exactly what is on screen — "
            f"got {_note_text(pane)!r}"
        )


def test_the_note_reads_processed_not_found():
    """The discriminator, pinned at the unit against both directions."""
    from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
        WatchlistsCollectionsScreen as Screen,
    )

    filtered_out = {"found_count": 5, "processed_count": 0}
    stored_then_lost = {"found_count": 5, "processed_count": 5}
    never_found = {"found_count": 0, "processed_count": 0}

    assert Screen._run_items_note(filtered_out, []) == (
        Screen._RUN_ITEMS_ALL_FILTERED_NOTE
    )
    assert Screen._run_items_note(stored_then_lost, []) == (
        Screen._RUN_ITEMS_REATTRIBUTED_NOTE
    )
    assert Screen._run_items_note(never_found, []) == Screen._RUN_ITEMS_EMPTY_NOTE


# --- Qodo PR #1348: a poll tick is not a selection -------------------------


def _running_run() -> dict[str, Any]:
    run = dict(RUNS[1])
    run.update(status="running", finished_at=None, processed_count=0, found_count=0)
    return run


@pytest.mark.asyncio
async def test_a_poll_tick_on_an_unchanged_run_schedules_no_detail_load():
    """`run_poll` fires once a second for up to a minute, with no user action.

    It used to re-post `RunSelected`, which the screen cannot tell from a
    click — so a selected running run ran a full `_load_run_detail`, worker
    and item query included, every single second.
    """
    from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunProgressTick

    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        running = _running_run()
        pane.runs = [dict(RUNS[0]), running]
        await pilot.pause(0.2)

        item_calls = _install_item_source(screen)

        async def unchanged(**_kwargs):
            return dict(running)

        screen._controller.get_run = unchanged

        await pilot.click("#runs-table", offset=SECOND_ROW_OFFSET)
        assert await _settle_until(pilot, lambda: pane.selected_run is not None)
        # The click itself is a real selection and loads the detail once.
        assert await _settle_until(pilot, lambda: len(item_calls) == 1)
        after_click = len(item_calls)

        for _ in range(3):
            screen.post_message(RunProgressTick(running["id"]))
            await pilot.pause(0.2)
        await pilot.pause(0.3)

        assert len(item_calls) == after_click, (
            "a tick on a run that has not changed must not re-query its items "
            f"({len(item_calls) - after_click} extra queries in three ticks)"
        )


@pytest.mark.asyncio
async def test_a_running_runs_visible_stats_still_follow_it_across_ticks():
    """The other half: the throttle must not freeze a live run's detail.

    A local run writes its stats, log and items in one go at the END, so the
    moment a tick has to notice is the transition out of `running`.
    """
    from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunProgressTick

    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        running = _running_run()
        pane.runs = [dict(RUNS[0]), running]
        await pilot.pause(0.2)

        record = {"value": dict(running)}

        async def current(**_kwargs):
            return dict(record["value"])

        async def three_items(**_kwargs):
            return [
                {"title": f"Fresh {n}", "status": "new", "alert_count": 0}
                for n in range(3)
            ]

        screen._controller.get_run = current
        screen._controller.list_items = three_items

        await pilot.click("#runs-table", offset=SECOND_ROW_OFFSET)
        assert await _settle_until(pilot, lambda: pane.selected_run is not None)
        assert "Status: running" in _stats_text(pane), "precondition: it is live"

        # The run finishes between one tick and the next.
        record["value"] = dict(
            running,
            status="completed",
            finished_at="2026-08-04T11:00:03+00:00",
            duration="3.0s",
            found_count=3,
            processed_count=3,
            log_text="fetched 3 items",
        )
        screen.post_message(RunProgressTick(running["id"]))
        assert await _settle_until(
            pilot, lambda: "Status: completed" in _stats_text(pane)
        ), "the detail froze at its first paint instead of following the run"

        stats = _stats_text(pane)
        assert "Found: 3" in stats and "Processed: 3" in stats
        assert "fetched 3 items" in _logs_text(pane)
        assert pane.query_one("#runs-detail-items", DataTable).row_count == 3, (
            "a run's items land when it completes; the tick is what notices"
        )
        # The row the user is looking at must agree with the detail below it.
        row = pane.query_one("#runs-table", DataTable).get_cell(
            running["id"], list(pane.query_one("#runs-table", DataTable).columns)[1]
        )
        assert str(row) == "completed"


@pytest.mark.asyncio
async def test_a_tick_for_a_run_the_user_has_left_does_nothing():
    """The tick races the user; the selection guard is what settles it."""
    from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunProgressTick

    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _runs_pane(pilot, host)
        running = _running_run()
        pane.runs = [dict(RUNS[0]), running]
        await pilot.pause(0.2)
        _install_item_source(screen)

        reads: list[Any] = []

        async def record_read(**kwargs):
            reads.append(kwargs.get("run_id"))
            return dict(running)

        screen._controller.get_run = record_read

        await pilot.click("#runs-table", offset=FIRST_ROW_OFFSET)
        assert await _settle_until(
            pilot, lambda: pane.selected_run["id"] == RUNS[0]["id"]
        )

        screen.post_message(RunProgressTick(running["id"]))
        await pilot.pause(0.4)

        assert reads == [], (
            "a tick for a run the user has navigated away from must not even "
            "read it"
        )
        assert pane.selected_run["id"] == RUNS[0]["id"]


# --- TASK-2331: authoritative, generation-checked Runs refresh ------------


@pytest.mark.asyncio
async def test_refresh_replaces_rows_and_reloads_selected_fresh_detail_by_id():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _refresh_test_pane(pilot, host)
        await _seed_refresh_snapshot(screen, pane, pilot)
        before = _mounted_run_snapshot(screen, pane)

        fresh_selected = dict(
            RUNS[1],
            status="failed",
            found_count=9,
            processed_count=8,
            error_count=1,
            log_text="authoritative refresh log",
        )
        fresh_rows = [dict(RUNS[0], source_title="Fresh Summit"), fresh_selected]
        list_started = asyncio.Event()
        release_list = asyncio.Event()
        detail_loaded = asyncio.Event()
        list_calls: list[dict[str, Any]] = []
        detail_calls: list[dict[str, Any]] = []

        async def list_runs(**kwargs):
            list_calls.append(kwargs)
            list_started.set()
            await release_list.wait()
            return deepcopy(fresh_rows)

        async def list_items(**kwargs):
            detail_calls.append(kwargs)
            detail_loaded.set()
            return [
                {
                    "id": "local:watchlist_item:99",
                    "item_id": 99,
                    "run_id": 2,
                    "title": "Fresh detail item",
                    "status": "new",
                    "alert_count": 0,
                }
            ]

        async def get_run_must_not_be_needed(**_kwargs):
            raise AssertionError("the selected normalized id is in the fresh page")

        screen._controller.list_runs = list_runs
        screen._controller.list_items = list_items
        screen._controller.get_run = get_run_must_not_be_needed

        await pilot.click("#runs-refresh-button")
        await asyncio.wait_for(list_started.wait(), timeout=2)
        assert _mounted_run_snapshot(screen, pane) == before, (
            "nothing may publish while authoritative reconciliation is incomplete"
        )

        release_list.set()
        await asyncio.wait_for(detail_loaded.wait(), timeout=2)
        assert await _wait_worker_group_idle(pilot, screen, "wc_runs")
        assert await _wait_worker_group_idle(pilot, screen, "wc_run_detail")
        await pilot.pause()

        assert list_calls == [{"runtime_backend": "local", "limit": 100}]
        assert screen._loaded_runs == fresh_rows
        assert pane.runs == fresh_rows
        assert screen.selected_run == fresh_selected
        assert pane.selected_run == fresh_selected
        assert detail_calls and detail_calls[-1]["run_id"] == 2
        assert pane.run_items[0]["title"] == "Fresh detail item"
        assert screen._run_detail_items == pane.run_items
        assert "authoritative refresh log" in _logs_text(pane)
        assert "Found: 9" in _stats_text(pane)


@pytest.mark.asyncio
async def test_refresh_fetches_and_appends_a_selected_run_outside_the_newest_100():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _refresh_test_pane(pilot, host)
        await _seed_refresh_snapshot(screen, pane, pilot)

        page = [
            dict(
                RUNS[0],
                id=f"local:watchlist_run:{1000 + index}",
                run_id=1000 + index,
                source_title=f"Page row {index}",
            )
            for index in range(100)
        ]
        pinned = dict(
            RUNS[1],
            status="completed",
            found_count=12,
            processed_count=12,
            log_text="pinned fresh detail",
        )
        pin_started = asyncio.Event()
        release_pin = asyncio.Event()
        pin_calls: list[dict[str, Any]] = []

        async def list_runs(**kwargs):
            assert kwargs == {"runtime_backend": "local", "limit": 100}
            return deepcopy(page)

        async def get_run(**kwargs):
            pin_calls.append(kwargs)
            pin_started.set()
            await release_pin.wait()
            return dict(pinned)

        async def list_items(**_kwargs):
            return [{"title": "Pinned item", "status": "new", "alert_count": 0}]

        screen._controller.list_runs = list_runs
        screen._controller.get_run = get_run
        screen._controller.list_items = list_items

        screen.post_message(RefreshRunsRequested())
        await asyncio.wait_for(pin_started.wait(), timeout=2)
        assert pane.runs[-1]["id"] == RUNS[1]["id"], (
            "the mounted page stays untouched until the pin lookup resolves"
        )

        release_pin.set()
        assert await _wait_worker_group_idle(pilot, screen, "wc_runs")
        assert await _wait_worker_group_idle(pilot, screen, "wc_run_detail")
        await pilot.pause()

        assert pin_calls == [{"runtime_backend": "local", "run_id": 2}], (
            "the service receives the selected record's raw run_id, never its "
            "namespaced UI id"
        )
        assert screen._loaded_runs[:100] == page
        assert screen._loaded_runs[100] == pinned
        assert pane.runs == screen._loaded_runs
        assert screen.selected_run == pinned
        assert pane.selected_run == pinned
        assert pane.run_items[0]["title"] == "Pinned item"


@pytest.mark.parametrize(
    ("backend", "missing_error"),
    [
        ("local", KeyError("local detail is gone")),
        ("server", APIResponseError(404, "server detail is gone")),
    ],
)
@pytest.mark.asyncio
async def test_refresh_authoritative_not_found_clears_selection_and_detail(
    backend, missing_error
):
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _refresh_test_pane(pilot, host)
        screen.runtime_backend = backend
        await pilot.pause()
        selected = dict(
            RUNS[1],
            id=f"{backend}:watchlist_run:2",
            backend=backend,
            job_id=2,
        )
        await _seed_refresh_snapshot(screen, pane, pilot, run=selected)

        lookup_started = asyncio.Event()
        release_lookup = asyncio.Event()
        worker_calls: list[dict[str, Any]] = []
        original_run_worker = screen.run_worker

        def recording_run_worker(coro, **kwargs):
            worker_calls.append(kwargs)
            return original_run_worker(coro, **kwargs)

        async def list_runs(**kwargs):
            assert kwargs == {"runtime_backend": backend, "limit": 100}
            return [dict(RUNS[0], id=f"{backend}:watchlist_run:1", backend=backend)]

        async def get_run(**kwargs):
            assert kwargs == {"runtime_backend": backend, "run_id": 2}
            lookup_started.set()
            await release_lookup.wait()
            raise missing_error

        screen.run_worker = recording_run_worker
        screen._controller.list_runs = list_runs
        screen._controller.get_run = get_run
        screen.post_message(RefreshRunsRequested())
        await asyncio.wait_for(lookup_started.wait(), timeout=2)

        release_lookup.set()
        assert await _wait_worker_group_idle(pilot, screen, "wc_runs")
        assert await _wait_worker_group_idle(pilot, screen, "wc_run_detail")
        await pilot.pause()

        assert [run["id"] for run in screen._loaded_runs] == [
            f"{backend}:watchlist_run:1"
        ]
        assert pane.runs == screen._loaded_runs
        assert screen.selected_run is None
        assert pane.selected_run is None
        assert screen._run_detail_items == pane.run_items == []
        assert screen._run_detail_logs == pane.run_logs == ""
        assert screen._run_detail_items_note == pane.run_items_note == ""
        assert "No run selected" in _stats_text(pane)
        assert any(
            call.get("exclusive") is True
            and call.get("group") == "wc_run_detail"
            for call in worker_calls
        ), "the authoritative clear must supersede older run-detail work"


@pytest.mark.parametrize("failure_stage", ["list", "pin"])
@pytest.mark.asyncio
async def test_refresh_transient_failure_preserves_the_complete_mounted_snapshot(
    failure_stage,
):
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    toasts: list[tuple[str, dict[str, Any]]] = []
    app.notify = lambda message, **kwargs: toasts.append((str(message), kwargs))
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _refresh_test_pane(pilot, host)
        await _seed_refresh_snapshot(screen, pane, pilot)
        before = _mounted_run_snapshot(screen, pane)
        failure_started = asyncio.Event()
        release_failure = asyncio.Event()

        async def list_runs(**_kwargs):
            if failure_stage == "list":
                failure_started.set()
                await release_failure.wait()
                raise RuntimeError("private local path")
            return [dict(RUNS[0])]

        async def get_run(**kwargs):
            assert kwargs["run_id"] == 2
            failure_started.set()
            await release_failure.wait()
            raise APIResponseError(503, "secret upstream URL")

        screen._controller.list_runs = list_runs
        screen._controller.get_run = get_run
        screen.post_message(RefreshRunsRequested())
        await asyncio.wait_for(failure_started.wait(), timeout=2)
        assert _mounted_run_snapshot(screen, pane) == before

        release_failure.set()
        assert await _wait_worker_group_idle(pilot, screen, "wc_runs")
        await pilot.pause()

        assert _mounted_run_snapshot(screen, pane) == before
        assert toasts
        message, kwargs = toasts[-1]
        assert "secret" not in message and "path" not in message
        assert kwargs.get("severity") == "error"
        assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_backend_switch_discards_a_gated_refresh_response():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _refresh_test_pane(pilot, host)
        await _seed_refresh_snapshot(screen, pane, pilot)
        list_started = asyncio.Event()
        release_list = asyncio.Event()

        async def list_runs(**kwargs):
            assert kwargs == {"runtime_backend": "local", "limit": 100}
            list_started.set()
            await release_list.wait()
            return [dict(RUNS[0], source_title="stale local row")]

        screen._controller.list_runs = list_runs
        screen.post_message(RefreshRunsRequested())
        await asyncio.wait_for(list_started.wait(), timeout=2)

        screen.runtime_backend = "server"
        await pilot.pause()
        assert screen._loaded_runs == []
        assert pane.runs == []
        assert screen.selected_run is pane.selected_run is None

        release_list.set()
        assert await _wait_worker_group_idle(pilot, screen, "wc_runs")
        await pilot.pause()

        assert screen.runtime_backend == "server"
        assert screen._loaded_runs == []
        assert pane.runs == []
        assert screen.selected_run is pane.selected_run is None
        assert screen._run_detail_items == pane.run_items == []
        assert "No run selected" in _stats_text(pane)


@pytest.mark.asyncio
async def test_backend_switch_discards_a_stale_refresh_failure_silently():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    toasts: list[str] = []
    app.notify = lambda message, **_kwargs: toasts.append(str(message))
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _refresh_test_pane(pilot, host)
        await _seed_refresh_snapshot(screen, pane, pilot)
        list_started = asyncio.Event()
        release_list = asyncio.Event()

        async def list_runs(**kwargs):
            if kwargs.get("runtime_backend") == "server":
                return []
            list_started.set()
            while not release_list.is_set():
                try:
                    await release_list.wait()
                except asyncio.CancelledError:
                    continue
            raise RuntimeError("obsolete backend failure")

        screen._controller.list_runs = list_runs
        screen.post_message(RefreshRunsRequested())
        await asyncio.wait_for(list_started.wait(), timeout=2)
        screen.runtime_backend = "server"
        await pilot.pause()
        toasts.clear()

        release_list.set()
        assert await _wait_worker_group_idle(pilot, screen, "wc_runs")

        assert screen.runtime_backend == "server"
        assert screen._loaded_runs == pane.runs == []
        assert toasts == [], "obsolete backend work is discarded without an error toast"


@pytest.mark.asyncio
async def test_second_refresh_generation_supersedes_an_older_late_response():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _refresh_test_pane(pilot, host)
        await _seed_refresh_snapshot(screen, pane, pilot)
        starts = [asyncio.Event(), asyncio.Event()]
        releases = [asyncio.Event(), asyncio.Event()]
        rows = [
            [dict(RUNS[0], source_title="older response"), dict(RUNS[1])],
            [dict(RUNS[0], source_title="newest response"), dict(RUNS[1])],
        ]
        call_count = 0

        async def list_runs(**_kwargs):
            nonlocal call_count
            index = call_count
            call_count += 1
            starts[index].set()
            while not releases[index].is_set():
                try:
                    await releases[index].wait()
                except asyncio.CancelledError:
                    # The generation check, not cooperative cancellation, is
                    # the publication authority this regression exercises.
                    continue
            return deepcopy(rows[index])

        async def list_items(**_kwargs):
            return []

        screen._controller.list_runs = list_runs
        screen._controller.list_items = list_items

        screen._request_runs_refresh()
        assert screen._runs_refresh_generation == 1
        await asyncio.wait_for(starts[0].wait(), timeout=2)

        screen._request_runs_refresh()
        assert screen._runs_refresh_generation == 2, (
            "the superseding token must advance when Refresh is accepted, "
            "not when its worker eventually starts"
        )
        await asyncio.wait_for(starts[1].wait(), timeout=2)

        releases[1].set()
        assert await _pump_until(
            pilot,
            lambda: bool(screen._loaded_runs)
            and screen._loaded_runs[0]["source_title"] == "newest response",
        )
        releases[0].set()
        assert await _wait_worker_group_idle(pilot, screen, "wc_runs")
        assert await _wait_worker_group_idle(pilot, screen, "wc_run_detail")
        await pilot.pause()

        assert screen._loaded_runs[0]["source_title"] == "newest response"
        assert pane.runs[0]["source_title"] == "newest response"


@pytest.mark.asyncio
async def test_later_user_clear_supersedes_an_older_run_detail_query():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _refresh_test_pane(pilot, host)
        await _seed_refresh_snapshot(screen, pane, pilot)
        query_started = asyncio.Event()
        release_query = asyncio.Event()
        worker_calls: list[dict[str, Any]] = []
        original_run_worker = screen.run_worker

        def recording_run_worker(coro, **kwargs):
            worker_calls.append(kwargs)
            return original_run_worker(coro, **kwargs)

        async def stale_items(**_kwargs):
            query_started.set()
            while not release_query.is_set():
                try:
                    await release_query.wait()
                except asyncio.CancelledError:
                    continue
            return [{"title": "Stale resurrected item", "status": "new"}]

        screen.run_worker = recording_run_worker
        screen._controller.list_items = stale_items

        pane.select_run_by_id(str(RUNS[0]["id"]))
        await asyncio.wait_for(query_started.wait(), timeout=2)
        pane.selected_run = None
        assert await _pump_until(pilot, lambda: screen.selected_run is None)
        assert screen._run_detail_items == pane.run_items == []

        release_query.set()
        assert await _wait_worker_group_idle(pilot, screen, "wc_run_detail")
        await pilot.pause()

        detail_workers = [
            call for call in worker_calls if call.get("group") == "wc_run_detail"
        ]
        assert len(detail_workers) >= 2
        assert all(call.get("exclusive") is True for call in detail_workers)
        assert screen.selected_run is pane.selected_run is None
        assert screen._run_detail_items == pane.run_items == []
        assert screen._run_detail_logs == pane.run_logs == ""
        assert screen._run_detail_items_note == pane.run_items_note == ""
        assert "Stale resurrected item" not in str(pane.run_items)
