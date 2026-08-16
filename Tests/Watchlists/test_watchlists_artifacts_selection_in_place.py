"""Task-15779: selecting a briefing must not destroy the briefings table.

Born-red pins, written against the pre-fix code: `ArtifactsPane.selected_
briefing` was `reactive(..., recompose=True)`, so the row click/arrow-key
that selects a briefing recomposed the WHOLE pane -- destroying and
rebuilding the very `DataTable` the user was navigating. Keyboard focus,
cursor and scroll all died with the old widget: the concrete symptom
(task-15461's Implementation Notes, where this was found and recorded) is
that a second arrow-key press immediately after a selection does nothing
until the user manually re-focuses the table.

The fix decomposes the recompose surface: everything below the table lives
in `BriefingDetailRegion` (a stateless recompose boundary rendering from
the pane's own reactives), and a selection updates the table's highlight,
the Export/Keep buttons and that region IN PLACE -- the table widget
itself is never torn down.

Same harness discipline as `test_watchlists_artifacts_pane.py` (whose
helpers this file reuses): the real screen, the real `SubscriptionsDB`,
real key presses through the real focused table. Nothing is faked at all
here -- no briefing is ever generated, only read.
"""

from __future__ import annotations

import pytest
from textual.coordinate import Coordinate
from textual.widgets import Button, DataTable, Static

from Tests.Watchlists.test_watchlists_artifacts_pane import (
    _briefing_rows,
    _build_test_app,
    _open_artifacts,
    _render_to_console,
    _seed_watchlist,
    _seeded_item_rows,
)
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import ArtifactsPane

# Same reason as `test_watchlists_artifacts_pane.py`'s own mark: the unit CI
# job selects `-m unit` and the UI job runs `Tests -m ui`, so an unmarked
# test in `Tests/Watchlists` is collected by nothing.
pytestmark = pytest.mark.ui


def _seed_briefings(app, watchlist_id: int, *, count: int, citation_id=None):
    """`count` complete briefings with distinct bodies, straight into the DB.

    A raw `insert_briefing`/`update_briefing` (the same seeding
    `test_only_a_focused_tables_highlight_selects` uses) rather than the
    Generate flow: these tests are about what a SELECTION does to the
    mounted table, not about generation. When `citation_id` names a real
    `subscription_items` row, every body cites it -- so the reload a
    selection dispatches lands with a non-empty `citations` value, which is
    exactly the kind of post-selection arrival that also used to recompose
    the pane out from under the user.
    """
    db = app.watchlist_bundle_service.db
    for index in range(count):
        briefing_id = db.insert_briefing(watchlist_id)
        citation = f"See [item {citation_id}].\n\n" if citation_id is not None else ""
        db.update_briefing(
            briefing_id,
            status="complete",
            body_markdown=f"{citation}BODY-{briefing_id}-UNIQUE paragraph.",
        )


async def _settle(pilot, host) -> None:
    """Let the selection's own `wl-briefings-load` worker land completely."""
    await pilot.pause(0.05)
    await host.workers.wait_for_complete()
    await pilot.pause(0.05)


def _table_plains(table: DataTable) -> list[list[str]]:
    """Every cell's painted characters, row-major -- styles excluded."""
    return [
        [
            cell.plain if hasattr(cell, "plain") else str(cell)
            for cell in table.get_row_at(index)
        ]
        for index in range(table.row_count)
    ]


def _row_styles(table: DataTable, index: int) -> set[str]:
    """The distinct styles across one row's cells, as plain strings."""
    return {str(getattr(cell, "style", "")) for cell in table.get_row_at(index)}


@pytest.mark.asyncio
async def test_a_second_arrow_key_immediately_after_selecting_still_moves_the_selection():
    """The AC's concrete symptom, end-to-end: down, then down again.

    Pre-fix, the first press's selection recomposed the pane; the rebuilt
    table held no focus, so the second press reached nothing and the
    selection stayed where it was until the user manually re-focused.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    _seed_briefings(app, watchlist_id, count=4)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-table", DataTable)
        rows = _briefing_rows(app, watchlist_id)
        assert table.row_count == len(rows) >= 4, "the fixture needs a populated table"
        table.focus()
        await pilot.pause()
        assert table.has_focus, "precondition: the user is in the table"

        await pilot.press("down")
        await _settle(pilot, host)
        assert pane.selected_briefing is not None
        assert pane.selected_briefing["id"] == rows[1]["id"], (
            "the first press must select the second row"
        )

        await pilot.press("down")
        await _settle(pilot, host)
        assert pane.selected_briefing["id"] == rows[2]["id"], (
            "the second arrow-key press right after a selection must move "
            "the selection on -- pre-task-15779 it did NOTHING, because the "
            "selection's recompose had destroyed the focused table"
        )


@pytest.mark.asyncio
async def test_selecting_a_briefing_keeps_the_table_widget_and_its_focus():
    """The mechanism behind the symptom: the widget must survive."""
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    _seed_briefings(app, watchlist_id, count=3)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-table", DataTable)
        table.focus()
        await pilot.pause()

        await pilot.press("down")
        await _settle(pilot, host)

        assert pane.selected_briefing is not None, "the press must select"
        assert pane.query_one("#artifacts-table", DataTable) is table, (
            "selecting a briefing must not destroy and rebuild the briefings "
            "table -- the mounted widget must be the SAME instance"
        )
        assert table.has_focus, (
            "and the table the user is navigating must still hold focus"
        )


@pytest.mark.asyncio
async def test_selecting_a_briefing_preserves_the_tables_scroll_position():
    """Deep in a long list, a selection must not throw the viewport away."""
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    _seed_briefings(app, watchlist_id, count=40)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-table", DataTable)
        assert table.row_count == 40

        # Park the cursor deep in the list WITHOUT selecting: an unfocused
        # table's highlight is not user-driven (`highlight_is_user_driven`),
        # so nothing recomposes yet even pre-fix. The cursor move scrolls
        # the viewport down to it.
        table.cursor_coordinate = Coordinate(34, 0)
        await pilot.pause(0.1)
        table.focus()
        await pilot.pause()
        scroll_before = table.scroll_y
        assert scroll_before > 0, (
            "precondition: the table must actually be scrolled, or this "
            f"test pins nothing (scroll_y={scroll_before}, "
            f"height={table.size.height})"
        )

        await pilot.press("down")
        await _settle(pilot, host)

        assert pane.query_one("#artifacts-table", DataTable) is table, (
            "the scrolled table must survive the selection"
        )
        assert scroll_before <= table.scroll_y <= scroll_before + 1, (
            "the viewport must stay where the user was (at most one row of "
            "cursor-follow), not reset by a rebuild: scroll_y was "
            f"{scroll_before}, now {table.scroll_y}"
        )


@pytest.mark.asyncio
async def test_the_row_highlight_moves_in_place_as_the_selection_moves():
    """The selected row's `reverse bold` must arrive -- and LEAVE -- without
    the table being rebuilt around it."""
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    _seed_briefings(app, watchlist_id, count=3)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-table", DataTable)
        table.focus()
        await pilot.pause()

        await pilot.press("down")
        await _settle(pilot, host)
        assert pane.query_one("#artifacts-table", DataTable) is table, (
            "the highlight must be painted onto the SAME table instance"
        )
        assert _row_styles(table, 1) == {ArtifactsPane._SELECTED_ROW_STYLE}, (
            "the newly selected row must carry the selected style"
        )
        assert _row_styles(table, 0) == {""}, "row 0 was never selected"

        await pilot.press("down")
        await _settle(pilot, host)
        assert pane.query_one("#artifacts-table", DataTable) is table
        assert _row_styles(table, 2) == {ArtifactsPane._SELECTED_ROW_STYLE}, (
            "the highlight must follow the selection"
        )
        assert _row_styles(table, 1) == {""}, (
            "and must LEAVE the previously selected row -- an in-place "
            "restyle that only ever adds would show two selected rows"
        )


@pytest.mark.asyncio
async def test_the_detail_updates_while_the_tables_content_stands_still():
    """The other half of the bargain: the detail region must still follow
    the selection -- including the citations its reload lands later -- and
    the surviving table's CONTENT must be byte-identical before and after
    (the in-place restyle touches styles, never characters)."""
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    cited_item_id = _seeded_item_rows(app)[0]["id"]
    _seed_briefings(app, watchlist_id, count=3, citation_id=cited_item_id)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-table", DataTable)
        rows = _briefing_rows(app, watchlist_id)
        plains_before = _table_plains(table)
        export_button = pane.query_one("#artifacts-export-button", Button)
        assert export_button.disabled, "precondition: nothing selected yet"

        table.focus()
        await pilot.pause()
        await pilot.press("down")
        await _settle(pilot, host)

        # The table survived, its painted characters untouched.
        assert pane.query_one("#artifacts-table", DataTable) is table
        assert _table_plains(table) == plains_before, (
            "a selection may restyle the table's rows but must not change "
            "one painted character of them"
        )

        # The detail shows the newly selected briefing's own body.
        selected = pane.selected_briefing
        assert selected is not None and selected["id"] == rows[1]["id"]
        detail = pane.query_one("#artifacts-detail", Static)
        plain, _ = _render_to_console(detail.renderable, width=160)
        assert f"BODY-{selected['id']}-UNIQUE" in plain, (
            "the detail region must render the selected briefing's body"
        )

        # The reload the selection dispatched has landed (citations resolved
        # from the body's own `[item N]` marker) -- and THAT arrival did not
        # destroy the table either.
        assert pane.citations, "the fixture's citation must have resolved"
        assert pane.query_one("#artifacts-citations-table", DataTable) is not None
        assert pane.query_one("#artifacts-table", DataTable) is table, (
            "the reload landing scripts/citations after a selection must "
            "not rebuild the table any more than the selection itself may"
        )

        # And the selection-dependent toolbar buttons were updated in place.
        assert pane.query_one("#artifacts-export-button", Button) is export_button, (
            "the toolbar must survive too -- its state is patched, not rebuilt"
        )
        assert not export_button.disabled, (
            "Export must arm for a complete selection without a recompose"
        )
