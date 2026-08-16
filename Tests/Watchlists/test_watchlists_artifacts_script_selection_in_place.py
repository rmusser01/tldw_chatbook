"""Task-16852: selecting a SCRIPT must not destroy the scripts table.

Task-15779 fixed a briefing selection destroying the briefings table by
moving the detail chrome into `BriefingDetailRegion`, a stateless recompose
boundary. Its Implementation Notes disclosed the exact same defect one level
down as deliberately unexpanded scope: `selected_script`/`script_audio`
still funnelled into `_refresh_detail_region()`, so selecting a script
recomposed the WHOLE `BriefingDetailRegion` -- scripts `DataTable` included
-- destroying the very table the user was navigating. Keyboard focus,
cursor and scroll all died with the old widget: the same symptom
task-15779's own module docstring records for the briefings table, one
level down.

The fix nests a second boundary, `ScriptDetailRegion`, inside
`BriefingDetailRegion`: it holds only the script detail `Static` and the
Synthesize/Play/Stop toolbar, and a script selection updates the scripts
table's highlight and that region IN PLACE -- the scripts table widget
itself is never torn down.

Same harness discipline as `test_watchlists_artifacts_selection_in_place.py`
(whose helpers this file reuses): the real screen, the real
`SubscriptionsDB`, real key presses through the real focused table. Scripts
are seeded directly via `SubscriptionsDB.insert_briefing_script` -- these
tests are about what a SCRIPT SELECTION does to the mounted table, not
about the real Cast flow (already covered elsewhere).
"""

from __future__ import annotations

import pytest
from textual.coordinate import Coordinate
from textual.widgets import DataTable, Static

from Tests.Watchlists.test_watchlists_artifacts_pane import (
    ONE_SPEAKER_ROSTER,
    _build_test_app,
    _open_artifacts,
    _render_to_console,
    _seed_watchlist,
)
from Tests.Watchlists.test_watchlists_artifacts_selection_in_place import (
    _row_styles,
    _settle,
    _table_plains,
)
from tldw_chatbook.Subscriptions.briefing_cast import dump_roster
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import ArtifactsPane

# Same reason as the sibling files' own mark: the unit CI job selects
# `-m unit` and the UI job runs `Tests -m ui`, so an unmarked test in
# `Tests/Watchlists` is collected by nothing.
pytestmark = pytest.mark.ui


def _seed_briefing_with_scripts(app, watchlist_id: int, *, count: int) -> int:
    """A `complete` briefing and `count` `complete` cast scripts under it,
    built directly via the DB -- these tests are about what a SCRIPT
    SELECTION does to the mounted table, not about the real Cast/LLM path
    (`_seed_complete_script`'s own precedent in `test_watchlists_artifacts_
    pane.py`).

    Each script gets a distinct `preset_name` (`Preset-000`, `Preset-001`,
    ...) so the scripts table's rows -- and the detail's own header, which
    names the preset -- are individually identifiable.

    Returns:
        The briefing's id.
    """
    db = app.watchlist_bundle_service.db
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(briefing_id, status="complete", body_markdown="Body")
    for index in range(count):
        db.insert_briefing_script(
            briefing_id,
            preset_id=None,
            preset_name=f"Preset-{index:03d}",
            roster_snapshot_json=dump_roster(ONE_SPEAKER_ROSTER),
            status="complete",
        )
    return briefing_id


def _script_rows(app, briefing_id: int) -> list[dict]:
    """Every seeded script, in the same newest-first order the pane itself
    reads them in (`list_briefing_scripts`) -- the authoritative row order
    for a test to key off, rather than assuming insertion order."""
    return app.watchlist_bundle_service.db.list_briefing_scripts(
        briefing_id, limit=200
    )


async def _select_briefing_and_settle(pane, pilot, host, briefing_id) -> None:
    """Select the one seeded briefing and let its reload land -- the
    scripts table does not exist until this settles (it renders only once
    `selected_briefing is not None`, and its ROWS only once the reload's
    `scripts` arrives)."""
    pane.select_briefing_by_id(str(briefing_id))
    await _settle(pilot, host)


@pytest.mark.asyncio
async def test_a_second_arrow_key_immediately_after_selecting_a_script_still_moves_the_selection():
    """The AC's concrete symptom, one level down: down, then down again.

    Pre-fix, the first press's script selection recomposed the WHOLE
    detail region (scripts table included); the rebuilt table held no
    focus, so the second press reached nothing and the selection stayed
    where it was until the user manually re-focused.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_briefing_with_scripts(app, watchlist_id, count=4)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        await _select_briefing_and_settle(pane, pilot, host, briefing_id)

        table = pane.query_one("#artifacts-scripts-table", DataTable)
        rows = _script_rows(app, briefing_id)
        assert table.row_count == len(rows) >= 4, "the fixture needs a populated table"
        table.focus()
        await pilot.pause()
        assert table.has_focus, "precondition: the user is in the table"

        await pilot.press("down")
        await _settle(pilot, host)
        assert pane.selected_script is not None
        assert pane.selected_script["id"] == rows[1]["id"], (
            "the first press must select the second row"
        )

        await pilot.press("down")
        await _settle(pilot, host)
        assert pane.selected_script["id"] == rows[2]["id"], (
            "the second arrow-key press right after a script selection must "
            "move the selection on -- pre-task-16852 it did NOTHING, because "
            "the selection's recompose had destroyed the focused table"
        )


@pytest.mark.asyncio
async def test_selecting_a_script_keeps_the_table_widget_and_its_focus():
    """The mechanism behind the symptom: the scripts table widget must
    survive a script selection, exactly as the briefings table already
    does (task-15779)."""
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_briefing_with_scripts(app, watchlist_id, count=3)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        await _select_briefing_and_settle(pane, pilot, host, briefing_id)

        table = pane.query_one("#artifacts-scripts-table", DataTable)
        table.focus()
        await pilot.pause()

        await pilot.press("down")
        await _settle(pilot, host)

        assert pane.selected_script is not None, "the press must select"
        assert pane.query_one("#artifacts-scripts-table", DataTable) is table, (
            "selecting a script must not destroy and rebuild the scripts "
            "table -- the mounted widget must be the SAME instance"
        )
        assert table.has_focus, (
            "and the table the user is navigating must still hold focus"
        )


@pytest.mark.asyncio
async def test_selecting_a_script_preserves_the_tables_scroll_position():
    """Deep in a long scripts list, a selection must not throw the
    viewport away -- the same guarantee the briefings table already has."""
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_briefing_with_scripts(app, watchlist_id, count=40)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        await _select_briefing_and_settle(pane, pilot, host, briefing_id)

        table = pane.query_one("#artifacts-scripts-table", DataTable)
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

        assert pane.query_one("#artifacts-scripts-table", DataTable) is table, (
            "the scrolled table must survive the selection"
        )
        assert scroll_before <= table.scroll_y <= scroll_before + 1, (
            "the viewport must stay where the user was (at most one row of "
            "cursor-follow), not reset by a rebuild: scroll_y was "
            f"{scroll_before}, now {table.scroll_y}"
        )


@pytest.mark.asyncio
async def test_the_script_row_highlight_moves_in_place_as_the_selection_moves():
    """The selected row's `reverse bold` must arrive -- and LEAVE -- without
    the scripts table being rebuilt around it."""
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_briefing_with_scripts(app, watchlist_id, count=3)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        await _select_briefing_and_settle(pane, pilot, host, briefing_id)

        table = pane.query_one("#artifacts-scripts-table", DataTable)
        table.focus()
        await pilot.pause()

        await pilot.press("down")
        await _settle(pilot, host)
        assert pane.query_one("#artifacts-scripts-table", DataTable) is table, (
            "the highlight must be painted onto the SAME table instance"
        )
        assert _row_styles(table, 1) == {ArtifactsPane._SELECTED_ROW_STYLE}, (
            "the newly selected row must carry the selected style"
        )
        assert _row_styles(table, 0) == {""}, "row 0 was never selected"

        await pilot.press("down")
        await _settle(pilot, host)
        assert pane.query_one("#artifacts-scripts-table", DataTable) is table
        assert _row_styles(table, 2) == {ArtifactsPane._SELECTED_ROW_STYLE}, (
            "the highlight must follow the selection"
        )
        assert _row_styles(table, 1) == {""}, (
            "and must LEAVE the previously selected row -- an in-place "
            "restyle that only ever adds would show two selected rows"
        )


@pytest.mark.asyncio
async def test_the_script_detail_updates_while_the_scripts_table_content_stands_still():
    """The other half of the bargain: the script detail must still follow
    the selection, and the surviving table's CONTENT must be byte-identical
    before and after (the in-place restyle touches styles, never
    characters)."""
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_briefing_with_scripts(app, watchlist_id, count=3)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        await _select_briefing_and_settle(pane, pilot, host, briefing_id)

        table = pane.query_one("#artifacts-scripts-table", DataTable)
        rows = _script_rows(app, briefing_id)
        plains_before = _table_plains(table)

        table.focus()
        await pilot.pause()
        await pilot.press("down")
        await _settle(pilot, host)

        # The table survived, its painted characters untouched.
        assert pane.query_one("#artifacts-scripts-table", DataTable) is table
        assert _table_plains(table) == plains_before, (
            "a script selection may restyle the table's rows but must not "
            "change one painted character of them"
        )

        # The script detail shows the newly selected script's own identity.
        selected = pane.selected_script
        assert selected is not None and selected["id"] == rows[1]["id"]
        detail = pane.query_one("#artifacts-script-detail", Static)
        plain, _ = _render_to_console(detail.renderable, width=160)
        assert rows[1]["preset_name"] in plain, (
            "the script detail must render the newly selected script's own "
            "preset name"
        )
