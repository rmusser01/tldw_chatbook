"""Sources pane: table-rebuild highlights must not hijack selection — TASK-38586.

`SourcesPane` handled `RowHighlighted`/`CellHighlighted` unconditionally
(TASK-1100's click-to-select). But a `DataTable` ALSO posts a highlight when
its rows are rebuilt — `_refresh_table_rows` clears and re-adds rows on every
filter keystroke, and the `sources` reactive recomposes the whole table — so
the announcement for whatever row the cursor falls back on acted like a user
choice. Selecting source X and typing in the search box re-selected the first
filtered row; `watch_selected_source` posted `SourceSelected`, and the screen
retargeted the Inspector — so Check now / Preview / Delete could act on a row
the user never chose (audit 1.6; phantom arrival verified with a pilot).

Every sibling pane in the package already gates on `highlight_is_user_driven`
(items/runs/rules/artifacts/notifications). These tests pin the same contract
for SourcesPane.

Widget-level harness, per the B6 brief: the full app cannot boot in this
environment, so the pane is mounted alone in a minimal app and its data
dependencies (normally seeded by the screen) are set directly. The existing
app-factory tests (`test_watchlists_source_row_click_selects.py`,
`test_watchlists_check_now_skipped.py`) cannot run here; these are their
widget-level equivalents.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input

from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane

# Bare-widget harness reading no config; keep the collection-time profile so
# the per-test sandbox redirect's config-participant admission
# (RecoveryRequired on this dev baseline, see the B6 brief) stays out of the
# way. Registered marker: Tests/conftest.py (TASK-32873).
pytestmark = pytest.mark.bootstrap_profile

# See the sibling test file for the two-source shape: a single-source fixture
# could pass even if click-to-select regressed entirely, because the default
# selection would stand in for the click.
SUMMIT = {
    "id": "local:subscription:1",
    "source_id": 1,
    "name": "Summit Route",
    "source_type": "rss",
    "active": True,
}
DARKNET = {
    "id": "local:subscription:2",
    "source_id": 2,
    "name": "Darknet Diaries",
    "source_type": "rss",
    "active": True,
}
SOURCES = [SUMMIT, DARKNET]


class SourcesPaneApp(App[None]):
    def compose(self) -> ComposeResult:
        yield SourcesPane(id="watchlists-sources-pane")


async def _pane_with_sources(pilot) -> SourcesPane:
    pane = pilot.app.query_one("#watchlists-sources-pane", SourcesPane)
    pane.sources = [dict(source) for source in SOURCES]
    await pilot.pause(0.2)
    return pane


@pytest.mark.asyncio
async def test_typing_in_the_search_box_does_not_retarget_the_selection():
    """AC#2: a filter rebuild must not move an existing user selection.

    Red on the unfixed pane: the rebuild's row-0 phantom highlight selects
    the first FILTERED row (Darknet), replacing the user's choice (Summit).
    """
    async with SourcesPaneApp().run_test(size=(160, 40)) as pilot:
        pane = await _pane_with_sources(pilot)
        pane.select_source_by_id(SUMMIT["id"])
        await pilot.pause()

        search = pane.query_one("#sources-search-input", Input)
        search.focus()
        await pilot.pause()
        for character in "dark":
            await pilot.press(character)
        await pilot.pause(0.2)

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 1, "precondition: the filter narrowed the table"
        assert pane.selected_source is not None
        assert pane.selected_source["id"] == SUMMIT["id"], (
            "typing in the search box retargeted the selection onto "
            f"{pane.selected_source.get('name')!r} — the rebuild's phantom "
            "highlight must not act like a user choice"
        )


@pytest.mark.asyncio
async def test_clicking_a_row_still_selects_that_row():
    """User-driven selection must keep working after the guard (TASK-1100).

    A click focuses the table before its highlight is dispatched, so the
    focus gate must let it through. The SECOND row is targeted because row 0
    is what the arrival selection already holds.
    """
    async with SourcesPaneApp().run_test(size=(160, 40)) as pilot:
        pane = await _pane_with_sources(pilot)
        await pilot.pause()

        await pilot.click("#sources-table", offset=(4, 2))
        await pilot.pause()

        assert pane.selected_source is not None, "the click must select a source"
        assert pane.selected_source["id"] == DARKNET["id"], (
            "the click must select the row that was clicked, not the arrival row"
        )


@pytest.mark.asyncio
async def test_keyboard_cursor_movement_still_selects_the_row_it_lands_on():
    """Arrow keys arrive at a focused table; they must keep selecting."""
    async with SourcesPaneApp().run_test(size=(160, 40)) as pilot:
        pane = await _pane_with_sources(pilot)
        table = pane.query_one("#sources-table", DataTable)
        table.focus()
        await pilot.pause()

        await pilot.press("down")
        await pilot.pause()

        assert pane.selected_source is not None
        assert pane.selected_source["id"] == DARKNET["id"]


@pytest.mark.asyncio
async def test_populate_arms_the_first_source_on_arrival():
    """TASK-1100's product behavior must survive the guard.

    A freshly populated table used to select its first row via the phantom
    row-0 highlight, arming Preview/Check now on arrival
    (pinned by the app-factory suite, which cannot run in this environment).
    The guard filters that announcement, so the pane must arm the arrival
    selection itself — deterministically, not by reacting to the phantom.
    """
    async with SourcesPaneApp().run_test(size=(160, 40)) as pilot:
        pane = await _pane_with_sources(pilot)

        assert pane.selected_source is not None, (
            "the first source must be selected on arrival so Preview and "
            "Check now are armed"
        )
        assert pane.selected_source["id"] == SUMMIT["id"]


@pytest.mark.asyncio
async def test_later_rebuilds_do_not_resurrect_a_cleared_selection():
    """The arrival selection must arm once, not on every rebuild.

    The screen clears the pane's selection on purpose when the tree moves to
    another scope (`_apply_tree_scope` -> `_clear_pane_selections`). A
    recompose triggered after that (filter editor toggle, create form) must
    not bring a selection back.
    """
    async with SourcesPaneApp().run_test(size=(160, 40)) as pilot:
        pane = await _pane_with_sources(pilot)
        assert pane.selected_source is not None, "precondition: arrival armed"

        pane.selected_source = None  # what the screen's tree-scope clear does
        pane.show_filter_editor = not pane.show_filter_editor  # forces a recompose
        await pilot.pause(0.2)

        assert pane.selected_source is None, (
            "a rebuild resurrected a selection the owner cleared on purpose"
        )


def _fabricated_event(*, has_focus: bool, row_value) -> SimpleNamespace:
    stopped: list[bool] = []
    return SimpleNamespace(
        data_table=SimpleNamespace(id="sources-table", has_focus=has_focus),
        row_key=SimpleNamespace(value=row_value),
        stop=lambda: stopped.append(True),
    ), stopped


def _bare_pane() -> SimpleNamespace:
    """A pane stub carrying exactly what the handlers read, sibling-test style."""
    pane = SimpleNamespace(
        sources=[dict(SUMMIT), dict(DARKNET)],
        selected_source=None,
    )
    pane.select_source_by_id = MethodType(SourcesPane.select_source_by_id, pane)
    return pane


def test_row_highlight_from_an_unfocused_table_selects_nothing():
    """The guard itself, fabricated: a rebuild's announcement (unfocused)."""
    pane = _bare_pane()
    event, stopped = _fabricated_event(has_focus=False, row_value=DARKNET["id"])

    SourcesPane.on_data_table_row_highlighted(pane, event)

    assert stopped == [True]
    assert pane.selected_source is None, (
        "an unfocused table's highlight is a rebuild artifact, not a choice"
    )


def test_row_highlight_from_a_focused_table_selects_that_row():
    pane = _bare_pane()
    event, stopped = _fabricated_event(has_focus=True, row_value=DARKNET["id"])

    SourcesPane.on_data_table_row_highlighted(pane, event)

    assert stopped == [True]
    assert pane.selected_source is not None
    assert pane.selected_source["id"] == DARKNET["id"]


def test_cell_highlight_from_an_unfocused_table_selects_nothing():
    """Same contract for the cell-shaped cursor variant."""
    pane = _bare_pane()
    stopped: list[bool] = []
    event = SimpleNamespace(
        data_table=SimpleNamespace(id="sources-table", has_focus=False),
        cell_key=SimpleNamespace(row_key=SimpleNamespace(value=DARKNET["id"])),
        stop=lambda: stopped.append(True),
    )

    SourcesPane.on_data_table_cell_highlighted(pane, event)

    assert stopped == [True]
    assert pane.selected_source is None
