"""The notes list toolbar: ten peers, three of which vanished (task-32617).

Two findings, one of them re-diagnosed by driving the canvas directly:

* the three note-placement actions were DROPPED whenever no row was
  selected -- the critique read the vanishing as filter-caused, but the
  filter is only what loses the selection; and
* the row carried ten actions with no grouping rule a reader could state,
  and no stated ceiling for the next one.

Every assertion here was first run against the unfixed tree and recorded
failing; the RED text is on the task.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.widgets import Button, Static

from Tests.textual_test_utils import widget_pilot  # noqa: F401
from Tests.Widgets.Library.test_library_notes_canvas import _list_state
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesTreeProjection,
    LibraryNotesTreeRow,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    NOTES_LIST_VISIBLE_ACTION_BUDGET,
    NOTES_TREE_ACTION_GROUP_HEADING,
    LibraryNotesCanvas,
)

pytestmark = pytest.mark.asyncio


def _tree_projection() -> LibraryNotesTreeProjection:
    return LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id="folder:work",
                kind="folder",
                label="Work",
                depth=0,
                folder_id="work",
            ),
            LibraryNotesTreeRow(
                placement_id="note:work:n1:m1",
                kind="note",
                label="Q3 plan",
                depth=1,
                note_id="n1",
                folder_id="work",
                membership_id="m1",
                breadcrumb="Work / Q3 plan",
            ),
        )
    )


@pytest.mark.parametrize("filter_value", ["", "plan"])
async def test_note_actions_are_disabled_with_a_reason_rather_than_removed(
    widget_pilot,  # noqa: F811
    filter_value: str,
) -> None:
    """task-32617 AC#1: unavailable is a state, not an absence.

    Born red with the three buttons simply absent. Both parameters matter:
    the critique read the vanishing as filter-caused, but driving the canvas
    directly proves it is the missing SELECTION -- the filter is only what
    loses it. Fixing the selection case therefore fixes the filtered one.
    """
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
        tree_selected_placement_id="",
        filter_value=filter_value,
    ) as pilot:
        await pilot.pause()
        for button_id in (
            "library-notes-placement-add",
            "library-notes-placement-move",
            "library-notes-placement-remove",
        ):
            action = pilot.app.query_one(f"#{button_id}", Button)
            assert action.disabled is True
            assert str(action.label).startswith("○ "), action.label
        reason = pilot.app.query_one(
            "#library-notes-tree-actions-disabled-reason", Static
        )
        assert (
            str(reason.renderable)
            == "Note actions unavailable — select a note in the list"
        )


async def test_selecting_a_note_enables_its_actions_and_drops_the_reason(
    widget_pilot,  # noqa: F811
) -> None:
    """task-32617 AC#1: the negative control -- the reason is not permanent."""
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
        tree_selected_placement_id="note:work:n1:m1",
    ) as pilot:
        await pilot.pause()
        assert (
            pilot.app.query_one("#library-notes-placement-add", Button).disabled
            is False
        )
        assert not pilot.app.query("#library-notes-tree-actions-disabled-reason")


async def test_the_notes_list_states_the_rule_its_second_action_group_follows(
    widget_pilot,  # noqa: F811
) -> None:
    """task-32617 AC#2: the group is named, in the screen's own grammar."""
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
        tree_selected_placement_id="note:work:n1:m1",
    ) as pilot:
        await pilot.pause()
        heading = pilot.app.query_one(
            "#library-notes-tree-actions-heading", Static
        )
        assert str(heading.renderable) == NOTES_TREE_ACTION_GROUP_HEADING
        assert "destination-section" in heading.classes
        # The heading introduces the group: everything above it acts on the
        # list, everything below on the folder tree and the selected row.
        actions = pilot.app.query_one("#library-notes-tree-actions")
        assert heading.region.y < actions.region.y


async def test_the_notes_list_action_budget_is_the_stated_number(
    widget_pilot,  # noqa: F811
) -> None:
    """task-32617 AC#3: the ceiling is a decision a later change trips over.

    Composed at the canvas's widest reachable state -- a filter (Clear
    filter), a selected note (three placement actions), a restorable folder,
    an import receipt and a sync root (Manage sync folders).
    """
    snapshot = initial_lasting_sync_snapshot(lasting_available=True)
    snapshot = replace(snapshot, root_page_count=2)
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
        tree_selected_placement_id="note:work:n1:m1",
        tree_deleted_folder_available=True,
        import_receipt_available=True,
        lasting_sync_snapshot=snapshot,
        filter_value="plan",
    ) as pilot:
        await pilot.pause()
        actions = [
            button
            for button in pilot.app.query(Button)
            if button.id
            and button.id.startswith("library-notes-")
            and not button.id.startswith("library-notes-tree-folder")
            and not button.id.startswith("library-notes-tree-note")
            and not button.id.startswith("library-notes-tree-pager")
        ]
        assert len(actions) == NOTES_LIST_VISIBLE_ACTION_BUDGET, [
            button.id for button in actions
        ]
