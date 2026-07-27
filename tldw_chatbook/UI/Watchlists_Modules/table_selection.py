"""Shared rule for turning a `DataTable` cursor move into a selection.

TASK-1105. The Watchlists panes select on `RowHighlighted`/`CellHighlighted`
rather than on `RowSelected`/`CellSelected`, because Textual fires the latter
pair only on *activation* — Enter, or a second click on the cell the cursor is
already on — so a single click on any other row moved the cursor and selected
nothing.

Highlights are not all equal, though. A `DataTable` also announces a highlight
when it is first built: every one of these panes holds its rows in a
`reactive(..., recompose=True)`, so assigning `runs`/`items`/`rules`/
`notifications` constructs a brand new table whose cursor starts on row 0 and
says so. Treating that as a user action makes the pane fight its own screen:

* the runs deep link (`apply_navigation_context(run_id=...)`) selected the
  requested run, and the queued row-0 highlight from the list load then
  replaced it with whatever happened to be first;
* `_apply_tree_scope` clearing a pane's selection was undone by the rebuild
  that clearing itself triggered, so moving the tree could not deselect.

Focus is the honest discriminator. A mouse click focuses the table on
`MouseDown`, *before* the `Click` that moves the cursor is dispatched, and the
keyboard cannot move a cursor in a table that is not focused — so every
user-driven highlight arrives at a focused table. A table built by a recompose
has just been mounted and holds no focus (Textual does not carry focus across
the remove/remount a recompose performs), so its opening announcement is
filtered out here.

`SourcesPane` deliberately does NOT use this gate: TASK-1100 relies on the
row-0 highlight of a freshly populated sources table selecting the first
source, which is what arms `Preview`/`Check now` on arrival.
"""

from __future__ import annotations

from typing import Any


def highlight_is_user_driven(event: Any) -> bool:
    """Whether a `DataTable` highlight came from the user, not from a rebuild.

    Args:
        event: A `DataTable.RowHighlighted` or `DataTable.CellHighlighted`.

    Returns:
        True when the table that posted the highlight currently has focus,
        which is the case for every mouse- or keyboard-driven cursor move and
        is not the case for the highlight a newly mounted table announces.
    """
    table = getattr(event, "data_table", None)
    return bool(table is not None and table.has_focus)
