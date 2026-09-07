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

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IdSelectionModel:
    """Canonical-ID selection state for a filtered, ordered table."""

    _selected_ids: set[str] = field(default_factory=set)
    anchor_id: str | None = None
    visible_ids: tuple[str, ...] = ()

    @property
    def selected_ids(self) -> frozenset[str]:
        """Return the selected canonical IDs as an immutable value."""
        return frozenset(self._selected_ids)

    @property
    def hidden_count(self) -> int:
        """Return how many selected IDs are outside the current filter."""
        return len(self._selected_ids.difference(self.visible_ids))

    @property
    def status_text(self) -> str:
        """Return the non-color selection summary shown beside the table."""
        count = len(self._selected_ids)
        hidden = self.hidden_count
        return f"{count} selected · {hidden} hidden by filters"

    def set_visible_ids(self, source_ids: tuple[str, ...]) -> None:
        """Replace the current filtered/sorted ID order without selecting rows."""
        self.visible_ids = tuple(source_ids)

    def toggle(self, source_id: str) -> None:
        """Toggle one canonical ID and make it the range anchor."""
        if source_id in self._selected_ids:
            self._selected_ids.remove(source_id)
        else:
            self._selected_ids.add(source_id)
        self.anchor_id = source_id

    def shift(self, source_id: str, direction: int) -> str:
        """Move one visible row and replace the anchored contiguous range."""
        if direction not in {-1, 1} or source_id not in self.visible_ids:
            return source_id
        current_index = self.visible_ids.index(source_id)
        target_index = min(
            max(current_index + direction, 0), len(self.visible_ids) - 1
        )
        target_id = self.visible_ids[target_index]
        if self.anchor_id not in self.visible_ids:
            self.anchor_id = source_id
        anchor_index = self.visible_ids.index(self.anchor_id)
        old_start, old_end = sorted((anchor_index, current_index))
        new_start, new_end = sorted((anchor_index, target_index))
        self._selected_ids.difference_update(
            self.visible_ids[old_start : old_end + 1]
        )
        self._selected_ids.update(self.visible_ids[new_start : new_end + 1])
        return target_id

    def toggle_visible(self) -> None:
        """Select or clear visible IDs while preserving hidden selections."""
        visible = set(self.visible_ids)
        if visible and visible.issubset(self._selected_ids):
            self._selected_ids.difference_update(visible)
        else:
            self._selected_ids.update(visible)

    def clear(self) -> None:
        """Clear visible and hidden selections."""
        self._selected_ids.clear()
        self.anchor_id = None

    def replace(self, source_ids: tuple[str, ...]) -> None:
        """Replace the selection with canonical IDs supplied by its owner."""
        self._selected_ids = set(source_ids)
        self.anchor_id = None

    def prune(self, existing_ids: tuple[str, ...]) -> None:
        """Drop only IDs that no longer exist in the authoritative source set."""
        existing = set(existing_ids)
        self._selected_ids.intersection_update(existing)
        if self.anchor_id not in existing:
            self.anchor_id = None


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
