"""Library media Trash view: browsable, restorable trashed-media list.

task-4025: the third ``_library_media_view`` value ("trash") of the Browse ▸
Media canvas -- entered from the media list's "Trash" toolbar action, exited
via "‹ Media"/Escape. Lists every ``is_trash=1`` item (via the
``list_media_trash`` seam) and restores per item through the existing
``restore_media_item`` seam. Restore deliberately has NO receipt: ADR-055's
receipts accompany destruction, and restore is recovery -- its feedback is
the row leaving this list, the counts moving, and the transient notice line.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_media_state import (
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_ERROR_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_LOADING_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP,
    LibraryMediaTrashState,
)
from tldw_chatbook.Library.library_shell_state import (
    library_disabled_action_label,
)
from tldw_chatbook.Widgets.Library.library_rail import _visible_row_title
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


class LibraryMediaTrashCanvas(PostRecomposeCallback, RecomposeCaptureGuard, Vertical):
    """Render the Library media Trash list with per-item restore.

    Attributes:
        canvas: Current Trash view display state.
    """

    def __init__(
        self,
        canvas: LibraryMediaTrashState,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.canvas = canvas
        # Same width contract as ``LibraryMediaCanvas`` -- this view swaps
        # in for the list inside the same canvas host. 1fr (never 13fr):
        # see that canvas's constructor comment -- an independent 13fr
        # resolves ~13x wider than the host and clips children.
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def sync_state(self, canvas: LibraryMediaTrashState) -> None:
        """Refresh the canvas from new state.

        Args:
            canvas: Latest Trash view display state.

        Returns:
            None.
        """
        self.canvas = canvas
        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        """Render the heading, status/notice lines, trash rows, and Restore.

        Returns:
            ComposeResult for the Trash view.
        """
        # Heading: the "‹ <list>" back affordance + a titled count, the
        # same heading shape the note-load view uses ("‹ Notes" + title).
        heading = Horizontal(classes="ds-toolbar", id="library-media-trash-heading")
        heading.styles.height = "auto"
        with heading:
            yield Button(
                "‹ Media",
                id="library-media-trash-back",
                classes="library-canvas-action",
                compact=True,
            )
            yield Static(
                f"Trash ({self.canvas.count})",
                id="library-media-trash-title",
                classes="library-toolbar-count",
                markup=False,
            )

        # Restore feedback (never a receipt -- see the module docstring).
        notice = Static(
            self.canvas.notice,
            id="library-media-trash-notice",
            markup=False,
        )
        notice.display = bool(self.canvas.notice)
        yield notice

        # One status line, in honesty order: a fetch error outranks
        # everything; loading outranks the empty copy (an unloaded Trash
        # must never claim to be empty); then the truncation line or the
        # honest empty state.
        if self.canvas.error:
            status_text = self.canvas.error
        elif self.canvas.loading:
            status_text = "Loading Trash…"
        else:
            status_text = self.canvas.status_copy or self.canvas.empty_copy
        status = Static(
            status_text,
            id="library-media-trash-status",
            markup=False,
        )
        status.display = bool(status_text)
        yield status

        trash_list = Vertical(id="library-media-trash-list")
        # PR-1505 review (the L3a clipping lesson -- a plain auto-height
        # Vertical clips content past the fold, and a 200-item trash page
        # pushed the Restore toolbar ~100 rows off a 24-row terminal): the
        # list owns the remaining height between the heading/status above
        # and the Restore toolbar below, and scrolls its own overflow --
        # the same geometry `#library-media-list` gets from the stylesheet
        # in the wide split. Inline like the rest of this widget's
        # geometry; min_height 0 so the 1fr can actually shrink.
        trash_list.styles.height = "1fr"
        trash_list.styles.min_height = 0
        trash_list.styles.overflow_y = "auto"
        trash_list.styles.overflow_x = "hidden"
        with trash_list:
            for index, row in enumerate(self.canvas.rows):
                # Selected-row grammar: leading "▸ " (task-4023 AC#5).
                marker = "▸" if row.selected else " "
                label_rest = (
                    f" {_visible_row_title(row.title)}\n    {row.secondary}"
                )
                button = Button(
                    f"{marker}{label_rest}",
                    id=f"library-media-trash-row-{index}",
                    classes="library-media-trash-row",
                    compact=True,
                )
                button.media_id = row.media_id
                button._library_row_label_rest = label_rest
                # Tooltips render markup -- escape user titles.
                button.tooltip = escape_markup(row.title)
                # Reuses the media list's selected-row styling class -- one
                # CSS contract, not a fork.
                button.set_class(row.selected, "library-media-row-selected")
                button.styles.height = 2
                button.styles.min_height = 2
                yield button

        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            restore_disabled = self.canvas.loading or not self.canvas.rows
            restore = Button(
                # F-018 + task-4023 AC#1: disabled carries the non-colour
                # "○" marker and a reason tooltip.
                library_disabled_action_label("Restore", restore_disabled),
                id="library-media-trash-restore",
                classes="library-canvas-action",
                compact=True,
            )
            restore._library_disabled_marker_base = "Restore"
            restore.disabled = restore_disabled
            # Disabled reasons in the status line's own honesty order
            # (error > loading > empty): a failed fetch also leaves zero
            # rows, so without the error branch the tooltip claimed
            # "Nothing in Trash" for a Trash that merely could not be
            # read (PR-1505 review; F-018 demands the TRUE reason).
            if not restore_disabled:
                restore.tooltip = LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP
            elif self.canvas.error:
                restore.tooltip = LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_ERROR_TOOLTIP
            elif self.canvas.loading:
                restore.tooltip = LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_LOADING_TOOLTIP
            else:
                restore.tooltip = LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP
            yield restore
